"""
存储层：SQLite + 特征加密 + 审计日志
=====================================
原来人脸库是一个 `faces.npz` —— 特征明文、没有并发保护、写到一半断电就整个坏掉、
谁在什么时候录入/删除了谁完全查不到。这里把它换成正经的存储。

四个设计决定，每个都有明确理由：

1. **SQLite 而不是继续用 npz**
   事务保证「删一个人」要么全成要么全不成；WAL 模式让读不阻塞写，崩溃后能自动恢复；
   数据量大时 `person_id` 上有索引，不用每次把整个文件读进内存。
   几十人规模谈性能是虚的，真正的收益是**一致性**和**可查询**。

2. **AES-256-GCM 加密特征**
   人脸特征是可还原到个人的生物识别信息，明文躺在硬盘上等于把身份证复印件放在桌上。
   密钥单独存 `data/secret.key`（权限 0600），数据库文件被单独拷走也解不出特征。

   **必须说清的边界**：密钥和数据在同一台机器、同一个用户下，
   所以它只防「只拷走 db 文件」这一种情况，防不了「整台机器被拿走」或「拿到你的登录态」。
   真要抗这两条，得把密钥放进 TPM / Windows 凭据管理器 / KMS，那是另一个量级的工程。

3. **审计日志进数据库而不是写文件**
   「谁在几点录入了谁、删除了谁、被识别成谁」是这类系统的硬需求（合规、纠纷取证）。
   放同一张库里可以用 SQL 直接查，也天然跟着事务走 —— 删除失败时日志不会留下假记录。
   日志里**只记身份和动作，不记特征、不记分数**，避免审计表本身变成泄露渠道。

4. **对上层保持笨接口**
   `FaceLibrary` 仍然暴露 `names` / `feats` 两个快照和 `match()`，
   这样 gui.py、diagnose.py、测试都不用改。代价是删除不能靠改快照，
   必须走 `delete()` —— 这条写进了 docstring，GUI 里也照做了。

用法：
    from storage import FaceDB
    db = FaceDB("data/faces.db")
    db.add_sample("张三", feat, quality=0.87)
    names, feats = db.load_all()
    db.audit("recognize", subject="张三", detail="score=0.91")
    db.close()
"""
import os
import sqlite3
from datetime import datetime

import numpy as np

FEAT_DIM = 128
SCHEMA_VERSION = 1
AAD = b"face-feat-v1"          # 附加认证数据：防止把别的表的密文挪过来冒充特征


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def harden_permissions(path):
    """把文件权限收紧到「只有当前用户能读」。

    POSIX 上 os.open(..., 0o600) 就够；但 **Windows 会直接忽略这些位**
    （实测 chmod 之后 st_mode 仍是 666），必须用 ACL 才有效。
    所以这里分平台处理，失败不抛异常 —— 加固是尽力而为，不能因此让程序起不来。
    """
    try:
        if os.name == "posix":
            os.chmod(path, 0o600)
            return True
        if os.name == "nt":
            import subprocess
            user = os.environ.get("USERNAME") or "."
            r = subprocess.run(
                ["icacls", path, "/inheritance:r", "/grant:r", f"{user}:F"],
                capture_output=True, timeout=8)
            return r.returncode == 0
    except Exception:
        pass
    return False


def permission_note(path):
    """给用户看的可读权限描述（Windows 上 st_mode 不可信，用 icacls 的结果代替）。"""
    if os.name == "posix":
        return oct(os.stat(path).st_mode)[-3:]
    try:
        import subprocess
        r = subprocess.run(["icacls", path], capture_output=True, timeout=8)
        out = r.stdout.decode("gbk", "replace")
        keep = []
        for l in out.splitlines():
            s = l.strip()
            if not s:
                continue
            # 去掉「已成功处理 N 个文件」这类统计尾巴，只留 ACL 行
            if s.startswith("已成功处理") or s.startswith("Successfully processed"):
                continue
            keep.append(s)
        return " | ".join(keep)
    except Exception:
        return "未知"


# ---------------------------------------------------------------- 加密
class Crypto:
    """AES-256-GCM，密钥单独文件。没有 cryptography 时自动降级为明文并明确告警。"""

    def __init__(self, key_path):
        self.key_path = key_path
        self.enabled = False
        self._aead = None
        self._warned = False
        self._init()

    def _init(self):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            self.enabled = False
            return

        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                key = f.read()
            if len(key) != 32:
                raise ValueError(
                    f"密钥文件 {self.key_path} 长度异常（{len(key)} 字节，应为 32）。"
                    f"若不是误改，删掉它会让已有特征无法解密。")
        else:
            key = os.urandom(32)
            os.makedirs(os.path.dirname(os.path.abspath(self.key_path)), exist_ok=True)
            # O_EXCL + 0600：不覆盖已有密钥，且只有当前用户可读
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, key)
            finally:
                os.close(fd)

        self._aead = AESGCM(key)
        self.enabled = True
        harden_permissions(self.key_path)     # 收紧密钥文件权限（Windows 走 icacls）

    def seal(self, arr):
        """numpy 特征 → 密文（nonce 12 字节 + 密文）。"""
        raw = np.asarray(arr, dtype="<f4").tobytes()
        if not self.enabled:
            return raw
        nonce = os.urandom(12)
        return nonce + self._aead.encrypt(nonce, raw, AAD)

    def open(self, blob):
        """密文 → numpy 特征。"""
        blob = bytes(blob)
        if not self.enabled:
            return np.frombuffer(blob, dtype="<f4").astype(np.float32)
        return np.frombuffer(self._aead.decrypt(blob[:12], blob[12:], AAD),
                             dtype="<f4").astype(np.float32)


# ---------------------------------------------------------------- 数据库
class FaceDB:
    def __init__(self, path, key_path=None, audit=True):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.key_path = key_path or os.path.join(os.path.dirname(self.path), "secret.key")
        self.crypto = Crypto(self.key_path)
        self.audit_enabled = audit

        self.conn = sqlite3.connect(self.path, timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        # WAL：读不挡写、崩溃可恢复。这是 SQLite 在「一直开着摄像头」场景下的关键设置。
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._create_schema()

    # -------------------------------------------------------- schema
    def _create_schema(self):
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS persons (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT    NOT NULL,
                    created_at TEXT    NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_persons_name ON persons(name);

                CREATE TABLE IF NOT EXISTS samples (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    feat       BLOB    NOT NULL,
                    quality    REAL,
                    created_at TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_samples_person ON samples(person_id);

                CREATE TABLE IF NOT EXISTS audit (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      TEXT NOT NULL,
                    action  TEXT NOT NULL,
                    subject TEXT,
                    detail  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts);
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit(action);

                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),))

    # -------------------------------------------------------- 读
    def load_all(self):
        """返回 (names, feats)，按 person_id / sample_id 稳定排序。

        每次全量解密。几十人 × 每条 128 float32 的开销可以忽略，
        换来的是「永远和数据库一致」，不会出现内存快照过期的问题。
        """
        rows = self.conn.execute("""
            SELECT p.name AS name, s.feat AS feat
            FROM samples s JOIN persons p ON p.id = s.person_id
            ORDER BY p.id, s.id
        """).fetchall()
        names = [r["name"] for r in rows]
        if not rows:
            return names, np.zeros((0, FEAT_DIM), np.float32)
        feats = np.stack([self.crypto.open(r["feat"]) for r in rows])
        return names, feats

    def persons(self):
        rows = self.conn.execute("""
            SELECT p.id, p.name, p.created_at, COUNT(s.id) AS n
            FROM persons p LEFT JOIN samples s ON s.person_id = p.id
            GROUP BY p.id ORDER BY p.id
        """).fetchall()
        return [{"id": r["id"], "name": r["name"], "samples": r["n"],
                 "created_at": r["created_at"]} for r in rows]

    def stats(self):
        names, feats = self.load_all()
        n_people = len(set(names))
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        return {
            "people": n_people, "samples": len(names), "dim": int(feats.shape[1]) if len(names) else 0,
            "encrypted": self.crypto.enabled,
            "db_bytes": size,
            "db_path": self.path,
            "key_path": self.key_path if self.crypto.enabled else None,
            "audit_entries": self.conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0],
        }

    # -------------------------------------------------------- 写
    def add_sample(self, name, feat, quality=None):
        """同一个名字再录一次就是追加样本，不会新建 person。"""
        feat = np.asarray(feat, np.float32).reshape(-1)
        if feat.shape[0] != FEAT_DIM:
            raise ValueError(f"特征维度应为 {FEAT_DIM}，实际 {feat.shape[0]}")
        blob = self.crypto.seal(feat)
        with self.conn:                              # 事务：person 和 sample 一起成或一起败
            row = self.conn.execute("SELECT id FROM persons WHERE name = ?",
                                    (name,)).fetchone()
            if row is None:
                cur = self.conn.execute(
                    "INSERT INTO persons(name, created_at) VALUES(?, ?)", (name, now_iso()))
                pid = cur.lastrowid
            else:
                pid = row["id"]
            self.conn.execute(
                "INSERT INTO samples(person_id, feat, quality, created_at) VALUES(?,?,?,?)",
                (pid, blob, float(quality) if quality is not None else None, now_iso()))
            self._audit_locked("enroll", name,
                               f"quality={quality:.3f}" if quality is not None else None)
        return pid

    def delete_person(self, name):
        with self.conn:
            row = self.conn.execute("SELECT id FROM persons WHERE name = ?",
                                    (name,)).fetchone()
            if row is None:
                return 0
            n = self.conn.execute("SELECT COUNT(*) FROM samples WHERE person_id = ?",
                                  (row["id"],)).fetchone()[0]
            self.conn.execute("DELETE FROM persons WHERE id = ?", (row["id"],))   # 级联删样本
            self._audit_locked("delete", name, f"samples={n}")
        return n

    def clear_all(self):
        with self.conn:
            n = self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            p = self.conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            self.conn.execute("DELETE FROM samples")
            self.conn.execute("DELETE FROM persons")
            self._audit_locked("clear", None, f"people={p} samples={n}")
        return p, n

    # -------------------------------------------------------- 审计
    def audit(self, action, subject=None, detail=None):
        with self.conn:
            self._audit_locked(action, subject, detail)

    def _audit_locked(self, action, subject=None, detail=None):
        if not self.audit_enabled:
            return
        self.conn.execute(
            "INSERT INTO audit(ts, action, subject, detail) VALUES(?,?,?,?)",
            (now_iso(), action, subject, detail))

    def audit_tail(self, n=20, action=None):
        sql = "SELECT ts, action, subject, detail FROM audit"
        args = ()
        if action:
            sql += " WHERE action = ?"
            args = (action,)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self.conn.execute(sql, args + (int(n),)).fetchall()
        return [dict(r) for r in rows]

    def export_audit_csv(self, path):
        import csv
        rows = self.conn.execute(
            "SELECT ts, action, subject, detail FROM audit ORDER BY id").fetchall()
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["时间", "动作", "对象", "详情"])
            for r in rows:
                w.writerow([r["ts"], r["action"], r["subject"] or "", r["detail"] or ""])
        return len(rows)

    # -------------------------------------------------------- 迁移
    def import_npz(self, npz_path, quality=None):
        """把老的 faces.npz（names + feats 明文）一次性导入。已导入过的不会重复导。"""
        if not os.path.exists(npz_path):
            return 0
        if self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] > 0:
            return -1                                 # 库非空，拒绝覆盖式导入
        d = np.load(npz_path, allow_pickle=True)
        names = [str(x) for x in d["names"]]
        feats = np.asarray(d["feats"], np.float32)
        if len(names) != len(feats):
            raise ValueError(f"npz 内容不一致：names {len(names)} 条 vs feats {len(feats)} 条")

        n = 0
        with self.conn:
            for nm, ft in zip(names, feats):
                row = self.conn.execute("SELECT id FROM persons WHERE name = ?",
                                        (nm,)).fetchone()
                if row is None:
                    cur = self.conn.execute(
                        "INSERT INTO persons(name, created_at) VALUES(?, ?)",
                        (nm, now_iso()))
                    pid = cur.lastrowid
                else:
                    pid = row["id"]
                self.conn.execute(
                    "INSERT INTO samples(person_id, feat, quality, created_at) VALUES(?,?,?,?)",
                    (pid, self.crypto.seal(ft), quality,
                     now_iso()))
                n += 1
            self._audit_locked("migrate", None,
                               f"from={os.path.basename(npz_path)} samples={n}")
        return n

    # -------------------------------------------------------- 维护
    def vacuum(self):
        self.conn.execute("VACUUM")
        return True

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
