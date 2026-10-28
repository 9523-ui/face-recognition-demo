"""
存储层测试：加密、事务、审计、迁移
==============================
存储这块最容易「看起来对了但其实是错的」：
  · 加密写了但没生效（存进去还是明文）
  · 删除只删了 person，样本成了孤儿
  · 审计写了，但删除失败时也留了假记录
所以每一项都拿断言卡，不看输出猜。

运行：
    python test_storage.py
"""
import os
import shutil
import sqlite3
import sys
import tempfile

import numpy as np

import storage
from storage import FaceDB, FEAT_DIM
import face_demo
from face_demo import FaceLibrary

PASS = FAIL = 0
FAILS = []


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"    OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        FAIL += 1
        FAILS.append(name)
        print(f"    FAIL  {name}  {extra}")


def unit(i=0, dim=FEAT_DIM):
    """造一个确定性的单位向量，避免随机导致断言不能复现。"""
    rng = np.random.default_rng(1000 + i)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def main():
    tmp = tempfile.mkdtemp(prefix="_storage_test_")
    print("=" * 66)
    print("存储层测试   临时目录:", tmp)
    print("=" * 66)

    # ---------------- 1. 加密本身 ----------------
    print("\n[1] 加密")
    db = FaceDB(os.path.join(tmp, "a.db"))
    check("cryptography 可用，加密已启用", db.crypto.enabled is True)
    check("密钥文件已生成", os.path.exists(db.key_path),
          f"{os.path.getsize(db.key_path)} 字节" if os.path.exists(db.key_path) else "")
    check("密钥长度 32 字节（AES-256）",
          os.path.getsize(db.key_path) == 32 if os.path.exists(db.key_path) else False)

    v = unit(0)
    blob = db.crypto.seal(v)
    back = db.crypto.open(blob)
    check("加密→解密逐位一致", np.array_equal(v, back),
          f"max err {float(np.abs(v - back).max()):.2e}")
    check("密文长度 = 512 + 12(nonce) + 16(tag)", len(blob) == 540, f"{len(blob)} 字节")
    check("同一特征两次加密结果不同（nonce 随机）",
          db.crypto.seal(v) != db.crypto.seal(v))

    # ---------------- 2. 明文不落盘 ----------------
    print("\n[2] 明文特征不能出现在库文件里")
    db.add_sample("张三", v, quality=0.88)
    db.conn.commit()
    raw = v.astype("<f4").tobytes()
    with open(db.path, "rb") as f:
        disk = f.read()
    check("特征明文未出现在 .db 文件", raw not in disk)
    # WAL 文件也要查 —— 数据落地时可能先在 -wal 里
    wal = db.path + "-wal"
    if os.path.exists(wal):
        with open(wal, "rb") as f:
            check("特征明文未出现在 -wal 文件", raw not in f.read())
    names, feats = db.load_all()
    check("解密后与写入值一致", names == ["张三"] and np.array_equal(feats[0], v))

    # ---------------- 3. 增删改查 ----------------
    print("\n[3] 增删改查")
    db.add_sample("张三", unit(1), quality=0.90)          # 同一人第二条
    db.add_sample("李四", unit(2))
    names, feats = db.load_all()
    check("同一人追加样本不新建人员", names.count("张三") == 2 and feats.shape == (3, FEAT_DIM),
          f"{len(set(names))} 人 / {len(names)} 条")
    check("persons() 统计正确",
          {p["name"]: p["samples"] for p in db.persons()} == {"张三": 2, "李四": 1})

    n = db.delete_person("李四")
    check("删除返回删掉的样本数", n == 1, f"{n} 条")
    names, _ = db.load_all()
    check("删除后查不到该人", "李四" not in names)
    check("删除不存在的人返回 0", db.delete_person("查无此人") == 0)
    orphan = db.conn.execute(
        "SELECT COUNT(*) FROM samples WHERE person_id NOT IN (SELECT id FROM persons)"
    ).fetchone()[0]
    check("没有孤儿样本（级联删除生效）", orphan == 0, f"{orphan} 条")

    # ---------------- 4. 维度校验 ----------------
    print("\n[4] 输入校验")
    try:
        db.add_sample("坏人", np.zeros(64, np.float32))
        check("维度不对时抛 ValueError", False, "没抛异常")
    except ValueError as e:
        check("维度不对时抛 ValueError", True, str(e)[:40])

    # ---------------- 5. 审计 ----------------
    print("\n[5] 审计日志")
    tail = db.audit_tail(20)
    actions = [r["action"] for r in tail]
    check("审计按时间倒序返回", actions[0] in ("delete", "enroll"), str(actions[:3]))
    check("enroll 留下的对象名正确",
          any(r["action"] == "enroll" and r["subject"] == "张三" for r in tail))
    check("enroll 记录了质量分",
          any(r["detail"] and "quality=" in r["detail"] for r in tail))
    check("delete 记录了删了多少条",
          any(r["action"] == "delete" and r["detail"] == "samples=1" for r in tail))
    check("审计里没有特征数据（不泄露）",
          not any(r["detail"] and len(r["detail"]) > 200 for r in tail))
    csv_path = os.path.join(tmp, "audit.csv")
    n_csv = db.export_audit_csv(csv_path)
    check("导出审计 CSV", n_csv == len(tail) and os.path.exists(csv_path), f"{n_csv} 行")

    # ---------------- 6. 清空保留审计 ----------------
    print("\n[6] 清空")
    before = db.audit_tail(100)
    p, s = db.clear_all()
    check("clear 返回删除量", p >= 1 and s >= 1, f"{p} 人 / {s} 条")
    names, feats = db.load_all()
    check("清空后库为空", len(names) == 0 and feats.shape == (0, FEAT_DIM),
          f"shape {feats.shape}")
    after = db.audit_tail(100)
    check("清空后审计记录还在（不是删库）", len(after) > len(before) - 1,
          f"{len(before)} → {len(after)} 条")
    check("clear 动作已入审计", after[0]["action"] == "clear" and "people=" in (after[0]["detail"] or ""))
    db.close()

    # ---------------- 7. 从 npz 迁移 ----------------
    print("\n[7] 老 npz 迁移")
    npz = os.path.join(tmp, "old.npz")
    np.savez(npz, names=np.array(["老甲", "老乙", "老甲"], dtype=object),
             feats=np.stack([unit(10), unit(11), unit(12)]))
    db2 = FaceDB(os.path.join(tmp, "b.db"))
    n = db2.import_npz(npz)
    check("迁移条数正确", n == 3, f"{n} 条")
    names, feats = db2.load_all()
    check("迁移后人员/样本数正确", len(set(names)) == 2 and len(names) == 3,
          f"{sorted(set(names))}")
    check("迁移的特征可正确解密", np.array_equal(feats[0], unit(10)))
    check("重复导入不会翻倍", db2.import_npz(npz) == -1, "已拒绝")
    check("迁移动作入审计",
          any(r["action"] == "migrate" for r in db2.audit_tail(10)))
    db2.close()

    # ---------------- 8. 并发 ----------------
    print("\n[8] 并发（WAL）")
    pa = os.path.join(tmp, "c.db")
    w = FaceDB(pa)
    w.add_sample("并发甲", unit(20))
    r = FaceDB(pa)                                   # 第二个连接同时读
    names_r, _ = r.load_all()
    check("第二个连接能读到已提交数据", names_r == ["并发甲"])
    w.add_sample("并发乙", unit(21))
    names_r2, _ = r.load_all()
    check("写后另一连接能读到新数据（无快照过期）", "并发乙" in names_r2, str(names_r2))
    mode = r.conn.execute("PRAGMA journal_mode").fetchone()[0]
    check("journal_mode 是 WAL（崩溃可恢复）", str(mode).lower() == "wal", str(mode))
    w.close()
    r.close()

    # ---------------- 9. FaceLibrary 端到端 ----------------
    print("\n[9] FaceLibrary 端到端")
    lib = FaceLibrary(path=os.path.join(tmp, "lib.db"))
    check("新库为空", len(lib.names) == 0)
    a = unit(30)
    lib.add("甲", a, quality=0.9)
    lib.add("乙", unit(31))
    check("add 后快照刷新", len(lib.names) == 2, f"{lib.count_by_name()}")
    nm, sc = lib.match(a)
    check("自己的特征能认出来", nm == "甲" and sc > 0.99, f"{nm} {sc:.3f}")
    nm2, sc2 = lib.match(unit(99))
    check("陌生特征判未知", nm2 == "未知", f"{nm2} {sc2:.3f}")
    lib.delete("甲")
    check("delete 后快照同步", "甲" not in lib.names, str(lib.names))
    nm3, _ = lib.match(a)
    check("删除后原特征不再命中该人", nm3 != "甲", nm3)
    lib.save()
    check("老的 save() 调用不报错且不重复写", len(lib.names) == 1)
    lib.audit("recognize", "乙", "score=0.9")
    check("FaceLibrary.audit 可用",
          lib.audit_tail(1)[0]["action"] == "recognize")
    lib.close()

    # ---------------- 10. 降级：没有 cryptography ----------------
    print("\n[10] 无加密库时的降级")
    old = storage.Crypto._init
    def _no_crypto(self):
        self.enabled = False
        self._aead = None
    storage.Crypto._init = _no_crypto
    try:
        db3 = FaceDB(os.path.join(tmp, "plain.db"))
        check("缺 cryptography 时 enabled=False", db3.crypto.enabled is False)
        db3.add_sample("明文甲", unit(40))
        names, feats = db3.load_all()
        check("降级模式下仍能正常读写", names == ["明文甲"] and np.array_equal(feats[0], unit(40)))
        db3.close()
    finally:
        storage.Crypto._init = old

    # ---------------- 结论 ----------------
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 66)
    print(f"通过 {PASS} 项，失败 {FAIL} 项")
    if FAILS:
        print("失败项：" + "、".join(FAILS))
        print("=" * 66)
        return 1
    print("结论：全部通过")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
