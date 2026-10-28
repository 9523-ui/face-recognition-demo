"""
摄像头人脸识别 demo
==================
基于 OpenCV 自带的 YuNet（检测）+ SFace（识别），无需 PyTorch / dlib / 深度学习框架。

用法（在 face_demo 目录下）:
    python face_demo.py                        实时识别（默认摄像头 0）
    python face_demo.py --enroll 张三          录入某人：窗口里按【空格】采样，按 q 结束
    python face_demo.py --enroll 张三 --image 照片.jpg     从图片录入
    python face_demo.py --list                 查看已录入的人
    python face_demo.py --clear                清空人脸库
    python face_demo.py --image 图.jpg         对静态图跑完整流程并保存标注图
    python face_demo.py --snapshot             从摄像头抓一帧跑一遍并保存（无需窗口）
    python face_demo.py --bench 100            跑 100 帧统计速度
    python face_demo.py --cam 1                换第二路摄像头

设计要点（面试可讲）:
  1. 检测每帧做、识别每 N 帧做 —— SFace 一次约 10ms，没必要每帧算，人在画面里不会突变。
  2. 多帧投票 —— 单帧识别一定会有闪变（转头、侧脸、光线），用滑动窗口取众数才稳定。
  3. 简易跟踪 —— 用框中心距离关联前后帧，保证"同一个人始终是同一个 track"，
     否则投票窗口会在不同人之间混，越投越错。
  4. 阈值判定 —— cosine 相似度 < 0.363 一律判"未知"，宁可认不出，也不能认错。
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

from config import S, get_logger
from storage import FaceDB

log = get_logger()

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
YUNET = os.path.join(MODELS, "yunet.onnx")
SFACE = os.path.join(MODELS, "sface.onnx")
DATA_DIR = os.path.join(BASE, "data")
DB_PATH = os.path.join(DATA_DIR, "faces.db")        # SQLite（原先是一个明文 faces.npz）
LEGACY_NPZ = os.path.join(DATA_DIR, "faces.npz")    # 老格式，首次运行会自动迁移
OUT_DIR = os.path.join(BASE, "out")

# 参数统一从 config.S 初始化（settings.json > 内置默认，环境变量可覆盖）。
# 改 settings.json 后重启程序生效 —— 刻意不做热加载，避免识别中途阈值突然变化。
DET_THRESH = float(S.get("det_thresh", 0.6))       # YuNet 检测置信度阈值
MATCH_THRESH = float(S.get("match_thresh", 0.45))  # 1:N 判定阈值（0.363 是 1:1 场景的，偏松）
MATCH_MARGIN = float(S.get("match_margin", 0.045)) # 第一名领先第二名不足此值时弃权
MIN_QUALITY = float(S.get("min_quality", 0.45))    # 质量分低于此值不采信（不投票、不入库）
VOTE_N = int(S.get("vote_n", 7))                   # 投票窗口长度（帧）
FEAT_AVG_N = int(S.get("feat_avg_n", 5))           # 特征滑动平均窗口（帧）
RECOG_EVERY = int(S.get("recog_every", 2))         # 每 N 帧做一次特征提取


LOST_TOLERANCE = 3        # 跟踪丢失容忍帧数


# ---------------------------------------------------------------- 模型封装
class FaceDetector:
    def __init__(self, model=YUNET, thresh=DET_THRESH):
        if not os.path.exists(model):
            raise FileNotFoundError(f"缺少检测模型: {model}")
        self.det = cv2.FaceDetectorYN.create(model, "", (320, 320), thresh, 0.3, 5000)
        self._size = None

    def __call__(self, img):
        h, w = img.shape[:2]
        if self._size != (w, h):
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.det.detect(img)
        return faces


class FaceEncoder:
    def __init__(self, model=SFACE):
        if not os.path.exists(model):
            raise FileNotFoundError(f"缺少识别模型: {model}")
        self.rec = cv2.FaceRecognizerSF.create(model, "")

    def encode(self, img, face_row):
        aligned = self.rec.alignCrop(img, face_row)
        feat = self.rec.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)


def face_quality(gray, face):
    """给这一帧的人脸打质量分（0~1），四项：

      尺寸  —— SFace 内部会把脸对齐到 112×112。脸在画面里只有 60px 宽时，
               这一步等于强行放大插值，纹理全糊，特征自然就飘。140px 以上才踏实。
      清晰度 —— 拉普拉斯方差。对焦不准、运动模糊、摄像头脏了都会掉。
      亮度   —— 平均灰度落在 60~200 之间最稳；过暗噪点大，过亮高光死白。
      正脸度 —— 拿鼻尖相对两眼中点的水平偏移除以眼距当 yaw 的近似。
               侧脸时这个比值会明显变大，5 个关键点也会跟着漂。

    返回 (总分, 明细 dict)。
    """
    h, w = gray.shape[:2]
    fx, fy, fw, fh = [float(v) for v in face[:4]]
    x0, y0 = max(0, int(fx)), max(0, int(fy))
    x1, y1 = min(w, int(fx + fw)), min(h, int(fy + fh))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return 0.0, {"尺寸": 0.0, "清晰": 0.0, "亮度": 0.0, "正脸": 0.0}
    roi = gray[y0:y1, x0:x1]

    size = float(np.clip((min(fw, fh) - 60) / 80.0, 0.0, 1.0))         # 60px → 0，140px → 1
    lap = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    sharp = float(np.clip((lap - 25.0) / 125.0, 0.0, 1.0))              # 25 → 0，150 → 1
    mean = float(roi.mean())
    if mean < 60:
        bright = float(np.clip((mean - 25.0) / 35.0, 0.0, 1.0))
    elif mean > 200:
        bright = float(np.clip((245.0 - mean) / 45.0, 0.0, 1.0))
    else:
        bright = 1.0

    eye_r, eye_l = float(face[4]), float(face[6])
    nose_x = float(face[8])
    eye_dist = abs(eye_l - eye_r) + 1e-6
    yaw = abs(nose_x - (eye_l + eye_r) / 2.0) / eye_dist
    front = float(np.clip(1.0 - yaw / 0.35, 0.0, 1.0))                  # yaw 比 0.35 以上算大侧脸

    total = 0.30 * size + 0.30 * sharp + 0.20 * bright + 0.20 * front
    # 亮度或清晰度任意一项塌到 0，说明这帧根本看不清 —— 尺寸再大、脸再正也不能算合格。
    # 用乘性惩罚而不是调权重，是因为「全黑图」靠尺寸和正脸也能拿到 0.5 分，加权平均压不下去。
    total *= (0.5 + 0.5 * min(bright, sharp))
    return total, {"尺寸": round(size, 3), "清晰": round(sharp, 3),
                   "亮度": round(bright, 3), "正脸": round(front, 3),
                   "lap": round(lap, 1), "px": int(min(fw, fh))}


class FaceLibrary:
    """人脸特征库。

    存储交给 `storage.FaceDB`（SQLite + AES-GCM 加密特征 + 审计日志），
    但对外仍然暴露 `names` / `feats` 两个**快照**和 `match()`，
    所以识别循环里的 `len(lib.names)`、`lib.match(feat)` 一行都不用改。

    ⚠️ `names` / `feats` 是只读快照 —— 直接改它们既不会写进数据库，也不会留审计。
       增删必须走 add() / delete() / clear()。
       （改造前 GUI 就是这么直接改快照的，那次改动一起修了。）
    """

    def __init__(self, path=None):
        is_default = path is None          # 只有用默认库时才自动去迁老的 npz
        # 兼容老调用：传 .npz 路径自动换成同名 .db，并把 npz 里的数据搬过来
        legacy = None
        if path and str(path).endswith(".npz"):
            legacy = str(path)
            path = legacy[:-4] + ".db"
        self.path = path or S.abspath("db_path")

        self.db = FaceDB(self.path)
        self.names, self.feats = [], np.zeros((0, 128), np.float32)

        if not self.db.crypto.enabled:
            log.warning("未安装 cryptography —— 特征将以明文存储")

        src = legacy or (LEGACY_NPZ if is_default else None)
        if src and os.path.exists(src):
            try:
                n = self.db.import_npz(src)
                if n > 0:
                    log.info("已把 %s 的 %d 条特征迁移进 SQLite（原文件保留未删）",
                             os.path.basename(src), n)
            except Exception as e:                       # 迁移失败不能拦着启动
                log.error("迁移 %s 失败：%s", src, e)
        self._load()

    # ------------------------------------------------------------ 内部
    def _load(self):
        self.names, self.feats = self.db.load_all()

    # ------------------------------------------------------------ 写
    def add(self, name, feat, quality=None):
        self.db.add_sample(name, feat, quality)
        self._load()

    def delete(self, name):
        """删掉某人的全部样本，返回删掉的样本条数。"""
        n = self.db.delete_person(name)
        self._load()
        return n

    def clear(self):
        p, s = self.db.clear_all()
        self._load()
        return p, s

    def save(self):
        """兼容老接口。数据在 add/delete/clear 时已经落盘，这里只刷新快照。"""
        self._load()

    def audit(self, action, subject=None, detail=None):
        self.db.audit(action, subject, detail)

    def audit_tail(self, n=20, action=None):
        return self.db.audit_tail(n, action)

    def stats(self):
        return self.db.stats()

    def close(self):
        self.db.close()

    # ------------------------------------------------------------ 判定
    def match(self, feat, margin=None):
        """返回 (名字, 最高相似度)。

        三道闸，都是为了防止「认错人」这种比认不出更糟的情况：
          1. 绝对阈值 —— 最高分不到阈值就是未知（阈值从 settings.json 读）
          2. 相对优势（ratio test）—— 第一名和第二名（不同人）咬得太近就弃权
          3. 只跟「别人」比 —— 同一个人的多条样本之间不互相竞争
        """
        if len(self.names) == 0:
            return "未注册", 0.0
        thr = float(S.get("match_thresh"))
        marg = float(S.get("match_margin")) if margin is None else float(margin)

        sims = self.feats @ feat
        order = np.argsort(-sims)
        i1 = int(order[0])
        s1 = float(sims[i1])
        n1 = self.names[i1]
        if s1 < thr:
            return "未知", s1

        s2 = -1.0
        for j in order[1:]:
            if self.names[int(j)] != n1:        # 同一个人的多条样本不算竞争
                s2 = float(sims[int(j)])
                break
        if s2 > 0 and s1 - s2 < marg and s2 > thr * 0.85:
            return "未知", s1                    # 两个人长得都像，不敢定
        return n1, s1

    def person_scores(self, feat, topn=5):
        """每个人取自己最高的那条样本分，返回前 topn 名 —— 给诊断工具看「差多少」。"""
        if len(self.names) == 0:
            return []
        sims = self.feats @ feat
        best = {}
        for i, n in enumerate(self.names):
            if n not in best or sims[i] > best[n]:
                best[n] = float(sims[i])
        return sorted(best.items(), key=lambda kv: -kv[1])[:topn]

    def count_by_name(self):
        out = {}
        for n in self.names:
            out[n] = out.get(n, 0) + 1
        return out


# ---------------------------------------------------------------- 跟踪 + 投票
class Track:
    def __init__(self, tid, box):
        self.id, self.box = tid, box
        self.votes = deque(maxlen=VOTE_N)
        self.feats = deque(maxlen=FEAT_AVG_N)      # 最近几帧的特征，用来平滑
        self.name, self.score = "未知", 0.0
        self.lost, self.idx = 0, None

    @property
    def center(self):
        x, y, w, h = self.box
        return x + w / 2, y + h / 2

    def push_feat(self, feat):
        """把这一帧的特征并进滑动窗口，返回窗口均值（已归一化）。

        单帧特征会被噪点、微小转头、自动曝光抖动带着跑；几帧平均把随机项压掉，
        留下的是这个人稳定的那部分。比「每帧各判一次再投票」更细腻 ——
        投票只能选，平均能减噪，两者可以叠加用。
        """
        self.feats.append(feat)
        if len(self.feats) == 1:
            return feat
        m = np.mean(self.feats, axis=0)
        return m / (np.linalg.norm(m) + 1e-9)

    def push_vote(self, name, score):
        self.votes.append(name)
        self.score = score
        if self.votes:                                  # 众数投票
            self.name = max(set(self.votes), key=self.votes.count)


class Tracker:
    def __init__(self, gate_ratio=0.25):
        self.tracks, self._id, self.gate_ratio = [], 0, gate_ratio

    def update(self, boxes):
        used = set()
        for t in self.tracks:
            gate = max(60.0, t.box[2] * self.gate_ratio + 40)
            best, bd = None, 1e9
            for i, b in enumerate(boxes):
                if i in used:
                    continue
                d = np.hypot(b[0] + b[2] / 2 - t.center[0], b[1] + b[3] / 2 - t.center[1])
                if d < bd:
                    best, bd = i, d
            if best is not None and bd < gate:
                used.add(best)
                t.box, t.lost, t.idx = boxes[best], 0, best
            else:
                t.lost += 1
                t.idx = None
        self.tracks = [t for t in self.tracks if t.lost <= LOST_TOLERANCE]
        for i, b in enumerate(boxes):
            if i not in used:
                self._id += 1
                t = Track(self._id, b)
                t.idx = i
                self.tracks.append(t)
        return self.tracks


# ---------------------------------------------------------------- 活体检测（眨眼）
EYE_KEYS = (4, 6)         # YuNet 关键点里左右眼中心的 x 下标（后面紧跟 y）
MIN_FACE_FOR_LIVENESS = 36


def eye_dark_ratio(gray, face):
    """眼区「暗像素占比」：睁眼时有虹膜/瞳孔/睫毛，比脸颊明显暗；闭眼被眼睑盖住，几乎全亮。

    用同一个人下半张脸（脸颊）的中位灰度当参考，而不是固定阈值，
    这样换个人、换光照都不用重新调参。
    返回 None 表示这块区域不可用（脸太小 / 太暗 / 出界），调用方应当跳过这一帧。
    """
    h, w = gray.shape[:2]
    fx, fy, fw, fh = [float(v) for v in face[:4]]
    if fw < MIN_FACE_FOR_LIVENESS or fh < MIN_FACE_FOR_LIVENESS:
        return None

    # 参考亮度取人脸下部中间（脸颊），避开眉毛、头发和背景
    cx0, cx1 = max(0, int(fx + 0.28 * fw)), min(w, int(fx + 0.72 * fw))
    cy0, cy1 = max(0, int(fy + 0.64 * fh)), min(h, int(fy + 0.94 * fh))
    cheek = gray[cy0:cy1, cx0:cx1]
    if cheek.size < 120:
        return None
    skin = float(np.median(cheek))
    if skin < 25:                       # 太暗，暗度阈值没有意义
        return None
    thr = skin * 0.62                   # 比皮肤暗 38% 以上算「暗」

    vals = []
    for k in EYE_KEYS:
        ex, ey = float(face[k]), float(face[k + 1])
        rw, rh = 0.44 * fw, 0.30 * fh
        x0, x1 = max(0, int(ex - rw / 2)), min(w, int(ex + rw / 2))
        y0, y1 = max(0, int(ey - rh / 2)), min(h, int(ey + rh / 2))
        if x1 - x0 < 8 or y1 - y0 < 6:
            continue
        roi = cv2.resize(gray[y0:y1, x0:x1], (40, 26), interpolation=cv2.INTER_AREA)
        vals.append(float((roi < thr).mean()))
    if not vals:
        return None
    return float(np.mean(vals))


class BlinkState:
    """单人份的眨眼状态机：攒一段历史 → 用 P90 当「睁眼水平」→ 看比例掉下去再回来。

    为什么不用固定阈值：暗度受光照、肤色、眼镜、刘海影响太大，绝对阈值换个场景就废。
    用「当前值 / 近 5 秒 P90」这个相对量，只要眼睛能闭上就能测出来。

    为什么照片骗不过：静态照片（不管睁眼还是闭眼）的比例恒在 1.0 附近，
    永远凑不出「睁 → 闭 → 睁」的完整状态转换，所以拿不到 blinks。
    """
    CLOSE_RATIO = 0.40        # 低于睁眼水平的 40% 判闭眼
    OPEN_RATIO = 0.70         # 回到 70% 以上判睁眼（中间是死区，防抖）
    WARMUP = 8                # 先攒几帧再判，否则基线不稳
    MIN_CLOSED = 2            # 闭眼至少 2 帧（30fps 下约 66ms）
    MAX_CLOSED = 22           # 超过 22 帧算「一直闭着」，不是眨眼
    COOLDOWN = 12             # 记一次后冷静 12 帧，避免同一个眨眼记成两次

    __slots__ = ("hist", "ref", "ratio", "blinks", "closed_run", "open_run",
                 "pending_closed", "cool", "n", "ok_frames")

    def __init__(self):
        self.hist = deque(maxlen=150)     # 5 秒 @30fps
        self.ref = 0.0                    # 睁眼参考水平（P90）
        self.ratio = 1.0                  # 当前值 / 参考水平
        self.blinks = 0
        self.closed_run = 0
        self.open_run = 0
        self.pending_closed = 0
        self.cool = 0
        self.n = 0
        self.ok_frames = 0

    @property
    def passed(self):
        return self.blinks >= 1

    @property
    def warming(self):
        return self.ok_frames < self.WARMUP

    def push(self, v):
        self.hist.append(v)
        self.n += 1
        self.ok_frames += 1
        if self.cool > 0:
            self.cool -= 1
        if self.warming:
            return
        # P90 作为睁眼参考；下限 0.008 防止全暗画面把参考压到 0 造成除零/爆表
        # 每 3 帧更新一次就够，省掉每帧排一次序
        if self.ref == 0.0 or self.n % 3 == 0:
            self.ref = max(float(np.percentile(self.hist, 90)), 0.008)
        self.ratio = v / self.ref

        if self.ratio < self.CLOSE_RATIO:
            self.closed_run += 1
            self.open_run = 0
        elif self.ratio > self.OPEN_RATIO:
            if self.open_run == 0 and self.closed_run:
                self.pending_closed = self.closed_run      # 记住这次闭了多久
            self.open_run += 1
            if (self.open_run >= 2 and self.cool == 0
                    and self.MIN_CLOSED <= self.pending_closed <= self.MAX_CLOSED):
                self.blinks += 1
                self.cool = self.COOLDOWN
                self.pending_closed = 0
            self.closed_run = 0
        # 死区（比例在 0.40~0.70）两个计数器都不动，避免抖动误判

    def label(self):
        if self.warming:
            return "校验中…"
        if self.passed:
            return "活体通过"
        return "请眨一下眼"


def update_liveness(tracks, gray, faces):
    """给每个 track 挂一个 BlinkState 并推进一帧。返回通过活体的人数。"""
    passed = 0
    for t in tracks:
        if t.idx is None or faces is None or t.idx >= len(faces):
            continue
        st = getattr(t, "blink", None)
        if st is None:
            st = BlinkState()
            t.blink = st
        d = eye_dark_ratio(gray, faces[t.idx])
        if d is not None:
            st.push(d)
        if st.passed:
            passed += 1
    return passed


def count_persons(faces, min_conf=0.62, min_side=36, area_ratio=0.25):
    """数「画面里有几个人」，顺带返回主脸（面积最大的那张）。

    过滤两道：置信度低的、太小的（背景海报/远处路人）不算人头；
    另外比主脸小太多的也不算，避免墙上一张照片触发多人告警。
    """
    if faces is None or len(faces) == 0:
        return 0, None
    usable = [f for f in faces
              if float(f[14]) >= min_conf and f[2] >= min_side and f[3] >= min_side]
    if not usable:
        return 0, None
    main = max(usable, key=lambda f: float(f[2]) * float(f[3]))
    a = float(main[2]) * float(main[3])
    n = sum(1 for f in usable if float(f[2]) * float(f[3]) >= a * area_ratio)
    return n, main


# ---------------------------------------------------------------- 中文绘制
_FONTS = {}
_FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc",
]


def _font(size):
    if size in _FONTS:
        return _FONTS[size]
    f = None
    if HAS_PIL:
        for p in _FONT_PATHS:
            if os.path.exists(p):
                try:
                    f = ImageFont.truetype(p, size)
                    break
                except Exception:
                    continue
    _FONTS[size] = f
    return f


def draw_labels(frame, items):
    """items: [(text, (x,y), (b,g,r), size)] —— 一次性过 PIL，避免反复转换。"""
    if not items:
        return frame
    f0 = _font(items[0][3])
    if f0 is None:                                     # 没字体就退回 cv2（中文会丢）
        for text, xy, color, size in items:
            cv2.putText(frame, text, (int(xy[0]), int(xy[1])), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 1, cv2.LINE_AA)
        return frame
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, xy, (b, g, r), size in items:
        f = _font(size) or f0
        bbox = d.textbbox((0, 0), text, font=f)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = int(xy[0]), int(xy[1])
        d.rectangle([x - 4, y - th - 7, x + tw + 5, y + 5], fill=(0, 0, 0))
        d.text((x, y - th - 5), text, font=f, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def annotate(frame, faces, tracks, fps=None, extra=None, require_liveness=False, warn=None):
    vis = frame.copy()
    labels = []
    # 人多时只给最大的几张脸写标签，否则文字会把画面糊住（合影场景实测 135 张脸）
    order = sorted(range(len(tracks)), key=lambda i: -tracks[i].box[2] * tracks[i].box[3])
    label_idx = set(order[:3]) if len(tracks) > 8 else set(range(len(tracks)))
    for i, t in enumerate(tracks):
        if t.idx is None or faces is None:
            continue
        x, y, w, h = [int(v) for v in faces[t.idx][:4]]
        known = t.name not in ("未知", "未注册")
        st = getattr(t, "blink", None)
        live = bool(st is not None and st.passed)
        q = getattr(t, "quality", None)
        if require_liveness and known and not live:
            # 脸是认识的，但还没眨眼 —— 可能是照片，先不给结论
            color = (60, 205, 245)
            text = f"{t.name}  待眨眼验证"
        else:
            color = (80, 220, 80) if known else (80, 160, 255)
            text = f"{t.name}  {t.score:.2f}" + ("  已验" if live else "")
        if q is not None:
            if q < MIN_QUALITY:
                color = (150, 150, 150)          # 灰框：这帧质量不够，结论暂不更新
                text += f"  画面不清 Q{q:.2f}"
            else:
                text += f"  Q{q:.2f}"
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        if i in label_idx:
            labels.append((text, (x, max(y - 8, 22)), color, 22))
    if fps is not None:
        labels.append((f"FPS {fps:.1f}", (12, 30), (240, 240, 240), 20))
    if warn:
        labels.append((warn, (12, 70), (60, 60, 255), 24))
    if extra:
        labels.append((extra, (12, vis.shape[0] - 14), (200, 200, 200), 17))
    return draw_labels(vis, labels)


# ---------------------------------------------------------------- 输入源
def open_camera(idx, width=640, height=480):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"打不开摄像头 {idx}（是否被其他程序占用？）")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # 减少缓冲，画面更跟手
    for _ in range(5):                            # 丢掉前几帧，等自动曝光稳定
        cap.read()
    return cap


def biggest_face(faces):
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


# ---------------------------------------------------------------- 各模式
def run_live(args, det, enc, lib):
    cap = open_camera(args.cam)
    tracker = Tracker()
    i, fps, t_last = 0, 0.0, time.perf_counter()
    hint = "空格=存图  q=退出" if args.snapshot_key else "q=退出"
    print(f"[实时识别] 活体={'开（需眨眼）' if args.liveness else '关'}  空格=截图存档  q=退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        faces = det(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = [] if faces is None else [f[:4] for f in faces]
        tracks = tracker.update(boxes)

        if args.liveness:
            update_liveness(tracks, gray, faces)

        if i % RECOG_EVERY == 0 and faces is not None:
            for t in tracks:
                if t.idx is None:
                    continue
                q, _ = face_quality(gray, faces[t.idx])
                t.quality = q
                if q < MIN_QUALITY:
                    continue                       # 画质太差，这一帧不采信
                name, score = lib.match(t.push_feat(enc.encode(frame, faces[t.idx])))
                t.push_vote(name, score)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_last, 1e-6))
        t_last = now

        n_person, _ = count_persons(faces)
        vis = annotate(frame, faces, tracks, fps,
                       f"{n_person} 人  |  {hint}", require_liveness=args.liveness)
        cv2.imshow("Face Demo - YuNet + SFace", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            os.makedirs(OUT_DIR, exist_ok=True)
            p = os.path.join(OUT_DIR, f"live_{time.strftime('%H%M%S')}.jpg")
            cv2.imwrite(p, vis)
            print("  已保存", p)
        i += 1
    cap.release()
    cv2.destroyAllWindows()


def run_enroll(args, det, enc, lib):
    name = args.enroll.strip()
    if not name:
        print("请用 --enroll 名字 指定要录入的人"); return

    if args.image:                                  # 从图片录入
        img = cv2.imread(args.image)
        if img is None:
            print("读不到图片:", args.image); return
        f = biggest_face(det(img))
        if f is None:
            print("这张图里没检测到人脸"); return
        lib.add(name, enc.encode(img, f))
        print(f"已从 {args.image} 录入「{name}」，当前库内 {len(lib.names)} 条特征")
        return

    cap = open_camera(args.cam)
    print(f"[录入「{name}」] 让脸正对镜头，按【空格】采样 {args.samples} 次，q 退出")
    print("              画面里同时出现多个人时会拒绝采样，请保持只有一个人")
    got = 0
    while got < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        faces = det(frame)
        n_person, main = count_persons(faces)
        f = main if main is not None else biggest_face(faces)
        vis = frame.copy()
        labels = [(f"录入: {name}  {got}/{args.samples}  空格=采样  q=退出", (12, 30), (240, 240, 240), 20)]
        if f is not None:
            x, y, w, h = [int(v) for v in f[:4]]
            color = (80, 220, 80) if n_person <= 1 else (60, 60, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        else:
            labels.append(("没检测到人脸，请正对镜头", (12, 60), (80, 160, 255), 20))
        if n_person > 1:
            labels.append((f"检测到 {n_person} 人：录入时请保持画面中只有一个人",
                           (12, 90), (60, 60, 255), 22))
        cv2.imshow("Enroll", draw_labels(vis, labels))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32 and f is not None:
            if n_person > 1:
                print(f"  拒绝采样：画面中有 {n_person} 个人，录入时请保持画面中只有一个人")
                time.sleep(0.15)
                continue
            lib.add(name, enc.encode(frame, f))
            got += 1
            print(f"  采样 {got}/{args.samples}")
            time.sleep(0.12)                        # 拉开一点间隔，避免采到几乎相同的帧
    cap.release()
    cv2.destroyAllWindows()
    print(f"完成：共 {got} 条，库内总计 {len(lib.names)} 条特征 / {len(set(lib.names))} 个人")


def run_image(args, det, enc, lib):
    img = cv2.imread(args.image)
    if img is None:
        print("读不到图片:", args.image); return
    t0 = time.perf_counter()
    faces = det(img)
    t_det = (time.perf_counter() - t0) * 1000
    n = 0 if faces is None else len(faces)
    print(f"图片 {img.shape[1]}x{img.shape[0]}  检出 {n} 张脸  检测耗时 {t_det:.1f} ms")

    tracker = Tracker()
    boxes = [] if faces is None else [f[:4] for f in faces]
    tracks = tracker.update(boxes)
    hits = {}
    for t in tracks:
        if t.idx is None:
            continue
        name, score = lib.match(enc.encode(img, faces[t.idx]))
        t.votes.append(name); t.name, t.score = name, score
        hits[name] = hits.get(name, 0) + 1
    if hits:
        print("识别结果:", "，".join(f"{k}×{v}" for k, v in sorted(hits.items(), key=lambda x: -x[1])))
    else:
        print("识别结果: （人脸库为空，先 --enroll 录入）")

    vis = annotate(img, faces, tracks, None, f"{n} face(s) | detect {t_det:.1f} ms")
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "image_" + os.path.splitext(os.path.basename(args.image))[0] + ".jpg")
    cv2.imwrite(p, vis)
    print("标注图已保存:", p)


def run_snapshot(args, det, enc, lib):
    cap = open_camera(args.cam)
    frame = None
    for _ in range(6):
        ok, frame = cap.read()
        if ok:
            break
        time.sleep(0.1)
    cap.release()
    if frame is None:
        print("抓帧失败"); return
    faces = det(frame)
    n = 0 if faces is None else len(faces)
    tracker = Tracker()
    tracks = tracker.update([] if faces is None else [f[:4] for f in faces])
    for t in tracks:
        if t.idx is not None:
            name, score = lib.match(enc.encode(frame, faces[t.idx]))
            t.votes.append(name); t.name, t.score = name, score
    print(f"抓帧 {frame.shape[1]}x{frame.shape[0]}  人脸 {n} 张")
    for t in tracks:
        print(f"  track#{t.id} {t.name} sim={t.score:.3f} box={[int(v) for v in t.box]}")
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "snapshot.jpg")
    cv2.imwrite(p, annotate(frame, faces, tracks, None, f"{n} face(s)"))
    print("已保存:", p)


def run_bench(args, det, enc, lib):
    cap = open_camera(args.cam)
    tracker = Tracker()
    frames, t_det, t_rec, t_all = [], [], [], time.perf_counter()
    face_frames = 0
    for i in range(args.bench):
        ok, frame = cap.read()
        if not ok:
            break
        a = time.perf_counter()
        faces = det(frame)
        b = time.perf_counter()
        t_det.append((b - a) * 1000)
        boxes = [] if faces is None else [f[:4] for f in faces]
        if faces is not None and len(faces):
            face_frames += 1
        tracks = tracker.update(boxes)
        if i % RECOG_EVERY == 0 and faces is not None:
            for t in tracks:
                if t.idx is not None:
                    c = time.perf_counter()
                    lib.match(enc.encode(frame, faces[t.idx]))
                    t_rec.append((time.perf_counter() - c) * 1000)
        frames.append(i)
    cap.release()
    total = time.perf_counter() - t_all
    if not frames:
        print("没有采到帧"); return
    print(f"帧数 {len(frames)}  总耗时 {total:.2f}s  端到端 {len(frames)/total:.1f} FPS")
    print(f"检测: 均值 {np.mean(t_det):.1f} ms  P95 {np.percentile(t_det,95):.1f} ms")
    if t_rec:
        print(f"识别: 均值 {np.mean(t_rec):.1f} ms  P95 {np.percentile(t_rec,95):.1f} ms  (每 {RECOG_EVERY} 帧一次)")
    else:
        print("识别: 本次画面内没有人脸，未采样")
    print(f"含人脸的帧: {face_frames}/{len(frames)}")


def run_list(lib):
    st = lib.stats()
    if not lib.names:
        print("人脸库是空的，先跑 --enroll 名字 录入")
        print(f"  库文件: {st['db_path']}")
        return
    print(f"人脸库: {st['people']} 个人 / {st['samples']} 条特征")
    for n, c in lib.count_by_name().items():
        print(f"  {n}  ({c} 条)")
    print(f"\n库文件   {st['db_path']}  ({st['db_bytes'] / 1024:.1f} KB)")
    print(f"特征加密 {'是（AES-256-GCM）' if st['encrypted'] else '否（未装 cryptography）'}")
    print(f"当前阈值 {S.get('match_thresh')}   margin {S.get('match_margin')}"
          f"   （来源：{S.origin('match_thresh')}）")
    print(f"审计记录 {st['audit_entries']} 条")


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser(description="摄像头人脸识别 demo (YuNet + SFace)")
    ap.add_argument("--cam", type=int, default=0, help="摄像头编号，默认 0")
    ap.add_argument("--enroll", metavar="名字", help="录入某人")
    ap.add_argument("--samples", type=int, default=8, help="录入时采样次数，默认 8")
    ap.add_argument("--image", help="对静态图片处理 / 从图片录入")
    ap.add_argument("--list", action="store_true", help="查看人脸库")
    ap.add_argument("--clear", action="store_true", help="清空人脸库")
    ap.add_argument("--snapshot", action="store_true", help="抓一帧跑一遍并保存")
    ap.add_argument("--bench", type=int, metavar="N", help="跑 N 帧统计性能")
    ap.add_argument("--mirror", action="store_true", default=True, help="画面镜像（默认开）")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--liveness", action="store_true", default=True,
                    help="开启活体检测（需眨眼，默认开）")
    ap.add_argument("--no-liveness", dest="liveness", action="store_false",
                    help="关闭活体检测，认得就直接显示名字")
    ap.add_argument("--snapshot-key", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    for p in (YUNET, SFACE):
        if not os.path.exists(p):
            print(f"缺少模型文件: {p}")
            print("下载地址见 README.md，或运行: python download_models.py")
            return 1

    lib = FaceLibrary()
    if args.clear:
        # 不再删文件：SQLite 删库会连带丢掉审计记录，还会留下 -wal/-shm 孤儿文件。
        # 清空数据但保留库结构和审计，这样「谁清空过」还查得到。
        p, s = lib.clear()
        print(f"人脸库已清空：{p} 人 / {s} 条样本（审计记录保留）")
        return 0
    if args.list:
        run_list(lib)
        return 0

    det, enc = FaceDetector(), FaceEncoder()
    if args.enroll:
        run_enroll(args, det, enc, lib)
    elif args.image:
        run_image(args, det, enc, lib)
    elif args.snapshot:
        run_snapshot(args, det, enc, lib)
    elif args.bench:
        run_bench(args, det, enc, lib)
    else:
        run_live(args, det, enc, lib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
