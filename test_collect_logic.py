"""
采集流程的分支测试（不需要摄像头）
==================================
把 FaceApp._collect_step 单独拎出来跑，验证四件事：

  1. 画面里出现 2 个人 → 立刻暂停、不采样、给出「录入时请保持画面中只有一个人」
  2. 恢复单人 → 告警清除、采集继续
  3. 一直没人脸 → 超时后自动取消
  4. 活体没通过 → 不采样；通过后才开始采满 8 帧

运行：
    python test_collect_logic.py
"""
import sys

import cv2
import numpy as np

import gui
from face_demo import BlinkState, face_quality

FAILED = []


class Widget:
    """最小化的 Tk 控件替身，只记录 config 过的属性。"""

    def __init__(self):
        self.v = {}

    def cget(self, k):
        return self.v.get(k, "")

    def config(self, **kw):
        self.v.update(kw)


class FakeEncoder:
    def __init__(self):
        self.rng = np.random.default_rng(7)

    def encode(self, frame, face):
        v = self.rng.random(128).astype(np.float32)
        return v / np.linalg.norm(v)


class Harness(gui.FaceApp):
    """绕过 __init__（不开口摄像头、不建窗口），只装 _collect_step 需要的状态。"""

    def __init__(self, live=True):
        self.samples = []
        self._since_sample = 0
        self._miss = 0
        self._multi_on = False
        self._warn = ""
        self._banner = ""
        self.mode = "collect"
        self._pending_name = "测试"
        self.hint = Widget()
        self.status = Widget()
        self.enc = FakeEncoder()
        self.saved = []
        self.lib = type("L", (), {})()
        self.lib.names = []
        self._sample_q = []
        self.lib.add = lambda n, f, quality=None: self.saved.append((n, f))
        self.lib.count_by_name = lambda: {n: 1 for n, _ in self.saved}
        self.lib.save = lambda: None
        self.root = type("R", (), {"bell": lambda s: None})()
        self.live_var = type("V", (), {"get": lambda s: live})()
        self.cancelled = None

    # --- 覆写掉需要真实控件的部分（finish_collect 用父类的，走真实入库逻辑）---
    def _reset_add_button(self):
        pass

    def refresh_list(self):
        pass

    def cancel_collect(self, why):
        self.cancelled = why
        self.mode = "live"


def fake_faces(n, w=160, h=190):
    """造 n 个人脸行（15 列：x,y,w,h + 5 组关键点 + score），面积依次递减。"""
    rows = []
    for i in range(n):
        rows.append([40 + i * 220, 100, w - i * 30, h - i * 30,
                     70, 150, 140, 150, 105, 180, 80, 210, 130, 210, 0.9 - i * 0.3])
    return np.array(rows, np.float32)


def fake_tracks(faces, passed=False):
    out = []
    for i in range(len(faces)):
        t = type("T", (), {})()
        t.idx = i
        st = BlinkState()
        st.ok_frames = 60
        st.blinks = 1 if passed else 0
        t.blink = st
        out.append(t)
    return out


def check(name, cond, detail=""):
    print(f"    {'OK  ' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


def main():
    # 有纹理的假画面：质量分能过门限；全黑画面用来测「质量不够就不收」
    rng = np.random.default_rng(3)
    frame = rng.integers(60, 190, (480, 640, 3), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = np.zeros((480, 640, 3), np.uint8)
    dark_gray = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)

    # ---------- 1. 多人 → 暂停 ----------
    print("\n[1] 画面里同时出现 2 个人")
    h = Harness()
    f2 = fake_faces(2)
    tr = fake_tracks(f2, passed=True)
    for _ in range(10):
        h._collect_step(frame, gray, f2, tr, len(f2))
    check("没有采样（samples 为空）", len(h.samples) == 0, f"samples={len(h.samples)}")
    check("画面红色告警已设置", h._warn == "录入时请保持画面中只有一个人", repr(h._warn))
    check("状态栏提示正确", "只有一个人" in h.status.cget("text"), repr(h.status.cget("text")))
    check("标记为多人暂停", h._multi_on is True)
    check("不会因为等太久而取消", h.cancelled is None)

    # ---------- 2. 恢复单人 → 继续 ----------
    print("\n[2] 另一个人离开，画面只剩 1 人")
    f1 = fake_faces(1)
    tr1 = fake_tracks(f1, passed=True)
    h._since_sample = 0
    for _ in range(200):
        h._collect_step(frame, gray, f1, tr1, 1)
        if h.mode == "live":
            break
    check("告警已清除", h._warn == "", repr(h._warn))
    check("多人标记已复位", h._multi_on is False)
    check("采集继续并采满 8 帧入库", len(h.saved) == 1 and h.mode == "live",
          f"入库={len(h.saved)} mode={h.mode}")

    # ---------- 3. 一直没人脸 → 超时取消 ----------
    print("\n[3] 画面里一直没有脸")
    h3 = Harness()
    empty = np.zeros((0, 15), np.float32)
    for _ in range(gui.COLLECT_TIMEOUT + 5):
        h3._collect_step(frame, gray, empty, [], 0)
        if h3.mode == "live":
            break
    check("超时后自动取消", h3.cancelled is not None and h3.mode == "live", repr(h3.cancelled))
    check("取消原因提到没检测到人脸", "没检测到人脸" in (h3.cancelled or ""), repr(h3.cancelled))

    # ---------- 4. 活体未通过 → 不采样 ----------
    print("\n[4] 活体校验没通过（没眨眼）")
    h4 = Harness(live=True)
    tr_pending = fake_tracks(f1, passed=False)
    for _ in range(150):
        h4._collect_step(frame, gray, f1, tr_pending, 1)
    check("一帧都没采", len(h4.samples) == 0, f"samples={len(h4.samples)}")
    check("画面提示要眨眼", "眨" in h4._banner, repr(h4._banner))
    check("状态栏提活体校验", "活体" in h4.status.cget("text"), repr(h4.status.cget("text")))

    # ---------- 5. 活体通过 → 采满 ----------
    print("\n[5] 眨过眼之后")
    tr_passed = fake_tracks(f1, passed=True)
    for _ in range(200):
        h4._collect_step(frame, gray, f1, tr_passed, 1)
        if h4.mode == "live":
            break
    check("采满 8 帧并入库", len(h4.saved) == 1, f"入库={len(h4.saved)}")
    check("入库样本是 128 维", len(h4.saved[0][1]) == 128 if h4.saved else False)
    check("入库名字正确", h4.saved[0][0] == "测试" if h4.saved else False)

    # ---------- 6. 关掉活体开关 → 无需眨眼 ----------
    print("\n[6] 关掉活体开关")
    h6 = Harness(live=False)
    for _ in range(200):
        h6._collect_step(frame, gray, f1, tr_pending, 1)   # 同一个「没眨眼」的人
        if h6.mode == "live":
            break
    check("不拦，直接采满入库", len(h6.saved) == 1, f"入库={len(h6.saved)}")

    # ---------- 7. 画质不够的帧不收 ----------
    print("\n[7] 画面全黑（画质不达标）")
    h7 = Harness(live=False)
    for _ in range(120):
        h7._collect_step(dark, dark_gray, f1, tr1, 1)
    check("一帧都没采", len(h7.samples) == 0, f"samples={len(h7.samples)}")
    msg = h7.status.cget("text") + " " + h7._banner + " " + h7.hint.cget("text")
    check("提示指向光线问题", ("开灯" in msg or "光线" in msg) and "质量" in msg,
          repr(msg[:50]))
    check("没有把低质量帧算进样本", h7.samples == [])

    print("\n" + "=" * 60)
    print("结论：" + ("全部通过" if not FAILED else f"★失败 {len(FAILED)} 项：{FAILED}"))
    print("=" * 60)
    return 0 if not FAILED else 2


if __name__ == "__main__":
    sys.exit(main())
