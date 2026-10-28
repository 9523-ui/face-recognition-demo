"""
活体（眨眼）与人数校验的量化验证
=================================
不接摄像头，用真实人脸图（从公开合影里裁出的单人样本）+ 人工合成的「闭眼帧」，
把几个关键指标跑出来：

  1. eye_dark_ratio 在睁眼 / 闭眼两态下差多少（这是眨眼判定的物理依据）
  2. BlinkState 只在「睁→闭→睁」时计一次眨眼；单帧抖动、连续睁眼、连续闭眼都不该计
  3. 照片攻击：连续喂同一张睁眼图 / 闭眼图，必须永远拿不到 blink
  4. count_persons 数人数对不对（单人 / 双人 / 角落里的小脸）
  5. 单帧耗时，确认加了活体不掉帧

运行：
    python test_liveness.py
"""
import os
import sys
import time

import cv2
import numpy as np

from face_demo import BlinkState, FaceDetector, count_persons, eye_dark_ratio

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_IMG = os.path.join(BASE, "testdata", "selfie.jpg")


def crop_single(img, faces, pad=1.5):
    """从合影里裁出一张单人样本：以最大脸为中心外扩 pad 倍，缩放到 480 宽。"""
    f = max(faces, key=lambda z: z[2] * z[3])
    x, y, w, h = [float(v) for v in f[:4]]
    cx, cy = x + w / 2, y + h / 2
    W, H = w * pad, h * pad * 1.25
    x0, y0 = int(max(0, cx - W / 2)), int(max(0, cy - H / 2))
    x1, y1 = int(min(img.shape[1], cx + W / 2)), int(min(img.shape[0], cy + H / 2))
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (480, int(480 * crop.shape[0] / crop.shape[1])),
                      interpolation=cv2.INTER_CUBIC)


def make_closed(img, face, eyelash=False):
    """把眼区用脸颊皮肤盖掉，模拟闭眼（真实闭眼就是眼睑=皮肤）。

    eyelash=True 时额外在眼缝位置压一条暗线，模拟睫毛阴影 —— 用来验证
    「闭眼不是纯粹的 0，带点残差也照样能判出来」。
    """
    out = img.copy()
    h, w = out.shape[:2]
    fx, fy, fw, fh = [float(v) for v in face[:4]]
    cx0, cx1 = max(0, int(fx + 0.28 * fw)), min(w, int(fx + 0.72 * fw))
    cy0, cy1 = max(0, int(fy + 0.66 * fh)), min(h, int(fy + 0.90 * fh))
    patch = out[cy0:cy1, cx0:cx1]
    if patch.size < 100:
        return out
    patch = cv2.GaussianBlur(patch, (0, 0), 3)

    for k in (4, 6):
        ex, ey = float(face[k]), float(face[k + 1])
        rw, rh = 0.46 * fw, 0.32 * fh
        x0, x1 = max(0, int(ex - rw / 2)), min(w, int(ex + rw / 2))
        y0, y1 = max(0, int(ey - rh / 2)), min(h, int(ey + rh / 2))
        if x1 - x0 < 6 or y1 - y0 < 5:
            continue
        out[y0:y1, x0:x1] = cv2.resize(patch, (x1 - x0, y1 - y0))
        if eyelash:                                  # 睫毛阴影
            shade = int(np.clip(np.median(patch) * 0.45, 8, 120))
            cv2.line(out, (x0 + 2, int(ey)), (x1 - 2, int(ey)), (shade, shade, shade), 2)
    return cv2.GaussianBlur(out, (3, 3), 0.6)


def feed(blink, value, n):
    for _ in range(n):
        blink.push(float(value))


def main():
    line = "=" * 68
    print(line)
    if not os.path.exists(TEST_IMG):
        print("缺少测试图:", TEST_IMG)
        return 1

    src = cv2.imread(TEST_IMG)
    det = FaceDetector()
    all_faces = det(src)
    if all_faces is None or len(all_faces) == 0:
        print("测试图里没检出人脸")
        return 1

    img = crop_single(src, all_faces)                # 单人样本
    faces = det(img)
    n_face = 0 if faces is None else len(faces)
    if n_face == 0:
        print("裁剪出的单人样本检不到人脸")
        return 1
    face = max(faces, key=lambda z: float(z[2]) * float(z[3]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"单人样本 {img.shape[1]}x{img.shape[0]}  检出 {n_face} 张脸  "
          f"主脸 {int(face[2])}x{int(face[3])} px  conf {float(face[14]):.3f}")

    # ---------------- 1. 单帧特征：睁眼 vs 闭眼 ----------------
    closed = make_closed(img, face)
    closed_lash = make_closed(img, face, eyelash=True)
    op = eye_dark_ratio(gray, face)
    cl = eye_dark_ratio(cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY), face)
    cl2 = eye_dark_ratio(cv2.cvtColor(closed_lash, cv2.COLOR_BGR2GRAY), face)
    print("\n[1] 眼区暗度（越大越「睁」）")
    print(f"    睁眼                {op:.4f}")
    print(f"    闭眼（纯皮肤）       {cl:.4f}")
    print(f"    闭眼（带睫毛阴影）   {cl2:.4f}")
    print(f"    闭/睁 比例          纯皮肤 {cl / op:.3f}   带睫毛 {cl2 / op:.3f}"
          f"   （判定线 {BlinkState.CLOSE_RATIO:.2f}）")

    # ---------------- 2. 状态机 ----------------
    print("\n[2] 状态机：睁 20 帧 → 闭 4 帧 → 睁 15 帧")
    b = BlinkState()
    feed(b, op, 20)
    r_open = b.ratio
    feed(b, cl2, 4)
    r_close = b.ratio
    feed(b, op, 15)
    print(f"    眨眼计数 {b.blinks}   {'通过' if b.blinks == 1 else '失败'}")
    print(f"    睁眼比例 {r_open:.3f}（要 > {BlinkState.OPEN_RATIO:.2f}）  "
          f"闭眼比例 {r_close:.3f}（要 < {BlinkState.CLOSE_RATIO:.2f}）")

    # 连眨三次应该记三次
    b2 = BlinkState()
    feed(b2, op, 15)
    for _ in range(3):
        feed(b2, cl2, 3)
        feed(b2, op, 12)
    print(f"    连眨 3 次 → {b2.blinks} 次  {'通过' if b2.blinks == 3 else '失败'}")

    # ---------------- 3. 照片攻击 ----------------
    print("\n[3] 照片攻击（连续喂同一状态，必须拿不到 blink）")
    ok_attack = True
    for name, v in (("一直睁眼（照片）", op), ("一直闭眼（照片）", cl)):
        bb = BlinkState()
        feed(bb, v, 300)
        passed = bb.blinks > 0
        ok_attack &= not passed
        print(f"    {name} 300 帧 → blinks={bb.blinks}  ratio={bb.ratio:.3f}  "
              f"{'★被攻破' if passed else '拦住'}")

    # ---------------- 4. 噪声 / 误记 ----------------
    print("\n[4] 噪声场景")
    rng = np.random.default_rng(0)
    b3 = BlinkState()
    for _ in range(1800):
        b3.push(float(op * (1.0 + rng.normal(0, 0.15))))
    print(f"    睁眼 ±15% 抖动 1800 帧（60s）→ blinks={b3.blinks}  "
          f"{'无虚警' if b3.blinks == 0 else '★虚警'}")

    b4 = BlinkState()
    for i in range(1800):
        if i % 40 == 20:                 # 每 40 帧掉一帧信号，模拟检测抖动
            b4.push(float(cl))
        else:
            b4.push(float(op))
    print(f"    单帧掉信号 每40帧一次       → blinks={b4.blinks}  "
          f"{'不误判（单帧闭眼不算眨眼）' if b4.blinks == 0 else '★误判'}")

    # ---------------- 5. 人数校验 ----------------
    print("\n[5] 人数校验")
    cases = []
    n1, _ = count_persons(faces)
    cases.append(("单人样本", n1, 1))

    two = np.hstack([img, img])
    f2 = det(two)
    n2, _ = count_persons(f2)
    cases.append((f"横向拼成两个人（检出 {0 if f2 is None else len(f2)} 张脸）", n2, 2))

    three = np.hstack([img, img, img])
    f3 = det(three)
    n3, _ = count_persons(f3)
    cases.append((f"拼成三个人（检出 {0 if f3 is None else len(f3)} 张脸）", n3, 3))

    corner = img.copy()
    hh, ww = corner.shape[:2]
    tiny = cv2.resize(img, (ww // 3, hh // 3))
    corner[0:hh // 3, ww - ww // 3:] = tiny          # 角落贴一张小脸
    f4 = det(corner)
    n4, _ = count_persons(f4)
    cases.append((f"主脸 + 角落小脸（检出 {0 if f4 is None else len(f4)} 张脸）", n4, 1))

    ok_person = True
    for name, got, want in cases:
        good = (got == want) if want == 1 else (got >= want)
        ok_person &= good
        print(f"    {name:<40} → {got} 人  {'正确' if good else '★失败'}")

    # ---------------- 6. 耗时 ----------------
    t0 = time.perf_counter()
    for _ in range(500):
        eye_dark_ratio(gray, face)
    dt = (time.perf_counter() - t0) / 500 * 1000
    print(f"\n[6] 活体单帧耗时 {dt:.3f} ms（两次 40x26 ROI 采样，可忽略）")

    # ---------------- 7. 留一张对照图 ----------------
    os.makedirs(os.path.join(BASE, "out"), exist_ok=True)
    p = os.path.join(BASE, "out", "liveness_compare.jpg")
    cv2.imwrite(p, np.hstack([img, closed]))
    print(f"睁眼 / 闭眼合成对照图：{p}")

    all_ok = (b.blinks == 1 and b2.blinks == 3 and ok_attack
              and b3.blinks == 0 and b4.blinks == 0 and ok_person)
    print("\n" + line)
    print("结论：" + ("全部通过" if all_ok else "★存在失败项，见上方标记"))
    print(line)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
