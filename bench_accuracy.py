"""
识别精度的退化曲线：到底什么条件下会开始认不出
==============================================
用公开合影里的真实人脸当测试对象，把「劣化因子」一个一个单独拉大，
看相似度在哪里掉线。这比笼统的「准确率 99%」有用得多 ——
它会告诉你：脸至少要多宽、多清楚、注册几张，才够用。

测三件事：
  1. 人脸像素宽度 → 相似度（决定「离镜头多远才算够」）
  2. 模糊程度     → 相似度（对焦不准 / 运动模糊 / 镜头脏了）
  3. 注册样本数   → 认对率（「多录几次」到底有没有用，几个才够）

运行：
    python bench_accuracy.py
"""
import os
import sys
import time

import cv2
import numpy as np

from face_demo import (FaceDetector, FaceEncoder, MATCH_THRESH, MATCH_MARGIN,
                       MIN_QUALITY, face_quality)

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_IMG = os.path.join(BASE, "testdata", "selfie.jpg")
MIN_FACE_W = 70


def crop_face(img, f, pad=1.7):
    x, y, w, h = [float(v) for v in f[:4]]
    cx, cy = x + w / 2, y + h / 2
    x0 = int(max(0, cx - w * pad / 2))
    y0 = int(max(0, cy - h * pad / 2))
    x1 = int(min(img.shape[1], cx + w * pad / 2))
    y1 = int(min(img.shape[0], cy + h * pad / 2))
    crop = img[y0:y1, x0:x1].copy()
    nf = f.copy()
    nf[0] -= x0
    nf[1] -= y0
    for k in (4, 6, 8, 10, 12):
        nf[k] -= x0
        nf[k + 1] -= y0
    return crop, nf


def scaled(img, f, s):
    out = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 \
        else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    nf = f.copy()
    nf[[0, 2]] *= s
    nf[[1, 3]] *= s
    nf[[4, 6, 8, 10, 12]] *= s
    nf[[5, 7, 9, 11, 13]] *= s
    return out, nf


def blurred(img, f, sigma):
    if sigma <= 0.01:
        return img, f
    return cv2.GaussianBlur(img, (0, 0), sigma), f


def bright(img, f, k):
    return np.clip(img.astype(np.float32) * k, 0, 255).astype(np.uint8), f


def jpeg(img, f, quality):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return (cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img), f


def bar(v, lo, hi, width=22):
    """把数值画成一条小横条，终端里看趋势比看数字快。"""
    n = int(round(np.clip((v - lo) / (hi - lo), 0, 1) * width))
    return "█" * n + "·" * (width - n)


def main():
    print("=" * 78)
    if not os.path.exists(TEST_IMG):
        print("缺少测试图:", TEST_IMG)
        return 1
    src = cv2.imread(TEST_IMG)
    det, enc = FaceDetector(), FaceEncoder()
    faces = det(src)
    big = [f for f in faces if float(f[2]) >= MIN_FACE_W]
    print(f"测试图 {src.shape[1]}x{src.shape[0]}  检出 {len(faces)} 张脸，"
          f"取宽 ≥{MIN_FACE_W}px 的 {len(big)} 张做实验")

    samples = []
    for f in big:
        crop, nf = crop_face(src, f)
        if crop.size == 0 or min(nf[2], nf[3]) < 60:
            continue
        try:
            ref = enc.encode(crop, nf)
        except cv2.error:
            continue
        samples.append((crop, nf, ref))
    if not samples:
        print("没有可用的样本")
        return 1
    print(f"有效样本 {len(samples)} 个（原始人脸宽度中位数 "
          f"{np.median([s[1][2] for s in samples]):.0f}px）")
    print(f"判定阈值 {MATCH_THRESH}（低于此值判「未知」）")

    # ---------------- 0. 同人 / 不同人的相似度分布 ----------------
    print("\n[0] 分不分得开？—— 同一个人 vs 不同的人")
    var_kinds = [("亮0.85", lambda c, f: bright(c, f, 0.85)),
                 ("原图", lambda c, f: (c, f)),
                 ("亮1.15", lambda c, f: bright(c, f, 1.15)),
                 ("模糊1.0", lambda c, f: blurred(c, f, 1.0)),
                 ("模糊2.0", lambda c, f: blurred(c, f, 2.0)),
                 ("缩70%", lambda c, f: scaled(c, f, 0.7)),
                 ("JPEG45", lambda c, f: jpeg(c, f, 45))]
    reps = []
    for crop, nf, ref in samples:
        fs = []
        for _, fn in var_kinds:
            im, ff = fn(crop, nf)
            try:
                fs.append(enc.encode(im, ff))
            except cv2.error:
                pass
        if len(fs) >= 5:
            reps.append(fs)

    same, diff = [], []
    for fs in reps:
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                same.append(float(fs[i] @ fs[j]))
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            diff.append(float(reps[i][0] @ reps[j][0]))

    if same and diff:
        print(f"    同一个人的不同条件（{len(same)} 对）  最低 {min(same):.3f}  "
              f"均值 {np.mean(same):.3f}  p5 {np.percentile(same, 5):.3f}")
        print(f"    不同人之间（{len(diff)} 对）         最高 {max(diff):.3f}  "
              f"均值 {np.mean(diff):.3f}  p95 {np.percentile(diff, 95):.3f}")
        gap = min(same) - max(diff)
        print(f"    两类之间的间隔（越大越安全）        {gap:.3f}")
        ideal = (max(diff) + min(same)) / 2
        print(f"    这批数据上的最优阈值 ≈ {ideal:.3f}   "
              f"（当前 {MATCH_THRESH}）")
        if max(diff) >= MATCH_THRESH:
            print(f"    ★ 注意：不同人最高能到 {max(diff):.3f}，已经超过当前阈值 —— "
                  f"只靠单一阈值会认错，必须加 ratio test")
        else:
            print(f"    不同人的最高分 {max(diff):.3f} 仍在阈值以下，单靠阈值就够")
        same_fail = sum(1 for s in same if s < MATCH_THRESH)
        print(f"    同一人被判「未知」的比例：{same_fail}/{len(same)} "
              f"= {same_fail / len(same) * 100:.1f}%（越大说明阈值定得太严）")

    # ---------------- 1. 尺寸 ----------------
    print("\n[1] 人脸像素宽度 → 与原始尺寸特征的相似度")
    print(f"    {'脸宽 px':>8} {'相似度':>8} {'质量分':>8}   趋势")
    scales = [1.0, 0.8, 0.6, 0.45, 0.35, 0.28, 0.22, 0.17, 0.13, 0.10]
    dim_curve = []
    for s in scales:
        coss, quals, px = [], [], []
        for crop, nf, ref in samples:
            im, ff = scaled(crop, nf, s)
            if min(ff[2], ff[3]) < 24:
                continue
            try:
                q = enc.encode(im, ff)
            except cv2.error:
                continue
            coss.append(float(q @ ref))
            qualifies, _ = face_quality(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), ff)
            quals.append(qualifies)
            px.append(float(ff[2]))
        if not coss:
            continue
        cos_m, qual_m, px_m = np.mean(coss), np.mean(quals), np.mean(px)
        dim_curve.append((px_m, cos_m, qual_m))
        mark = "认得出" if cos_m >= MATCH_THRESH else "认不出"
        print(f"    {px_m:>8.0f} {cos_m:>8.3f} {qual_m:>8.3f}   "
              f"{bar(cos_m, 0, 1)} {mark}")

    # ---------------- 2. 模糊 ----------------
    print("\n[2] 模糊程度 → 相似度（模拟对焦不准 / 运动模糊）")
    print(f"    {'sigma':>8} {'相似度':>8} {'清晰度':>8}   趋势")
    for sigma in (0.0, 0.6, 1.0, 1.5, 2.0, 2.8, 3.5, 4.5):
        coss, laps = [], []
        for crop, nf, ref in samples:
            im, ff = blurred(crop, nf, sigma)
            try:
                q = enc.encode(im, ff)
            except cv2.error:
                continue
            coss.append(float(q @ ref))
            _, d = face_quality(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), ff)
            coss_q = d["清晰"]
            laps.append(coss_q)
        if not coss:
            continue
        cos_m, lap_m = np.mean(coss), np.mean(laps)
        mark = "认得出" if cos_m >= MATCH_THRESH else "认不出"
        print(f"    {sigma:>8.1f} {cos_m:>8.3f} {lap_m:>8.3f}   "
              f"{bar(cos_m, 0, 1)} {mark}")

    # ---------------- 3. 注册样本数 ----------------
    print("\n[3] 注册样本数 → 认对率（查询用 4 种劣化条件）")
    variants = [("原图", lambda c, f: (c, f)),
                ("模糊1.0", lambda c, f: blurred(c, f, 1.0)),
                ("暗0.78", lambda c, f: bright(c, f, 0.78)),
                ("缩到60%", lambda c, f: scaled(c, f, 0.6))]
    print(f"    {'注册张数':>10} {'认对率':>9} {'误识':>7} {'拒识':>7}   说明")
    for n_reg in (1, 2, 3, 5, 8):
        correct = wrong = reject = 0
        for i, (crop, nf, ref) in enumerate(samples):
            # 注册照：轻微亮度扰动，模拟多次录入
            regs = []
            for k in range(n_reg):
                im, ff = bright(crop, nf, 0.94 + 0.03 * (k % 3))
                try:
                    regs.append(enc.encode(im, ff))
                except cv2.error:
                    pass
            if not regs:
                continue
            others = [s[2] for j, s in enumerate(samples) if j != i]
            for _, fn in variants:
                im, ff = fn(crop, nf)
                try:
                    q = enc.encode(im, ff)
                except cv2.error:
                    continue
                # 与自己的注册照比
                s_self = max(float(r @ q) for r in regs)
                s_other = max(float(o @ q) for o in others) if others else -1
                if s_self < MATCH_THRESH:
                    reject += 1
                elif s_other > s_self:
                    wrong += 1
                elif s_self - s_other < MATCH_MARGIN and s_other > MATCH_THRESH * 0.85:
                    reject += 1
                else:
                    correct += 1
        tot = correct + wrong + reject or 1
        note = "← 只录一次就很勉强" if n_reg == 1 else ("← 推荐" if n_reg in (3, 5) else "")
        print(f"    {n_reg:>10} {correct / tot * 100:>8.1f}% {wrong:>7} "
              f"{reject:>7}   {note}")

    print("\n" + "=" * 78)
    print("怎么用这组数据：")
    print("  · 第 1 张表告诉你人脸在画面里应该占多宽 —— 太窄时相似度直接掉到阈值以下，")
    print("    这时再怎么调参数都没用，只能让人坐近一点或者换高分辨率摄像头。")
    print("  · 第 2 张表说明对焦/抖动的影响，模糊到某个程度同样救不回来。")
    print("  · 第 3 张表证明「多录几次」不是玄学：样本多了以后，同一人的最高分更稳。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
