"""
1:N 开集检索基准：底库涨到多少人会开始认错
================================================
`bench_accuracy.py` 测的是「两张图是不是同一个人」（1:1 验证），
但程序真实干的活是「来一个人，从库里挑一个，或者说不认识」（1:N 检索）。
这两件事的风险完全不同：1:1 只要一个阈值，1:N 里**库内每个人都是一个陷阱**，
底库越大，「刚好长得像」撞上来的概率越高。

本脚本回答四个问题：
  1. 底库 5 / 20 / 80 人时，陌生人被误认的概率各是多少？
  2. 四种判定策略在同一误识率下，谁多认对几个？
  3. z-norm（看「比库内其他人突出多少」而不是绝对相似度）值不值得上？
  4. 阈值该不该随底库规模调整，调成多少？

方法：用合影里 130+ 张真实人脸当 130+ 个身份。
  注册照 = 原图裁剪（模拟高质量录入）
  现场照 = 缩到 60% + 模糊 0.7 + 亮度 0.9 + JPEG 40（模拟另一台设备、另一个时间）

运行：
    python bench_scale.py
"""
import os
import sys
import time

import cv2
import numpy as np

from face_demo import (FaceDetector, FaceEncoder, MATCH_THRESH, MATCH_MARGIN,
                       face_quality)

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_IMG = os.path.join(BASE, "testdata", "selfie.jpg")
MIN_W = 90              # 只取够大的脸，小脸会让所有策略一起崩，掩盖差异
TRIALS = 8              # 每个规模重复随机抽样的次数
# 原图里脸只有 7 张够 80px，凑不出「几十人的底库」。
# 放大 2.5 倍再检测，YuNet 对小人脸的召回会跟着上来（135 → 350 张），
# 够 90px 的身份数从 7 涨到 60+，才谈得上「规模」实验。
# 注意这只放大了像素、不增加真实信息，所以偏小的身份仍然偏难 —— 对结论是保守的。
UPSCALE = 2.5
# 放大后还检出一批「脸宽 370px 但清晰度 0.00」的框 —— 是背景/海报被误检，不是脸。
# 这类特征没有区分度（人人都像），会把不同人相似度整体抬到 0.5+，污染 FAR 统计。
# 按程序自己的质量门控筛掉：注册照质量不合格，本来也不允许入库。
QUAL_MIN = 0.55
RNG = np.random.default_rng(20261028)


# ---------------------------------------------------------------- 数据准备
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


def degrade(img, f, s=0.6):
    """模拟「另一台设备 + 另一个时间」的现场照。"""
    h, w = img.shape[:2]
    out = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(out, (0, 0), 0.7)
    out = np.clip(out.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    nf = f.copy()
    nf[[0, 2]] *= s
    nf[[1, 3]] *= s
    nf[[4, 6, 8, 10, 12]] *= s
    nf[[5, 7, 9, 11, 13]] *= s
    return out, nf


def enc_pair(enc, img, row):
    """一次对齐，出两个特征：原图 + 水平镜像（用于 test-time augmentation）。"""
    aligned = enc.rec.alignCrop(img, row)
    a = enc.rec.feature(aligned).flatten().astype(np.float32)
    b = enc.rec.feature(cv2.flip(aligned, 1)).flatten().astype(np.float32)
    return a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)


def load_identities():
    if not os.path.exists(TEST_IMG):
        raise SystemExit("缺少测试图: " + TEST_IMG)
    src = cv2.imread(TEST_IMG)
    src = cv2.resize(src, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    det, enc = FaceDetector(), FaceEncoder()
    faces = det(src)
    if faces is None:
        raise SystemExit("测试图里没检出人脸")

    reg_a, reg_b, qry_a, qry_b = [], [], [], []
    dropped = 0
    for f in faces:
        if float(f[2]) < MIN_W:
            continue
        crop, nf = crop_face(src, f)
        if crop.size == 0 or min(nf[2], nf[3]) < 70:
            continue
        q, _ = face_quality(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), nf)
        if q < QUAL_MIN:                      # 注册照本身不合格，不入库
            dropped += 1
            continue
        try:
            ra, rb = enc_pair(enc, crop, nf)          # 注册照（清晰）
            dim, df = degrade(crop, nf)               # 现场照（退化）
            qa, qb = enc_pair(enc, dim, df)
        except cv2.error:
            continue
        reg_a.append(ra)
        reg_b.append(rb)
        qry_a.append(qa)
        qry_b.append(qb)

    if len(reg_a) < 20:
        raise SystemExit(f"有效身份只有 {len(reg_a)} 个，太少，无法做规模实验")
    print(f"   （按质量 ≥{QUAL_MIN} 筛掉 {dropped} 个不合格框）")
    return (np.stack(reg_a), np.stack(reg_b),
            np.stack(qry_a), np.stack(qry_b))


# ---------------------------------------------------------------- 打分
def build_scores(reg_a, reg_b, qry_a, qry_b):
    """四种策略的 (查询 × 底库) 分数矩阵。每行 = 这个查询对库内每个人的相似度。"""
    return {
        "0 基线": qry_a @ reg_a.T,
        "1 +TTA": np.maximum(qry_a @ reg_a.T, qry_b @ reg_a.T),
        "2 +注册镜像": np.maximum(qry_a @ reg_a.T, qry_a @ reg_b.T),
        "3 +两者都加": np.maximum.reduce([qry_a @ reg_a.T, qry_b @ reg_a.T,
                                          qry_a @ reg_b.T, qry_b @ reg_b.T]),
    }


def trial_records(S, n_lib, rng):
    """一次随机抽样：抽 n_lib 人建库，其余当陌生人。返回判定所需的全部数组。"""
    M = S.shape[0]
    n_lib = min(n_lib, M - 1)
    lib = rng.choice(M, size=n_lib, replace=False)
    sub = S[:, lib]                                    # (M, n_lib)

    order = np.argsort(-sub, axis=1)[:, :2]
    s1 = sub[np.arange(M), order[:, 0]]
    s2 = sub[np.arange(M), order[:, 1]] if n_lib > 1 else np.full(M, -1.0)
    pred = lib[order[:, 0]]

    mu = sub.mean(axis=1)
    sd = sub.std(axis=1) + 1e-6
    z = (s1 - mu) / sd                                 # 比库内整体突出多少

    in_lib = np.zeros(M, bool)
    in_lib[lib] = True
    return {
        "s1": s1, "s2": s2, "z": z,
        "correct": in_lib & (pred == np.arange(M)),     # 在库且认对自己
        "stranger": ~in_lib,                            # 根本不在库
        "swap": in_lib & (pred != np.arange(M)),        # 在库却认成别人
    }


def stack(recs):
    keys = ("s1", "s2", "z", "correct", "stranger", "swap")
    return {k: np.stack([r[k] for r in recs]) for k in keys}


def evaluate(st, thr, margin, use_z=False):
    """一次算出 认对 / 拒识 / 陌生人误接受 / 在库认错。全部向量化。"""
    base = st["z"] if use_z else st["s1"]
    acc = base >= thr
    if not use_z and margin > 0:
        # 第一名领先第二名不足 margin 时弃权（只在第二名也够像时才弃权）
        acc &= ~((st["s2"] > 0) & ((st["s1"] - st["s2"]) < margin)
                 & (st["s2"] > thr * 0.85))
    tp = int((acc & st["correct"]).sum())
    fa_open = int((acc & st["stranger"]).sum())
    fa_swap = int((acc & st["swap"]).sum())
    fn = int((~acc & (st["correct"] | st["swap"])).sum())
    n_genuine = int((st["correct"] | st["swap"]).sum())
    n_stranger = int(st["stranger"].sum())
    return {
        "tp": tp, "fn": fn, "fa_open": fa_open, "fa_swap": fa_swap,
        "dir": tp / max(n_genuine, 1),
        "far": fa_open / max(n_stranger, 1),
        "swap_rate": fa_swap / max(n_genuine, 1),
    }


def sweep(st, margins, use_z=False):
    """扫阈值 × margin，返回每个阈值下表现最好（认对率与误识折中）的一档。"""
    grid = np.arange(-1.0, 6.01, 0.1) if use_z else np.arange(0.15, 0.751, 0.005)
    rows = []
    for t in grid:
        best = None
        for m in margins:
            r = evaluate(st, t, m, use_z)
            obj = r["dir"] - 4.0 * r["far"] - 1.0 * r["swap_rate"]   # 误识比拒识严重得多
            if best is None or obj > best["obj"]:
                best = {"obj": obj, "thr": t, "margin": m, **r}
        rows.append(best)
    return rows


def at_far(rows, target):
    ok = [r for r in rows if r["far"] <= target]
    return max(ok, key=lambda r: r["dir"]) if ok else None


# ---------------------------------------------------------------- 主流程
def main():
    t0 = time.time()
    print("=" * 84)
    print("1:N 开集检索基准 —— 底库规模 × 判定策略")
    print("=" * 84)

    reg_a, reg_b, qry_a, qry_b = load_identities()
    M = reg_a.shape[0]
    print(f"身份数 {M}（脸宽 ≥{MIN_W}px；注册=原图，现场=缩60%+模糊0.7+JPEG40）")
    print(f"当前代码的判定：阈值 {MATCH_THRESH} / margin {MATCH_MARGIN}")

    sim_same = np.sum(reg_a * qry_a, axis=1)
    print(f"\n同一个人的 注册 vs 现场  相似度：均值 {sim_same.mean():.3f}  "
          f"最低 {sim_same.min():.3f}  p5 {np.percentile(sim_same, 5):.3f}")
    sim_diff = reg_a @ reg_a.T
    np.fill_diagonal(sim_diff, -1)
    print(f"不同人的 注册 vs 注册    相似度：最高 {sim_diff.max():.3f}  "
          f"均值(每人最强对手) {sim_diff.max(axis=1).mean():.3f}  ← 1:N 里真正的威胁")

    S = build_scores(reg_a, reg_b, qry_a, qry_b)
    margins = (0.0, 0.02, 0.045, 0.07)

    # ---------------- 1. 规模税 ----------------
    print("\n" + "-" * 84)
    print("[1] 规模税 & 阈值错得有多离谱（两件事分开看）")
    print("-" * 84)
    print(f"    {'底库人数':>8} {'陌生人最高分':>12} {'陌生人p95':>10} "
          f"{'FAR@阈值.363':>13} {'FAR@阈值.55':>12}")
    for n_lib in [n for n in (5, 10, 20, 30, 40) if n <= M - 8]:
        st = stack([trial_records(S["0 基线"], n_lib, RNG) for _ in range(TRIALS)])
        s_str = st["s1"][st["stranger"]]
        lo = evaluate(st, MATCH_THRESH, MATCH_MARGIN)
        hi = evaluate(st, 0.55, MATCH_MARGIN)
        print(f"    {n_lib:>8} {s_str.max():>12.3f} {np.percentile(s_str, 95):>10.3f} "
              f"{lo['far'] * 100:>12.1f}% {hi['far'] * 100:>11.1f}%")
    print("    └ 陌生人最高分随库变大而上升（规模税）；但 FAR 主要由**阈值**决定，"
          "0.363 时无论几个人都是三成误接受")

    # ---------------- 2. 策略对比 ----------------
    n_lib = min(30, M - 8)
    print("\n" + "-" * 84)
    print(f"[2] 四种策略：同样误识率下，谁能把阈值抬得更高（底库 {n_lib} 人）")
    print("-" * 84)
    print("    阈值越高越安全。认对率都是 100% 时，**「能抬到多高」就是策略的收益**。")
    print(f"    {'策略':<14} {'认对率':>8} {'FAR≤1%需阈值':>13} {'FAR≤0.5%需阈值':>15} {'相对基线':>10}")
    base_thr = None
    for name, mat in S.items():
        st = stack([trial_records(mat, n_lib, RNG) for _ in range(TRIALS)])
        rows = sweep(st, margins)
        p1 = at_far(rows, 0.01)
        p05 = at_far(rows, 0.005)
        if base_thr is None:
            base_thr = p1["thr"] if p1 else 0.0
        gain = f"+{p1['thr'] - base_thr:.3f}" if p1 else "—"
        d = f"{p1['dir'] * 100:.1f}%" if p1 else "—"
        t1 = f"{p1['thr']:.3f}" if p1 else "—"
        t05 = f"{p05['thr']:.3f}" if p05 else "—"
        print(f"    {name:<14} {d:>8} {t1:>13} {t05:>15} {gain:>10}")

    # ---------------- 3. z-norm ----------------
    print("\n" + "-" * 84)
    print("[3] 固定判据的稳定性：绝对阈值 vs z-norm（后者随底库规模自动适应）")
    print("-" * 84)
    print(f"    {'底库人数':>8} {'FAR@0.363':>11} {'FAR@z=3.0':>11} {'认对率@z=3.0':>13}")
    for n_lib_i in [n for n in (5, 20, 40) if n <= M - 8]:
        st = stack([trial_records(S["0 基线"], n_lib_i, RNG) for _ in range(TRIALS)])
        a = evaluate(st, MATCH_THRESH, MATCH_MARGIN)
        b = evaluate(st, 3.0, 0.0, use_z=True)
        print(f"    {n_lib_i:>8} {a['far'] * 100:>10.1f}% {b['far'] * 100:>10.2f}% "
              f"{b['dir'] * 100:>12.1f}%")
    print("    └ z-norm 用一个固定 z 就能跨规模稳定，不用手工按人数改阈值")

    # ---------------- 4. 阈值随规模 ----------------
    print("\n" + "-" * 84)
    print("[4] 阈值该取多少：按底库规模分别标定（目标 FAR≈1%，用最强的「+两者都加」）")
    print("-" * 84)
    print(f"    {'底库人数':>8} {'建议阈值':>10} {'建议margin':>12} {'预期认对率':>11} {'预期FAR':>9}")
    for n_lib_i in [n for n in (5, 20, 40) if n <= M - 8]:
        st = stack([trial_records(S["3 +两者都加"], n_lib_i, RNG) for _ in range(TRIALS)])
        r = at_far(sweep(st, margins), 0.01)
        if r:
            print(f"    {n_lib_i:>8} {r['thr']:>10.3f} {r['margin']:>12.2f} "
                  f"{r['dir'] * 100:>10.1f}% {r['far'] * 100:>8.2f}%")

    print("\n" + "=" * 84)
    print(f"耗时 {time.time() - t0:.1f}s")
    print("""
怎么读这份数据：
  · [1] 两个数要分开看：陌生人最高分随库变大而上升（规模税，数学必然）；
    但当前 FAR 三成的主因是**阈值 0.363 本身** —— 它来自 SFace 论文的 1:1 验证场景，
    搬到 1:N 检索上根本不成立。
  · [2] 认对率打平时，「FAR≤1% 需要多高的阈值」就是策略的收益：抬得越高越安全。
  · [3] z-norm 的价值在于**免调参**：底库从 5 人涨到 40 人，绝对阈值得跟着挪，z 不用。
  · [4] 能直接抄进代码的那张表。

重要局限：几十个身份都来自同一张合影 —— 同相机、同场景、同一天，
「不同人」的相似度被系统性低估（缺跨种族/跨年龄/跨光照的真实多样性），
同人跨设备的落差也被低估。真实部署时两类分数会更靠拢，
绝对阈值还得往上压，z 阈值也要重新标。这里的数只能当**乐观下限**看。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
