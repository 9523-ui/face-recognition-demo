"""
识别诊断：为什么认不出 / 为什么认错
==================================
不开界面，抓几帧（或读一张图），把这个人的「画面质量」和「库内相似度」
摊开来看，然后直接给可执行建议。

用法：
    python diagnose.py                  摄像头抓 30 帧，取质量最高的一帧来诊断
    python diagnose.py --frames 60      多抓一会儿
    python diagnose.py --image 照片.jpg  对静态图诊断（比如想检查录入用的照片行不行）
    python diagnose.py --cam 1          换摄像头

看什么：
    质量分  —— 下划线在 0.60 以上才踏实；低于 0.45 程序会直接拒绝采信
    相似度  —— 跟库里每个人差多少，离阈值 0.363 还有多远
    建议    —— 按最拖后腿的那一项给，不废话
"""
import argparse
import os
import sys

import cv2
import numpy as np

from face_demo import (MATCH_MARGIN, MATCH_THRESH, MIN_QUALITY, FaceDetector,
                       FaceEncoder, FaceLibrary, count_persons, face_quality,
                       open_camera)

GOOD_Q = 0.60          # 这个分数以上算「画质可靠」


def analyse(frame, det, enc):
    """返回这一帧的主脸信息；没有人脸返回 None。"""
    faces = det(frame)
    n, main = count_persons(faces)
    if main is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    q, qd = face_quality(gray, main)
    try:
        feat = enc.encode(frame, main)
    except cv2.error:
        return None
    return {"q": q, "qd": qd, "feat": feat, "n": n}


def bar(v, width=24):
    n = int(round(np.clip(v, 0, 1) * width))
    return "█" * n + "·" * (width - n)


def main():
    ap = argparse.ArgumentParser(description="人脸识别诊断")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--frames", type=int, default=30, help="抓多少帧，默认 30")
    ap.add_argument("--image", help="对静态图诊断")
    ap.add_argument("--mirror", action="store_true", help="画面镜像（与 gui 一致时不加）")
    args = ap.parse_args()

    det, enc = FaceDetector(), FaceEncoder()
    lib = FaceLibrary()

    print("=" * 70)
    results = []
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print("读不到图片:", args.image)
            return 1
        r = analyse(img, det, enc)
        if r is None:
            print("这张图里没检出可用的人脸")
            return 1
        results.append(r)
        print(f"图片来源 {args.image}  {img.shape[1]}x{img.shape[0]}")
    else:
        cap = open_camera(args.cam)
        print(f"采集 {args.frames} 帧中，请保持自然姿势看着镜头…")
        for i in range(args.frames):
            ok, fr = cap.read()
            if not ok:
                break
            if args.mirror:
                fr = cv2.flip(fr, 1)
            r = analyse(fr, det, enc)
            if r is not None:
                results.append(r)
        cap.release()
        if not results:
            print("一帧可用的人脸都没抓到 —— 检查摄像头是否被占用、人是否在画面里")
            return 1
        print(f"抓到 {len(results)}/{args.frames} 帧含人脸，分辨率 "
              f"{fr.shape[1]}x{fr.shape[0]}")

    # ---------------- 质量 ----------------
    qs = [r["q"] for r in results]
    best = max(results, key=lambda r: r["q"])
    qd = best["qd"]
    print(f"\n[画面质量]  最好的一帧 {best['q']:.3f}   中位数 "
          f"{np.median(qs):.3f}   最差 {min(qs):.3f}")
    print(f"    {'尺寸':<4} {bar(qd['尺寸'])} {qd['尺寸']:.2f}   人脸宽 {qd['px']}px")
    print(f"    {'清晰':<4} {bar(qd['清晰'])} {qd['清晰']:.2f}   "
          f"Laplacian 方差 {qd['lap']}")
    print(f"    {'亮度':<4} {bar(qd['亮度'])} {qd['亮度']:.2f}")
    print(f"    {'正脸':<4} {bar(qd['正脸'])} {qd['正脸']:.2f}")
    print(f"    合计 {best['q']:.3f}   （低于 {MIN_QUALITY} 会被拒绝采信，"
          f"{GOOD_Q} 以上算可靠）")

    # ---------------- 相似度 ----------------
    print("\n[库内相似度]")
    if len(lib.names) == 0:
        print("    人脸库是空的，先录入：python face_demo.py --enroll 名字")
        scores = []
    else:
        # 用质量最好的几帧特征求平均，比单帧更稳
        top = sorted(results, key=lambda r: -r["q"])[:5]
        mean_feat = np.mean([r["feat"] for r in top], axis=0)
        mean_feat /= (np.linalg.norm(mean_feat) + 1e-9)
        scores = lib.person_scores(mean_feat, topn=6)
        cnt = lib.count_by_name()
        for i, (nm, s) in enumerate(scores):
            mark = "✓ 过阈值" if s >= MATCH_THRESH else "✗ 低于阈值"
            print(f"    {i + 1}. {nm:<10} {s:.3f}  {bar(s, 18)}  {mark}   "
                  f"库内 {cnt.get(nm, 0)} 条")

    # ---------------- 建议 ----------------
    print("\n[建议]")
    tips = []
    if qd["尺寸"] < 0.8:
        tips.append(f"人脸只有 {qd['px']}px 宽，偏小。坐到离镜头 40~60cm，"
                    f"让脸占到画面高度的 1/3 以上（目标 ≥140px）")
    if qd["清晰"] < 0.6:
        tips.append(f"画面偏虚（清晰度 {qd['清晰']:.2f}）。端稳、别晃，"
                    f"或者擦一下镜头；光线不足时摄像头会拉长曝光，更容易糊")
    if qd["亮度"] < 0.6:
        tips.append("光线不理想。别背对窗户，让光从正面来（台灯/屏幕光都行），"
                    "避免顶光和强烈侧光")
    if qd["正脸"] < 0.7:
        tips.append("脸偏得比较多。识别时尽量正对镜头，左右转头控制在 30° 以内")

    if best["q"] < MIN_QUALITY:
        tips.insert(0, f"★ 当前画质 {best['q']:.2f} 达不到采信标准 {MIN_QUALITY}，"
                       f"这种情况程序会一直判「未知」，先按下面的条目改善画质")

    if scores:
        top_nm, top_s = scores[0]
        n_samples = lib.count_by_name().get(top_nm, 0)
        if top_s < MATCH_THRESH:
            gap = MATCH_THRESH - top_s
            tips.append(f"最像的是「{top_nm}」但只有 {top_s:.3f}，差 {gap:.3f}。"
                        f"常见原因是录入和现在条件差太多 —— 换到同样的光线下再录 2~3 次")
            if n_samples <= 1:
                tips.append(f"「{top_nm}」库里只有 {n_samples} 条样本。"
                            f"建议录 3~5 条：正面、稍微左转、稍微右转、戴/不戴眼镜各来一次")
        elif len(scores) > 1 and top_s - scores[1][1] < MATCH_MARGIN:
            tips.append(f"「{top_nm}」({top_s:.3f}) 和「{scores[1][0]}」"
                        f"({scores[1][1]:.3f}) 咬得太近，差 {top_s - scores[1][1]:.3f} < "
                        f"{MATCH_MARGIN}，程序会保守地判「未知」。"
                        f"给这两个人各多录几条可拉开差距")
        else:
            print(f"    识别正常：「{top_nm}」{top_s:.3f}，领先第二名 "
                  f"{top_s - scores[1][1] if len(scores) > 1 else float('nan'):.3f}")

    if not tips:
        print("    没有明显问题，画质和匹配度都够用。")
    else:
        for t in tips:
            print(f"    · {t}")

    # ---------------- 帧间稳定性 ----------------
    if len(results) >= 5:
        feats = [r["feat"] for r in results]
        base = feats[0]
        cos = [float(f @ base) for f in feats[1:]]
        print(f"\n[稳定性]  各帧特征与首帧的相似度  均值 {np.mean(cos):.3f}  "
              f"最低 {min(cos):.3f}")
        if np.mean(cos) < 0.85:
            print("    偏低：人在动 / 光线在变 / 画面偏糊，特征会一直漂，"
                  "这种状态下多帧投票也救不回来")
        else:
            print("    正常：同一个人的特征在各帧之间是稳的")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
