"""验证 gui.py 里「多帧平均入库」的核心逻辑（不依赖摄像头）。

模拟：对同一个人的照片做 8 种亮度变化当作 8 帧采样，求平均后入库；
再用另一个人测试是否会被误认。
"""
import os
import shutil
import tempfile

import cv2
import numpy as np

from face_demo import FaceDetector, FaceEncoder, FaceLibrary

img = cv2.imread("testdata/selfie.jpg")
det, enc = FaceDetector(), FaceEncoder()
faces = det(img)
faces = sorted(faces, key=lambda f: -f[2] * f[3])
print(f"合影检出 {len(faces)} 张脸")


def sample_variants(f, n=8):
    """把一张脸做 n 种亮度变化，模拟连续采样 8 帧。"""
    x, y, w, h = [int(v) for v in f[:4]]
    pad = 0.25
    x0, y0 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
    x1 = min(img.shape[1], int(x + w * (1 + pad)))
    y1 = min(img.shape[0], int(y + h * (1 + pad)))
    base = img[y0:y1, x0:x1]
    out = []
    for k in range(n):
        gain = 0.72 + 0.08 * k
        v = np.clip(base.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        ff = det(v)
        if ff is None or len(ff) == 0:
            continue
        j = int(np.argmax([q[2] * q[3] for q in ff]))
        out.append(enc.encode(v, ff[j]))
    return out


sA = sample_variants(faces[0])
sB = sample_variants(faces[1])
print(f"A 有效样本 {len(sA)}/{8}    B 有效样本 {len(sB)}/{8}")

meanA = np.stack(sA).mean(axis=0)
meanA = meanA / (np.linalg.norm(meanA) + 1e-9)
meanB = np.stack(sB).mean(axis=0)
meanB = meanB / (np.linalg.norm(meanB) + 1e-9)

consistency = float(np.mean([float(f @ meanA) for f in sA]))
print(f"A 内部一致性（8 帧与均值平均余弦）: {consistency:.3f}")
print(f"A均值 vs B均值 余弦: {float(meanA @ meanB):.3f}   (阈值 0.45，来自 settings.json)")

# 存储已换成 SQLite：用独立的临时目录，连密钥一起隔离，避免污染真实库
tmpdir = tempfile.mkdtemp(prefix="_gui_test_")
tmp = os.path.join(tmpdir, "faces.db")
lib = FaceLibrary(path=tmp)
lib.add("测试甲", meanA.astype(np.float32))

nm1, sc1 = lib.match(meanA)
nm2, sc2 = lib.match(meanB)
print(f"\n用 A 的均值去识别   -> {nm1}  ({sc1:.3f})   {'✓ 命中' if nm1 == '测试甲' else '✗ 失败'}")
print(f"用 B 的均值去识别   -> {nm2}  ({sc2:.3f})   {'✓ 正确判未知' if nm2 == '未知' else '✗ 误认!'}")
print(f"库里条目: {dict(lib.count_by_name())}")

# 清理（SQLite 还有 -wal / -shm 两个伴生文件）
shutil.rmtree(tmpdir, ignore_errors=True)
