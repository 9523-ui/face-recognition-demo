"""
下载 demo 需要的两个模型文件（首次运行前执行一次）
    python download_models.py

YuNet : 人脸检测，约 227 KB
SFace : 人脸识别（128 维特征），约 37 MB
"""
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")

TARGETS = [
    ("yunet.onnx", 232589,
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    ("sface.onnx", 38696353,
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
]

MIRROR = "https://cdn.jsdelivr.net/gh/opencv/opencv_zoo@main/models/{d}/{f}"


def fetch(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                sys.stdout.write(f"\r    {pct:3d}%  {done/1048576:.1f} / {total/1048576:.1f} MB")
                sys.stdout.flush()
    sys.stdout.write("\n")


def main():
    os.makedirs(MODELS, exist_ok=True)
    for name, expect, url in TARGETS:
        dst = os.path.join(MODELS, name)
        if os.path.exists(dst) and os.path.getsize(dst) > expect * 0.9:
            print(f"[跳过] {name} 已存在 ({os.path.getsize(dst)/1048576:.1f} MB)")
            continue
        print(f"[下载] {name}")
        ok = False
        for u in (url, MIRROR.format(d=("face_detection_yunet" if "yunet" in name
                                        else "face_recognition_sface"),
                                     f=("face_detection_yunet_2023mar.onnx" if "yunet" in name
                                        else "face_recognition_sface_2021dec.onnx"))):
            try:
                fetch(u, dst)
                size = os.path.getsize(dst)
                if size > expect * 0.9:
                    print(f"  完成 {size/1048576:.1f} MB")
                    ok = True
                    break
                print(f"  文件偏小({size} B)，换下一个源")
            except Exception as e:
                print(f"  失败: {type(e).__name__}: {e}")
        if not ok:
            print(f"[错误] {name} 下载失败。可手动下载后放到 {MODELS}\\{name}")
            print(f"       {url}")
            return 1
    print("\n模型就绪，可以运行: python face_demo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
