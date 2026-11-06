# -*- coding: utf-8 -*-
"""调参实验：低照度场景下不同预处理链路对"形状判定"的影响。

用于回答一个具体问题：暗场里颜色召回已经能靠 V 通道 gamma 修回来，
但形状判定为什么仍然差？是 CLAHE/锐化放大噪声，还是 Canny 阈值问题？
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.detectors.color import ColorDetector          # noqa: E402
from src.detectors.shape import detect_circles        # noqa: E402
from src.preprocessing.enhance import Preprocessor    # noqa: E402

ANN = os.path.join(ROOT, "dataset", "annotations.json")


def evaluate(process_fn, label):
    ann = json.load(open(ANN, encoding="utf-8"))
    items = [i for i in ann["images"] if i["condition"] == "dim"]
    det = ColorDetector("red", min_area=120, confidence_threshold=0.15)
    hit_color = hit_shape = n_gt = 0
    for it in items:
        img = cv2.imread(os.path.join(ROOT, "dataset", "images", it["file"]))
        proc = process_fn(img)
        gts = [o for o in it["objects"] if o["label"] == "red_circle"]
        n_gt += len(gts)
        h, w = proc.shape[:2]
        circles, _ = detect_circles(proc, (h, w), min_area=180, min_circularity=0.75)
        centers = [(c["center"][0], c["center"][1]) for c in circles]
        for g in gts:
            # 颜色召回：预处理后该点是否仍落在红色掩码内
            hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
            px = hsv[g["cy"], g["cx"]]
            in_red = px[2] >= 100 and px[1] >= 100 and (px[0] <= 10 or px[0] >= 160)
            hit_color += int(in_red)
            hit_shape += int(any(np.hypot(cx - g["cx"], cy - g["cy"]) < 20 for cx, cy in centers))
    return {
        "label": label,
        "color_recall": round(hit_color / n_gt, 3),
        "shape_recall": round(hit_shape / n_gt, 3),
    }


def main():
    rows = []
    rows.append(evaluate(lambda im: im, "原图（不处理）"))
    rows.append(evaluate(lambda im: Preprocessor(mode="clahe_sharpen").process(im), "CLAHE+锐化"))
    rows.append(evaluate(lambda im: Preprocessor(mode="hsv_gamma").process(im), "V通道gamma+CLAHE+锐化"))
    rows.append(evaluate(
        lambda im: Preprocessor(mode="hsv_gamma", clip_limit=1.0, sharpen=False).process(im),
        "V通道gamma+弱CLAHE(1.0)+不锐化"))
    rows.append(evaluate(lambda im: Preprocessor(mode="hsv_gamma", tile_grid=4,
                                                 clip_limit=2.0, sharpen=False).process(im),
                         "V通道gamma+CLAHE(4x4,2.0)+不锐化"))
    print(f"{'方案':32s}{'颜色召回':>10s}{'形状召回':>10s}")
    for r in rows:
        print(f"{r['label']:32s}{r['color_recall']:>10.3f}{r['shape_recall']:>10.3f}")


if __name__ == "__main__":
    main()
