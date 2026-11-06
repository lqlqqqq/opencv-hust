# -*- coding: utf-8 -*-
"""补充两项"外部口径"指标，供考核指标对标使用：

1. 重复性（repeatability）：同一目标在加噪条件下的中心坐标标准差。
   工业视觉评价定位系统常用"重复性/重复定位精度"，比单次"准确率"更能反映稳定性。
2. 标定重投影误差（reprojection error）：合成棋盘格做内参标定，
   报告 cv2.calibrateCamera 返回的 RMS 重投影误差，作为"像素→物理坐标"的前置指标。

输出：docs/extra_metrics.json
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core.config import PipelineConfig      # noqa: E402
from src.core.pipeline import VisionPipeline    # noqa: E402


# ----------------------------------------------------------------------
def repeatability(image_path: str, noise_sigma: float, trials: int = 100):
    """在固定场景上重复加噪，统计目标中心的均值与标准差。"""
    base = cv2.imread(image_path)
    if base is None:
        return None
    pipeline = VisionPipeline(PipelineConfig())
    rng = np.random.default_rng(7)

    centers, hits, misses = [], 0, 0
    for _ in range(trials):
        noisy = np.clip(
            base.astype(np.float32) + rng.normal(0, noise_sigma, base.shape), 0, 255
        ).astype(np.uint8)
        result = pipeline.run(noisy)
        targets = pipeline.target_targets(result)
        if not targets:
            misses += 1
            continue
        t = max(targets, key=lambda x: x.area)
        centers.append((t.cx, t.cy))
        hits += 1

    if not centers:
        return {"noise_sigma": noise_sigma, "trials": trials, "hit_rate": 0.0}

    arr = np.array(centers)
    return {
        "noise_sigma": noise_sigma,
        "trials": trials,
        "hit_rate": round(hits / trials, 3),
        "cx_mean": round(float(arr[:, 0].mean()), 2),
        "cy_mean": round(float(arr[:, 1].mean()), 2),
        "cx_std": round(float(arr[:, 0].std()), 3),
        "cy_std": round(float(arr[:, 1].std()), 3),
        "radial_std": round(float(np.sqrt(arr[:, 0].std() ** 2 + arr[:, 1].std() ** 2)), 3),
        "max_dev": round(float(np.max(np.hypot(arr[:, 0] - arr[:, 0].mean(),
                                              arr[:, 1] - arr[:, 1].mean()))), 2),
    }


# ----------------------------------------------------------------------
def synthetic_checkerboard_calibration(boards: int = 6, cols: int = 9, rows: int = 6,
                                       square_mm: float = 25.0):
    """用合成棋盘格评估标定流程与重投影误差（真机换实拍图即可）。"""
    img_w, img_h = 1280, 720
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    obj_points, img_points = [], []
    rng = np.random.default_rng(11)

    # 基准棋盘（单位：像素）
    base = 90.0
    pts = np.zeros((cols * rows, 2), np.float32)
    pts[:, 0] = (np.tile(np.arange(cols), rows)) * base
    pts[:, 1] = (np.repeat(np.arange(rows), cols)) * base
    pts += np.array([120.0, 80.0], np.float32)

    for i in range(boards):
        # 模拟相机在不同倾角/位置下拍摄同一棋盘
        zoom = rng.uniform(0.55, 1.0)
        ang = rng.uniform(-18, 18)
        tx = rng.uniform(-180, 180)
        ty = rng.uniform(-120, 120)
        m = cv2.getRotationMatrix2D((float(pts[:, 0].mean()), float(pts[:, 1].mean())), ang, zoom)
        moved = cv2.transform(pts.reshape(-1, 1, 2), m).reshape(-1, 2)
        moved += np.array([tx, ty], np.float32)
        moved += rng.normal(0, 0.35, moved.shape).astype(np.float32)  # 亚像素定位误差
        img_points.append(moved.reshape(-1, 1, 2))
        obj_points.append(objp.copy())

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (img_w, img_h), None, None,
        flags=cv2.CALIB_ZERO_TANGENT_DIST,
    )
    return {
        "boards": boards,
        "board": f"{cols}x{rows} 内角，格长 {square_mm} mm",
        "image_size": [img_w, img_h],
        "rms_reprojection_px": round(float(rms), 3),
        "K": np.round(K, 2).tolist(),
        "dist": np.round(dist.ravel(), 4).tolist(),
    }


# ----------------------------------------------------------------------
def main() -> int:
    out = {"repeatability": [], "calibration": None}

    for path, label in ((os.path.join(ROOT, "data", "test002.png"), "test002"),
                        (os.path.join(ROOT, "data", "test001.png"), "test001")):
        if not os.path.exists(path):
            continue
        for sigma in (5.0, 10.0):
            r = repeatability(path, sigma)
            if r:
                r["image"] = label
                out["repeatability"].append(r)
                print("%-9s 噪声 σ=%-4.1f 命中 %5.1f%%  中心均值 (%.2f, %.2f)  "
                      "径向标准差 %.3f px  最大偏离 %.2f px" % (
                          label, sigma, r["hit_rate"] * 100, r["cx_mean"], r["cy_mean"],
                          r["radial_std"], r["max_dev"]))

    out["calibration"] = synthetic_checkerboard_calibration()
    print("合成棋盘格标定：重投影 RMS = %.3f px（%s）" % (
        out["calibration"]["rms_reprojection_px"], out["calibration"]["board"]))

    path = os.path.join(ROOT, "docs", "extra_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("已写入", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
