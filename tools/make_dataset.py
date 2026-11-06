# -*- coding: utf-8 -*-
"""生成带真值标注的合成测试集，用于量化评测识别准确率与耗时。

设计思路
--------
项目书里的指标（识别准确率、延迟、帧率）必须在可复现的数据上测量。
真实拍摄成本高、标注慢，因此先构造 5 类典型场景：

============  ============================================================
条件          作用
============  ============================================================
normal        均匀背景 + 正常光照：基线准确率
clutter       纹理背景 + 大量同色/异色干扰物：抗干扰能力
dim           低照度（gamma=0.35）+ 噪声：低照度增强的有效性
bright_blur   过曝 + 运动模糊：强光/低对比鲁棒性
occlusion     目标相互遮挡：粘连与遮挡场景
============  ============================================================

每张图都输出真值：每个物体的类别（red_circle / red_rect / other）、
中心坐标、半径或宽高。目标类别固定为"红色圆形靶标"。

用法::

    python tools/make_dataset.py --out dataset --per-condition 40
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np

W, H = 640, 480

# 目标与干扰物颜色（BGR）
RED = (30, 30, 220)
OTHER_COLORS = {
    "green": (60, 180, 60),
    "blue": (200, 90, 40),
    "yellow": (40, 200, 220),
    "orange": (30, 130, 240),
    "purple": (170, 60, 140),
}


# ----------------------------------------------------------------------
def _draw_shape(img, shape, color, center, size, angle=0.0):
    """在图上绘制一个实心形状，并叠加轻微纹理避免颜色过于理想。"""
    cx, cy = int(center[0]), int(center[1])
    layer = img.copy()
    if shape == "circle":
        r = int(size)
        cv2.circle(layer, (cx, cy), r, color, -1, cv2.LINE_AA)
        bbox = (cx - r, cy - r, 2 * r, 2 * r)
    else:
        w, h = int(size[0]), int(size[1])
        box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.int32)
        cv2.fillPoly(layer, [box], color, cv2.LINE_AA)
        bbox = (cx - w // 2, cy - h // 2, w, h)

    # 仅在形状内部叠加 ±6 的亮度扰动，模拟真实物体表面
    x, y, bw, bh = bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + bw), min(img.shape[0], y + bh)
    if x1 > x0 and y1 > y0:
        noise = np.random.randint(-6, 7, (y1 - y0, x1 - x0, 1), dtype=np.int16)
        patch = layer[y0:y1, x0:x1].astype(np.int16) + noise
        layer[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return layer


def _background(kind: str, rng: np.random.Generator) -> np.ndarray:
    img = np.full((H, W, 3), 235, dtype=np.uint8)
    if kind == "texture":
        # 网格 + 斜线纹理，模拟桌面/墙面的干扰
        for _ in range(60):
            p1 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            p2 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            gray = int(rng.integers(180, 225))
            cv2.line(img, p1, p2, (gray, gray, gray), 1, cv2.LINE_AA)
    elif kind == "gradient":
        grad = np.linspace(0.55, 1.25, W, dtype=np.float32)
        img = np.clip(img.astype(np.float32) * grad[None, :, None], 0, 255).astype(np.uint8)
    # 轻微高斯噪声
    noise = rng.normal(0, 2.5, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _lighting(img: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32) / 255.0
    if mode == "dim":
        out = np.power(out, 1.0 / 0.30)  # gamma < 1 的逆操作 → 明显变暗
        out *= 0.60
    elif mode == "bright":
        out = np.power(out, 1.0 / 1.6) * 1.15
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    if mode == "dim":
        noise = rng.normal(0, 9.0, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if mode == "bright":
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def _place_non_overlapping(rng, existing, w, h, margin=8, tries=120):
    """随机找一个不与已有物体重叠的位置。"""
    for _ in range(tries):
        cx = rng.integers(w // 2 + margin, W - w // 2 - margin)
        cy = rng.integers(h // 2 + margin, H - h // 2 - margin)
        ok = True
        for (ox, oy, ow, oh) in existing:
            if abs(cx - ox) < (w + ow) / 2 + margin and abs(cy - oy) < (h + oh) / 2 + margin:
                ok = False
                break
        if ok:
            return int(cx), int(cy)
    return None


# ----------------------------------------------------------------------
def build_scene(condition: str, rng: np.random.Generator) -> Tuple[np.ndarray, List[Dict]]:
    bg_kind = {"normal": "uniform", "clutter": "texture", "dim": "uniform",
               "bright_blur": "gradient", "occlusion": "uniform"}[condition]
    light = {"normal": "normal", "clutter": "normal", "dim": "dim",
             "bright_blur": "bright", "occlusion": "normal"}[condition]

    img = _background(bg_kind, rng)
    objects: List[Dict] = []
    occupied: List[Tuple[int, int, int, int]] = []

    n_targets = 2 if condition != "occlusion" else 2
    n_red_rects = 2 if condition == "clutter" else 0
    n_others = {"normal": 4, "clutter": 9, "dim": 4, "bright_blur": 5, "occlusion": 3}[condition]

    def add(shape, color, label, size, angle=0.0):
        if shape == "circle":
            r = int(size)
            pos = _place_non_overlapping(rng, occupied, 2 * r, 2 * r)
            if pos is None:
                return False
            cx, cy = pos
            img2 = _draw_shape(img, "circle", color, (cx, cy), r)
            img[:, :, :] = img2
            occupied.append((cx, cy, 2 * r, 2 * r))
            objects.append({"label": label, "color": color, "shape": "circle",
                            "cx": cx, "cy": cy, "radius": r})
        else:
            w, h = int(size[0]), int(size[1])
            pos = _place_non_overlapping(rng, occupied, w, h)
            if pos is None:
                return False
            cx, cy = pos
            img2 = _draw_shape(img, "rectangle", color, (cx, cy), (w, h), angle)
            img[:, :, :] = img2
            occupied.append((cx, cy, w, h))
            objects.append({"label": label, "color": color, "shape": "rectangle",
                            "cx": cx, "cy": cy, "width": w, "height": h, "angle": angle})
        return True

    # 1) 目标：红色圆形靶标
    for _ in range(n_targets):
        r = int(rng.integers(22, 40))
        add("circle", RED, "red_circle", r)

    # 2) 同色干扰：红色矩形（考验"形状"这一路）
    for _ in range(n_red_rects):
        w = int(rng.integers(50, 95))
        h = int(rng.integers(35, 70))
        add("rectangle", RED, "red_rect", (w, h), angle=float(rng.integers(-30, 30)))

    # 3) 异色干扰：其它颜色的圆/矩形（颜色这一路应当直接滤掉）
    other_names = list(OTHER_COLORS.keys())
    for _ in range(n_others):
        name = other_names[int(rng.integers(0, len(other_names)))]
        if rng.random() < 0.5:
            add("circle", OTHER_COLORS[name], "other", int(rng.integers(18, 38)))
        else:
            add("rectangle", OTHER_COLORS[name], "other",
                (int(rng.integers(40, 90)), int(rng.integers(30, 70))),
                angle=float(rng.integers(-30, 30)))

    # 4) 遮挡场景：再叠一个部分遮挡目标的大圆
    if condition == "occlusion" and objects:
        base = objects[0]
        r = int(base["radius"]) - 6
        off = (int(rng.integers(-r // 2, r // 2)), int(rng.integers(-r // 2, r // 2)))
        img = _draw_shape(img, "circle", (90, 90, 90),
                          (base["cx"] + off[0], base["cy"] + off[1]), r)

    img = _lighting(img, light, rng)

    if condition == "bright_blur":
        k = int(rng.choice([5, 7]))
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)

    return img, objects


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="生成带真值的合成测试集")
    ap.add_argument("--out", default="dataset", help="输出目录")
    ap.add_argument("--per-condition", type=int, default=40, help="每个条件的图片数")
    ap.add_argument("--seed", type=int, default=20260401)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    annotations: List[Dict] = []
    conditions = ["normal", "clutter", "dim", "bright_blur", "occlusion"]

    for cond in conditions:
        for i in range(args.per_condition):
            img, objects = build_scene(cond, rng)
            name = f"{cond}_{i:03d}.png"
            cv2.imwrite(os.path.join(img_dir, name), img)
            annotations.append(
                {
                    "file": name,
                    "condition": cond,
                    "width": W,
                    "height": H,
                    "objects": objects,
                }
            )

    meta = {
        "image_size": [W, H],
        "conditions": conditions,
        "per_condition": args.per_condition,
        "seed": args.seed,
        "count": len(annotations),
        "target_label": "red_circle",
        "note": "label: red_circle=目标, red_rect=同色异形干扰, other=异色干扰",
    }
    with open(os.path.join(out_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "images": annotations}, f, ensure_ascii=False, indent=1)

    print(f"已生成 {len(annotations)} 张图片 → {img_dir}")
    print(f"标注文件 → {os.path.join(out_dir, 'annotations.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
