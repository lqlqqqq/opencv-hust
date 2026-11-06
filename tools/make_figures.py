# -*- coding: utf-8 -*-
"""生成中期检查报告用的插图。

产出（docs/figures/）::

    fig_pipeline.png        系统处理流程框图
    fig_lowlight_compare.png 低照度场景：预处理前后对比（含检测框）
    fig_real_cases.png      真实测试图检测结果（成功 + 失败对照）

用法::

    python tools/make_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core.config import PipelineConfig       # noqa: E402
from src.core.pipeline import VisionPipeline     # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# ----------------------------------------------------------------------
def draw_pipeline(plt) -> str:
    """画一张处理流程框图。"""
    fig, ax = plt.subplots(figsize=(10.5, 3.6), dpi=170)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.2, 1.6, 1.6, 1.0, "图像输入\n640×480", "#e8f1fb"),
        (2.3, 2.5, 1.9, 0.9, "颜色路预处理\nV 通道 γ 校正", "#fde9d9"),
        (2.3, 0.6, 1.9, 0.9, "形状路预处理\n去噪（不锐化）", "#e2f0d9"),
        (4.7, 2.5, 1.9, 0.9, "HSV 颜色分割\n形态学 + 主导度", "#fde9d9"),
        (4.7, 0.6, 1.9, 0.9, "Canny + 轮廓分析\n矩形 / 圆形", "#e2f0d9"),
        (7.1, 1.6, 1.7, 1.0, "颜色-形状\n融合判定", "#fff2cc"),
        (9.3, 2.4, 2.3, 0.85, "目标输出\n(ID, 类别, x, y, 置信度)", "#ddebf7"),
        (9.3, 0.9, 2.3, 0.85, "串口下发\n0xAA 0xBB … CRC16", "#ddebf7"),
    ]
    for (x, y, w, h, text, color) in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#4a4a4a", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    arrows = [
        ((1.8, 2.1), (2.3, 2.95)),
        ((1.8, 2.1), (2.3, 1.05)),
        ((4.2, 2.95), (4.7, 2.95)),
        ((4.2, 1.05), (4.7, 1.05)),
        ((6.6, 2.95), (7.1, 2.3)),
        ((6.6, 1.05), (7.1, 1.9)),
        ((8.8, 2.1), (9.3, 2.8)),
        ((8.8, 2.1), (9.3, 1.3)),
    ]
    for (start, end) in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color="#4a4a4a", lw=1.2))

    ax.set_title("视觉识别与定位系统处理流程（1.3 版）", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_pipeline.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
def draw_lowlight_compare(plt) -> str:
    """低照度场景：无预处理 vs 双路预处理。"""
    ann = json.load(open(os.path.join(ROOT, "dataset", "annotations.json"), encoding="utf-8"))
    item = [i for i in ann["images"] if i["condition"] == "dim"][1]
    img = cv2.imread(os.path.join(ROOT, "dataset", "images", item["file"]))

    cfg_off = PipelineConfig()
    cfg_on = PipelineConfig.preset("lowlight")

    p_off, p_on = VisionPipeline(cfg_off), VisionPipeline(cfg_on)
    r_off, r_on = p_off.run(img), p_on.run(img)

    canvas_off = VisionPipeline.annotate(img, p_off.target_targets(r_off))
    canvas_on = VisionPipeline.annotate(img, p_on.target_targets(r_on))
    mask_off = r_off.masks["color"]
    mask_on = r_on.masks["color"]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), dpi=170)
    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"原图（均值亮度 {img.mean():.0f}）\n目标像素 V≈88，低于阈值 V≥100"),
        (mask_off, f"未开启预处理：红色掩码\n检出目标 {len(p_off.target_targets(r_off))} 个"),
        (cv2.cvtColor(canvas_off, cv2.COLOR_BGR2RGB), "未开启预处理：检测结果（目标漏检）"),
        (cv2.cvtColor(canvas_on, cv2.COLOR_BGR2RGB),
         f"V 通道 γ 校正 + 双路预处理\n检出目标 {len(p_on.target_targets(r_on))} 个"),
    ]
    for ax, (data, title) in zip(axes.ravel(), panels):
        if data.ndim == 2:
            ax.imshow(data, cmap="gray")
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")
    fig.suptitle("低照度场景下的预处理效果对比", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_lowlight_compare.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
def draw_real_cases(plt) -> str:
    """真实测试图：成功案例与复杂场景失败案例对照。"""
    cases = [
        ("data/test002.png", "合成风格场景：红色圆形靶标准确检出"),
        ("data/test001.png", "复杂彩色场景：粉/品红块造成颜色误检（下阶段改进点）"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=170)
    pipeline = VisionPipeline(PipelineConfig())
    for ax, (rel, title) in zip(axes, cases):
        img = cv2.imread(os.path.join(ROOT, rel))
        result = pipeline.run(img)
        canvas = VisionPipeline.annotate(img, result.targets)
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title}\n耗时 {result.total_ms:.1f} ms", fontsize=9.5)
        ax.axis("off")
    fig.suptitle("真实测试图检测结果", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_real_cases.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    plt = _plt()
    for path in (draw_pipeline(plt), draw_lowlight_compare(plt), draw_real_cases(plt)):
        print("图:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
