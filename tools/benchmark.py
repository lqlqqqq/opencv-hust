# -*- coding: utf-8 -*-
"""量化评测脚本：在合成数据集上测量召回率、精确率、定位误差与耗时。

评测口径（与项目书指标对齐）
----------------------------
* **颜色召回率 / 精确率**：只看"红色块"这一级，检验颜色分割是否把目标区域找出来。
* **形状判定准确率**：对已匹配到的红色块，判断形状（圆形/矩形）是否正确。
* **目标判定准确率**：同时满足"红色 + 圆形"才算命中目标类别（红色圆形靶标）。
  同色异形干扰（红色矩形）会被计入误检，这正是颜色单路与融合路的核心差别。
* **定位误差**：匹配成功的目标中心与真值中心的欧氏距离（px），报告均值与 RMSE。
* **耗时 / 帧率**：单帧总耗时与各阶段耗时，报告均值与 P95。

用法::

    python tools/benchmark.py --dataset dataset --out docs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.config import PipelineConfig, PreprocessConfig   # noqa: E402
from src.core.pipeline import VisionPipeline                   # noqa: E402


# ----------------------------------------------------------------------
def make_variants() -> Dict[str, PipelineConfig]:
    """构造对照实验用的三套配置。"""
    color_only = PipelineConfig()
    color_only.shape.enabled = False
    color_only.preprocess.enabled = False

    fusion = PipelineConfig()

    fusion_lowlight = PipelineConfig()
    fusion_lowlight.preprocess = PreprocessConfig(enabled=True, color_mode="hsv_gamma", shape_mode="none")

    return {"color_only": color_only, "fusion": fusion, "fusion_lowlight": fusion_lowlight}


# ----------------------------------------------------------------------
def gt_size(obj: Dict) -> float:
    if obj["shape"] == "circle":
        return float(obj["radius"]) * 2.0
    return float(max(obj["width"], obj["height"]))


def greedy_match(gts: List[Dict], preds: List[Dict]) -> Tuple[List[Tuple[Dict, Dict, float]], List[Dict]]:
    """按中心距离贪心匹配，返回 (匹配对, 未匹配预测)。"""
    used = set()
    pairs: List[Tuple[Dict, Dict, float]] = []
    for gt in gts:
        tol = max(15.0, 0.35 * gt_size(gt))
        best_i, best_d = -1, float("inf")
        for i, p in enumerate(preds):
            if i in used:
                continue
            d = math.hypot(p["cx"] - gt["cx"], p["cy"] - gt["cy"])
            if d <= tol and d < best_d:
                best_i, best_d = i, d
        if best_i >= 0:
            used.add(best_i)
            pairs.append((gt, preds[best_i], best_d))
    unmatched = [p for i, p in enumerate(preds) if i not in used]
    return pairs, unmatched


def to_pred_list(targets) -> List[Dict]:
    return [
        {
            "cx": t.cx,
            "cy": t.cy,
            "color": t.color,
            "shape": t.shape,
            "confidence": t.confidence,
            "source": t.source,
            "area": t.area,
        }
        for t in targets
    ]


# ----------------------------------------------------------------------
def wilson_interval(k: int, n: int, z: float = 1.96):
    """二项比例的 Wilson 置信区间（小样本下比正态近似更稳）。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _new_stats() -> Dict:
    """单个场景的累计计数器。"""
    return {
        "n_gt_red": 0, "tp_color": 0, "fp_color": 0,
        "n_gt_target": 0, "tp_target": 0, "fp_target": 0,
        "tp_circle_matched": 0,
        "n_matched_shape": 0, "n_shape_ok": 0,
        "err_sum": 0.0, "err_sq": 0.0, "n_err": 0, "errs": [],
        "ms": [], "ms_pre": [], "ms_color": [], "ms_shape": [],
    }


def evaluate(annotations: Dict, dataset_dir: str, variants: Dict[str, PipelineConfig]) -> Dict:
    images = annotations["images"]
    results: Dict[str, Dict] = {}
    per_image_rows: List[Dict] = []

    for name, cfg in variants.items():
        pipeline = VisionPipeline(cfg)
        supports_shape = name != "color_only"
        agg: Dict[str, Dict] = defaultdict(_new_stats)

        for item in images:
            path = os.path.join(dataset_dir, "images", item["file"])
            img = cv2.imread(path)
            if img is None:
                continue
            result = pipeline.run(img)
            preds = to_pred_list(result.targets)

            cond = item["condition"]
            a = agg[cond]

            red_gts = [o for o in item["objects"] if o["label"] in ("red_circle", "red_rect")]
            target_gts = [o for o in item["objects"] if o["label"] == "red_circle"]

            red_preds = [p for p in preds if p["color"] == "red"]
            pairs, unmatched = greedy_match(red_gts, red_preds)

            a["n_gt_red"] += len(red_gts)
            a["tp_color"] += len(pairs)
            a["fp_color"] += len(unmatched)
            for gt, pred, dist in pairs:
                if gt["label"] == "red_circle":
                    a["tp_circle_matched"] += 1
                a["n_matched_shape"] += 1
                if pred["shape"] == gt["shape"]:
                    a["n_shape_ok"] += 1
                a["err_sum"] += dist
                a["err_sq"] += dist * dist
                a["n_err"] += 1
                a["errs"].append(dist)

            # 目标类别（红色圆形）级别
            circle_preds = [
                p for p in preds
                if p["color"] == "red" and p["shape"] == "circle"
            ]
            t_pairs, t_unmatched = greedy_match(target_gts, circle_preds)
            a["n_gt_target"] += len(target_gts)
            a["tp_target"] += len(t_pairs)
            a["fp_target"] += len(t_unmatched)

            a["ms"].append(result.total_ms)
            a["ms_pre"].append(result.timings_ms["preprocess"])
            a["ms_color"].append(result.timings_ms["color"])
            a["ms_shape"].append(result.timings_ms["shape"])

            per_image_rows.append(
                {
                    "variant": name,
                    "file": item["file"],
                    "condition": cond,
                    "gt_red": len(red_gts),
                    "tp_red": len(pairs),
                    "fp_red": len(unmatched),
                    "gt_target": len(target_gts),
                    "tp_target": len(t_pairs),
                    "fp_target": len(t_unmatched),
                    "ms": round(result.total_ms, 3),
                }
            )

        # 汇总（含全部场景合计；小样本比例给 Wilson 95% 置信区间）
        overall = _new_stats()
        for _cond, _s in agg.items():
            for _k, _v in _s.items():
                if isinstance(_v, list):
                    overall[_k].extend(_v)
                else:
                    overall[_k] += _v
        agg["all"] = overall
        summary = {}
        for cond, a in agg.items():
            def ratio(num, den):
                return round(num / den, 4) if den else 0.0

            color_recall = ratio(a["tp_color"], a["n_gt_red"])
            color_precision = ratio(a["tp_color"], a["tp_color"] + a["fp_color"])
            target_recall = ratio(a["tp_target"], a["n_gt_target"])
            target_precision = ratio(a["tp_target"], a["tp_target"] + a["fp_target"])
            err_mean = round(a["err_sum"] / a["n_err"], 2) if a["n_err"] else 0.0
            err_rmse = round(math.sqrt(a["err_sq"] / a["n_err"]), 2) if a["n_err"] else 0.0
            # 端到端口径：把系统输出直接当作靶标坐标时的召回/精确率。
            # 融合变体输出的是"红色圆形"，颜色单路输出的是全部红色块，
            # 因此颜色单路会把同色矩形也算作目标，精确率会明显下降。
            if supports_shape:
                e2e_recall = ratio(a["tp_target"], a["n_gt_target"])
                e2e_precision = ratio(a["tp_target"], a["tp_target"] + a["fp_target"])
            else:
                e2e_recall = ratio(a["tp_circle_matched"], a["n_gt_target"])
                e2e_precision = ratio(
                    a["tp_circle_matched"], a["tp_color"] + a["fp_color"]
                )
            summary[cond] = {
                "n_gt_red": a["n_gt_red"],
                "color_recall": color_recall,
                "color_precision": color_precision,
                "color_f1": round(
                    2 * color_recall * color_precision / (color_recall + color_precision), 4
                ) if (color_recall + color_precision) else 0.0,
                "shape_acc": ratio(a["n_shape_ok"], a["n_matched_shape"]),
                "target_recall": target_recall,
                "target_precision": target_precision,
                "target_f1": round(
                    2 * target_recall * target_precision / (target_recall + target_precision), 4
                ) if (target_recall + target_precision) else 0.0,
                "e2e_recall": e2e_recall,
                "e2e_precision": e2e_precision,
                "center_err_mean": err_mean,
                "center_err_rmse": err_rmse,
                "center_err_p50": round(float(np.percentile(a["errs"], 50)), 2) if a["errs"] else 0.0,
                "center_err_p95": round(float(np.percentile(a["errs"], 95)), 2) if a["errs"] else 0.0,
                "e2e_recall_ci95": [round(v, 4) for v in wilson_interval(a["tp_target"], a["n_gt_target"])],
                "e2e_precision_ci95": [round(v, 4) for v in
                                       wilson_interval(a["tp_target"], a["tp_target"] + a["fp_target"])],
                "ms_mean": round(float(np.mean(a["ms"])), 2),
                "ms_p95": round(float(np.percentile(a["ms"], 95)), 2),
                "ms_preprocess": round(float(np.mean(a["ms_pre"])), 2),
                "ms_color": round(float(np.mean(a["ms_color"])), 2),
                "ms_shape": round(float(np.mean(a["ms_shape"])), 2),
                "fps_mean": round(1000.0 / float(np.mean(a["ms"])), 1),
            }
        results[name] = summary

    return {"variants": results, "per_image": per_image_rows}


# ----------------------------------------------------------------------
def to_markdown(results: Dict, dataset_meta: Dict) -> str:
    lines: List[str] = []
    lines.append("# 合成数据集评测结果\n")
    lines.append(
        f"数据集：{dataset_meta['count']} 张 640×480 合成图（每类 "
        f"{dataset_meta['per_condition']} 张），目标类别 = 红色圆形靶标。\n"
    )
    cond_names = {
        "normal": "均匀背景/正常光照",
        "clutter": "纹理背景/同色异形干扰",
        "dim": "低照度 + 噪声",
        "bright_blur": "过曝 + 运动模糊",
        "occlusion": "目标遮挡",
    }

    for variant, summary in results["variants"].items():
        supports_shape = variant != "color_only"
        lines.append(f"\n## 变体：{variant}\n")
        if not supports_shape:
            lines.append(
                "> 该变体关闭了形状检测，只评估颜色分割与定位；"
                "形状/目标类别指标以「—」表示，"
                "端到端列按\"把所有红色块都当作靶标\"统计。\n"
            )
        lines.append(
            "| 场景 | 颜色召回 | 颜色精确率 | 形状准确率 | 目标召回 | 目标精确率 | "
            "端到端召回 | 端到端精确率 | 中心误差均值(px) | 中心误差RMSE(px) | "
            "平均耗时(ms) | P95(ms) | FPS |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

        def fmt(value: float, enabled: bool = True) -> str:
            return f"{value:.3f}" if enabled else "—"

        for cond, m in summary.items():
            lines.append(
                f"| {cond_names.get(cond, cond)} | {m['color_recall']:.3f} | "
                f"{m['color_precision']:.3f} | {fmt(m['shape_acc'], supports_shape)} | "
                f"{fmt(m['target_recall'], supports_shape)} | "
                f"{fmt(m['target_precision'], supports_shape)} | "
                f"{m['e2e_recall']:.3f} | {m['e2e_precision']:.3f} | "
                f"{m['center_err_mean']:.2f} | {m['center_err_rmse']:.2f} | "
                f"{m['ms_mean']:.2f} | {m['ms_p95']:.2f} | {m['fps_mean']:.1f} |"
            )
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
def plot_figures(results: Dict, out_dir: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[警告] matplotlib 不可用，跳过绘图: {exc}")
        return []

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    conds = list(next(iter(results["variants"].values())).keys())
    labels = {
        "normal": "正常光照", "clutter": "复杂背景", "dim": "低照度",
        "bright_blur": "过曝+模糊", "occlusion": "目标遮挡",
    }
    x = np.arange(len(conds))
    width = 0.2

    # 图 1：融合变体下各场景的颜色/目标指标
    fusion = results["variants"].get("fusion_lowlight", results["variants"]["fusion"])
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
    ax.bar(x - 1.5 * width, [fusion[c]["color_recall"] for c in conds], width, label="颜色召回")
    ax.bar(x - 0.5 * width, [fusion[c]["color_precision"] for c in conds], width, label="颜色精确率")
    ax.bar(x + 0.5 * width, [fusion[c]["target_recall"] for c in conds], width, label="目标召回")
    ax.bar(x + 1.5 * width, [fusion[c]["target_precision"] for c in conds], width, label="目标精确率")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in conds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("指标值")
    ax.set_title("融合检测（含双路预处理）在各场景下的表现")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_metrics_by_scene.png")
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    # 图 2：颜色单路 vs 融合（目标精确率对比，重点看同色异形干扰）
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
    names = {"color_only": "颜色单路", "fusion": "颜色+形状融合",
             "fusion_lowlight": "融合+双路预处理"}
    for i, (variant, summary) in enumerate(results["variants"].items()):
        vals = [summary[c]["e2e_precision"] for c in conds]
        ax.bar(x + (i - 1) * width * 1.4, vals, width * 1.3,
               label=names.get(variant, variant))
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in conds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("端到端精确率")
    ax.set_title("不同配置下端到端精确率（红色圆形靶标）")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_variant_precision.png")
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    # 图 3：耗时分解
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    pre = [fusion[c]["ms_preprocess"] for c in conds]
    col = [fusion[c]["ms_color"] for c in conds]
    sha = [fusion[c]["ms_shape"] for c in conds]
    ax.bar(x, pre, 0.5, label="预处理")
    ax.bar(x, col, 0.5, bottom=pre, label="颜色检测")
    ax.bar(x, sha, 0.5, bottom=np.array(pre) + np.array(col), label="形状检测")
    for xi, c in zip(x, conds):
        total = fusion[c]["ms_mean"]
        ax.text(xi, total + 0.4, f"{total:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in conds])
    ax.set_ylabel("耗时 (ms)")
    ax.set_title("单帧耗时分解（含预处理，640×480）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_timing.png")
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)
    return saved


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="合成数据集量化评测")
    ap.add_argument("--dataset", default="dataset", help="数据集目录")
    ap.add_argument("--out", default="docs", help="结果输出目录")
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset)
    ann_path = os.path.join(dataset_dir, "annotations.json")
    if not os.path.exists(ann_path):
        print(f"[错误] 找不到标注文件: {ann_path}，请先运行 tools/make_dataset.py")
        return 2
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    variants = make_variants()
    t0 = time.time()
    report = evaluate(annotations, dataset_dir, variants)
    elapsed = time.time() - t0

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    payload = {"meta": annotations["meta"], "eval_seconds": round(elapsed, 2), **report}
    with open(os.path.join(out_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    md = to_markdown(report, annotations["meta"])
    with open(os.path.join(out_dir, "benchmark_table.md"), "w", encoding="utf-8") as f:
        f.write(md)

    figures = plot_figures(report, os.path.join(out_dir, "figures"))

    print(md)
    print(f"评测用时 {elapsed:.1f} s")
    for p in figures:
        print(f"图: {p}")
    print(f"结果: {os.path.join(out_dir, 'benchmark_results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
