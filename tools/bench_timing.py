# -*- coding: utf-8 -*-
"""固定协议的耗时测量：给"延迟/帧率"指标一个可复现的口径。

协议（写进验收方案，缺一条数字就不可比）：
1. 预热 10 帧后开始计时；
2. 每张图重复 3 轮，取每帧总耗时，再对所有帧取中位数与 P95；
3. 分两组报：合成集（200 张，算法本身开销）与真实集（data/ 下实拍图，含复杂背景的开销）；
4. 记录机器信息与并发状态（本机无其它负载）。

输出：docs/timing_bench.json
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core.config import PipelineConfig      # noqa: E402
from src.core.pipeline import VisionPipeline    # noqa: E402

REPEATS = 3
WARMUP = 10


def collect(paths):
    return [p for p in paths if os.path.exists(p)]


def measure(pipeline, paths, repeats=REPEATS, warmup=WARMUP):
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append((os.path.basename(p), img))
    if not images:
        return None

    # 预热（含 OpenCV/CLAHE 等首次调用的懒初始化开销）
    for i in range(warmup):
        pipeline.run(images[i % len(images)][1])

    totals, stages = [], {"preprocess": [], "color": [], "shape": []}
    per_file = {}
    for _ in range(repeats):
        for name, img in images:
            r = pipeline.run(img)
            totals.append(r.total_ms)
            for k in stages:
                stages[k].append(r.timings_ms[k])
            per_file.setdefault(name, []).append(r.total_ms)

    def stat(values):
        return {
            "mean": round(statistics.fmean(values), 2),
            "median": round(statistics.median(values), 2),
            "p95": round(float(__import__("numpy").percentile(values, 95)), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "n": len(values),
        }

    return {
        "frames": len(totals),
        "total_ms": stat(totals),
        "stages_ms": {k: stat(v) for k, v in stages.items()},
        "worst_files": sorted(
            ((round(max(v), 1), k) for k, v in per_file.items()), reverse=True
        )[:3],
        "fps_median": round(1000.0 / statistics.median(totals), 1),
        "fps_p95": round(1000.0 / float(__import__("numpy").percentile(totals, 95)), 1),
    }


def main() -> int:
    dataset_dir = os.path.join(ROOT, "dataset", "images")
    synthetic = sorted(
        os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".png")
    ) if os.path.isdir(dataset_dir) else []
    real = sorted(
        os.path.join(ROOT, "data", f) for f in os.listdir(os.path.join(ROOT, "data"))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    out = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
        },
        "protocol": {
            "warmup_frames": WARMUP,
            "repeats": REPEATS,
            "statistic": "median & p95 over all frames",
            "resolution": "640x480",
        },
        "configs": {},
    }

    for name, cfg in (
        ("default", PipelineConfig()),
        ("lowlight", PipelineConfig.preset("lowlight")),
    ):
        pipeline = VisionPipeline(cfg)
        out["configs"][name] = {
            "synthetic_200": measure(pipeline, synthetic),
            "real_4": measure(pipeline, real, repeats=10),
        }

    path = os.path.join(ROOT, "docs", "timing_bench.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    for name, groups in out["configs"].items():
        print(f"[{name}]")
        for g, m in groups.items():
            if not m:
                continue
            t = m["total_ms"]
            print(f"  {g:<14} 中位数 {t['median']:6.2f} ms  P95 {t['p95']:6.2f} ms  "
                  f"均值 {t['mean']:6.2f} ms  → 中位帧率 {m['fps_median']:.1f} FPS")
            print(f"  {'':14} 分阶段中位: 预处理 {m['stages_ms']['preprocess']['median']:.2f} / "
                  f"颜色 {m['stages_ms']['color']['median']:.2f} / "
                  f"形状 {m['stages_ms']['shape']['median']:.2f} ms")
            print(f"  {'':14} 最慢: {m['worst_files']}")
    print(f"\n已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
