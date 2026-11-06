# -*- coding: utf-8 -*-
"""兼容旧版入口的冒烟测试脚本。

问题背景：原 ``main_test.py`` 与 ``src/detectors/shape.py`` 的接口签名不一致
（前者按 ``detect_rectangles(path, min_area=...)`` 调用，后者要求
``(img, img_shape, min_area)``），导致形状检测一路直接抛异常；同时脚本里
调用了 ``cv2.waitKey(0)``，无法在无界面环境中运行。

1.3 版本统一走 :class:`src.core.pipeline.VisionPipeline`，该脚本保留为
最小调用示例与回归冒烟测试::

    python main_test.py                     # 默认处理 data/test002.png
    python main_test.py data/test001.png
"""

from __future__ import annotations

import os
import sys

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import print_targets, save_outputs          # noqa: E402
from src.core.config import PipelineConfig            # noqa: E402
from src.core.pipeline import VisionPipeline          # noqa: E402


def main() -> int:
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "test002.png")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[错误] 无法读取图片: {image_path}")
        return 2

    cfg = PipelineConfig()
    pipeline = VisionPipeline(cfg)
    result = pipeline.run(image)
    targets = pipeline.target_targets(result)

    print(f"图片: {image_path}  尺寸: {result.image_size[0]}x{result.image_size[1]}")
    print(
        f"颜色块: {result.debug['color_count']} 个 | 形状: {result.debug['shape_count']} 个 | "
        f"融合后目标: {len(result.targets)} 个"
    )
    print_targets(targets, prefix="  ")
    print(f"耗时: {result.total_ms:.2f} ms ({result.fps:.1f} FPS)")

    save_dir = os.path.join(ROOT, "result")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for key, path in save_outputs(image, result, save_dir, stem).items():
        print(f"  [{key}] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
