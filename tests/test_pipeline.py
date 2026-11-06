# -*- coding: utf-8 -*-
"""流水线与配置的单元测试 / 冒烟测试。"""

import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.config import PipelineConfig, PreprocessConfig   # noqa: E402
from src.core.pipeline import VisionPipeline                    # noqa: E402
from src.core.result import FrameResult, Target                 # noqa: E402
from src.preprocessing.enhance import Preprocessor, estimate_brightness  # noqa: E402


def make_scene(w=640, h=480, dim=False):
    """构造一张可控测试图：白底 + 红圆 + 蓝方 + 红方。"""
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    cv2.circle(img, (150, 120), 40, (30, 30, 220), -1)      # 红色圆形（目标）
    cv2.rectangle(img, (350, 80), (470, 180), (200, 90, 40), -1)  # 蓝色矩形（异色干扰）
    cv2.rectangle(img, (180, 300), (320, 380), (30, 30, 220), -1)  # 红色矩形（同色异形）
    if dim:
        # 先归一化到 0~1 再做幂次，否则直接对 0~255 取幂会被 clip 成全白
        norm = img.astype(np.float32) / 255.0
        img = np.clip((norm ** 1.9) * 0.55 * 255.0, 0, 255).astype(np.uint8)
    return img


class TestConfig(unittest.TestCase):
    def test_presets(self):
        cfg = PipelineConfig.preset("lowlight")
        self.assertTrue(cfg.preprocess.enabled)
        self.assertEqual(cfg.preprocess.color_mode, "hsv_gamma")
        self.assertEqual(cfg.preprocess.shape_mode, "none")
        self.assertEqual(PipelineConfig.preset("default").target_shape, "circle")

    def test_json_roundtrip(self):
        import json
        import tempfile

        cfg = PipelineConfig()
        cfg.color.colors = ["red", "green"]
        cfg.preprocess.enabled = True
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.json")
            cfg.save(path)
            loaded = PipelineConfig.load(path)
        self.assertEqual(loaded.color.colors, ["red", "green"])
        self.assertTrue(loaded.preprocess.enabled)
        self.assertEqual(json.dumps(cfg.to_dict()), json.dumps(loaded.to_dict()))


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.img = make_scene()

    def test_detects_red_circle(self):
        pipeline = VisionPipeline(PipelineConfig())
        result = pipeline.run(self.img)
        targets = pipeline.target_targets(result)
        self.assertGreaterEqual(len(targets), 1)
        best = targets[0]
        self.assertAlmostEqual(best.cx, 150, delta=6)
        self.assertAlmostEqual(best.cy, 120, delta=6)

    def test_fusion_rejects_same_color_rectangle(self):
        pipeline = VisionPipeline(PipelineConfig())
        result = pipeline.run(self.img)
        # 红色矩形应被识别为 rectangle，而不是 circle
        red_rects = [t for t in result.targets if t.color == "red" and t.shape == "rectangle"]
        self.assertGreaterEqual(len(red_rects), 1)
        circles = [t for t in result.targets if t.color == "red" and t.shape == "circle"]
        self.assertTrue(all(abs(t.cx - 150) < 30 for t in circles))

    def test_timings_and_fps(self):
        pipeline = VisionPipeline(PipelineConfig())
        result = pipeline.run(self.img)
        self.assertIn("color", result.timings_ms)
        self.assertGreater(result.total_ms, 0)
        self.assertGreater(result.fps, 0)

    def test_target_only_output_filter(self):
        cfg = PipelineConfig()
        cfg.target_color = "red"
        cfg.target_shape = "circle"
        pipeline = VisionPipeline(cfg)
        targets = pipeline.target_targets(pipeline.run(self.img))
        for t in targets:
            self.assertEqual(t.color, "red")
            self.assertEqual(t.shape, "circle")

    def test_empty_image_rejected(self):
        pipeline = VisionPipeline(PipelineConfig())
        with self.assertRaises(ValueError):
            pipeline.run(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_annotation_image(self):
        pipeline = VisionPipeline(PipelineConfig())
        result = pipeline.run(self.img)
        canvas = VisionPipeline.annotate(self.img, result.targets)
        self.assertEqual(canvas.shape, self.img.shape)

    def test_frame_result_serialization(self):
        pipeline = VisionPipeline(PipelineConfig())
        payload = pipeline.run(self.img).to_dict()
        self.assertIn("targets", payload)
        self.assertEqual(payload["targets"][0]["label"].split("_")[0], "red")


class TestPreprocess(unittest.TestCase):
    def test_modes_run(self):
        img = make_scene()
        for mode in Preprocessor.MODES:
            out = Preprocessor(mode=mode).process(img)
            self.assertEqual(out.shape, img.shape)
            self.assertEqual(out.dtype, np.uint8)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            Preprocessor(mode="not_a_mode")

    def test_brightness_preserved(self):
        """锐化核归一化后整体亮度不应塌陷（原 /9 写法会掉到 1/9）。"""
        img = make_scene()
        out = Preprocessor(mode="clahe_sharpen").process(img)
        before = estimate_brightness(img)
        after = estimate_brightness(out)
        self.assertGreater(after, before * 0.8)

    def test_dim_image_recall_improves(self):
        """低照度下 V 通道 gamma 应把暗红目标抬回检测阈值。"""
        img = make_scene(dim=True)
        cfg_off = PipelineConfig()
        cfg_on = PipelineConfig.preset("lowlight")
        n_off = len(VisionPipeline(cfg_off).target_targets(VisionPipeline(cfg_off).run(img)))
        pipeline_on = VisionPipeline(cfg_on)
        n_on = len(pipeline_on.target_targets(pipeline_on.run(img)))
        self.assertGreaterEqual(n_on, 1)
        self.assertGreaterEqual(n_on, n_off)


if __name__ == "__main__":
    unittest.main(verbosity=2)
