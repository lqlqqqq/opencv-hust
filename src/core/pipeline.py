# -*- coding: utf-8 -*-
"""统一视觉流水线：预处理 → 颜色检测 → 形状检测 → 颜色/形状融合 → 目标列表。

对外只暴露 :class:`VisionPipeline`，供三种入口复用：

* ``main.py``              单图 / 批量 / 摄像头
* ``tools/benchmark.py``   离线评测
* 以后 C++ 版本的 ``main.cpp``（结构与阶段顺序保持一致）
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.config import PipelineConfig
from src.core.result import FrameResult, Target
from src.detectors.color import ColorDetector
from src.detectors.shape import detect_circles, detect_rectangles
from src.preprocessing.enhance import Preprocessor


class VisionPipeline:
    """可复用的检测流水线实例（内部缓存检测器，适合逐帧调用）。"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._color_detectors: Dict[str, ColorDetector] = {}
        self._build_preprocessors()
        self._build_color_detectors()

    # ------------------------------------------------------------------
    def _make_preprocessor(self, mode: str) -> Preprocessor:
        pcfg = self.config.preprocess
        return Preprocessor(
            mode=mode,
            gaussian_ksize=pcfg.gaussian_ksize,
            median_ksize=pcfg.median_ksize,
            clip_limit=pcfg.clip_limit,
            tile_grid=pcfg.tile_grid,
            sharpen=pcfg.sharpen,
            gamma=pcfg.gamma,
            target_luma=pcfg.target_luma,
        )

    def _build_preprocessors(self) -> None:
        pcfg = self.config.preprocess
        if pcfg.enabled:
            self._color_pp = self._make_preprocessor(pcfg.color_mode)
            self._shape_pp = self._make_preprocessor(pcfg.shape_mode)
        else:
            self._color_pp = self._make_preprocessor("none")
            self._shape_pp = self._make_preprocessor("none")

    def _build_color_detectors(self) -> None:
        cfg = self.config.color
        self._color_detectors.clear()
        if not cfg.enabled:
            return
        for color in cfg.colors:
            self._color_detectors[color] = ColorDetector(
                target_color=color,
                min_area=cfg.min_area,
                confidence_threshold=cfg.confidence_threshold,
                enable_morphology=cfg.enable_morphology,
                morph_kernel_size=cfg.morph_kernel_size,
                dominance_ratio=cfg.dominance_ratio,
            )

    def set_config(self, config: PipelineConfig) -> None:
        """热更新配置（例如命令行改参数后重新构建检测器）。"""
        self.config = config
        self._build_preprocessors()
        self._build_color_detectors()

    # ------------------------------------------------------------------
    def _detect_color(self, image: np.ndarray) -> Tuple[List[Target], np.ndarray]:
        """多颜色检测，返回目标列表与合并后的颜色掩码。"""
        h, w = image.shape[:2]
        merged_mask = np.zeros((h, w), dtype=np.uint8)
        targets: List[Target] = []

        for color, detector in self._color_detectors.items():
            detections, mask = detector.detect_from_bgr(image, verbose=False)
            merged_mask = cv2.bitwise_or(merged_mask, mask)
            for d in detections:
                targets.append(
                    Target(
                        id=len(targets) + 1,
                        color=color,
                        shape="unknown",
                        center=(float(d["center_x"]), float(d["center_y"])),
                        width=float(d["width"]),
                        height=float(d["height"]),
                        area=float(d["area"]),
                        confidence=float(d["confidence"]),
                        angle=float(d.get("angle", 0.0)),
                        source="color",
                        extra={"circularity": float(d.get("circularity", 0.0))},
                    )
                )
        return targets, merged_mask

    def _detect_shape(self, image: np.ndarray) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
        """矩形 + 圆形检测，返回形状字典列表与掩码。"""
        cfg = self.config.shape
        h, w = image.shape[:2]
        shapes: List[Dict] = []
        masks: Dict[str, np.ndarray] = {}

        if cfg.detect_rectangles:
            rects, rect_mask = detect_rectangles(
                image, (h, w), min_area=cfg.min_rectangle_area
            )
            masks["rectangle"] = rect_mask
            for r in rects:
                shapes.append(
                    {
                        "shape": "rectangle",
                        "center": (float(r["center"][0]), float(r["center"][1])),
                        "width": float(r["long_side"]),
                        "height": float(r["short_side"]),
                        "angle": float(r.get("angle", 0.0)),
                        "area": float(r.get("area", 0.0)),
                    }
                )

        if cfg.detect_circles:
            circles, circ_mask = detect_circles(
                image,
                (h, w),
                min_area=cfg.min_circle_area,
                min_circularity=cfg.min_circularity,
                max_aspect_ratio_diff=cfg.max_aspect_ratio_diff,
            )
            masks["circle"] = circ_mask
            for c in circles:
                shapes.append(
                    {
                        "shape": "circle",
                        "center": (float(c["center"][0]), float(c["center"][1])),
                        "width": float(c["diameter"]),
                        "height": float(c["diameter"]),
                        "radius": float(c["radius"]),
                        "area": float(c["area"]),
                        "circularity": float(c["circularity"]),
                    }
                )
        return shapes, masks

    # ------------------------------------------------------------------
    def _fuse(self, color_targets: List[Target], shapes: List[Dict]) -> List[Target]:
        """把颜色块与形状按中心距离配对，配对成功即判定为 (颜色, 形状) 目标。

        设计动机：只看颜色时，同色的其它形状也会被报为目标；只看形状时，
        背景里同形状的干扰物也会被报为目标。两者同时成立才认为命中了目标类别。
        """
        fcfg = self.config.fusion
        if not fcfg.enabled or not shapes:
            return list(color_targets)

        used = [False] * len(shapes)
        fused: List[Target] = []

        for ct in color_targets:
            short_side = max(1.0, min(ct.width, ct.height))
            max_dist = fcfg.max_center_distance or min(30.0, 0.5 * short_side)

            best_idx = -1
            best_dist = float("inf")
            for i, s in enumerate(shapes):
                if used[i]:
                    continue
                sx, sy = s["center"]
                dist = float(np.hypot(sx - ct.cx, sy - ct.cy))
                # 形状中心必须落在颜色块附近；同时用「短边」做尺寸一致性约束，
                # 避免把相邻的大形状配到小色块上（长边差异由宽高比天然排除）。
                sw = max(float(s.get("width", 0.0)), 1.0)
                sh = max(float(s.get("height", 0.0)), 1.0)
                shape_short = min(sw, sh)
                size_ratio = min(shape_short, short_side) / max(shape_short, short_side)
                if dist <= max_dist and size_ratio >= 0.45 and dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                s = shapes[best_idx]
                used[best_idx] = True
                fused.append(
                    Target(
                        id=ct.id,
                        color=ct.color,
                        shape=s["shape"],
                        center=(s["center"][0], s["center"][1]),
                        width=float(s.get("width", ct.width)),
                        height=float(s.get("height", ct.height)),
                        radius=float(s.get("radius", 0.0)),
                        area=float(s.get("area", ct.area)),
                        confidence=round(0.5 * ct.confidence + 0.5 * min(1.0, s.get("circularity", 1.0)), 4),
                        angle=float(s.get("angle", ct.angle)),
                        source="fusion",
                        extra={"match_dist": round(best_dist, 2), "color_conf": ct.confidence},
                    )
                )
            elif fcfg.keep_unmatched_color:
                fused.append(ct)

        if fcfg.keep_unmatched_shape:
            for i, s in enumerate(shapes):
                if used[i]:
                    continue
                fused.append(
                    Target(
                        id=0,
                        color="unknown",
                        shape=s["shape"],
                        center=(s["center"][0], s["center"][1]),
                        width=float(s.get("width", 0.0)),
                        height=float(s.get("height", 0.0)),
                        radius=float(s.get("radius", 0.0)),
                        area=float(s.get("area", 0.0)),
                        confidence=0.3,
                        angle=float(s.get("angle", 0.0)),
                        source="shape",
                    )
                )
        return fused

    # ------------------------------------------------------------------
    def run(self, image_bgr: np.ndarray) -> FrameResult:
        """执行一次完整检测。

        Args:
            image_bgr: OpenCV BGR 图像

        Returns:
            FrameResult，包含目标列表、各阶段耗时、掩码与图像尺寸
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("输入图像为空")

        cfg = self.config
        timings: Dict[str, float] = {}
        h, w = image_bgr.shape[:2]
        result = FrameResult(image_size=(w, h))

        # 1) 预处理（双路：颜色路与形状路各自增强）
        t0 = time.perf_counter()
        if cfg.preprocess.enabled:
            color_image = (
                self._color_pp.process(image_bgr)
                if cfg.preprocess.color_mode != "none"
                else image_bgr
            )
            shape_image = (
                self._shape_pp.process(image_bgr)
                if cfg.preprocess.shape_mode != "none"
                else image_bgr
            )
        else:
            color_image = image_bgr
            shape_image = image_bgr
        timings["preprocess"] = (time.perf_counter() - t0) * 1000.0

        # 2) 颜色检测
        t0 = time.perf_counter()
        color_targets: List[Target] = []
        if cfg.color.enabled:
            color_targets, color_mask = self._detect_color(color_image)
        else:
            color_mask = np.zeros((h, w), dtype=np.uint8)
        timings["color"] = (time.perf_counter() - t0) * 1000.0

        # 3) 形状检测
        t0 = time.perf_counter()
        shapes: List[Dict] = []
        shape_masks: Dict[str, np.ndarray] = {}
        if cfg.shape.enabled:
            shapes, shape_masks = self._detect_shape(shape_image)
        timings["shape"] = (time.perf_counter() - t0) * 1000.0

        # 4) 融合
        t0 = time.perf_counter()
        targets = self._fuse(color_targets, shapes)
        targets.sort(key=lambda t: t.area, reverse=True)
        for i, t in enumerate(targets, 1):
            t.id = i
        timings["fusion"] = (time.perf_counter() - t0) * 1000.0

        result.targets = targets
        result.timings_ms = timings
        result.masks = {"color": color_mask, "shape": shape_masks}
        result.debug = {
            "color_count": len(color_targets),
            "shape_count": len(shapes),
            "working_image": color_image if cfg.preprocess.enabled else None,
            "color_image": color_image if cfg.preprocess.enabled else None,
            "shape_image": shape_image if cfg.preprocess.enabled else None,
        }
        return result

    # ------------------------------------------------------------------
    def target_targets(self, result: FrameResult) -> List[Target]:
        """按配置的目标类别（目标颜色 + 目标形状）过滤目标列表。"""
        color = self.config.target_color
        shape = self.config.target_shape
        return [
            t
            for t in result.targets
            if t.color == color and (shape == "any" or t.shape == shape)
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def annotate(image_bgr: np.ndarray, targets: List[Target]) -> np.ndarray:
        """在图像上绘制检测结果，用于结果保存与答辩插图。"""
        canvas = image_bgr.copy()
        for t in targets:
            cx, cy = int(round(t.cx)), int(round(t.cy))
            if t.shape == "circle" and t.radius > 0:
                cv2.circle(canvas, (cx, cy), int(round(t.radius)), (0, 255, 0), 2)
            else:
                half_w = int(round(max(t.width, 1.0) / 2))
                half_h = int(round(max(t.height, 1.0) / 2))
                cv2.rectangle(
                    canvas,
                    (cx - half_w, cy - half_h),
                    (cx + half_w, cy + half_h),
                    (0, 255, 0),
                    2,
                )
            cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            label = f"#{t.id} {t.color}/{t.shape} {t.confidence:.2f}"
            cv2.putText(
                canvas,
                label,
                (max(0, cx - 60), max(12, cy - int(max(t.height, 8) / 2) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas
