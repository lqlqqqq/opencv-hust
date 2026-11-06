# -*- coding: utf-8 -*-
"""流水线配置：集中管理所有可调参数，支持 JSON 存取与命令行覆盖。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class PreprocessConfig:
    """图像预处理参数（颜色路 / 形状路可分别配置）。

    实测结论（见 tools/exp_preprocess_ab.py）：暗场下

    * 颜色路需要 ``hsv_gamma`` 把 V 通道抬回检测阈值，否则暗红目标 V≈88 直接漏检；
    * 形状路不能做锐化——锐化会把 Canny 边缘打碎，圆形召回从 0.99 掉到 0.35。

    因此默认采用"双路预处理"：颜色路增强、形状路用较温和的设置。
    """

    enabled: bool = False
    color_mode: str = "hsv_gamma"   # 颜色路：none/denoise/clahe/clahe_sharpen/auto_gamma/hsv_gamma
    shape_mode: str = "none"        # 形状路：同上，默认不处理
    gaussian_ksize: int = 5
    median_ksize: int = 5
    clip_limit: float = 3.0
    tile_grid: int = 8
    sharpen: bool = True
    gamma: float = 1.0
    target_luma: float = 160.0  # 暗场提亮的目标亮度（0~255）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ColorConfig:
    """颜色检测参数。"""

    enabled: bool = True
    colors: List[str] = field(default_factory=lambda: ["red"])
    min_area: int = 120
    confidence_threshold: float = 0.15
    enable_morphology: bool = True
    morph_kernel_size: int = 5
    dominance_ratio: float = 1.5  # 主通道主导度门限，0 表示关闭

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShapeConfig:
    """形状检测参数。"""

    enabled: bool = True
    detect_rectangles: bool = True
    detect_circles: bool = True
    min_rectangle_area: int = 150
    min_circle_area: int = 180
    min_circularity: float = 0.75
    max_aspect_ratio_diff: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FusionConfig:
    """颜色与形状的融合参数。"""

    enabled: bool = True
    # 颜色块中心与形状中心的最大匹配距离（像素）；
    # 0 表示自动取 min(30, 0.5 * 颜色块较短边)
    max_center_distance: float = 0.0
    # 融合后是否保留未匹配的颜色块（作为 color_only 目标）
    keep_unmatched_color: bool = True
    # 是否保留未匹配的形状（作为 shape_only 目标，默认丢弃以抑制边缘误检）
    keep_unmatched_shape: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineConfig:
    """完整流水线配置。"""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    # 只输出"目标类别"，即同时满足指定颜色 + 指定形状的目标
    target_color: str = "red"
    target_shape: str = "circle"

    # ---------- 序列化 ----------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preprocess": self.preprocess.to_dict(),
            "color": self.color.to_dict(),
            "shape": self.shape.to_dict(),
            "fusion": self.fusion.to_dict(),
            "target_color": self.target_color,
            "target_shape": self.target_shape,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PipelineConfig":
        data = data or {}
        cfg = cls()
        if "preprocess" in data:
            cfg.preprocess = PreprocessConfig(**data["preprocess"])
        if "color" in data:
            cfg.color = ColorConfig(**data["color"])
        if "shape" in data:
            cfg.shape = ShapeConfig(**data["shape"])
        if "fusion" in data:
            cfg.fusion = FusionConfig(**data["fusion"])
        for key in ("target_color", "target_shape"):
            if key in data:
                setattr(cfg, key, str(data[key]))
        return cfg

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    # ---------- 常用预设 ----------
    @classmethod
    def preset(cls, name: str) -> "PipelineConfig":
        """内置预设：默认 / 低照度 / 复杂背景 / 实时跟踪。"""
        cfg = cls()
        name = (name or "default").lower()
        if name in ("default", ""):
            return cfg
        if name == "lowlight":
            cfg.preprocess.enabled = True
            cfg.preprocess.color_mode = "hsv_gamma"
            cfg.preprocess.shape_mode = "none"
            cfg.color.min_area = 80
            cfg.color.confidence_threshold = 0.1
            return cfg
        if name == "cluttered":
            cfg.color.min_area = 200
            cfg.color.confidence_threshold = 0.2
            cfg.shape.min_rectangle_area = 300
            cfg.shape.min_circle_area = 300
            cfg.shape.min_circularity = 0.78
            return cfg
        if name == "realtime":
            cfg.preprocess.enabled = False
            cfg.color.min_area = 150
            return cfg
        raise ValueError(f"未知预设: {name}")
