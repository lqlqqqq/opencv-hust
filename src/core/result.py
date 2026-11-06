# -*- coding: utf-8 -*-
"""统一的检测结果数据结构。

所有检测器（颜色 / 形状 / 融合）都把结果收敛到 :class:`Target`，
便于统一序列化（JSON）、统一走串口协议、统一做评测。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# 形状编号（个位），与串口协议中的"目标类别"编码保持一致
SHAPE_CODE = {"circle": 1, "rectangle": 2, "unknown": 3}

# 颜色编号（十位）
COLOR_CODE = {
    "red": 1,
    "green": 2,
    "blue": 3,
    "yellow": 4,
    "orange": 5,
    "purple": 6,
    "unknown": 9,
}


def encode_class(color: str, shape: str) -> int:
    """把 (颜色, 形状) 编码为一个字节的目标类别编号。

    十位表示颜色，个位表示形状，例如红色圆形 = 0x11 = 17。
    未知颜色/形状使用 9 / 3，保证编码始终落在 0~99 范围内。
    """
    c = COLOR_CODE.get(color, COLOR_CODE["unknown"])
    s = SHAPE_CODE.get(shape, SHAPE_CODE["unknown"])
    return c * 10 + s


def decode_class(code: int) -> Tuple[str, str]:
    """encode_class 的逆运算，用于上位机解析串口数据。"""
    code = int(code)
    c = code // 10
    s = code % 10
    color = next((k for k, v in COLOR_CODE.items() if v == c), "unknown")
    shape = next((k for k, v in SHAPE_CODE.items() if v == s), "unknown")
    return color, shape


@dataclass
class Target:
    """一个被识别到的目标。"""

    id: int
    color: str = "unknown"
    shape: str = "unknown"
    center: Tuple[float, float] = (0.0, 0.0)
    width: float = 0.0
    height: float = 0.0
    radius: float = 0.0
    area: float = 0.0
    confidence: float = 0.0
    angle: float = 0.0
    source: str = "fusion"  # color / shape / fusion  —— 标记由哪一路得出
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---------- 派生属性 ----------
    @property
    def cx(self) -> float:
        return float(self.center[0])

    @property
    def cy(self) -> float:
        return float(self.center[1])

    @property
    def class_code(self) -> int:
        return encode_class(self.color, self.shape)

    @property
    def label(self) -> str:
        return f"{self.color}_{self.shape}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["center"] = [round(float(self.center[0]), 2), round(float(self.center[1]), 2)]
        d["label"] = self.label
        d["class_code"] = self.class_code
        for key in ("width", "height", "radius", "area", "angle"):
            d[key] = round(float(d[key]), 2)
        d["confidence"] = round(float(self.confidence), 4)
        return d


@dataclass
class FrameResult:
    """单帧（单图）的完整检测结果。"""

    targets: List[Target] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    image_size: Tuple[int, int] = (0, 0)  # (w, h)
    masks: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings_ms.values()))

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0

    def to_dict(self, with_targets: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "image_size": list(self.image_size),
            "timings_ms": {k: round(v, 3) for k, v in self.timings_ms.items()},
            "total_ms": round(self.total_ms, 3),
            "fps": round(self.fps, 2),
            "count": len(self.targets),
        }
        if with_targets:
            d["targets"] = [t.to_dict() for t in self.targets]
        return d
