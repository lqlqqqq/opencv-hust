# -*- coding: utf-8 -*-
"""图像预处理：为低照度 / 低对比度场景提供可切换的增强链路。

该模块把原先耦合在 ``Launcher_aim.GreenDotTracker`` 中的预处理逻辑解耦出来，
既可在离线的颜色/形状检测前调用，也可在实时视频循环中逐帧调用。
"""

from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """可配置的预处理链。

    mode:
        - ``none``          直接返回原图
        - ``denoise``       高斯 + 中值滤波
        - ``clahe``         LAB 空间 L 通道 CLAHE 增强（不改变色相）
        - ``clahe_sharpen`` CLAHE 后做锐化（默认）
        - ``auto_gamma``    依据平均亮度做 RGB 三通道 gamma 校正 + CLAHE
        - ``hsv_gamma``     只在 HSV 的 V 通道做 gamma 校正 + CLAHE（保色相，低照度首选）
    """

    MODES = ("none", "denoise", "clahe", "clahe_sharpen", "auto_gamma", "hsv_gamma")

    def __init__(
        self,
        mode: str = "clahe_sharpen",
        gaussian_ksize: int = 5,
        median_ksize: int = 5,
        clip_limit: float = 3.0,
        tile_grid: int = 8,
        sharpen: bool = True,
        gamma: float = 1.0,
        target_luma: float = 160.0,
    ):
        if mode not in self.MODES:
            raise ValueError(f"未知预处理模式: {mode}，可选 {self.MODES}")
        self.mode = mode
        self.gaussian_ksize = self._odd(gaussian_ksize)
        self.median_ksize = self._odd(median_ksize)
        self.clip_limit = float(clip_limit)
        self.tile_grid = (int(tile_grid), int(tile_grid))
        self.sharpen = bool(sharpen)
        self.gamma = float(gamma)
        self.target_luma = float(target_luma)
        self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)

    # ------------------------------------------------------------------
    @staticmethod
    def _odd(k: int) -> int:
        k = int(k)
        if k <= 1:
            return 1
        return k if k % 2 == 1 else k + 1

    # ------------------------------------------------------------------
    def _denoise(self, img: np.ndarray) -> np.ndarray:
        out = img
        if self.gaussian_ksize > 1:
            out = cv2.GaussianBlur(out, (self.gaussian_ksize, self.gaussian_ksize), 0)
        if self.median_ksize > 1:
            out = cv2.medianBlur(out, self.median_ksize)
        return out

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _apply_sharpen(self, img: np.ndarray) -> np.ndarray:
        # 注意：卷积核元素之和必须为 1，否则整体亮度会被缩放。
        # 原 Launcher_aim.preprocess_frame 里写成 kernel/9（核和为 1/9），
        # 会把画面压暗到原来的 1/9，属于必须修正的缺陷。
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    # ------------------------------------------------------------------
    @staticmethod
    def _gamma_from_mean(mean: float, target: float) -> float:
        """由平均亮度反解 gamma；画面不暗时返回 1.0（不做改动）。"""
        mean = float(np.clip(mean, 1.0, 254.0))
        target = float(np.clip(target, 1.0, 254.0))
        if mean >= target:
            return 1.0
        return float(np.clip(np.log(target / 255.0) / np.log(mean / 255.0), 0.35, 1.0))

    def _hsv_gamma(self, img: np.ndarray) -> np.ndarray:
        """只在 V 通道做幂次校正。

        RGB 三通道分别做 gamma 会改变通道比例，进而使色相漂移：
        实测暗红色（H≈178, V≈88）经 RGB gamma 后 H 漂到 12，落出红色阈值区间。
        在 HSV 空间只缩放 V，可把暗目标抬进 V≥100 的检测区间而不动色相。
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        gamma = self._gamma_from_mean(float(v.mean()), self.target_luma)
        if gamma != 1.0:
            lut = np.array(
                [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
            )
            v = cv2.LUT(v, lut)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    @staticmethod
    def _auto_gamma(img: np.ndarray, target_luma: float) -> np.ndarray:
        """按灰度均值估计 gamma：仅在画面偏暗时提亮，避免把正常/过曝画面压暗。

        归一化亮度 y=mean/255，希望输出均价接近 target_luma/255，
        由 y**gamma = target 得 gamma = ln(target)/ln(y)；
        画面本身不暗（y >= target）时保持 gamma=1，即不做改动。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        target = float(np.clip(target_luma, 1.0, 254.0))
        if mean <= 1.0 or mean >= target:
            gamma = 1.0
        else:
            gamma = float(np.log(target / 255.0) / np.log(mean / 255.0))
            gamma = float(np.clip(gamma, 0.35, 1.0))
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        return cv2.LUT(img, lut)

    # ------------------------------------------------------------------
    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        """对 BGR 图像执行预处理，返回新的图像（不修改入参）。"""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("输入图像为空")
        if self.mode == "none":
            return image_bgr.copy()

        img = image_bgr
        if self.mode in ("denoise", "clahe", "clahe_sharpen", "auto_gamma", "hsv_gamma"):
            img = self._denoise(img)

        if self.mode == "denoise":
            return img

        if self.mode == "hsv_gamma":
            img = self._hsv_gamma(img)
            img = self._apply_clahe(img)
            return self._apply_sharpen(img) if self.sharpen else img

        if self.mode == "auto_gamma":
            img = self._auto_gamma(img, self.target_luma)
            img = self._apply_clahe(img)
            return self._apply_sharpen(img) if self.sharpen else img

        img = self._apply_clahe(img)
        if self.mode == "clahe_sharpen":
            img = self._apply_sharpen(img) if self.sharpen else img
        return img

    # 兼容原有调用习惯
    __call__ = process


def estimate_brightness(image_bgr: np.ndarray) -> float:
    """返回灰度均值，用于判断是否需要启用低照度增强。"""
    return float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).mean())
