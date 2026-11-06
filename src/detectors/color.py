

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time

class ColorDetector:
    """
    颜色检测器
    检测指定颜色的目标，返回检测结果和掩码
    """
    
    def __init__(self, 
                 target_color: str = "red",
                 min_area: int = 100,
                 confidence_threshold: float = 0.3,
                 enable_morphology: bool = True,
                 morph_kernel_size: int = 5,
                 dominance_ratio: float = 0.0):
        """
        初始化颜色检测器
        
        Args:
            target_color: 检测的目标颜色 ('red', 'green', 'blue', 'yellow', 'orange', 'purple')
            min_area: 最小检测面积（像素）
            confidence_threshold: 置信度阈值
            enable_morphology: 是否启用形态学优化
            morph_kernel_size: 形态学核大小
            dominance_ratio: 通道主导度约束（0 表示不启用）。启用后要求主通道
                明显强于其它通道，例如红色要求 R > ratio * max(G, B)。
                用于排除粉/品红/紫这类"色相落进红色区间但并非纯红"的干扰物。
        """
        self.target_color = target_color.lower()
        self.min_area = min_area
        self.confidence_threshold = confidence_threshold
        self.enable_morphology = enable_morphology
        self.dominance_ratio = float(dominance_ratio)
        
        # HSV颜色范围
        self.color_ranges = {
            'red': [[[0, 100, 100], [10, 255, 255]], 
                   [[160, 100, 100], [180, 255, 255]]],
            'green': [[[40, 100, 100], [80, 255, 255]]],
            'blue': [[[100, 100, 100], [130, 255, 255]]],
            'yellow': [[[20, 100, 100], [40, 255, 255]]],
            'orange': [[[10, 100, 100], [20, 255, 255]]],
            'purple': [[[130, 100, 100], [150, 255, 255]]]
        }
        
        # 形态学核
        if enable_morphology:
            self.morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (morph_kernel_size, morph_kernel_size)
            )
        
        # 统计
        self.total_detections = 0
        self.processing_times = []
    
    def detect(self, hsv_image: np.ndarray, verbose: bool = True,
               bgr_image: Optional[np.ndarray] = None) -> Tuple[List[Dict], np.ndarray]:
        """
        从HSV图像中检测颜色目标
        
        Args:
            hsv_image: HSV格式的图像
            verbose: 是否输出详细信息
            
        Returns:
            Tuple[List[Dict], np.ndarray]: 
                - 检测结果列表，每个元素是包含目标信息的字典
                - 二值掩码图像，检测到的区域为255
        """
        start_time = time.time()
        
        # 1. 检查颜色是否支持
        if self.target_color not in self.color_ranges:
            if verbose:
                print(f"错误: 不支持的颜色 '{self.target_color}'")
            return [], np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        # 2. 创建颜色掩膜
        mask = self._create_color_mask(hsv_image)
        
        # 3. 优化掩膜
        if self.enable_morphology:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # 4. 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4.5 通道主导度筛选：按轮廓平均色判断，比逐像素判断快一个数量级，
        #     且不会被单块高光/阴影带偏（详见 _passes_dominance）
        if self.dominance_ratio > 0 and bgr_image is not None:
            contours = [c for c in contours if self._passes_dominance(c, bgr_image)]
        
        # 5. 分析轮廓
        detections = []
        for i, cnt in enumerate(contours, 1):
            detection = self._analyze_contour(cnt, i)
            if detection:
                detections.append(detection)
        
        # 6. 计算处理时间
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        self.total_detections += len(detections)
        
        # 7. 输出信息
        if verbose:
            self._print_detection_info(detections, processing_time, mask)
        
        return detections, mask
    
    def detect_from_bgr(self, bgr_image: np.ndarray, verbose: bool = True) -> Tuple[List[Dict], np.ndarray]:
        """
        从BGR图像中检测颜色目标
        
        Args:
            bgr_image: BGR格式的图像
            verbose: 是否输出详细信息
            
        Returns:
            Tuple[List[Dict], np.ndarray]: 检测结果和掩码
        """
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        return self.detect(hsv_image, verbose, bgr_image=bgr_image)
    
    def _create_color_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """创建颜色掩膜"""
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.color_ranges[self.target_color]:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            color_mask = cv2.inRange(hsv_image, lower_np, upper_np)
            mask = cv2.bitwise_or(mask, color_mask)
        
        return mask

    def _passes_dominance(self, contour: np.ndarray, bgr_image: np.ndarray) -> bool:
        """按轮廓区域的平均色判断主通道是否主导。

        以红色为例：纯红块 R 远大于 G/B（实测 R/max(G,B) ≈ 2.6~3.6），
        而粉、品红、紫块的 R/max(G,B) 只有 1.3 左右。
        要求 R > ratio * max(G, B) 即可排除后一类，而不影响真正的红色目标。
        """
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return False
        roi = bgr_image[y:y + h, x:x + w]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(roi_mask, [contour - (x, y)], -1, 255, -1)
        b, g, r = cv2.mean(roi, mask=roi_mask)[:3]
        m = max(g, b, 1.0)
        ratio = float(self.dominance_ratio)

        if self.target_color in ("red", "orange"):
            return r > ratio * m
        if self.target_color == "green":
            return g > ratio * max(r, b, 1.0)
        if self.target_color == "blue":
            return b > ratio * max(r, g, 1.0)
        if self.target_color == "yellow":
            return min(r, g) > ratio * max(b, 1.0)
        if self.target_color == "purple":
            return min(r, b) > ratio * max(g, 1.0)
        return True
    
    def _analyze_contour(self, contour: np.ndarray, detection_id: int) -> Optional[Dict]:
        """分析单个轮廓"""
        # 计算面积
        area = cv2.contourArea(contour)
        if area < self.min_area:
            return None
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect
        
        # 计算周长和圆形度
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # 计算置信度
        confidence = min(circularity, 1.0)
        if confidence < self.confidence_threshold:
            return None
        
        # 创建检测结果
        return {
            'id': detection_id,
            'color': self.target_color,
            'center_x': float(x),
            'center_y': float(y),
            'width': float(w),
            'height': float(h),
            'area': float(area),
            'aspect_ratio': float(w / h) if h > 0 else 0,
            'circularity': float(circularity),
            'confidence': float(confidence),
            'angle': float(angle)
        }
    
    def _print_detection_info(self, detections: List[Dict], processing_time: float, mask: np.ndarray):
        """打印检测信息"""
        if not detections:
            print(f"未检测到 {self.target_color} 目标")
            print(f"处理时间: {processing_time:.1f}ms")
            print(f"掩码尺寸: {mask.shape}")
            return
        
        print(f"检测到 {len(detections)} 个 {self.target_color} 目标")
        print(f"处理时间: {processing_time:.1f}ms")
        print(f"掩码尺寸: {mask.shape}, 非零像素: {cv2.countNonZero(mask)}")
        print("-" * 40)
        
        for det in detections:
            print(f"  ID:{det['id']} 中心({det['center_x']:.0f},{det['center_y']:.0f}) "
                  f"尺寸:{det['width']:.0f}x{det['height']:.0f} "
                  f"面积:{det['area']:.0f} 置信度:{det['confidence']:.2f}")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.processing_times:
            return {}
        
        return {
            'total_detections': self.total_detections,
            'avg_processing_time_ms': np.mean(self.processing_times) if self.processing_times else 0,
            'config': {
                'target_color': self.target_color,
                'min_area': self.min_area,
                'confidence_threshold': self.confidence_threshold
            }
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_detections = 0
        self.processing_times = []

# 快速使用函数
def detect_color_with_mask(image: np.ndarray, 
                          target_color: str = "red",
                          min_area: int = 100,
                          verbose: bool = True) -> Tuple[List[Dict], np.ndarray]:
    """
    快速检测颜色并返回掩码
    
    Args:
        image: 输入图像（BGR或HSV）
        target_color: 目标颜色
        min_area: 最小面积
        verbose: 是否输出信息
        
    Returns:
        Tuple[List[Dict], np.ndarray]: 检测结果和掩码
    """
    detector = ColorDetector(
        target_color=target_color,
        min_area=min_area
    )
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        # 假设是BGR图像
        return detector.detect_from_bgr(image, verbose)
    else:
        # 假设是HSV图像
        return detector.detect(image, verbose)