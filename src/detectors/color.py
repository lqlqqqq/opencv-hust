import cv2
import numpy as np

def detect_colors_from_hsv(hsv_image, target_color="red", min_area=100):
    """
    从已预处理的HSV图像中检测特定颜色区域，标记并输出几何信息
    
    参数:
        hsv_image: 已预处理的HSV图像
        target_color: 要检测的颜色 ['red', 'green', 'blue', 'yellow']
        min_area: 最小检测面积（过滤小噪点）
    
    返回:
        img_result: 绘制结果的BGR图像
        mask: 颜色掩膜
        detection_info: 检测结果列表
    """
    # 1. 验证输入图像
    if hsv_image is None or len(hsv_image.shape) != 3 or hsv_image.shape[2] != 3:
        print("错误：输入必须是3通道的HSV图像")
        return None, None, None
    
    # 检查是否为有效的HSV范围
    h, s, v = cv2.split(hsv_image)
    if h.max() > 180 or s.max() > 255 or v.max() > 255:
        print("警告：输入的HSV值可能超出标准范围")
    
    # 2. 转换为BGR用于显示结果
    img_result = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    # 3. 定义颜色范围（HSV空间）- 针对已预处理图像的标准范围
    color_ranges = {
        'red': [
            ([0, 100, 100], [10, 255, 255]),     # 红色范围1
            ([160, 100, 100], [180, 255, 255])   # 红色范围2
        ],
        'green': [([40, 100, 100], [80, 255, 255])],      # 绿色范围
        'blue': [([100, 100, 100], [130, 255, 255])],     # 蓝色范围
        'yellow': [([20, 100, 100], [40, 255, 255])],     # 黄色范围
        'orange': [([10, 100, 100], [20, 255, 255])],     # 橙色范围
        'purple': [([130, 100, 100], [150, 255, 255])],   # 紫色范围
        'cyan': [([85, 100, 100], [100, 255, 255])],      # 青色范围
        'pink': [([150, 50, 150], [170, 200, 255])]       # 粉色范围
    }
    
    if target_color not in color_ranges:
        print(f"错误：不支持的颜色 '{target_color}'")
        print(f"支持的颜色: {list(color_ranges.keys())}")
        return None, None, None
    
    # 4. 创建颜色掩膜
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    
    for lower, upper in color_ranges[target_color]:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(hsv_image, lower_np, upper_np)
        mask = cv2.bitwise_or(mask, color_mask)
    
    # 5. 形态学优化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去除小噪声
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填充小孔洞
    
    # 6. 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    color_count = 0
    detection_info = []
    
    print("="*50)
    print(f"{target_color.capitalize()}颜色区域识别结果：")
    print("="*50)
    print(f"输入图像形状: {hsv_image.shape}")
    print(f"掩膜中白色像素数: {np.count_nonzero(mask)}")
    
    # 7. 遍历轮廓并筛选有效区域
    for cnt in contours:
        # 计算轮廓面积
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        color_count += 1
        
        # 计算外接矩形
        rect = cv2.minAreaRect(cnt)
        (center_x, center_y), (width, height), angle = rect
        
        # 修正角度
        if width < height:
            width, height = height, width
            angle = angle + 90
        angle = round(angle, 2) if abs(angle) > 0.01 else 0.0
        
        # 计算长宽比
        aspect_ratio = round(width / height, 4) if height != 0 else 0
        
        # 计算圆形度
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = round(4 * np.pi * area / (perimeter * perimeter), 4)
        else:
            circularity = 0
        
        # 计算凸包和实心度
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = round(area / hull_area, 4) if hull_area > 0 else 0
        
        # 获取边界框顶点
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 8. 绘制结果
        # 颜色映射
        color_map = {
            'red': (0, 0, 255),        # 红色
            'green': (0, 255, 0),      # 绿色
            'blue': (255, 0, 0),       # 蓝色
            'yellow': (0, 255, 255),   # 黄色
            'orange': (0, 165, 255),   # 橙色
            'purple': (128, 0, 128),   # 紫色
            'cyan': (255, 255, 0),     # 青色
            'pink': (203, 192, 255)    # 粉色
        }
        
        contour_color = color_map.get(target_color, (0, 255, 0))
        
        # 绘制轮廓
        cv2.drawContours(img_result, [box], 0, contour_color, 2)
        
        # 绘制中心点
        center = (int(center_x), int(center_y))
        cv2.circle(img_result, center, 6, contour_color, -1)
        cv2.circle(img_result, center, 8, (255, 255, 255), 2)
        
        # 标注区域编号
        cv2.putText(img_result, f"{target_color[0].upper()}{color_count}", 
                   (center[0] - 20, center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, contour_color, 2)
        
        # 标注面积
        cv2.putText(img_result, f"Area: {int(area)}", 
                   (center[0] - 30, center[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, contour_color, 1)
        
        # 9. 存储检测信息
        detection_info.append({
            'id': color_count,
            'color': target_color,
            'center_x': round(center_x, 2),
            'center_y': round(center_y, 2),
            'width': round(width, 2),
            'height': round(height, 2),
            'area': int(area),
            'aspect_ratio': aspect_ratio,
            'angle': angle,
            'circularity': circularity,
            'solidity': solidity,
            'perimeter': round(perimeter, 2)
        })
        
        # 10. 输出信息
        print(f"\n【{target_color.capitalize()}区域 {color_count}】")
        print(f"  中心坐标: ({round(center_x, 2)}, {round(center_y, 2)})")
        print(f"  尺寸: {round(width, 2)} × {round(height, 2)} 像素")
        print(f"  长宽比: {aspect_ratio}")
        print(f"  旋转角度: {angle}°")
        print(f"  面积: {int(area)} 像素²")
        print(f"  圆形度: {circularity} (1.0=完美圆形)")
        print(f"  实心度: {solidity} (1.0=完美实心)")
    
    # 11. 最终输出
    if color_count == 0:
        print(f"\n未检测到符合条件的{target_color}区域！")
    else:
        print(f"\n共检测到 {color_count} 个{target_color}区域")
    
    return img_result, mask, detection_info
