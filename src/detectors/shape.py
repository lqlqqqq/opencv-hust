import cv2
import numpy as np

def detect_rectangles(image_path, min_area=1):
    """
    检测图片中的矩形，标记并返回几何信息
    
    参数:
        image_path: 图片文件路径
        min_area: 最小检测面积（过滤小噪点）
    
    返回:
        dict: 包含检测结果的字典，格式如下：
            {
                'success': bool,  # 是否成功读取图片
                'image_path': str,  # 图片路径
                'rectangle_count': int,  # 检测到的矩形数量
                'rectangles': [  # 矩形列表
                    {
                        'id': int,  # 矩形编号
                        'center': (float, float),  # 中心坐标
                        'long_side': float,  # 长边长度
                        'short_side': float,  # 短边长度
                        'aspect_ratio': float,  # 长宽比
                        'angle': float,  # 旋转角度
                        'area': float,  # 面积
                        'contour': numpy.ndarray  # 轮廓点集
                    },
                    ...
                ],
                'result_image': numpy.ndarray,  # 标注后的图像
                'mask': numpy.ndarray  # 二值掩码图像（检测到的矩形区域为255，其他为0）
            }
    """
    # 1. 读取图像并预处理
    img = cv2.imread(image_path)
    if img is None:
        return {
            'success': False,
            'image_path': image_path,
            'rectangle_count': 0,
            'rectangles': [],
            'result_image': None,
            'mask': None
        }
    
    # 创建副本用于绘制结果
    img_result = img.copy()
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    # 二值化（增强边缘）
    _, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    #edges_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    img1=thresh.copy()

    img1 = cv2.dilate(img1, kernel, iterations=1)

    img1 = cv2.Canny(img1, 50, 150)

    # 2. 查找轮廓
    contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_result=img.copy()
    
    # 创建掩码图像（单通道，与原始图像相同尺寸）
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    rectangle_count = 0  # 统计识别到的矩形数量
    rectangles = []  # 存储矩形信息

    # 3. 遍历轮廓并筛选矩形
    for cnt in contours:
        # 计算轮廓面积，过滤小轮廓
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # 轮廓逼近（多边形拟合）
        # approxPolyDP参数：轮廓，逼近精度（周长的1.5%），是否闭合
        epsilon = 0.015 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        
        # 判断是否为四边形（矩形的基础特征）
        if len(approx) == 4:
            rectangle_count += 1
            
            # 计算最小外接矩形（支持旋转矩形）
            rect = cv2.minAreaRect(cnt)
            (center_x, center_y), (width, height), angle = rect
            
            # 修正角度和长宽的对应关系（统一以长边为参照）
            if width < height:
                width, height = height, width
                angle = angle + 90 if angle != 0 else 90
            # 处理接近0度的角度（避免-0.0等异常值）
            angle = round(angle, 2) if abs(angle) > 0.01 else 0.0
            
            # 计算长宽比
            aspect_ratio = round(width / height, 4) if height != 0 else 0
            
            # 获取矩形的四个顶点坐标（用于绘制）
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # 转为整数坐标
            
            # 4. 绘制矩形和标注信息
            # 绘制矩形轮廓（黑色，线宽2）
            cv2.drawContours(img_result, [box], 0, (0, 0, 0), 2)
            # 在掩码上填充矩形区域
            cv2.drawContours(mask, [box], 0, 255, -1)
            
            # 标注矩形编号（在中心位置）
            center = (int(center_x), int(center_y))
            cv2.putText(img_result, f"Rect {rectangle_count}", 
                        (center[0]-30, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
            # 5. 收集矩形的几何信息
            rectangles.append({
                'id': rectangle_count,
                'center': (round(center_x, 2), round(center_y, 2)),
                'long_side': round(width, 2),
                'short_side': round(height, 2),
                'aspect_ratio': aspect_ratio,
                'angle': angle,
                'area': round(area, 2),
                'contour': box
            })

    # 6. 返回检测结果
    return {
        'success': True,
        'image_path': image_path,
        'rectangle_count': rectangle_count,
        'rectangles': rectangles,
        'result_image': img_result,
        'mask': mask
    }


def detect_circles(image_path, min_area=1, min_circularity=0.5, max_aspect_ratio_diff=0.2):
    """
    检测图片中的圆形，标记并返回几何信息
    
    参数:
        image_path: 图片文件路径
        min_area: 最小检测面积（过滤小噪点）
        min_circularity: 最小圆度阈值（0-1之间，默认0.75，越接近1越圆）
        max_aspect_ratio_diff: 最大宽高比差异（默认0.2，用于排除矩形）
    
    返回:
        dict: 包含检测结果的字典，格式如下：
            {
                'success': bool,  # 是否成功读取图片
                'image_path': str,  # 图片路径
                'circle_count': int,  # 检测到的圆形数量
                'circles': [  # 圆形列表
                    {
                        'id': int,  # 圆形编号
                        'center': (float, float),  # 中心坐标
                        'radius': float,  # 半径
                        'diameter': float,  # 直径
                        'circularity': float,  # 圆度
                        'area': float  # 面积
                    },
                    ...
                ],
                'result_image': numpy.ndarray,  # 标注后的图像
                'mask': numpy.ndarray  # 二值掩码图像（检测到的圆形区域为255，其他为0）
            }
    """
    # 1. 读取图像并预处理
    img = cv2.imread(image_path)
    if img is None:
        return {
            'success': False,
            'image_path': image_path,
            'circle_count': 0,
            'circles': [],
            'result_image': None,
            'mask': None
        }
    
    # 创建副本用于绘制结果
    img_result = img.copy()
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪
    img1 = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 二值化（增强边缘）
    thresh = cv2.adaptiveThreshold(
img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    img1 = thresh.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 膨胀操作（闭合断裂的边缘）
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # img1 = cv2.dilate(img1, kernel, iterations=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    img1 = cv2.erode(img1, kernel, iterations=3)

    

    # Canny边缘检测
    img1 = cv2.Canny(img1, 50, 150)
    # cv2.imshow("qwq", img1)
    
    # img1 = cv2.dilate(img1, kernel, iterations=1)
    # img1 = cv2.Canny(img1, 50, 150)

    # 2. 查找轮廓
    contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_result = img.copy()
    
    # 创建掩码图像（单通道，与原始图像相同尺寸）
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    circle_count = 0  # 统计识别到的圆形数量
    circles = []  # 存储圆形信息

    # 3. 遍历轮廓并筛选圆形
    for cnt in contours:
        # 计算轮廓面积，过滤小轮廓
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # 计算轮廓周长
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        # 计算圆度：4π * 面积 / 周长²，理想圆形为1
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 计算最小外接矩形，检查宽高比（排除正方形/矩形）
        rect = cv2.minAreaRect(cnt)
        (center_x, center_y), (width, height), angle = rect
        if width < height:
            width, height = height, width
        aspect_ratio = width / height if height != 0 else 1
        aspect_ratio_diff = abs(aspect_ratio - 1)  # 与1的差异，圆形应该接近1:1
        
        # 判断是否为圆形：圆度足够高 + 宽高比接近1 + 不是四边形
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        is_quadrilateral = len(approx) == 4
        
        if circularity >= min_circularity and aspect_ratio_diff <= max_aspect_ratio_diff and not is_quadrilateral:
            circle_count += 1
            
            # 计算最小外接圆
            (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
            center_x = round(center_x, 2)
            center_y = round(center_y, 2)
            radius = round(radius, 2)
            diameter = round(radius * 2, 2)
            circularity = round(circularity, 4)
            area_rounded = round(area, 2)
            
            # 4. 绘制圆形和标注信息
            # 绘制圆形轮廓（蓝色，线宽2）
            center = (int(center_x), int(center_y))
            cv2.circle(img_result, center, int(radius), (255, 0, 0), 2)
            # 在掩码上填充圆形区域
            cv2.circle(mask, center, int(radius), 255, -1)
            # 绘制圆心（红色小点）
            cv2.circle(img_result, center, 3, (0, 0, 255), -1)
            
            # 标注圆形编号
            cv2.putText(img_result, f"Circle {circle_count}", 
                        (center[0]-30, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 5. 收集圆形的几何信息
            circles.append({
                'id': circle_count,
                'center': (center_x, center_y),
                'radius': radius,
                'diameter': diameter,
                'circularity': circularity,
                'area': area_rounded
            })

    # 6. 返回检测结果
    return {
        'success': True,
        'image_path': image_path,
        'circle_count': circle_count,
        'circles': circles,
        'result_image': img_result,
        'mask': mask
    }


# 主程序入口
if __name__ == "__main__":
    # 替换为你的图片路径（支持jpg/png等格式）
    image_path = "data/test002.png"
    
    # 读取原始图像
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"错误：无法读取图片 {image_path}")
    else:
        # ========== 矩形检测 ==========
        # 调用检测函数（可调整min_area过滤小矩形）
        result = detect_rectangles(image_path, min_area=100)
        
        # 处理检测结果
        if not result['success']:
            print(f"错误：无法处理矩形检测")
        else:
            print("="*50)
            print("矩形识别结果：")
            print("="*50)
            
            # 输出每个矩形的几何信息
            for rect in result['rectangles']:
                print(f"\n【矩形 {rect['id']}】")
                print(f"  中心坐标：{rect['center']}")
                print(f"  长边长度：{rect['long_side']} 像素")
                print(f"  短边长度：{rect['short_side']} 像素")
                print(f"  长宽比：{rect['aspect_ratio']}")
                print(f"  旋转角度：{rect['angle']} 度（相对于水平轴）")
                print(f"  面积：{rect['area']} 像素²")
            
            if result['rectangle_count'] == 0:
                print("\n未检测到符合条件的矩形！")
            else:
                print(f"\n共检测到 {result['rectangle_count']} 个矩形")
            
            # 显示原图像和掩码
            cv2.imshow("Original Image", original_img)
            cv2.imshow("Rectangle Mask", result['mask'])
            # 保存掩码图片
            cv2.imwrite("result/rectangle_mask.jpg", result['mask'])
            
            # 等待按键关闭窗口
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # ========== 圆形检测 ==========
        # 调用圆形检测函数（可调整min_area、min_circularity和max_aspect_ratio_diff）
        circle_result = detect_circles(image_path, min_area=200, min_circularity=0.80, max_aspect_ratio_diff=0.2)
        
        # 处理检测结果
        if not circle_result['success']:
            print(f"错误：无法处理圆形检测")
        else:
            print("\n" + "="*50)
            print("圆形识别结果：")
            print("="*50)
            
            # 输出每个圆形的几何信息
            for circle in circle_result['circles']:
                print(f"\n【圆形 {circle['id']}】")
                print(f"  中心坐标：{circle['center']}")
                print(f"  半径：{circle['radius']} 像素")
                print(f"  直径：{circle['diameter']} 像素")
                print(f"  圆度：{circle['circularity']}")
                print(f"  面积：{circle['area']} 像素²")
            
            if circle_result['circle_count'] == 0:
                print("\n未检测到符合条件的圆形！")
            else:
                print(f"\n共检测到 {circle_result['circle_count']} 个圆形")
            
            # 显示原图像和掩码
            cv2.imshow("Original Image", original_img)
            cv2.imshow("Circle Mask", circle_result['mask'])
            # 保存掩码图片
            cv2.imwrite("result/circle_mask.jpg", circle_result['mask'])
            
            # 等待按键关闭窗口
            cv2.waitKey(0)
            cv2.destroyAllWindows()