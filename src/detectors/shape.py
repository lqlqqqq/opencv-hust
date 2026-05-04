import cv2
import numpy as np


def detect_rectangles(img, img_shape, min_area=1, return_intermediate=False):
    """
    检测图像中的矩形，返回几何信息和掩码
    输入为BGR彩色图像，对RGB三通道分别进行Canny后合并
    """
    intermediate_images = {}

    # 1. 高斯模糊降噪（对彩色图）
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    if return_intermediate:
        intermediate_images['01_blurred'] = blurred

    # 2. 分离RGB三通道，分别Canny后合并
    b, g, r = cv2.split(blurred)
    edges_b = cv2.Canny(b, 50, 150)
    edges_g = cv2.Canny(g, 50, 150)
    edges_r = cv2.Canny(r, 50, 150)
    edges_canny = cv2.bitwise_or(edges_b, edges_g)
    edges_canny = cv2.bitwise_or(edges_canny, edges_r)
    if return_intermediate:
        intermediate_images['02_edges_canny'] = edges_canny
    
    # 4. 膨胀操作（闭合断裂的边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(edges_canny, kernel, iterations=1)
    if return_intermediate:
        intermediate_images['04_dilated'] = dilated
    
    # 5. 再次Canny边缘检测
    final_edges = cv2.Canny(dilated, 50, 150)
    if return_intermediate:
        intermediate_images['05_final_edges'] = final_edges
    
    # 6. 查找轮廓
    contours, _ = cv2.findContours(final_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建掩码
    mask = np.zeros(img_shape, dtype=np.uint8)
    
    rectangle_count = 0
    rectangles = []

    # 7. 遍历轮廓并筛选矩形
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # 优化轮廓近似，使用动态epsilon值
        arc_length = cv2.arcLength(cnt, True)
        epsilon = 0.03 * arc_length  # 更大的epsilon值，适应倾斜矩形
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 确保approx是正确的形状
        approx = np.squeeze(approx)
        
        if len(approx) == 4:
            # 计算轮廓面积和凸包面积的比例，确保形状接近凸形
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.8:  # 排除凹形
                    continue
            
            # 使用minAreaRect来辅助判断
            rect = cv2.minAreaRect(cnt)
            (center, (width, height), angle) = rect
            
            # 计算宽高比，排除过于细长的形状
            min_dim = min(width, height)
            max_dim = max(width, height)
            if min_dim > 0:
                aspect_ratio = max_dim / min_dim
                if aspect_ratio > 10:  # 更宽松的宽高比限制
                    continue
            
            # 改进的内角计算
            is_rectangle = True
            angles = []
            
            for i in range(4):
                # 三个连续的点
                p1 = approx[i]
                p2 = approx[(i+1)%4]
                p3 = approx[(i+2)%4]
                
                # 计算向量
                v1 = p1 - p2
                v2 = p3 - p2
                
                # 计算夹角（弧度）
                dot_product = np.dot(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 > 0 and norm_v2 > 0:
                    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
                    angle_deg = np.degrees(angle_rad)
                    angles.append(angle_deg)
                    # 更宽松的角度范围，适应45度倾斜的正方形
                    if angle_deg < 70 or angle_deg > 110:
                        is_rectangle = False
                        break
            
            # 额外的矩形判断条件
            if is_rectangle and len(angles) == 4:
                # 检查角度的一致性
                angle_std = np.std(angles)
                if angle_std > 15:  # 角度标准差过大，不是矩形
                    is_rectangle = False
            
            if is_rectangle:
                rectangle_count += 1
                
                rect = cv2.minAreaRect(cnt)
                (center_x, center_y), (width, height), angle = rect
                
                if width < height:
                    width, height = height, width
                    angle = angle + 90 if angle != 0 else 90
                angle = round(angle, 2) if abs(angle) > 0.01 else 0.0
                
                aspect_ratio = round(width / height, 4) if height != 0 else 0
                
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                cv2.drawContours(mask, [box], 0, 255, -1)
                
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

    if return_intermediate:
        return rectangles, mask, intermediate_images
    return rectangles, mask


def detect_circles(img, img_shape, min_area=1, min_circularity=0.75, max_aspect_ratio_diff=0.2, return_intermediate=False):
    """
    检测图像中的圆形，返回几何信息和掩码
    输入为BGR彩色图像，对RGB三通道分别进行Canny后合并
    """
    intermediate_images = {}

    # 1. 高斯模糊降噪（对彩色图）
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    if return_intermediate:
        intermediate_images['01_blurred'] = blurred

    # 2. 分离RGB三通道，分别Canny后合并
    b, g, r = cv2.split(blurred)
    edges_b = cv2.Canny(b, 50, 150)
    edges_g = cv2.Canny(g, 50, 150)
    edges_r = cv2.Canny(r, 50, 150)
    edges_canny = cv2.bitwise_or(edges_b, edges_g)
    edges_canny = cv2.bitwise_or(edges_canny, edges_r)
    if return_intermediate:
        intermediate_images['02_edges_canny'] = edges_canny
    
    # 4. 膨胀操作（闭合断裂的边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilated = cv2.dilate(edges_canny, kernel, iterations=1)
    if return_intermediate:
        intermediate_images['04_dilated'] = dilated
    
    # 5. 再次Canny边缘检测
    final_edges = cv2.Canny(dilated, 50, 150)
    if return_intermediate:
        intermediate_images['05_final_edges'] = final_edges
    
    # 6. 查找轮廓
    contours, _ = cv2.findContours(final_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建掩码
    mask = np.zeros(img_shape, dtype=np.uint8)
    
    circle_count = 0
    circles = []

    # 7. 遍历轮廓并筛选圆形
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        rect = cv2.minAreaRect(cnt)
        (center_x, center_y), (width, height), angle = rect
        if width < height:
            width, height = height, width
        aspect_ratio = width / height if height != 0 else 1
        aspect_ratio_diff = abs(aspect_ratio - 1)
        
        # 优化四边形检测
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx = np.squeeze(approx)
        is_quadrilateral = len(approx) == 4
        
        # 额外的正方形检测
        is_square = False
        if is_quadrilateral:
            # 计算四个边的长度
            side_lengths = []
            for i in range(4):
                p1 = approx[i]
                p2 = approx[(i+1)%4]
                length = np.linalg.norm(p1 - p2)
                side_lengths.append(length)
            
            # 检查边长是否接近相等
            if len(side_lengths) == 4:
                mean_length = np.mean(side_lengths)
                length_std = np.std(side_lengths)
                if length_std / mean_length < 0.1:  # 边长标准差小于10%
                    is_square = True
        
        # 提高圆度阈值，防止正方形被误识别
        if circularity >= min_circularity and aspect_ratio_diff <= max_aspect_ratio_diff and not is_quadrilateral and not is_square:
            circle_count += 1
            
            (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
            center_x = round(center_x, 2)
            center_y = round(center_y, 2)
            radius = round(radius, 2)
            diameter = round(radius * 2, 2)
            circularity = round(circularity, 4)
            area_rounded = round(area, 2)
            
            center = (int(center_x), int(center_y))
            cv2.circle(mask, center, int(radius), 255, -1)
            
            circles.append({
                'id': circle_count,
                'center': (center_x, center_y),
                'radius': radius,
                'diameter': diameter,
                'circularity': circularity,
                'area': area_rounded
            })

    if return_intermediate:
        return circles, mask, intermediate_images
    return circles, mask


def process_shape_detection(img, img_shape, shape_type, detect_func, min_area, **detect_kwargs):
    """
    处理形状检测并显示结果（合并后的简化函数）
    """
    # 执行检测
    shapes, mask, intermediate = detect_func(
        img, img_shape, min_area=min_area, return_intermediate=True, **detect_kwargs
    )
    
    # 显示检测结果
    print("="*50)
    print(f"{shape_type}识别结果：")
    print("="*50)
    
    if shape_type == 'rectangle':
        for shape in shapes:
            print(f"\n【{shape_type} {shape['id']}】")
            print(f"  中心坐标：{shape['center']}")
            print(f"  长边长度：{shape['long_side']} 像素")
            print(f"  短边长度：{shape['short_side']} 像素")
            print(f"  长宽比：{shape['aspect_ratio']}")
            print(f"  旋转角度：{shape['angle']} 度")
            print(f"  面积：{shape['area']} 像素²")
    else:  # circle
        for shape in shapes:
            print(f"\n【{shape_type} {shape['id']}】")
            print(f"  中心坐标：{shape['center']}")
            print(f"  半径：{shape['radius']} 像素")
            print(f"  直径：{shape['diameter']} 像素")
            print(f"  圆度：{shape['circularity']}")
            print(f"  面积：{shape['area']} 像素²")
    
    if len(shapes) == 0:
        print(f"\n未检测到符合条件的{shape_type}！")
    else:
        print(f"\n共检测到 {len(shapes)} 个{shape_type}")
    
    # 显示中间处理结果
    for name, intermediate_img in intermediate.items():
        cv2.imshow(f"{shape_type.capitalize()} - {name}", intermediate_img)
        print(f"显示: {shape_type.capitalize()} - {name}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 显示最终掩码
    cv2.imshow(f"{shape_type.capitalize()} - Final_Mask", mask)
    print(f"显示: {shape_type.capitalize()} - Final_Mask")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 绘制并显示带标记的原图
    result_img = img.copy()
    if shape_type == 'rectangle':
        for shape in shapes:
            box = shape['contour']
            cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)
            center = (int(shape['center'][0]), int(shape['center'][1]))
            cv2.putText(result_img, f"Rect {shape['id']}", 
                        (center[0]-30, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:  # circle
        for shape in shapes:
            center = (int(shape['center'][0]), int(shape['center'][1]))
            radius = int(shape['radius'])
            cv2.circle(result_img, center, radius, (255, 0, 0), 2)
            cv2.circle(result_img, center, 3, (0, 255, 0), -1)
            cv2.putText(result_img, f"Circle {shape['id']}", 
                        (center[0]-30, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.imshow(f"{shape_type.capitalize()} - Marked_Original", result_img)
    print(f"显示: {shape_type.capitalize()} - Marked_Original")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    cv2.imwrite(f"result/{shape_type}_mask.jpg", mask)
    cv2.imwrite(f"result/{shape_type}_marked.jpg", result_img)


# 主程序入口
if __name__ == "__main__":
    image_path = "C:/Users/A/Documents/opencv-hust/data/test002.png"
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
    else:
        img_shape = img.shape[:2]

        # 显示原始图像
        cv2.imshow("00_Original", img)
        print("显示: 00_Original")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 执行矩形检测
        process_shape_detection(
            img, img_shape, 'rectangle',
            detect_rectangles, min_area=700
        )

        # 显示原始图像
        cv2.imshow("00_Original", img)
        print("显示: 00_Original")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 执行圆形检测
        process_shape_detection(
            img, img_shape, 'circle',
            detect_circles, min_area=700,
            min_circularity=0.80, max_aspect_ratio_diff=0.15
        )
        
        print("\n程序结束。")
