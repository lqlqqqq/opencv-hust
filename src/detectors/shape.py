import cv2
import numpy as np


def detect_rectangles(gray, img_shape, min_area=1, return_intermediate=False):
    """
    检测灰度图像中的矩形，返回几何信息和掩码
    
    参数:
        gray: 输入灰度图像
        img_shape: 原始图像尺寸 (height, width)
        min_area: 最小检测面积（过滤小噪点）
        return_intermediate: 是否返回中间处理结果
    
    返回:
        tuple: (rectangles, mask, intermediate_images) 或 (rectangles, mask)
            rectangles: 矩形列表
            mask: 二值掩码图像
            intermediate_images: 中间处理结果字典（仅当 return_intermediate=True 时返回）
    """
    intermediate_images = {}
    
    # 1. 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if return_intermediate:
        intermediate_images['01_blurred'] = blurred
    
    # 2. Canny边缘检测
    edges_canny = cv2.Canny(blurred, 50, 150)
    if return_intermediate:
        intermediate_images['02_edges_canny'] = edges_canny
    
    # 3. 二值化（增强边缘）
    _, binary_thresh = cv2.threshold(edges_canny, 127, 255, cv2.THRESH_BINARY)
    if return_intermediate:
        intermediate_images['03_binary_thresh'] = binary_thresh
    
    # 4. 膨胀操作（闭合断裂的边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(binary_thresh, kernel, iterations=1)
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
        
        epsilon = 0.015 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        
        if len(approx) == 4:
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


def detect_circles(gray, img_shape, min_area=1, min_circularity=0.75, max_aspect_ratio_diff=0.2, return_intermediate=False):
    """
    检测灰度图像中的圆形，返回几何信息和掩码
    
    参数:
        gray: 输入灰度图像
        img_shape: 原始图像尺寸 (height, width)
        min_area: 最小检测面积（过滤小噪点）
        min_circularity: 最小圆度阈值
        max_aspect_ratio_diff: 最大宽高比差异
        return_intermediate: 是否返回中间处理结果
    
    返回:
        tuple: (circles, mask, intermediate_images) 或 (circles, mask)
            circles: 圆形列表
            mask: 二值掩码图像
            intermediate_images: 中间处理结果字典（仅当 return_intermediate=True 时返回）
    """
    intermediate_images = {}
    
    # 1. 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if return_intermediate:
        intermediate_images['01_blurred'] = blurred
    
    # 2. Canny边缘检测
    edges_canny = cv2.Canny(blurred, 50, 150)
    if return_intermediate:
        intermediate_images['02_edges_canny'] = edges_canny
    
    # 3. 二值化（增强边缘）
    _, binary_thresh = cv2.threshold(edges_canny, 127, 255, cv2.THRESH_BINARY)
    if return_intermediate:
        intermediate_images['03_binary_thresh'] = binary_thresh
    
    # 4. 膨胀操作（闭合断裂的边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(binary_thresh, kernel, iterations=1)
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
        
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        is_quadrilateral = len(approx) == 4
        
        if circularity >= min_circularity and aspect_ratio_diff <= max_aspect_ratio_diff and not is_quadrilateral:
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


def draw_rectangles_on_image(img, rectangles):
    """
    在图像上绘制矩形
    """
    result_img = img.copy()
    for rect in rectangles:
        # 绘制矩形轮廓
        box = rect['contour']
        cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)
        # 标注矩形编号
        center = (int(rect['center'][0]), int(rect['center'][1]))
        cv2.putText(result_img, f"Rect {rect['id']}", 
                    (center[0]-30, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return result_img


def draw_circles_on_image(img, circles):
    """
    在图像上绘制圆形
    """
    result_img = img.copy()
    for circle in circles:
        # 绘制圆形轮廓
        center = (int(circle['center'][0]), int(circle['center'][1]))
        radius = int(circle['radius'])
        cv2.circle(result_img, center, radius, (255, 0, 0), 2)
        # 绘制圆心
        cv2.circle(result_img, center, 3, (0, 255, 0), -1)
        # 标注圆形编号
        cv2.putText(result_img, f"Circle {circle['id']}", 
                    (center[0]-30, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return result_img


def display_intermediate_images_step_by_step(intermediate_images, title_prefix):
    """
    逐张显示中间处理结果图像，按回车键显示下一张
    """
    for name, img in intermediate_images.items():
        cv2.imshow(f"{title_prefix} - {name}", img)
        print(f"显示: {title_prefix} - {name}（按回车键继续，按ESC键退出）")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 按ESC键退出
        if key == 27:
            break


def display_image_with_prompt(img, window_name, prompt):
    """
    显示单张图像，按回车键继续
    """
    cv2.imshow(window_name, img)
    print(f"显示: {window_name}（{prompt}）")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 按ESC键退出
    if key == 27:
        return False
    return True


# 主程序入口
if __name__ == "__main__":
    image_path = "C:/Users/A/Documents/opencv-hust/data/test001.png"
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = img.shape[:2]
        
        # 显示原始灰度图
        if not display_image_with_prompt(gray, "00_Original_Gray", "按回车键继续"):
            exit()
        
        # ========== 矩形检测 ==========
        rectangles, rect_mask, rect_intermediate = detect_rectangles(
            gray, img_shape, min_area=100, return_intermediate=True
        )
        
        print("="*50)
        print("矩形识别结果：")
        print("="*50)
        
        for rect in rectangles:
            print(f"\n【矩形 {rect['id']}】")
            print(f"  中心坐标：{rect['center']}")
            print(f"  长边长度：{rect['long_side']} 像素")
            print(f"  短边长度：{rect['short_side']} 像素")
            print(f"  长宽比：{rect['aspect_ratio']}")
            print(f"  旋转角度：{rect['angle']} 度（相对于水平轴）")
            print(f"  面积：{rect['area']} 像素²")
        
        if len(rectangles) == 0:
            print("\n未检测到符合条件的矩形！")
        else:
            print(f"\n共检测到 {len(rectangles)} 个矩形")
        
        # 逐张显示矩形检测的中间结果
        display_intermediate_images_step_by_step(rect_intermediate, "Rect")
        
        # 显示最终掩码
        if not display_image_with_prompt(rect_mask, "Rect - Final_Mask", "按回车键继续"):
            exit()
        
        # 显示带矩形标记的原图
        rect_marked_img = draw_rectangles_on_image(img, rectangles)
        if not display_image_with_prompt(rect_marked_img, "Rect - Marked_Original", "按回车键继续圆形检测"):
            exit()
        
        cv2.imwrite("result/rectangle_mask.jpg", rect_mask)
        cv2.imwrite("result/rectangle_marked.jpg", rect_marked_img)
        
        # ========== 圆形检测 ==========
        circles, circle_mask, circle_intermediate = detect_circles(
            gray, img_shape, min_area=1200, min_circularity=0.80, max_aspect_ratio_diff=0.2, return_intermediate=True
        )
        
        print("\n" + "="*50)
        print("圆形识别结果：")
        print("="*50)
        
        for circle in circles:
            print(f"\n【圆形 {circle['id']}】")
            print(f"  中心坐标：{circle['center']}")
            print(f"  半径：{circle['radius']} 像素")
            print(f"  直径：{circle['diameter']} 像素")
            print(f"  圆度：{circle['circularity']}")
            print(f"  面积：{circle['area']} 像素²")
        
        if len(circles) == 0:
            print("\n未检测到符合条件的圆形！")
        else:
            print(f"\n共检测到 {len(circles)} 个圆形")
        
        # 显示原始灰度图
        if not display_image_with_prompt(gray, "00_Original_Gray", "按回车键继续"):
            exit()
        
        # 逐张显示圆形检测的中间结果
        display_intermediate_images_step_by_step(circle_intermediate, "Circle")
        
        # 显示最终掩码
        if not display_image_with_prompt(circle_mask, "Circle - Final_Mask", "按回车键继续"):
            exit()
        
        # 显示带圆形标记的原图
        circle_marked_img = draw_circles_on_image(img, circles)
        if not display_image_with_prompt(circle_marked_img, "Circle - Marked_Original", "按回车键退出"):
            exit()
        
        cv2.imwrite("result/circle_mask.jpg", circle_mask)
        cv2.imwrite("result/circle_marked.jpg", circle_marked_img)
        
        print("\n程序结束。")
