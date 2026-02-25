import cv2
import numpy as np

def detect_rectangles(image_path, min_area=1):
    """
    检测图片中的矩形，标记并输出几何信息
    
    参数:
        image_path: 图片文件路径
        min_area: 最小检测面积（过滤小噪点）
    """
    # 1. 读取图像并预处理
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    
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
    
    rectangle_count = 0  # 统计识别到的矩形数量
    print("="*50)
    print("矩形识别结果：")
    print("="*50)

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
            
            # 标注矩形编号（在中心位置）
            center = (int(center_x), int(center_y))
            cv2.putText(img_result, f"Rect {rectangle_count}", 
                        (center[0]-30, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
            # 5. 输出矩形的几何信息
            print(f"\n【矩形 {rectangle_count}】")
            print(f"  中心坐标：({round(center_x, 2)}, {round(center_y, 2)})")
            print(f"  长边长度：{round(width, 2)} 像素")
            print(f"  短边长度：{round(height, 2)} 像素")
            print(f"  长宽比：{aspect_ratio}")
            print(f"  旋转角度：{angle} 度（相对于水平轴）")
            print(f"  面积：{round(area, 2)} 像素²")

    # 6. 结果展示与保存
    if rectangle_count == 0:
        print("\n未检测到符合条件的矩形！")
    else:
        print(f"\n共检测到 {rectangle_count} 个矩形")
    
    # 显示结果
    cv2.imshow("Rectangle Detection Result", img_result)
    # 保存结果图片
    cv2.imwrite("result/rectangle_detection_result.jpg", img_result)
    
    # 等待按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主程序入口
if __name__ == "__main__":
    # 替换为你的图片路径（支持jpg/png等格式）
    image_path = "data/test002.png"
    # 调用检测函数（可调整min_area过滤小矩形）
    detect_rectangles(image_path, min_area=100)