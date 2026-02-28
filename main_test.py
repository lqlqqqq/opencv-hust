
import cv2
import numpy as np
import os
from src.detectors.color import ColorDetector
import sys

def main():
    """主程序 - 使用ColorDetector类"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 1. 读取现有图片
    print("1. 读取图片...")
    relative_path = "data\\test001.png" # 测试用图片相对路径（）
    absolute_path = os.path.join(current_dir, relative_path)
    image_path = absolute_path 
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"错误: 无法读取图片 {image_path}")
        return None, None
    
    print(f"✓ 图片尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 2. 创建颜色检测器
    print("\n2. 创建颜色检测器...")
    detector = ColorDetector(
        target_color="red",      # 检测红色
        min_area=50,            # 最小面积
        confidence_threshold=0  # 置信度阈值
    )
    
    # 3. 执行检测
    print("\n3. 执行颜色检测...")
    print("-" * 50)
    
    # 调用detect_from_bgr方法
    detections, mask = detector.detect_from_bgr(image, verbose=True)
    
    print("-" * 50)
    print("检测完成")
    
    # 4. 处理返回结果
    print("\n4. 处理检测结果...")
    
    if detections:
        print(f"✓ 总共检测到 {len(detections)} 个目标")
        
        # 遍历所有检测结果
        for detection in detections:
            print(f"\n目标 {detection['id']}:")
            print(f"  颜色: {detection['color']}")
            print(f"  中心坐标: ({detection['center_x']:.1f}, {detection['center_y']:.1f})")
            print(f"  尺寸: {detection['width']:.1f} x {detection['height']:.1f}")
            print(f"  面积: {detection['area']:.0f} 像素²")
            print(f"  置信度: {detection['confidence']:.2f}")
        
        # 统计信息
        total_area = sum(d['area'] for d in detections)
        avg_confidence = np.mean([d['confidence'] for d in detections])
        print(f"\n统计信息:")
        print(f"  总面积: {total_area:.0f} 像素²")
        print(f"  平均置信度: {avg_confidence:.2f}")
    else:
        print("✗ 未检测到目标")
    
    # 5. 处理掩码
    print("\n5. 处理掩码...")
    if mask is not None:
        print(f"  掩码尺寸: {mask.shape}")
        print(f"  掩码数据类型: {mask.dtype}")
        print(f"  掩码像素范围: {mask.min()} - {mask.max()}")
        print(f"  检测区域像素数: {cv2.countNonZero(mask)}")
        
         #可选的：保存掩码
        cv2.imwrite("detection_mask.jpg", mask)
        print(f"  掩码已保存为 detection_mask.jpg")
    
    # 6. 获取检测器统计
    print("\n6. 检测器统计信息:")
    stats = detector.get_statistics()
    print(f"  总检测目标数: {stats['total_detections']}")
    print(f"  平均处理时间: {stats['avg_processing_time_ms']:.1f}ms")
    
    if detections and mask is not None:
        # 统计高置信度目标数量
        high_conf_detections = [d for d in detections if d['confidence'] > 0]#暂时不考虑置信度
        high_conf_count = len(high_conf_detections)
            # 创建高置信度掩码
        high_conf_mask = np.zeros_like(mask, dtype=np.uint8)
            
        for detection in high_conf_detections:
                # 获取目标中心坐标
                center_x = int(detection['center_x'])
                center_y = int(detection['center_y'])
                
                # 在原始掩码中找到包含该中心点的轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # 检查中心点是否在轮廓内
                    if cv2.pointPolygonTest(contour, (center_x, center_y), False) >= 0:
                        # 填充这个轮廓到高置信度掩码
                        cv2.drawContours(high_conf_mask, [contour], -1, 255, -1)
                        break
            
            # 应用高置信度掩码
            if cv2.countNonZero(high_conf_mask) > 0:
                # 基本掩码处理
                masked_result = cv2.bitwise_and(image, image, mask=high_conf_mask)
                
                # 显示结果
                print("\n显示掩码处理结果 (按任意键关闭窗口):")
                cv2.imshow("High Confidence Mask (置信度>0.8)", high_conf_mask)
                cv2.waitKey(0)
                cv2.imshow("Masked Image (掩码处理后)", masked_result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # 保存结果
                cv2.imwrite("high_confidence_mask.jpg", high_conf_mask)
                cv2.imwrite("high_confidence_masked.jpg", masked_result)
                print("  高置信度掩码处理结果已保存")
            else:
                print("  高置信度掩码中没有检测到区域")
        else:
            print("  没有置信度高于0.8的目标")
    else:
        print("  没有检测结果或掩码，跳过掩码处理")
    return detections, mask

if __name__ == "__main__":
    # 运行主程序
    detections, mask = main()
    
    # 程序结束
    print("\n程序执行完成")