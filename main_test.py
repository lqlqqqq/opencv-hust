import cv2
import numpy as np
import os
import sys
from src.preprocessing.Launcher_aim import *
from src.detectors.color import ColorDetector
from src.detectors.shape import detect_rectangles, detect_circles
from src.tools.preprocess import apply_masks_to_image

# 确保结果文件夹存在
def ensure_result_folder():
    """确保result文件夹存在"""
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"✓ 创建结果文件夹: {result_dir}")
    return result_dir

def get_result_path(filename):
    """获取result文件夹下的完整文件路径"""
    result_dir = ensure_result_folder()
    return os.path.join(result_dir, filename)

def main():
    """主程序 - 可选择性地进行图形检测"""
    # 确保result文件夹存在
    result_dir = ensure_result_folder()
    
    # 配置选项
    DETECTION_CONFIG = {
        'enable_color_detection': True,  # 是否启用颜色检测
        'enable_shape_detection': True,   # 是否启用图形检测
        'detect_rectangles': True,        # 是否检测矩形
        'detect_circles': True,           # 是否检测圆形
        'use_preprocess': False,          # 是否使用预处理
        'show_process': False,            # 是否显示处理过程中的掩码
        'mask_applied_image': True,       # 是否应用掩码 
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 读取现有图片
    print("1. 读取图片...")
    relative_path = "data\\test002.png"  # 测试用图片相对路径
    absolute_path = os.path.join(current_dir, relative_path)
    image_path = absolute_path
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"错误: 无法读取图片 {image_path}")
        return None, None
    
    print(f"✓ 图片尺寸: {original_image.shape[1]}x{original_image.shape[0]}")
    
    # 2. 只调用预处理部分
    if DETECTION_CONFIG['use_preprocess']:
        print("\n2. 执行图像预处理（不包含颜色分割）...")
        try:
            # 创建GreenDotTracker实例但不初始化硬件
            tracker = GreenDotTracker(port=None, baud_rate=115200)
            
            # 步骤1: 只调用图像预处理
            print("  a) 图像预处理（降噪、对比度增强、锐化）...")
            preprocessed_image = tracker.preprocess_frame(original_image)
            
            masked_result = preprocessed_image
            color_mask = None
            enhanced_mask = None
            
            print(f"✓ 预处理完成")
            
            # 显示结果
            if DETECTION_CONFIG['show_process']:
                cv2.imshow('原始图像', original_image)
                cv2.imshow('预处理后图像', preprocessed_image)
                cv2.waitKey(500)
                
                if DETECTION_CONFIG['enable_color_detection'] and color_mask is not None:
                    cv2.imshow('颜色掩码', color_mask)
                    cv2.imshow('增强掩码', enhanced_mask)
                    cv2.imshow('掩码处理结果', masked_result)
                    cv2.waitKey(500)
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # 保存结果到result文件夹
            cv2.imwrite(get_result_path('original.jpg'), original_image)
            cv2.imwrite(get_result_path('preprocessed.jpg'), preprocessed_image)
            
            if DETECTION_CONFIG['enable_color_detection'] and color_mask is not None:
                cv2.imwrite(get_result_path('color_mask.jpg'), color_mask)
                cv2.imwrite(get_result_path('enhanced_mask.jpg'), enhanced_mask)
                cv2.imwrite(get_result_path('masked_result.jpg'), masked_result)
                print("  预处理结果已保存到result文件夹")
            
            # 使用预处理后的图像进行后续检测
            working_image = preprocessed_image
            
        except Exception as e:
            print(f"✗ 预处理失败: {e}")
            print("  将使用原始图像")
            working_image = original_image
    else:
        print("\n2. 跳过预处理，使用原始图像")
        working_image = original_image
    
    # 3. 颜色检测（可选）
    if DETECTION_CONFIG['enable_color_detection']:
        print("\n3. 执行颜色检测...")
        detector = ColorDetector(
            target_color="red",      # 检测红色
            min_area=50,            # 最小面积
            confidence_threshold=0  # 置信度阈值
        )
        
        detections, mask = detector.detect_from_bgr(working_image, verbose=True)
        
        if detections:
            print(f"✓ 颜色检测: 总共检测到 {len(detections)} 个目标")
        else:
            print("✗ 颜色检测: 未检测到目标")
    else:
        print("\n3. 跳过颜色检测")
        detections, mask = None, None
    
    # 4. 图形检测（可选）
    rectangle_result = None
    circle_result = None
    shape_mask = None
    
    if DETECTION_CONFIG['enable_shape_detection']:
        print("\n4. 执行图形检测...")
        print("-" * 50)
        
        # 保存临时图像用于图形检测
        temp_image_path = os.path.join(result_dir, "temp_for_shape_detection.jpg")
        cv2.imwrite(temp_image_path, working_image)
        
        # 检测矩形
        if DETECTION_CONFIG['detect_rectangles']:
            print("检测矩形...")
            rectangle_result = detect_rectangles(temp_image_path, min_area=100)
            
            if rectangle_result['success'] and rectangle_result['rectangle_count'] > 0:
                print(f"  ✓ 检测到 {rectangle_result['rectangle_count']} 个矩形")
                for rect in rectangle_result['rectangles']:
                    print(f"      矩形 {rect['id']}: 中心{rect['center']}, "
                          f"尺寸{rect['long_side']}x{rect['short_side']}, "
                          f"角度{rect['angle']}°, 面积{rect['area']}")
                
                # 显示和保存结果
                if DETECTION_CONFIG['show_process']:
                    cv2.imshow("Rectangles Detected", rectangle_result['result_image'])
                cv2.imwrite(get_result_path("rectangles_detection_result.jpg"), rectangle_result['result_image'])
                
                if rectangle_result['mask'] is not None:
                    cv2.imwrite(get_result_path("rectangles_mask.jpg"), rectangle_result['mask'])
            else:
                print("  ✗ 未检测到矩形")
        
        # 检测圆形
        if DETECTION_CONFIG['detect_circles']:
            print("\n检测圆形...")
            circle_result = detect_circles(
                temp_image_path, 
                min_area=200, 
                min_circularity=0.80, 
                max_aspect_ratio_diff=0.2
            )
            
            if circle_result['success'] and circle_result['circle_count'] > 0:
                print(f"  ✓ 检测到 {circle_result['circle_count']} 个圆形")
                for circle in circle_result['circles']:
                    print(f"      圆形 {circle['id']}: 中心{circle['center']}, "
                          f"半径{circle['radius']}, 圆度{circle['circularity']}, "
                          f"面积{circle['area']}")
                
                # 显示和保存结果
                if DETECTION_CONFIG['show_process']:
                    cv2.imshow("Circles Detected", circle_result['result_image'])
                cv2.imwrite(get_result_path("circles_detection_result.jpg"), circle_result['result_image'])
                
                if circle_result['mask'] is not None:
                    cv2.imwrite(get_result_path("circles_mask.jpg"), circle_result['mask'])
            else:
                print("  ✗ 未检测到圆形")
        
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        print("-" * 50)
    
    # 5. 创建图形检测掩码
    shape_mask = None
    if DETECTION_CONFIG['enable_shape_detection']:
        # 创建全0掩码
        shape_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        
        # 合并矩形和圆形掩码
        if (DETECTION_CONFIG['detect_rectangles'] and 
            rectangle_result is not None and 
            'mask' in rectangle_result and 
            rectangle_result['mask'] is not None):
            shape_mask = cv2.bitwise_or(shape_mask, rectangle_result['mask'])
        
        if (DETECTION_CONFIG['detect_circles'] and 
            circle_result is not None and 
            'mask' in circle_result and 
            circle_result['mask'] is not None):
            shape_mask = cv2.bitwise_or(shape_mask, circle_result['mask'])
    
    # 6. 应用掩码到原图
    if DETECTION_CONFIG['mask_applied_image'] and (mask is not None or shape_mask is not None):
        print("\n5. 应用掩码到原始图像...")
        mask_results = apply_masks_to_image(
            original_image=original_image,
            color_mask=mask,  # 颜色检测的掩码
            shape_mask=shape_mask  # 图形检测的掩码
        )
        
        # 显示和保存结果
        if mask_results['mask_applied_image'] is not None:
            print("  掩码应用结果:")
            if mask_results['mask_applied_image'] is not None:
                cv2.imshow("Mask Applied Result", mask_results['mask_applied_image'])
                cv2.imwrite(get_result_path("mask_applied_result.jpg"), mask_results['mask_applied_image'])
            
            if mask_results['combined_mask'] is not None:
                cv2.imwrite(get_result_path("combined_mask.jpg"), mask_results['combined_mask'])
                print("  ✓ 合并掩码已保存到result文件夹")
            
            if mask_results['color_only_image'] is not None:
                cv2.imshow("Color Only Result", mask_results['color_only_image'])
                cv2.imwrite(get_result_path("color_only_result.jpg"), mask_results['color_only_image'])
            
            if mask_results['shape_only_image'] is not None:
                cv2.imshow("Shape Only Result", mask_results['shape_only_image'])
                cv2.imwrite(get_result_path("shape_only_result.jpg"), mask_results['shape_only_image'])
    
    # 7. 保存所有检测结果到result文件夹
    print(f"\n所有检测结果已保存到: {result_dir}")
    print("保存的文件:")
    
    # 列出result文件夹中的所有文件
    for filename in os.listdir(result_dir):
        file_path = os.path.join(result_dir, filename)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  - {filename} ({file_size} 字节)")
    
    # 8. 等待按键关闭所有窗口
    print("\n检测完成，按任意键关闭所有窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detections, mask


if __name__ == "__main__":
    # 运行主程序
    detections, mask = main()
    
    # 程序结束
    print("\n程序执行完成")