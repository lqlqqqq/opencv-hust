def apply_masks_to_image(original_image, color_mask=None, shape_mask=None):
    """
    将颜色和图形掩码合并并应用于原图像
    
    参数:
        original_image: 原始BGR图像
        color_mask: 颜色检测的二值掩码 (0-255)，可为None
        shape_mask: 图形检测的二值掩码 (0-255)，可为None
        
    返回:
        dict: 包含处理结果的字典
    """
    import cv2
    import numpy as np
    
    # 确保掩码尺寸与原图一致
    h, w = original_image.shape[:2]
    
    # 初始化掩码
    if color_mask is not None and color_mask.shape[:2] != (h, w):
        color_mask = cv2.resize(color_mask, (w, h))
    
    if shape_mask is not None and shape_mask.shape[:2] != (h, w):
        shape_mask = cv2.resize(shape_mask, (w, h))
    
    # 合并掩码
    if color_mask is not None and shape_mask is not None:
        # 合并掩码（逻辑或）
        combined_mask = cv2.bitwise_and(color_mask, shape_mask)
    elif color_mask is not None:
        combined_mask = color_mask
    elif shape_mask is not None:
        combined_mask = shape_mask
    else:
        # 没有掩码，返回原图
        return {
            'mask_applied_image': original_image,
            'combined_mask': None,
            'color_only_image': original_image,
            'shape_only_image': original_image
        }
    
    # 应用掩码到原图
    mask_applied_image = cv2.bitwise_and(original_image, original_image, mask=combined_mask)
    
    # 可选：分别应用各个掩码
    color_only_image = None
    if color_mask is not None:
        color_only_image = cv2.bitwise_and(original_image, original_image, mask=color_mask)
    
    shape_only_image = None
    if shape_mask is not None:
        shape_only_image = cv2.bitwise_and(original_image, original_image, mask=shape_mask)
    
    return {
        'mask_applied_image': mask_applied_image,
        'combined_mask': combined_mask,
        'color_only_image': color_only_image,
        'shape_only_image': shape_only_image
    }