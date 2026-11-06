# **视觉检测系统API调用说明**

## **1. 快速开始**

### **1.1 运行完整检测**

```python
# 运行主程序
python main.py
```

### **1.2 命令行参数（可选）**

```bash
# 指定配置文件
python main.py --config config.yaml

# 指定图片路径
python main.py --image data/test002.png
```

## **2. 配置文件**

### **2.1 配置参数**

```python
# 在main.py中修改
DETECTION_CONFIG = {
    'enable_color_detection': True,   # 颜色检测开关
    'enable_shape_detection': True,   # 形状检测开关
    'detect_rectangles': True,        # 矩形检测开关
    'detect_circles': True,           # 圆形检测开关
    'use_preprocess': False,          # 预处理开关
    'show_process': False,            # 显示过程开关
    'mask_applied_image': True,       # 应用掩码开关
}
```

## **3. 模块调用API**

### **3.1 颜色检测**

```python
from src.detectors.color import ColorDetector

# 创建检测器
detector = ColorDetector(
    target_color="red",       # 目标颜色
    min_area=50,              # 最小面积
    confidence_threshold=0.3  # 置信度阈值
)

# 执行检测
detections, mask = detector.detect_from_bgr(image, verbose=True)
```

### **3.2 形状检测**

```python
from src.detectors.shape import detect_rectangles, detect_circles

# 矩形检测
rect_result = detect_rectangles(image_path, min_area=100)

# 圆形检测
circle_result = detect_circles(
    image_path, 
    min_area=200, 
    min_circularity=0.8
)
```

### **3.3 掩码处理**

```python
from src.tools.preprocess import apply_masks_to_image

# 应用掩码
result = apply_masks_to_image(
    original_image=image,
    color_mask=color_mask,
    shape_mask=shape_mask
)
```

## **4. 输入输出格式**

### **4.1 输入图片**

```python
# 支持格式
image = cv2.imread("data/test.png")  # PNG, JPG, BMP, TIFF
# 要求: BGR格式, 建议640x480以上分辨率
```

### **4.2 检测结果格式**

```python
# 颜色检测结果
detection = {
    'id': 1,                    # 目标ID
    'color': 'red',             # 颜色
    'center_x': 150.5,          # 中心X
    'center_y': 120.3,          # 中心Y
    'width': 45.2,              # 宽度
    'height': 42.8,             # 高度
    'area': 1935.6,             # 面积
    'confidence': 0.85          # 置信度
}
```

## **5. 结果保存**

### **5.1 自动保存位置**

```python
# 所有结果自动保存到result目录
result/
├── original.jpg              # 原始图片
├── preprocessed.jpg          # 预处理结果
├── color_mask.jpg            # 颜色掩码
├── mask_applied_result.jpg   # 掩码处理结果
├── rectangles_result.jpg     # 矩形检测结果
└── circles_result.jpg        # 圆形检测结果
```

## **6. 示例代码**

### **6.1 完整流程示例**

```python
import cv2
from src.detectors.color import ColorDetector
from src.detectors.shape import detect_rectangles
from src.tools.preprocess import apply_masks_to_image

# 1. 加载图片
image = cv2.imread("data/test.png")

# 2. 颜色检测
color_detector = ColorDetector("red", 50)
detections, color_mask = color_detector.detect_from_bgr(image)

# 3. 形状检测
cv2.imwrite("temp.jpg", image)
rect_result = detect_rectangles("temp.jpg")

# 4. 应用掩码
result = apply_masks_to_image(
    original_image=image,
    color_mask=color_mask,
    shape_mask=rect_result.get('mask')
)

# 5. 显示结果
cv2.imshow("Final Result", result['mask_applied_image'])
cv2.waitKey(0)
```

## **7. 环境配置**

### **7.1 依赖安装**

```bash
pip install opencv-python numpy
```

### **7.2 目录结构**

```markdown
项目目录/
├── data/           # 测试图片
├── src/            # 源码
│   ├── detectors/  # 检测器
│   └── tools/      # 工具
├── result/         # 结果保存
└── main.py         # 主程序
```

## **8. 常见问题**

### **8.1 图片无法加载**

```python
# 检查路径
if image is None:
    print("图片加载失败，检查路径")
```

### **8.2 无检测结果**

```python
# 调整参数
detector = ColorDetector(target_color="red", min_area=20)
```

### **8.3 保存失败**

```python
# 检查目录权限
import os
os.makedirs("result", exist_ok=True)
```

## **9. 技术支持**

- 

  问题反馈：检查控制台输出

- 

  参数调整：参考默认配置

- 

  结果查看：查看result目录