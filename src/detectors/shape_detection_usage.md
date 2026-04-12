# 形状检测函数使用说明

本文档介绍 `shape.py` 中提供的形状检测函数的使用方法，包括矩形检测和圆形检测。

## 目录
- [矩形检测函数](#矩形检测函数)
- [圆形检测函数](#圆形检测函数)
- [处理和显示函数](#处理和显示函数)
- [使用示例](#使用示例)

## 矩形检测函数

### 函数签名
```python
def detect_rectangles(gray, img_shape, min_area=1, return_intermediate=False):
```

### 参数说明
- **gray**: 输入灰度图像（numpy.ndarray）
- **img_shape**: 原始图像尺寸 (height, width)，用于创建掩码
- **min_area**: 最小检测面积，默认值为1（用于过滤小噪点）
- **return_intermediate**: 是否返回中间处理结果，默认值为False

### 返回值
- **rectangles**: 矩形列表，每个矩形包含以下信息：
  - `id`: 矩形编号
  - `center`: 中心坐标 (x, y)
  - `long_side`: 长边长度
  - `short_side`: 短边长度
  - `aspect_ratio`: 长宽比
  - `angle`: 旋转角度（相对于水平轴）
  - `area`: 面积
  - `contour`: 轮廓点坐标
- **mask**: 二值掩码图像
- **intermediate_images**: 中间处理结果字典（仅当 return_intermediate=True 时返回）

## 圆形检测函数

### 函数签名
```python
def detect_circles(gray, img_shape, min_area=1, min_circularity=0.75, max_aspect_ratio_diff=0.2, return_intermediate=False):
```

### 参数说明
- **gray**: 输入灰度图像（numpy.ndarray）
- **img_shape**: 原始图像尺寸 (height, width)，用于创建掩码
- **min_area**: 最小检测面积，默认值为1（用于过滤小噪点）
- **min_circularity**: 最小圆度阈值，默认值为0.75
- **max_aspect_ratio_diff**: 最大宽高比差异，默认值为0.2
- **return_intermediate**: 是否返回中间处理结果，默认值为False

### 返回值
- **circles**: 圆形列表，每个圆形包含以下信息：
  - `id`: 圆形编号
  - `center`: 中心坐标 (x, y)
  - `radius`: 半径
  - `diameter`: 直径
  - `circularity`: 圆度
  - `area`: 面积
- **mask**: 二值掩码图像
- **intermediate_images**: 中间处理结果字典（仅当 return_intermediate=True 时返回）

## 处理和显示函数

### 函数签名
```python
def process_shape_detection(img, gray, img_shape, shape_type, detect_func, min_area, **detect_kwargs):
```

### 参数说明
- **img**: 原始彩色图像
- **gray**: 灰度图像
- **img_shape**: 图像尺寸
- **shape_type**: 形状类型 ('rectangle' 或 'circle')
- **detect_func**: 检测函数（detect_rectangles 或 detect_circles）
- **min_area**: 最小面积
- **detect_kwargs**: 检测函数的其他参数

### 功能
- 执行形状检测
- 显示检测结果和中间处理步骤
- 在原彩色图上标注检测到的形状
- 保存结果图像

## 使用示例

### 基本使用示例

```python
import cv2
from shape import detect_rectangles, detect_circles, process_shape_detection

# 读取图像
image_path = "path/to/image.jpg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_shape = img.shape[:2]

# 检测矩形
rectangles, rect_mask = detect_rectangles(gray, img_shape, min_area=100)

# 检测圆形
circles, circle_mask = detect_circles(
    gray, img_shape, 
    min_area=1200, 
    min_circularity=0.80, 
    max_aspect_ratio_diff=0.15
)

# 处理并显示结果
process_shape_detection(
    img, gray, img_shape, 'rectangle',
    detect_rectangles, min_area=100
)

process_shape_detection(
    img, gray, img_shape, 'circle',
    detect_circles, min_area=1200,
    min_circularity=0.80, max_aspect_ratio_diff=0.15
)
```

### 自定义使用示例

```python
import cv2
from shape import detect_rectangles, detect_circles

# 读取图像
image_path = "path/to/image.jpg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_shape = img.shape[:2]

# 检测矩形（返回中间结果）
rectangles, rect_mask, rect_intermediate = detect_rectangles(
    gray, img_shape, min_area=100, return_intermediate=True
)

# 检测圆形（返回中间结果）
circles, circle_mask, circle_intermediate = detect_circles(
    gray, img_shape, 
    min_area=1200, 
    min_circularity=0.80, 
    max_aspect_ratio_diff=0.15,
    return_intermediate=True
)

# 显示检测结果
print(f"检测到 {len(rectangles)} 个矩形")
print(f"检测到 {len(circles)} 个圆形")

# 保存掩码
cv2.imwrite("rectangle_mask.jpg", rect_mask)
cv2.imwrite("circle_mask.jpg", circle_mask)
```

## 参数调优建议

### 矩形检测
- **min_area**: 根据图像中矩形的实际大小调整，一般设置为预期最小矩形面积的80%
- 对于复杂背景，可适当增加 `min_area` 值以减少误检

### 圆形检测
- **min_area**: 根据图像中圆形的实际大小调整
- **min_circularity**: 要求越严格，值越大（最大为1.0）
- **max_aspect_ratio_diff**: 要求越严格，值越小（最小为0）

### 通用建议
- 对于高对比度图像，检测效果更好
- 对于复杂背景，可能需要先进行预处理（如高斯模糊、阈值处理等）
- 可通过调整 `return_intermediate=True` 查看中间处理结果，帮助调优参数

## 注意事项
- 输入图像应为RGB或灰度图像
- 函数返回的掩码图像尺寸与输入图像相同
- 标注操作在原彩色图的副本上进行，不会修改原始图像

## 依赖项
- OpenCV (cv2)
- NumPy (numpy)
