# shape.py API 文档

## 文件位置

`src/detectors/shape.py`

---

## 一、核心检测函数

### `detect_rectangles`

检测图像中的矩形，返回几何信息和掩码。

#### 函数签名

```python
def detect_rectangles(img, img_shape, min_area=1, return_intermediate=False):
```

#### 参数说明

| 参数                  | 类型          | 默认值 | 说明                                                   |
| --------------------- | ------------- | ------ | ------------------------------------------------------ |
| `img`                 | numpy.ndarray | -      | HSV彩色图像，内部会自动分离RGB三通道分别提取边缘后合并 |
| `img_shape`           | tuple         | -      | 原始图像尺寸 `(height, width)`，用于创建掩码           |
| `min_area`            | float         | 1      | 最小检测面积，用于过滤小噪点                           |
| `return_intermediate` | bool          | False  | 是否返回中间处理结果（用于可视化调试）                 |

#### 返回值

**当 `return_intermediate=False` 时：**

```python
return rectangles, mask
```

| 返回值       | 类型          | 说明                             |
| ------------ | ------------- | -------------------------------- |
| `rectangles` | list[dict]    | 矩形列表                         |
| `mask`       | numpy.ndarray | 二值掩码图像（与输入图像同尺寸） |

**当 `return_intermediate=True` 时：**

```python
return rectangles, mask, intermediate_images
```

| 返回值                | 类型 | 说明                   |
| --------------------- | ---- | ---------------------- |
| `intermediate_images` | dict | 每一步预处理的结果图像 |

#### 矩形信息结构

`rectangles` 列表中的每个字典包含以下字段：

```python
{
    'id': int,              # 矩形编号（从1开始）
    'center': (float, float),   # 中心坐标 (x, y)
    'long_side': float,     # 长边长度（像素）
    'short_side': float,    # 短边长度（像素）
    'aspect_ratio': float,  # 长宽比
    'angle': float,         # 旋转角度（度，相对于水平轴）
    'area': float,          # 面积（像素²）
    'contour': numpy.ndarray  # 四个角点坐标（np.int32）
}
```

---

### `detect_circles`

检测图像中的圆形，返回几何信息和掩码。

#### 函数签名

```python
def detect_circles(img, img_shape, min_area=1, min_circularity=0.75,
                   max_aspect_ratio_diff=0.2, return_intermediate=False):
```

#### 参数说明

| 参数                    | 类型          | 默认值 | 说明                                                   |
| ----------------------- | ------------- | ------ | ------------------------------------------------------ |
| `img`                   | numpy.ndarray | -      | HSV彩色图像，内部会自动分离RGB三通道分别提取边缘后合并 |
| `img_shape`             | tuple         | -      | 原始图像尺寸 `(height, width)`                         |
| `min_area`              | float         | 1      | 最小检测面积                                           |
| `min_circularity`       | float         | 0.75   | 最小圆度阈值（1.0为正圆，值越大要求越严格）            |
| `max_aspect_ratio_diff` | float         | 0.2    | 最大宽高比差异（值越小要求越接近正圆）                 |
| `return_intermediate`   | bool          | False  | 是否返回中间处理结果                                   |

#### 返回值

**当 `return_intermediate=False` 时：**

```python
return circles, mask
```

| 返回值    | 类型          | 说明         |
| --------- | ------------- | ------------ |
| `circles` | list[dict]    | 圆形列表     |
| `mask`    | numpy.ndarray | 二值掩码图像 |

**当 `return_intermediate=True` 时：**

```python
return circles, mask, intermediate_images
```

#### 圆形信息结构

`circles` 列表中的每个字典包含以下字段：

```python
{
    'id': int,              # 圆形编号（从1开始）
    'center': (float, float),   # 圆心坐标 (x, y)
    'radius': float,        # 半径（像素）
    'diameter': float,      # 直径（像素）
    'circularity': float,   # 圆度
    'area': float           # 面积（像素²）
}
```

---

## 二、辅助函数

### `process_shape_detection`

调用检测函数并自动完成结果打印、中间步骤图像显示、标注图像绘制和结果保存。

#### 函数签名

```python
def process_shape_detection(img, img_shape, shape_type, detect_func, min_area, **detect_kwargs):
```

#### 参数说明

| 参数              | 类型          | 说明                                                                   |
| ----------------- | ------------- | ---------------------------------------------------------------------- |
| `img`             | numpy.ndarray | 原始BGR彩色图像                                                        |
| `img_shape`       | tuple         | 图像尺寸                                                               |
| `shape_type`      | str           | 形状类型：`'rectangle'` 或 `'circle'`                                  |
| `detect_func`     | callable      | 检测函数（传入 `detect_rectangles` 或 `detect_circles`）               |
| `min_area`        | float         | 最小面积                                                               |
| `**detect_kwargs` | -             | 检测函数的其他参数（如 `min_circularity`、`max_aspect_ratio_diff` 等） |

#### 功能说明

1. 调用检测函数并传入 `return_intermediate=True`
2. 终端打印每个检测到的图形的几何信息
3. 按回车键逐张显示各预处理步骤的结果图
4. 显示最终二值掩码图
5. 在原彩色图上绘制检测到的图形轮廓和编号
6. 将掩码和标注图保存到 `result/` 目录

---

## 三、使用示例

### 基本使用

```python
import cv2
from shape import detect_rectangles, detect_circles

# 读取图像
img = cv2.imread("test.png")
img_shape = img.shape[:2]

# 检测矩形（仅获取几何信息）
rects, rect_mask = detect_rectangles(img, img_shape, min_area=100)
for r in rects:
    print(f"矩形{r['id']}: 中心{r['center']}, 长边{r['long_side']:.1f}, 短边{r['short_side']:.1f}")

# 检测圆形（仅获取几何信息）
circles, circle_mask = detect_circles(
    img, img_shape, min_area=200,
    min_circularity=0.80, max_aspect_ratio_diff=0.15
)
for c in circles:
    print(f"圆形{c['id']}: 圆心{c['center']}, 半径{c['radius']:.1f}")
```

### 查看中间结果

```python
rects, rect_mask, intermediate = detect_rectangles(
    img, img_shape, min_area=100, return_intermediate=True
)

for name, img_step in intermediate.items():
    cv2.imshow(name, img_step)
    cv2.waitKey(0)
```

### 使用一体化处理函数

```python
from shape import process_shape_detection, detect_rectangles, detect_circles

# 矩形检测与展示
process_shape_detection(img, img_shape, 'rectangle', detect_rectangles, min_area=100)

# 圆形检测与展示
process_shape_detection(img, img_shape, 'circle', detect_circles, min_area=200,
                        min_circularity=0.80, max_aspect_ratio_diff=0.15)
```

---

## 四、参数调优建议

| 参数                    | 适用检测  | 调优方向                                                  |
| ----------------------- | --------- | --------------------------------------------------------- |
| `min_area`              | 矩形/圆形 | 根据目标实际大小设置，越大过滤的噪点越多                  |
| `min_circularity`       | 圆形      | 要求越严格值越大（最大1.0），调高可过滤方形但可能漏检椭圆 |
| `max_aspect_ratio_diff` | 圆形      | 要求越严格值越小，调低可排除形状不规则的检测结果          |

---

## 五、预处理流程（内部实现）

```
输入BGR图像
    ↓
高斯模糊 (5×5)
    ↓
分离RGB三通道
    ↓
各通道分别执行Canny边缘检测
    ↓
bitwise_or 合并三通道边缘
    ↓
形态学膨胀（矩形用8×8矩形核，圆形用8×8椭圆核）
    ↓
再次Canny边缘检测
    ↓
查找轮廓 → 筛选图形
```
