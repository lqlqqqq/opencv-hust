# 基于 OpenCV 的轻量化视觉识别与定位系统（1.3 版）

面向低算力嵌入式平台的视觉识别模块：**图像采集 → 双路预处理 → 颜色分割 → 形状判定 →
颜色-形状融合 → 目标坐标输出（JSON / 串口）**，目标类别为"红色圆形靶标"，
串口按 8N1、115200 与主控通信。

> 本版本对应 2026 年 11 月中期检查节点。相对 1.2 版的主要变化：
> 统一了处理流水线（1.2 的 `main_test.py` 与 `shape.py` 接口签名不一致，形状检测一路直接抛异常）、
> 补齐了串口协议代码、新增量化评测与单元测试、修正了预处理锐化核归一化错误。

---

## 1. 快速开始

```bash
# 环境（Python 3.10+）
pip install -r requirements.txt

# 单张图片
python main.py --image data/test002.png

# 批量目录（保存标注图 + 导出 JSON）
python main.py --dir data --save-dir result --json docs/real_images_result.json

# 低照度场景（V 通道 γ 校正 + 双路预处理）
python main.py --image data/test001.png --preset lowlight --show

# 摄像头实时检测，并向主控下发串口帧
python main.py --camera 0 --serial COM3 --baud 115200
```

一键回归（生成数据集 → 评测 → 单元测试 → 插图）：

```powershell
pwsh ./run_all.ps1
```

---

## 2. 目录结构

```
opencv-hust-main1.3/
├── main.py                    # 命令行入口（单图 / 批量 / 摄像头）
├── main_test.py               # 兼容旧入口的冒烟脚本（无窗口）
├── run_all.ps1                # 一键回归脚本
├── requirements.txt
├── src/
│   ├── core/
│   │   ├── config.py          # 配置数据类 + 预设 + JSON 存取
│   │   ├── result.py          # Target / FrameResult 统一数据结构
│   │   └── pipeline.py        # 视觉流水线（预处理→颜色→形状→融合）
│   ├── comm/
│   │   ├── protocol.py        # 串口帧封装 / CRC16 / 流式解析
│   │   └── link.py            # SerialLink / LoopbackLink / NullLink
│   ├── detectors/
│   │   ├── color.py           # 颜色检测（HSV 阈值 + 形态学 + 主导度约束）
│   │   ├── shape.py           # 矩形 / 圆形检测（Canny + 轮廓几何判定）
│   │   ├── color_api.md       # 颜色模块接口说明
│   │   └── shape_api.md       # 形状模块接口说明
│   ├── preprocessing/
│   │   ├── enhance.py         # 预处理链（去噪 / CLAHE / γ 校正 / 锐化）
│   │   └── Launcher_aim.py    # 早期实时跟踪脚本（保留作历史参考）
│   └── tools/preprocess.py    # 掩码合并工具
├── tools/
│   ├── make_dataset.py        # 合成测试集生成（5 类场景 + 真值）
│   ├── benchmark.py           # 量化评测（召回 / 精确 / 误差 / 耗时）
│   ├── make_figures.py        # 报告插图
│   ├── exp_preprocess_ab.py   # 预处理对比实验
│   ├── exp_protocol.py        # 协议抗干扰实验
│   ├── bench_timing.py         # 固定协议的耗时测量
│   ├── build_metrics_doc.py    # 生成考核指标与验收方案
│   ├── make_form_text.py       # 生成网上填报用纯文本
│   ├── build_midterm_report.py # 按学院模板生成中期检查表
│   └── build_attachment.py    # 生成中期检查附件（图表材料）
├── tests/                     # 单元测试（协议 / 流水线 / 预处理）
├── dataset/                   # 合成数据集（200 张图 + annotations.json）
├── docs/                      # 评测结果、图表、进展说明
└── data/  result/             # 真实测试图 / 输出结果
```

---

## 3. 命令行参数

| 参数 | 说明 |
|---|---|
| `--image` / `--dir` / `--camera` | 三选一的输入源 |
| `--config` | JSON 配置文件路径（与 `--preset` 二选一，优先生效） |
| `--preset` | `default` / `lowlight` / `cluttered` / `realtime` |
| `--colors` | 要检测的颜色，逗号分隔，如 `red,green` |
| `--target-shape` | 目标形状：`circle` / `rectangle` / `any` |
| `--save-dir` | 结果保存目录（默认 `result/`） |
| `--json` | 检测结果导出路径 |
| `--show` | 显示可视化窗口（按 q 退出） |
| `--serial` / `--baud` | 串口设备与波特率 |
| `--frames` | 摄像头模式下的最大帧数（0 = 不限） |
| `--quiet` | 只输出汇总 |

---

## 4. 接口与数据结构

### 4.1 流水线调用

```python
import cv2
from src.core.config import PipelineConfig
from src.core.pipeline import VisionPipeline

pipeline = VisionPipeline(PipelineConfig.preset("lowlight"))
result = pipeline.run(cv2.imread("data/test002.png"))

for t in pipeline.target_targets(result):      # 仅保留目标类别（红色圆形）
    print(t.id, t.color, t.shape, t.cx, t.cy, t.confidence)

print(result.timings_ms, result.total_ms, result.fps)
```

`FrameResult.to_dict()` 的输出可直接 `json.dump`：

```json
{
  "image_size": [640, 480],
  "timings_ms": {"preprocess": 0.0, "color": 4.2, "shape": 26.4, "fusion": 0.2},
  "total_ms": 30.8, "fps": 32.5, "count": 1,
  "targets": [{
    "id": 1, "color": "red", "shape": "circle",
    "center": [386.1, 271.0], "width": 60.7, "height": 60.7,
    "radius": 30.35, "area": 2455.0, "confidence": 0.89,
    "label": "red_circle", "class_code": 11
  }]
}
```

### 4.2 串口协议 v1（上行，14 字节）

```
AA BB | LEN | ID(1) X(2) Y(2) CONF(1) CLS(1) | CRC16(2) | CC DD
```

* `LEN` = 有效数据长度（v1 为 7；带距离与状态标志的 v2 为 10）
* `CONF` = 置信度 × 100（0~100）
* `CLS` = 目标类别编号：十位为颜色（红1 绿2 蓝3 黄4 橙5 紫6），个位为形状（圆1 矩2 其它3），
  例如 `0x11 = 17` 表示红色圆形
* `CRC16` = CRC-16/MODBUS，校验范围 `LEN + 有效数据`，小端
* v2 在 v1 之后追加 `距离(2, mm) + 标志(1)`，标志 bit0 锁定、bit1 允许发射

```python
from src.comm.protocol import encode_target, FrameParser

frame = encode_target(1, 386, 271, 0.89, 17)          # 14 字节
items = FrameParser().feed(frame)                      # 断帧/粘包/CRC 校验均可处理
```

### 4.3 关键配置项

```python
cfg = PipelineConfig()
cfg.preprocess.enabled = True
cfg.preprocess.color_mode = "hsv_gamma"   # 颜色路：低照度首选
cfg.preprocess.shape_mode = "none"        # 形状路：不要锐化
cfg.color.dominance_ratio = 1.5           # 主通道主导度门限（排除粉/品红）
cfg.shape.min_circularity = 0.75
cfg.target_color, cfg.target_shape = "red", "circle"
```

---

## 5. 实测指标（合成数据集，200 张 640×480）

完整表格见 [docs/benchmark_table.md](docs/benchmark_table.md)，原始数据见
[docs/benchmark_results.json](docs/benchmark_results.json)。目标类别 = 红色圆形靶标，
匹配容差 = max(15 px, 0.35×目标尺寸)。

| 场景 | 端到端召回 | 端到端精确率 | 中心误差均值 | 耗时 | 对应配置 |
|---|---|---|---|---|---|
| 均匀背景 / 正常光照 | 0.988 | 1.000 | 0.01 px | 19–23 ms | 默认 |
| 纹理背景 / 同色异形干扰 | 0.988 | 1.000 | 0.24 px | 19–25 ms | 默认 |
| 低照度 + 噪声 | 0.988 | 0.878 | 0.05 px | 28–34 ms | lowlight |
| 过曝 + 运动模糊 | 0.975 | 1.000 | 0.04 px | 19–31 ms | 默认 |
| 目标遮挡 | 0.950 | 1.000 | 1.05 px | 19–30 ms | 默认 |

对照结论：

* **颜色单路**在同色异形干扰场景下端到端精确率仅 **0.500**（红色矩形被当成靶标），
  加入形状融合后回升到 **1.000**；
* **低照度**下不预处理时端到端召回为 **0.000**（暗红目标 V≈88 < 阈值 100），
  V 通道 γ 校正后回升到 **0.988**；
* **锐化会让形状召回从 0.988 掉到 0.350**，因此采用"颜色路增强、形状路不锐化"的双路预处理。

> 以上均为 PC（i5 级别 CPU，640×480，单线程）实测；树莓派 4B 上的延迟与帧率尚未测，
> 属下一阶段任务。

---

## 6. 已知限制

1. **粘连目标**：相邻同色目标在形态学闭运算后连成一块，颜色块中心偏移，形状匹配失败
   （`data/test002.png` 中可见 2 处）。
2. **相近色相**：粉、品红、紫色块在弱光/阴影下会落进红色阈值区间；已加入
   `dominance_ratio` 主导度约束缓解，但 `data/test001.png` 这类高饱和彩色场景仍有误检。
3. **形状检测耗时占比高**（复杂真实图约 46 ms / 帧，占总耗时 80%）。原因是三通道
   Canny 合并 + 8×8 膨胀 + 全图轮廓，下一阶段用 ROI 限定与降采样优化。
4. **C++ 移植**仅完成颜色模块（`opencv.c/`），形状、预处理、串口尚未移植；
   本机无 OpenCV C++ SDK，需在树莓派上配置工具链后完成。
5. **相机标定**未开展，当前只输出像素坐标，未做畸变校正与像素→世界坐标映射。

---

## 7. 文档索引

| 文档 | 内容 |
|---|---|
| [docs/中期检查表-刘淇-2026年11月.docx](docs/中期检查表-刘淇-2026年11月.docx) | 按学院《项目中期检查表》模板填写的中期检查报告（4 页） |
| [docs/中期检查附件-图表材料.pdf](docs/中期检查附件-图表材料.pdf) | 中期检查附件：流程图、指标表、对比图、真实场景结果 |
| [docs/考核指标与验收方案.docx](docs/考核指标与验收方案.docx) | 25 项可验收指标：定义、三档阈值、测量口径、样本量论证 |\n| [docs/中期进展说明.md](docs/中期进展说明.md) | 对照项目书四阶段的实际进度、实验与结论 |
| [docs/benchmark_table.md](docs/benchmark_table.md) | 三种配置 × 五类场景的完整指标表 |
| [docs/接口说明.md](docs/接口说明.md) | 流水线 / 检测器 / 串口协议接口与字段 |
| [src/detectors/color_api.md](src/detectors/color_api.md) | 颜色检测模块 API |
| [src/detectors/shape_api.md](src/detectors/shape_api.md) | 形状检测模块 API |
| [Material/串口通信协议.docx](Material) | 协议原始定义（与 `src/comm/protocol.py` 对应） |

## 8. 测试

```bash
python -m unittest discover -s tests -v     # 29 个用例：协议编解码/断帧重同步/流水线/预处理
```

## 9. 团队分工

| 成员 | 分工 |
|---|---|
| 刘淇 | 团队统筹；颜色检测、主程序与流水线整合 |
| 刘绅楠 | 图像预处理（去噪 / CLAHE / γ 校正） |
| 张哲灵 | 形状识别（矩形 / 圆形几何判定） |
| 毛承耀 | 使用场景调研、嵌入式方案 |
| 李若楠 | 使用场景调研、串口协议设计 |
