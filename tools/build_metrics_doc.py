# -*- coding: utf-8 -*-
"""生成《考核指标与验收方案》：把"准确率>98%、延迟<50ms"拆成可执行、可复现的指标。

同时输出：
    docs/考核指标与验收方案.docx
    docs/考核指标与验收方案.md
"""

from __future__ import annotations

import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCX = os.path.join(ROOT, "docs", "考核指标与验收方案.docx")
MD = os.path.join(ROOT, "docs", "考核指标与验收方案.md")

SONG, HEI = "宋体", "黑体"

# ----------------------------------------------------------------------
# 指标表：编号, 指标, 定义/公式, 合格, 良好, 优秀, 当前值, 验收方法
FUNC = [
    ("A1", "多目标识别与编号",
     "同帧 ≥3 个同色目标，全部给出不重复编号，漏检 0",
     "通过", "—", "—", "通过（clutter 每帧 4 个红色目标）",
     "benchmark 的 clutter 场景逐图核对编号"),
    ("A2", "串口帧格式符合协议",
     "帧头/长度/ID/X/Y/CONF/CLS/CRC16/帧尾逐字节一致",
     "通过", "—", "—", "通过",
     "3000 帧编解码逐字节比对，见 tests/test_protocol.py"),
    ("A3", "断帧与噪声重同步",
     "帧前注入 4~64 字节随机噪声后仍能解出后续帧",
     "100%", "—", "—", "100%",
     "exp_protocol.py（每组 3000 帧）"),
    ("A4", "三种输入入口可用",
     "单图 / 批量目录 / 摄像头均可运行并输出结果",
     "通过", "—", "—", "通过（main.py --image/--dir/--camera）",
     "smoke 脚本 main_test.py + 手工摄像头测试"),
    ("A5", "结果可追溯",
     "每帧输出 JSON，含目标列表与四阶段耗时",
     "通过", "—", "—", "通过",
     "main.py --json，字段见 docs/接口说明.md"),
]

ACC = [
    ("B1", "目标级 F1（五场景合计）",
     "F1 = 2PR/(P+R)，目标类别=红色圆形靶标，容差 τ",
     "≥0.90", "≥0.95", "≥0.97", "0.975", "benchmark.py 合计行"),
    ("B2", "召回率 R（五场景合计）",
     "R = TP/(TP+FN)，并给出 Wilson 95% 置信区间",
     "≥0.90", "≥0.95", "≥0.97", "0.978（CI 0.958~0.988）", "benchmark.py 合计行"),
    ("B3", "精确率 P（五场景合计）",
     "P = TP/(TP+FP)",
     "≥0.90", "≥0.95", "≥0.97", "0.973（CI 0.952~0.985）", "benchmark.py 合计行"),
    ("B4", "单场景最差召回",
     "五类场景召回取最小值（防止均值掩盖短板）",
     "≥0.85", "≥0.90", "≥0.95", "0.950（遮挡场景）", "benchmark.py 分场景表"),
    ("B5", "中心定位误差 P95",
     "匹配目标中心与真值中心的欧氏距离的 95 分位",
     "≤5 px", "≤3 px", "≤2 px", "0.81 px（遮挡单列 4.78 px）",
     "benchmark.py 误差分位数"),
    ("B6", "形状判定准确率",
     "已匹配红块中，形状（圆/矩）判对的比例",
     "≥0.90", "≥0.94", "≥0.96", "0.967", "benchmark.py 形状列"),
    ("B7", "同色异形误判率",
     "红色矩形被判成红色圆形的比例（越低越好）",
     "≤5%", "≤2%", "≤1%", "0%", "benchmark.py（clutter 场景 FP 计数）"),
    ("B8", "低照度召回（必达）",
     "目标区亮度 V∈[60,120] 场景下的召回率",
     "≥0.90", "≥0.95", "≥0.97", "0.988", "benchmark.py 的 dim 场景"),
    ("B9", "标定重投影误差",
     "棋盘格内参标定后 cv2.calibrateCamera 的 RMS 重投影误差（像素）",
     "≤1.0 px", "≤0.5 px", "≤0.3 px", "0.476 px（合成棋盘格，真机待测）",
     "exp_extra_metrics.py"),
]

REAL = [
    ("C1", "端到端延迟 P95（板端）",
     "t1−t0，t0=cap.read() 返回，t1=串口整帧写出完成+链路传输 "
     "（14 B @115200 ≈ 1.2 ms）",
     "≤50 ms", "≤40 ms", "≤30 ms", "PC 基线 17.5 ms（真实集）/ 7.6 ms（合成集）",
     "bench_timing.py + 板端串口回环，取 P95"),
    ("C2", "稳态帧率（60 s 平均）",
     "连续运行 60 s 的帧数/时间，且不允许出现 <10 FPS 的 1 s 窗口",
     "≥15 FPS", "≥20 FPS", "≥25 FPS", "PC 基线 86.7 FPS（中位）",
     "板端 60 s 计数，同时记录每秒帧率曲线"),
    ("C3", "冷启动首帧耗时",
     "从进程启动到输出第一帧结果（含 OpenCV/CLAHE 初始化）",
     "≤1 s", "≤0.6 s", "≤0.4 s", "约 0.21 s（含首次调用懒初始化）",
     "计时脚本包一层 main.py 启动"),
]

RELY = [
    ("D1", "连续运行稳定性",
     "连续 2 h 不崩溃，进程 RSS 增长 ≤5%",
     "2 h 通过", "4 h 通过", "8 h 通过", "待板端测试",
     "板端 nohup 运行 + psutil 采样 RSS"),
    ("D2", "丢帧率",
     "读取失败/超时帧数 ÷ 总帧数",
     "≤1%", "≤0.5%", "≤0.1%", "待板端测试",
     "摄像头循环统计 ok=False 次数"),
    ("D3", "错帧被误收率",
     "被破坏的帧被当作有效帧接收的比例（越低越好）",
     "0", "0", "0", "0（1~8 比特翻转，5 组×3000 帧）",
     "exp_protocol.py"),
    ("D4", "实链路 CRC 错帧率",
     "2 h 运行中 CRC 校验失败帧数 ÷ 总帧数",
     "≤0.1%", "≤0.05%", "≤0.01%", "待板端测试",
     "FrameParser.stats 的 crc_errors 累计值"),
    ("D5", "安全联锁",
     "无目标或置信度低于阈值时，帧内\u201c允许发射\u201d位必须为 0",
     "100%", "—", "—", "待实现（协议已预留 flags 位）",
     "回放无目标视频，检查 flags 位"),
    ("D6", "异常恢复",
     "拔插摄像头/串口后 30 s 内自动恢复出帧",
     "≤30 s", "≤15 s", "≤5 s", "待实现",
     "人工拔插测试"),
    ("D7", "定位重复性（重复精度）",
     "同一目标在加噪（σ=10）条件下连续 100 帧，中心坐标的径向标准差",
     "≤1.0 px", "≤0.7 px", "≤0.4 px", "0.153 px（合成）/ 0.634 px（真实图）",
     "exp_extra_metrics.py"),
]

ENG = [
    ("E1", "C++ 与 Python 结果一致性",
     "同一批 200 张图：目标数差异 ≤2%，中心误差差 ≤2 px",
     "通过", "差异≤1%", "差异≤0.5%", "未完成（C++ 仅颜色模块）",
     "两版输出 JSON 逐图对比"),
    ("E2", "可构建与部署",
     "CMake 一键构建；板端可直接运行",
     "通过", "—", "—", "未完成",
     "板端 cmake .. && make && ./vision_demo"),
    ("E3", "交付物完整性",
     "源码/协议文档/接口文档/使用指南/测试报告/演示视频/一键复现脚本",
     "7 项齐全", "—", "—", "6 项（缺 C++ 与演示视频）",
     "对照清单逐项核对"),
]


# ----------------------------------------------------------------------
def set_font(run, name, size, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:eastAsia", "w:ascii", "w:hAnsi"):
        rfonts.set(qn(attr), name)


def para(doc, text, size=10.5, font=SONG, bold=False,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=4, line=16):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = Pt(line)
    run = p.add_run(text)
    set_font(run, font, size, bold)
    return p


def h1(doc, text):
    p = para(doc, text, size=13, font=HEI, bold=True, before=12, after=5, line=20)
    p.paragraph_format.keep_with_next = True
    p.runs[0].font.color.rgb = RGBColor(0x1F, 0x3B, 0x63)


def h2(doc, text):
    p = para(doc, text, size=11, font=HEI, bold=True, before=9, after=3, line=18,
             align=WD_ALIGN_PARAGRAPH.LEFT)
    p.paragraph_format.keep_with_next = True


def _no_split(row):
    """禁止该行在分页处被拆断。"""
    tr_pr = row._tr.get_or_add_trPr()
    el = tr_pr.makeelement(qn("w:cantSplit"), {})
    tr_pr.append(el)


def _repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    el = tr_pr.makeelement(qn("w:tblHeader"), {})
    tr_pr.append(el)


def table(doc, header, rows, widths, size=8.0):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    def fill(cells, texts, bold=False, center=False):
        for cell, text in zip(cells, texts):
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = Pt(11)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_font(p.add_run(str(text)), SONG, size, bold)

    fill(t.rows[0].cells, header, bold=True, center=True)
    _repeat_header(t.rows[0])
    _no_split(t.rows[0])
    for row in rows:
        cells = t.add_row().cells
        fill(cells, row)
        _no_split(t.rows[-1])
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
    return t


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("\n", " ") for c in r) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------
def main() -> int:
    md = ["# 考核指标与验收方案", ""]

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.9)
    sec.top_margin = sec.bottom_margin = Cm(1.9)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("基于 OpenCV 的轻量化视觉识别与定位"), HEI, 15, True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    set_font(p.add_run("考核指标与验收方案"), HEI, 14, True)
    para(doc, "项目负责人：刘淇　　版本：2026-11-04　　适用阶段：中期检查—结题验收",
         size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)

    # 0
    h1(doc, "一、为什么要重写指标")
    para(doc, "原项目书写的\u201c识别准确率高于 98%、延迟低于 50 ms、帧率不低于 15 FPS、"
              "数据传输错误率低于 0.1%\u201d方向正确，但缺少可验收的口径，实际无法判定是否达标：")
    for line in [
        "1. \u201c准确率\u201d未区分召回率、精确率与分类正确率——把多检出的干扰物当目标，"
        "或漏掉一半目标，都可能报出 98%；",
        "2. 未定义\u201c命中\u201d的容差：中心偏差 2 px 与 20 px 都算对？没有容差就没有定位精度指标；",
        "3. 未规定测试集（数量、场景分布、真值来源）：同一套代码在本项目数据上报 0.975，"
        "在仅用颜色的配置上报 0.863，在低照度未开预处理时报 0.000——数据集不定，数字就没有意义；",
        "4. 延迟未定义起止点（含不含采集、含不含串口发送）；",
        "5. 帧率未规定测量窗口与抖动容忍（瞬时 30 FPS 与稳态 30 FPS 是两回事）；",
        "6. 误码率在没有物理误码注入设备的条件下无法直接测量。",
    ]:
        para(doc, line, after=3)

    para(doc, "举证：同一版本代码，本机空闲时单帧中位耗时 5.8 ms，机器有负载时实测 40 ms 以上，"
              "相差约 7 倍；同一测试集上，仅用颜色的配置端到端精确率 0.500，"
              "加入形状融合后 1.000。口径不写死，任何百分比都可以被\u201c调\u201d出来。",
         after=4)

    # 1
    h1(doc, "二、任务定义与命中判据")
    for line in [
        "输入：单帧 BGR 图像 640×480（板端来自 USB 摄像头，MJPG，目标 30 FPS）。",
        "输出：目标列表，每项含编号、类别（颜色+形状）、中心像素坐标、置信度、可选距离；"
        "按协议经串口下发主控。",
        "命中判据：类别正确 **且** 中心误差 ≤ τ，其中 τ = max(15 px, 0.35 × 目标外接圆直径)。"
        "之所以取 0.35 D，是因为该偏差下串口输出的坐标仍落在靶标有效区域内。",
        "目标类别：主目标为\u201c红色圆形靶标\u201d；\u201c红色矩形\u201d作为同色异形排斥项单独统计误判率。",
        "场景边界：单帧 ≤8 个目标；目标直径 20~90 px；遮挡面积 ≤40%；照度覆盖目标区亮度 V 60~240。",
    ]:
        para(doc, line.replace("**", ""), after=3)

    # 2
    h1(doc, "三、指标体系（5 层 27 项）")
    para(doc, "每项指标只回答一个问题，并绑定一个可执行的验收动作与产物文件。"
              "\u201c合格\u201d为结题必须达到的下限，\u201c良好/优秀\u201d为加分档，"
              "\u201c当前值\u201d为截至 2026-11-04 的实测，来源见目录 docs/。", after=5)

    h2(doc, "A. 功能完整性（二值判定，不达标即不合格）")
    table(doc, ["编号", "指标", "定义 / 判据", "合格", "良好", "优秀", "当前值", "验收方法"],
          FUNC, [1.1, 2.6, 3.8, 1.5, 1.5, 1.5, 2.7, 2.5])
    md += ["## A. 功能完整性", "", md_table(
        ["编号", "指标", "定义 / 判据", "合格", "良好", "优秀", "当前值", "验收方法"], FUNC), ""]

    h2(doc, "B. 精度指标（核心，按三档给分）")
    table(doc, ["编号", "指标", "定义 / 判据", "合格", "良好", "优秀", "当前值", "验收方法"],
          ACC, [1.1, 2.6, 3.8, 1.5, 1.5, 1.5, 2.7, 2.5])
    md += ["## B. 精度指标", "", md_table(
        ["编号", "指标", "定义 / 判据", "合格", "良好", "优秀", "当前值", "验收方法"], ACC), ""]

    h2(doc, "C. 实时性指标（板端必达）")
    table(doc, ["编号", "指标", "定义 / 判据", "合格", "良好", "优秀", "当前值", "验收方法"],
          REAL, [1.1, 2.6, 3.8, 1.5, 1.5, 1.5, 2.7, 2.5])
    md += ["## C. 实时性指标", "", md_table(
        ["编号", "指标", "定义 / 判据", "合格", "良好", "优秀", "当前值", "验收方法"], REAL), ""]

    h2(doc, "D. 可靠性指标")
    table(doc, ["编号", "指标", "定义 / 判据", "合格", "良好", "优秀", "当前值", "验收方法"],
          RELY, [1.1, 2.6, 3.8, 1.5, 1.5, 1.5, 2.7, 2.5])
    md += ["## D. 可靠性指标", "", md_table(
        ["编号", "指标", "定义 / 判据", "合格", "良好", "优秀", "当前值", "验收方法"], RELY), ""]

    h2(doc, "E. 工程与交付指标")
    table(doc, ["编号", "指标", "定义 / 判据", "合格", "良好", "优秀", "当前值", "验收方法"],
          ENG, [1.1, 2.6, 3.8, 1.5, 1.5, 1.5, 2.7, 2.5])
    md += ["## E. 工程与交付指标", "", md_table(
        ["编号", "指标", "定义 / 判据", "合格", "良好", "优秀", "当前值", "验收方法"], ENG), ""]

    # 3
    h1(doc, "四、测量口径（缺一条，数字就不可比）")
    for line in [
        "测试集：合成集 200 张（正常/复杂背景/低照度/过曝模糊/遮挡 各 40 张，随机种子 20260401 固定）；"
        "真实集为实拍图（当前 4 张，结题前补至 ≥30 张）。两组分别报告，不混算。",
        "真值：合成集由生成器直接给出；真实集人工标注，标注文件与图像同目录。",
        "匹配：贪心最近邻，容差 τ = max(15 px, 0.35 × 目标外接圆直径)，一对一不重复。",
        "比例类指标：同时给点估计与 Wilson 95% 置信区间；误差类指标给均值、RMSE、P50、P95。",
        "耗时：预热 10 帧 → 重复 3 轮 → 取中位数与 P95；报告机器型号与是否空闲；"
        "分\u201c合成集/真实集\u201d两组，不用单一数字代表全部。",
        "帧率：连续 60 s 计数，同时记录每秒帧率曲线，任一 1 s 窗口低于 10 FPS 视为抖动不合格。",
        "复现：make_dataset.py → benchmark.py → bench_timing.py → exp_protocol.py → unittest，"
        "全程无人工干预。",
    ]:
        para(doc, line, after=3)

    # 4
    h1(doc, "五、样本量：为什么现在不能宣称\u201c98%\u201d")
    para(doc, "合成集共 480 个红色目标（其中靶标 400 个）。按 Wilson 区间，召回 0.978 的 "
              "95% 置信区间为 [0.958, 0.988]：下界 0.958 意味着\u201c不少于 95%\u201d是有证据的，"
              "\u201c不少于 98%\u201d在当前样本量下不成立。", after=3)
    table(doc, ["期望的估计精度", "所需目标样本量", "当前样本量", "结论"],
          [["±2.0%", "约 188 个", "480 个", "已满足"],
           ["±1.5%", "约 335 个", "480 个", "已满足"],
           ["±1.0%", "约 753 个", "480 个", "不足，需补样本"],
           ["±0.5%", "约 3010 个", "480 个", "不现实"]],
          [3.6, 3.6, 3.6, 6.0])
    md += ["## 五、样本量", "",
           md_table(["期望的估计精度", "所需目标样本量", "当前样本量", "结论"],
                    [["±2.0%", "约 188 个", "480 个", "已满足"],
                     ["±1.5%", "约 335 个", "480 个", "已满足"],
                     ["±1.0%", "约 753 个", "480 个", "不足，需补样本"],
                     ["±0.5%", "约 3010 个", "480 个", "不现实"]]), ""]
    para(doc, "因此建议二选一：①把指标写成\u201c召回率 95% 置信下界 ≥0.95\u201d（当前 0.958，成立）；"
              "②把合成集扩到 ≥750 个靶标后，再宣称 ≥0.98。这也是本项目\u201c指标模糊\u201d问题的根源之一："
              "不是达不到 98%，而是原来的表述在现有样本量下不可证。", before=3)

    # 6
    h1(doc, "六、外部对标：这些数字定在什么量级合理")
    para(doc, "公开检索的结果先说结论：同类项目（飞镖/自瞄/嵌入式视觉）极少给出可核验的量化指标，"
              "多数只有\u201c识别稳定、延迟低\u201d这类描述。能拿来对标的只有三类量级，"
              "其余口径只能借用工业界与学术界的通行做法。", after=4)
    table(doc, ["来源", "同类做法 / 数值", "对本项目指标的作用", "可核验程度"],
          [["深圳大学 RobotPilots《RM2024 飞镖镖架制导设计方案》"
            "（bbs.robomaster.com/article/54097）",
            "识别靶位绿灯像素偏差，传电控用 PID 跟随 YAW；不承诺绝对定位精度",
            "同类\u201c飞镖靶标\u201d方案以\u201c像素偏差 + 闭环收敛\u201d为目标，"
            "指标应落在定位稳定而非绝对精度",
            "正文需登录，仅摘要可见"],
           ["开源飞镖视觉 Elucidater01/2024_Dart_Algorithm（GitHub）",
            "同样是\u201c识别基准点→算像素差→串口下发\u201d，README 未列精度与帧率",
            "同类项目普遍不量化；本项目给出可复现指标本身即是区分度",
            "README 可核验"],
           ["Arducam 论坛帖：Low FPS (15) with cv2 color detection on Raspberry Pi 4B",
            "Pi4B + OpenCV 颜色检测约 15 FPS",
            "板端 \u201c≥15 FPS\u201d作合格线与现实吻合，并非放水",
            "仅标题可见，正文被 Cloudflare 拦截"],
           ["智能车竞赛 FPGA 方案讨论（z.shaonianxue.cn）",
            "高性能方案追求处理延迟 <10 ms",
            "给优秀档留出 30 ms 的目标空间",
            "讨论页可见"],
           ["嵌入式农业视觉论文（智慧农业 2025，搜索摘要）",
            "全流程单帧 43.8 ms（预处理 20 + 推理 23.8）",
            "嵌入式视觉 40~50 ms 属常态，50 ms 门槛合理",
            "仅摘要可见，PDF 未能解析"],
           ["工业机器视觉重复性评估（ScienceDirect，针床对位）",
            "用像素—毫米标定与重复性评价系统，而非单一\u201c准确率\u201d",
            "据此增设 B9 重投影误差、D7 定位重复性两项工业口径指标",
            "摘要可见"]],
          [4.2, 4.6, 4.6, 3.2])
    md += ["## 六、外部对标", "",
           md_table(["来源", "同类做法 / 数值", "对本项目指标的作用", "可核验程度"],
                    [["深大 RobotPilots RM2024 飞镖镖架制导方案", "绿灯像素偏差→PID 跟随 YAW",
                      "同类方案以像素偏差闭环为目标", "仅摘要可见"],
                     ["Elucidater01/2024_Dart_Algorithm", "像素差→串口下发，无量化指标",
                      "量化本身是区分度", "README 可核验"],
                     ["Arducam 论坛（Pi4B + cv2 颜色检测）", "约 15 FPS", "15 FPS 作合格线合理", "仅标题"],
                     ["智能车 FPGA 方案讨论", "<10 ms 处理延迟", "30 ms 作优秀档", "可见"],
                     ["嵌入式农业视觉论文", "单帧 43.8 ms", "50 ms 作合格线合理", "仅摘要"],
                     ["工业视觉重复性评估", "用重复性与重投影误差", "增设 B9、D7", "摘要"]]), ""]
    para(doc, "阈值依据分四类，写在报告里便于答辩时被追问：", before=4, after=2)
    for line in [
        "① 外部对标：C1（50/40/30 ms）、C2（15/20/25 FPS），依据上表；",
        "② 行业通用口径：B9 重投影误差（≤1.0/0.5/0.3 px，棋盘格标定常见 0.3~1.0 px）、"
        "D7 定位重复性（≤1.0/0.7/0.4 px）；",
        "③ 自身实测锚定：B1~B8、C3、D3 的阈值取在实测值上下浮动一档，保证\u201c当前能过、"
        "退步会被发现\u201d；",
        "④ 工程通行量级：D1 连续运行 2 h、D2 丢帧 ≤1%、D4 CRC 错帧 ≤0.1%。",
    ]:
        para(doc, line, after=2)
    # 7
    h1(doc, "七、与原项目书指标的对照改写")
    table(doc, ["原表述", "存在的问题", "建议表述"],
          [["识别准确率高于 98%",
            "未区分召回/精确率，未定义容差与测试集，样本量不足以支撑 98%",
            "在容差 τ=max(15px,0.35D) 下，五场景合计目标级 F1 ≥0.95（优秀 ≥0.97）、"
            "召回 95% 置信下界 ≥0.95、单场景召回 ≥0.90；精确率 ≥0.95"],
           ["延迟低于 50 ms",
            "未定义起止点，未说明测试平台与统计量",
            "板端端到端延迟 P95 ≤50 ms（t0=read() 返回，t1=串口整帧写出+链路传输），"
            "测试集与机器型号在报告中注明"],
           ["帧率不低于 15 FPS",
            "未规定测量窗口与抖动",
            "连续 60 s 稳态平均 ≥15 FPS，且不出现低于 10 FPS 的 1 s 窗口"],
           ["数据传输错误率低于 0.1%",
            "无物理误码注入设备时无法直接测量",
            "注入实验：1~8 比特翻转下错帧被误收率为 0；实链路 2 h 运行 CRC 错帧率 ≤0.1%"]],
          [3.4, 5.2, 8.2])
    md += ["## 七、与原项目书指标的对照改写", "",
           md_table(["原表述", "存在的问题", "建议表述"],
                    [["识别准确率高于 98%", "未区分召回/精确率，未定义容差与测试集，样本量不足以支撑 98%",
                      "F1 ≥0.95、召回 95% 置信下界 ≥0.95、单场景召回 ≥0.90、精确率 ≥0.95"],
                     ["延迟低于 50 ms", "未定义起止点，未说明测试平台与统计量",
                      "板端端到端 P95 ≤50 ms（t0=read() 返回，t1=串口写出+链路传输）"],
                     ["帧率不低于 15 FPS", "未规定测量窗口与抖动",
                      "连续 60 s 稳态平均 ≥15 FPS，无 <10 FPS 的 1 s 窗口"],
                     ["数据传输错误率 <0.1%", "无注入设备时无法直接测",
                      "比特翻转错帧误收率 0；实链路 2 h CRC 错帧率 ≤0.1%"]]), ""]

    # 6
    h1(doc, "八、当前状态与缺口")
    table(doc, ["状态", "指标", "说明"],
          [["已达标", "A1~A5、B1~B9、C3、D3、D7",
            "精度与协议可靠性均有自动化脚本与产物文件，可当场复现"],
           ["待板端验证", "C1、C2、D1、D2、D4",
            "硬件未到位；PC 基线已达标，板端需重测后填数"],
           ["未达标 / 待实现", "D5、D6、E1、E2",
            "安全联锁位未接入；C++ 仅颜色模块；异常恢复未做"],
           ["交付缺口", "E3",
            "缺 C++ 可执行程序与演示视频"]],
          [3.0, 4.8, 9.2])
    md += ["## 八、当前状态与缺口", "",
           md_table(["状态", "指标", "说明"],
                    [["已达标", "A1~A5、B1~B9、C3、D3、D7", "精度与协议可靠性均有自动化脚本与产物文件"],
                     ["待板端验证", "C1、C2、D1、D2、D4", "PC 基线已达标，板端需重测后填数"],
                     ["未达标/待实现", "D5、D6、E1、E2", "安全联锁位未接入；C++ 仅颜色模块；异常恢复未做"],
                     ["交付缺口", "E3", "缺 C++ 可执行程序与演示视频"]]), ""]

    # 7
    h1(doc, "九、验收执行清单（评委可现场抽测）")
    for line in [
        "1. 生成数据集：python tools/make_dataset.py --out dataset --per-condition 40",
        "2. 精度与耗时：python tools/benchmark.py --dataset dataset --out docs   → benchmark_table.md",
        "3. 固定协议计时：python tools/bench_timing.py   → timing_bench.json",
        "4. 协议抗干扰：python tools/exp_protocol.py",
        "5. 单元测试：python -m unittest discover -s tests -v（29 项）",
        "6. 单图抽查：python main.py --image data/test002.png --json out.json",
        "7. 板端（硬件到位后）：python main.py --camera 0 --serial /dev/ttyUSB0 连续 2 h，"
        "记录 FPS、CRC 错帧数、RSS 曲线",
    ]:
        para(doc, line, after=2)

    para(doc, "说明：以上指标全部由脚本产出，不依赖人工读数；任一指标不达标时，报告中需给出"
              "实测值、差距与整改计划，而不是只写\u201c基本达到\u201d。", before=6, after=2)

    doc.save(DOCX)
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("已生成:", DOCX)
    print("已生成:", MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
