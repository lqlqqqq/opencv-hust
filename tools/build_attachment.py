# -*- coding: utf-8 -*-
"""生成《中期检查附件材料》：处理流程图、指标表、对比图、真实场景结果图。

数据来源：
    docs/figures/*.png
    docs/benchmark_results.json

输出：
    docs/中期检查附件-图表材料.docx
"""

from __future__ import annotations

import json
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(SRC, "docs", "figures")
OUT = os.path.join(SRC, "docs", "中期检查附件-图表材料.docx")

SONG = "宋体"
HEI = "黑体"


def set_font(run, name: str, size: float, bold: bool = False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)


def para(doc, text, size=10.5, font=SONG, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         space_after=6, first_indent=0.0):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    pf.line_spacing = Pt(16)
    pf.first_line_indent = Cm(first_indent)
    run = p.add_run(text)
    set_font(run, font, size, bold)
    return p


def heading(doc, text, size=12):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_font(run, HEI, size, True)
    run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x63)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(text)
    set_font(run, SONG, 9)
    return p


def add_image(doc, path, width_cm=15.5, cap=None):
    if not os.path.exists(path):
        para(doc, f"（缺少插图：{os.path.basename(path)}）")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(path, width=Cm(width_cm))
    if cap:
        caption(doc, cap)


def add_table(doc, header, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, text in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        set_font(run, SONG, 9, True)
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(text))
            set_font(run, SONG, 9)
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    return table


def main() -> int:
    with open(os.path.join(SRC, "docs", "benchmark_results.json"), encoding="utf-8") as f:
        data = json.load(f)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.2)

    # 标题
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run("基于 OpenCV 的轻量化视觉识别与定位")
    set_font(run, HEI, 16, True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("项目中期检查 · 附件材料")
    set_font(run, HEI, 14, True)
    para(doc, "项目负责人：刘淇　　填表时间：2026 年 11 月",
         size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

    # 1 处理流程
    heading(doc, "一、系统处理流程")
    add_image(doc, os.path.join(FIG, "fig_pipeline.png"), 15.5, "图 1  视觉识别与定位系统处理流程")
    para(doc, "说明：颜色路与形状路采用不同的预处理——颜色路做 V 通道 γ 校正以把暗目标抬进检测阈值，"
              "形状路仅做轻降噪、不做锐化（实验表明锐化会使圆形召回由 98.8% 降至 35.0%）。"
              "两路结果按中心距离与短边尺寸一致性融合，同时满足颜色与形状才判为目标类别。")

    # 2 量化指标
    heading(doc, "二、量化评测结果")
    para(doc, "评测数据：200 张 640×480 合成图，5 类场景各 40 张；目标类别为红色圆形靶标；"
              "匹配容差 max(15px, 0.35×目标尺寸)。端到端指标指“把系统输出直接当作靶标坐标”时的召回与精确率。")

    cond_names = {
        "normal": "正常光照", "clutter": "复杂背景", "dim": "低照度",
        "bright_blur": "过曝+模糊", "occlusion": "目标遮挡",
    }
    variants = {
        "color_only": "颜色单路",
        "fusion": "颜色+形状融合",
        "fusion_lowlight": "融合+双路预处理",
    }
    for key, label in variants.items():
        summary = data["variants"][key]
        para(doc, f"表（{label}）", size=9.5, bold=True, space_after=2)
        rows = []
        for cond, m in summary.items():
            rows.append([
                cond_names.get(cond, cond),
                f"{m['color_recall']:.3f}",
                f"{m['color_precision']:.3f}",
                f"{m['e2e_recall']:.3f}",
                f"{m['e2e_precision']:.3f}",
                f"{m['center_err_mean']:.2f}",
                f"{m['ms_mean']:.1f}",
                f"{m['fps_mean']:.0f}",
            ])
        add_table(
            doc,
            ["场景", "颜色召回", "颜色精确率", "端到端召回", "端到端精确率", "中心误差(px)", "耗时(ms)", "FPS"],
            rows,
        )
        para(doc, "", space_after=2)

    add_image(doc, os.path.join(FIG, "fig_metrics_by_scene.png"), 15.5,
              "图 2  融合检测（含双路预处理）在各场景下的表现")
    add_image(doc, os.path.join(FIG, "fig_variant_precision.png"), 15.5,
              "图 3  不同配置下目标判定精确率对比")
    add_image(doc, os.path.join(FIG, "fig_timing.png"), 15.5,
              "图 4  单帧耗时分解（预处理 / 颜色检测 / 形状检测）")

    # 3 关键实验
    heading(doc, "三、关键对比实验")
    para(doc, "1. 颜色-形状融合的必要性：仅用颜色时，同色异形干扰（红色矩形）会被当作靶标，"
              "复杂背景下端到端精确率只有 0.500；加入形状判定后提升到 1.000，代价约为每帧增加 15ms。")
    para(doc, "2. 低照度增强：暗场目标像素 V≈88，低于颜色检测阈值 V≥100，未预处理时端到端召回为 0；"
              "V 通道 γ 校正后 V>130，召回恢复到 0.988。")
    para(doc, "3. 预处理链路对比（低照度 40 张）：不处理时颜色召回 0.325、形状召回 0.988；"
              "CLAHE+锐化时颜色召回 0.338、形状召回降至 0.225；V 通道 γ 校正且不锐化时颜色召回 1.000、"
              "形状召回 0.988，为最终采用方案。")
    add_image(doc, os.path.join(FIG, "fig_lowlight_compare.png"), 15.0,
              "图 5  低照度场景下的预处理效果对比")

    # 4 真实场景
    heading(doc, "四、真实测试图检测结果")
    add_image(doc, os.path.join(FIG, "fig_real_cases.png"), 15.5,
              "图 6  真实测试图检测结果（左：正常场景；右：高饱和彩色场景的误检）")
    para(doc, "右图中的粉、品红、紫色块在弱光与阴影下会进入红色阈值区间，虽已加入通道主导度约束"
              "（R>1.5·max(G,B)）过滤掉大部分干扰，仍是下一阶段需要继续改进的问题点。")

    # 5 协议
    heading(doc, "五、串口通信协议与可靠性")
    para(doc, "帧格式：AA BB | LEN | ID(1) X(2) Y(2) CONF(1) CLS(1) | CRC16(2) | CC DD，"
              "CRC 采用 CRC-16/MODBUS，校验范围为长度字节与有效数据；接口参数 115200、8N1。"
              "目标类别编码为“十位颜色 + 个位形状”，红色圆形为 0x11（17）。")
    para(doc, "抗干扰实验（每组 3000 帧）：1/2/3/5/8 比特随机翻转下，错帧被当作有效帧的漏检数均为 0；"
              "帧前混入 4/16/64 字节随机噪声时，重同步成功率均为 100%。")

    # 6 文件清单
    heading(doc, "六、随附材料清单")
    for line in [
        "1. 源码与文档：opencv-hust-main1.3（含 src、tools、tests、docs 目录）；",
        "2. 测试数据集：dataset/（200 张合成图 + annotations.json 真值）；",
        "3. 评测结果：docs/benchmark_results.json、docs/benchmark_table.md；",
        "4. 进度说明：docs/中期进展说明.md；接口说明：docs/接口说明.md；",
        "5. 单元测试：python -m unittest discover -s tests（29 个用例）。",
    ]:
        para(doc, line, space_after=2)

    doc.save(OUT)
    print("已生成:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
