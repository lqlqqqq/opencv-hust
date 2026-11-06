# -*- coding: utf-8 -*-
"""按学院《项目中期检查表》原版式填写中期检查报告。

输入：docs/中期检查模板.docx（由学院下发的 .doc 转换而来，保留原表格与字体）
输出：docs/中期检查表-刘淇-2026年11月.docx

填写原则：
* 不改动模板的页面设置、表格边框、栏目文字与签名行；
* 正文格内的空段落替换为正文（宋体五号、固定行距）；
* 签名/日期、指导教师意见、院系审核意见保持空白，交由老师与学院填写；
* 需要本人确认的字段（项目编号、经费金额）留空或留横线，不臆造数据。
"""

from __future__ import annotations

import os

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "docs", "中期检查模板.docx")
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "docs", "中期检查表-刘淇-2026年11月.docx")

SONG = "宋体"
KAI = "楷体_GB2312"

BODY_SIZE = 10.5      # 五号
BODY_LINE_PT = 16.0   # 固定行距，约为字号的 1.5 倍
HEADER_SIZE = 12.0


# ----------------------------------------------------------------------
def set_run_font(run, chinese_font: str, size_pt: float, bold: bool = False):
    run.font.name = chinese_font
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), chinese_font)
    rfonts.set(qn("w:ascii"), chinese_font)
    rfonts.set(qn("w:hAnsi"), chinese_font)


def style_paragraph(p, line_pt: float,
                    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                    first_indent_pt: float = 0.0):
    pf = p.paragraph_format
    pf.alignment = align
    pf.line_spacing = Pt(line_pt)          # 固定行距
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.first_line_indent = Pt(first_indent_pt)
    pf.left_indent = Pt(0)
    pf.right_indent = Pt(0)


def set_cell_value(cell, text: str, font: str = KAI, size: float = HEADER_SIZE,
                   align=WD_ALIGN_PARAGRAPH.LEFT):
    """填写表头单元格（单行值）。"""
    para = cell.paragraphs[0]
    for run in list(para.runs):
        run._element.getparent().remove(run._element)
    run = para.add_run(text)
    set_run_font(run, font, size)
    para.alignment = align
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def fill_cell(cell, lines, line_pt: float = BODY_LINE_PT, size: float = BODY_SIZE,
              keep_signature: bool = True):
    """把若干行正文写入单元格，保留（签名）行及其后的日期行。"""
    paras = list(cell.paragraphs)
    sig_idx = None
    if keep_signature:
        for i, p in enumerate(paras):
            if "签名" in p.text:
                sig_idx = i
                break

    if sig_idx is None:
        for p in paras:
            p._element.getparent().remove(p._element)
        anchor = None
    else:
        for p in paras[:sig_idx]:
            if not p.text.strip():
                p._element.getparent().remove(p._element)
        anchor = paras[sig_idx]

    for text in lines:
        if anchor is not None:
            new_p = anchor.insert_paragraph_before("")
        else:
            new_p = cell.add_paragraph()
        run = new_p.add_run(text)
        set_run_font(run, SONG, size)
        style_paragraph(new_p, line_pt)


def get_cell(table, r: int, c: int):
    return table.rows[r].cells[c]


def trim_empty_paragraphs(cell, n: int = 1):
    """删除签名行之前的若干空段落（用于压缩留白，避免内容被挤到下一页）。"""
    removed = 0
    for p in list(cell.paragraphs):
        if removed >= n:
            break
        if "签名" in p.text:
            break
        if not p.text.strip():
            p._element.getparent().remove(p._element)
            removed += 1
    return removed


def add_progress_table(cell, rows, widths=(5.5, 8.3, 2.3), size=9.0):
    """在单元格内插入"对照项目书研究内容"的小表。"""
    table = cell.add_table(rows=0, cols=3)
    table.style = "Table Grid"
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    def write(cells, texts, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT):
        for c, text in zip(cells, texts):
            p = c.paragraphs[0]
            p.alignment = align
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = Pt(12)
            run = p.add_run(text)
            set_run_font(run, SONG, size, bold)

    header = table.add_row().cells
    write(header, ["项目书研究内容", "已落实情况", "进度"], bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER)
    for row in rows:
        cells = table.add_row().cells
        write(cells, row, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT)
        cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for row in table.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)

    # 单元格必须以段落结尾，否则部分 Word 版本会判为非法结构
    tail = cell.add_paragraph()
    tail.paragraph_format.space_before = Pt(0)
    tail.paragraph_format.space_after = Pt(0)
    tail.paragraph_format.line_spacing = Pt(4)
    return table


# ----------------------------------------------------------------------
HEADER = [
    ("", "项目编号"),
    ("基于OpenCV的轻量化视觉识别与定位", "项目名称"),
    ("刘淇", "项目负责人"),
    ("电子信息与通信学院", "所在院（系）"),
    ("2026.01—2026.12", "实施时间"),
    (os.environ.get("DSH_PHONE", "＿＿＿＿＿＿＿＿"), "联系电话"),
    ("2026年11月4日", "填表时间"),
]

# 对照项目书"2.项目研究内容"的五条，如实标注落实情况
MAPPING = [
    ("①低照度与动态光照预处理；评估 RGB/HSV/YCrCb 色彩空间",
     "V 通道 γ 校正、CLAHE、去噪、颜色/形状双路预处理；HSV 分割；YCrCb 对比待做",
     "部分完成"),
    ("②自适应阈值分割与形态学优化",
     "形态学开闭 + 通道主导度约束；自适应/动态阈值策略待实现（现为固定 HSV 阈值 + 自动 γ）",
     "部分完成"),
    ("③特征过滤与多目标管理",
     "面积/宽高比/圆形度/凸度过滤；多目标编号与排序；轮廓矩与跨帧跟踪待补",
     "部分完成"),
    ("④相机标定与像素→世界坐标转换",
     "尚未开展（依赖硬件到位）",
     "未开始"),
    ("⑤嵌入式集成、轻量串口协议与上位机界面",
     "帧封装/CRC16/断帧重同步/v2 扩展、回环联调完成；C++ 仅颜色模块；GUI 与 NEON/多线程未做",
     "部分完成"),
]
SEC1 = [
    "1. 总体进度：对照项目书四阶段计划，第一阶段的场景分析与 Python 原型、第二阶段的算法优化与"
    "接口/串口协议设计已完成；第三阶段嵌入式移植尚未开展，第四阶段测试工具链已提前搭好。"
    "受开发板、摄像头等硬件采购延迟影响，整体较计划滞后约三个月，算法侧已具备上板条件。",
    "2. 目标与场景：以“红色圆形靶标”为主要识别与定位对象，输入 640×480 图像，输出目标编号、类别、"
    "像素坐标与置信度，经 115200、8N1 串口下发主控，为瞄准与分拣机构提供坐标输入。",
    "3. 处理链路：打通“图像输入 → 双路预处理 → HSV 颜色分割 → Canny 轮廓形状判定 → 颜色-形状融合 → "
    "目标输出”完整流水线，替代 1.2 版各自独立、接口互不兼容的脚本。",
    "4. 颜色识别：HSV 阈值分割 + 形态学开闭 + 通道主导度约束（红色要求 R>1.5·max(G,B)）。"
    "依据实测：纯红色块 R/max(G,B) 约 2.6~3.6，粉、品红、紫色块仅约 1.3，据此剔除相近色相干扰。",
    "5. 形状识别：矩形按轮廓四点近似、内角 70°~110°、凸度≥0.8、宽高比≤10 判定；"
    "圆形按圆形度（4πA/P²）≥0.75、最小外接矩形长短边差≤20%、排除四点近似方形判定。",
    "6. 融合判定：颜色块与形状按中心距离（≤min(30px, 0.5×短边)）与短边尺寸一致性（≥0.45）配对，"
    "两者同时成立才判为目标类别，输出结构化目标列表。",
    "7. 预处理与串口：修正原预处理锐化核归一化错误（核和误写为 1/9，整幅画面被压暗到 1/9），"
    "改为颜色路 V 通道 γ 校正、形状路仅轻降噪的双路处理；按协议文档实现帧封装与流式解析"
    "（AA BB | LEN | ID | X | Y | CONF | CLS | CRC16 | CC DD），CRC 采用 CRC-16/MODBUS，"
    "支持分片、粘包、噪声前缀与长度非法时的重同步，并扩展 v2 帧（距离与锁定/发射标志）。",
    "8. 工程化与测试：统一配置（4 个预设 + JSON）与命令行入口（单图/批量/摄像头，结果可导出 JSON），"
    "修复 1.2 版 main_test.py 与 shape.py 接口签名不一致导致的形状检测崩溃；建立 200 张带真值的"
    "合成测试集（5 类场景）与 29 个单元测试用例，识别效果由定性描述转为可量化指标。",
]

SEC2 = [
    "1. 软件成果：完成 1.3 版代码库，含流水线与配置、颜色检测、形状检测、融合判定、预处理、"
    "串口协议与链路、命令行入口共 12 个模块，代码托管于 github.com/lqlqqqq/opencv-hust。",
    "2. 量化指标（200 张 640×480 合成图，匹配容差 max(15px, 0.35×目标尺寸)）：",
    "（1）均匀背景/正常光照：端到端召回 98.8%、精确率 100%、中心误差 0.01px、约 20ms/帧；",
    "（2）纹理背景/同色异形干扰：召回 98.8%、精确率 100%、约 24ms/帧；",
    "（3）低照度+噪声：开启 V 通道 γ 校正后召回 98.8%、精确率 87.8%（未开启时召回为 0）；",
    "（4）过曝+运动模糊：召回 97.5%；目标遮挡：召回 95.0%；两者精确率均为 100%。",
    "3. 对比实验结论：①仅用颜色时同色异形干扰下精确率仅 50.0%，加入形状融合后提升至 100%；"
    "②暗场目标像素 V≈88 低于检测阈值 100，γ 校正后 V>130，召回由 0 提升至 98.8%；"
    "③对颜色路做锐化会使圆形召回由 98.8% 降至 35.0%，故形状路不锐化。",
    "4. 协议可靠性：每组 3000 帧，1/2/3/5/8 比特随机翻转下错帧漏检均为 0；帧前混入 4~64 字节"
    "随机噪声时重同步成功率 100%，验证了校验与重同步机制有效。",
    "5. 文档成果：接口说明、颜色/形状模块 API 文档、串口协议说明、评测指标表与对比图、单元测试报告。",
    "6. 团队进展：5 人分工稳定、每周例会一次；视觉算法组完成算法整合与参数定型，系统集成组完成"
    "使用场景确认与串口协议设计，并用回环链路完成协议联调。",
    "7. 阶段结论：算法链路与通信链路均已闭环，具备向嵌入式平台移植的条件；剩余工作集中在 C++ "
    "完整移植、树莓派部署、相机标定与真实场景验证。",
]

SEC3 = [
    "1. 硬件未到位：开发板、摄像头、下位机尚未采购齐全，第三阶段嵌入式移植未开展，延迟与帧率"
    "指标仅在 PC 上验证（15~62ms/帧），尚不能证明树莓派 4B 上满足 <50ms、≥15FPS。",
    "2. C++ 移植不完整：仅完成颜色检测模块，主程序依赖的预处理、掩码、串口模块尚未实现；"
    "本机缺少 OpenCV C++ SDK，需在目标板上配置工具链后补齐并逐帧对拍。",
    "3. 粘连目标：相邻同色目标经形态学闭运算后连成一块（真实图 test002 中有 2 处），颜色块中心"
    "偏移导致与形状配对失败，需增加距离变换分水岭或凸包分割。",
    "4. 相近色相干扰：粉、品红、紫色块在弱光或阴影下会进入红色阈值区间，主导度约束可缓解，"
    "但高饱和彩色场景（test001）仍有误检，需引入色差（ΔE）度量与饱和度联合门限。",
    "5. 实时性尚未优化：复杂真实图 50ms/帧中形状检测占 46ms，且双路预处理约 20ms/帧"
    "（其中 CLAHE 约占一半），均未做 ROI 限定、降采样与开关裁剪，是板端实时性的主要风险点。",
    "6. 无相机标定：只输出像素坐标，未做畸变校正与像素→世界坐标映射；真实样本仅 4 张，"
    "泛化性结论主要基于合成数据。",
]

SEC5 = [
    "1. 11 月上旬：实现粘连目标分离（距离变换+分水岭、凸包分割），完成形状检测 ROI 限定与降采样"
    "优化，把复杂场景单帧耗时压到 30ms 以内。",
    "2. 11 月中旬：补齐 C++ 移植（预处理、形状检测、掩码工具、串口），编写 CMake 构建脚本，"
    "并用同一批合成图与 Python 版逐帧对拍，指标差异超过 2% 即定位排查。",
    "3. 11 月下旬：搭建树莓派 4B 运行环境、交叉编译并部署，接入摄像头采集，与下位机完成串口"
    "联调，实测板端延迟、帧率与连续运行稳定性。",
    "4. 12 月上旬：完成棋盘格相机标定与像素→世界坐标转换，补充不少于 50 张真实场景样本并验证。",
    "5. 12 月中下旬：开展不少于 2 小时连续运行压力测试，统计丢帧与误码、固化参数；整理结题材料"
    "（源码、文档、测试报告、演示视频）。",
    "主要措施：坚持每周例会与 GitHub 分支管理，按里程碑检查；硬件未到位期间先用 PC+USB 摄像头"
    "跑通全链路；对实时性风险预留“ROI 检测+跳帧检测+跟踪补帧”降级方案，并建立问题清单逐项销号；对照项目书研究内容，YCrCb 色彩空间评估、自适应（动态）阈值、多目标跟踪与上位机调试界面安排在 11—12 月穿插完成。",
]

FUNDING = [
    "1. 视觉算法开发与测试用图像采集设备（USB 摄像头）：＿＿＿＿元；",
    "2. 嵌入式平台（树莓派 4B 及配件、存储卡）：＿＿＿＿元；",
    "3. 串口通信与调试耗材（USB-TTL、杜邦线、电源）：＿＿＿＿元；",
    "4. 打印、资料及其它：＿＿＿＿元。",
    "合计：＿＿＿＿元（已获资助与已使用经费以学校拨付及实际报销凭证为准）。",
]


# ----------------------------------------------------------------------
def main() -> int:
    doc = Document(TEMPLATE)

    # 1) 表头
    t0 = doc.tables[0]
    for i, (value, _label) in enumerate(HEADER):
        if value:
            set_cell_value(get_cell(t0, i, 1), value)

    # 2) 一、项目研究进展情况说明（正文 + 对照项目书研究内容的进度表）
    cell1 = get_cell(doc.tables[1], 1, 0)
    fill_cell(cell1, SEC1)
    add_progress_table(cell1, MAPPING)

    # 3) 二、已取得的阶段性成果
    fill_cell(get_cell(doc.tables[1], 3, 0), SEC2)

    # 4) 三、目前存在问题
    fill_cell(get_cell(doc.tables[2], 1, 0), SEC3)

    # 5) 四、经费使用明细（金额需本人确认，金额处留横线）
    fill_cell(get_cell(doc.tables[2], 5, 1), FUNDING)

    # 6) 五、下阶段研究计划及主要措施（保留签名行与日期行）
    fill_cell(get_cell(doc.tables[3], 1, 0), SEC5, keep_signature=True)

    # 7) 六、七 为教师与学院填写，保持空白；仅压缩少量留白段落，
    #    避免 "审核人（签名）" 的日期行被挤到下一页成为孤立页。
    trim_empty_paragraphs(get_cell(doc.tables[3], 3, 0), 2)
    trim_empty_paragraphs(get_cell(doc.tables[3], 5, 0), 1)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    doc.save(OUTPUT)
    print("已生成:", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
