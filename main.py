# -*- coding: utf-8 -*-
"""视觉识别与定位系统 —— 统一命令行入口。

用法示例::

    # 单张图片
    python main.py --image data/test002.png

    # 批量目录 + 保存结果图 + 导出 JSON
    python main.py --dir data --save-dir result --json result/report.json

    # 摄像头实时检测（USB 摄像头，端口 0），并向主控下发串口帧
    python main.py --camera 0 --serial COM3 --baud 115200

    # 使用低照度预设
    python main.py --image data/test001.png --preset lowlight --show

不带 ``--show`` 时完全无窗口运行，便于在服务器/评测脚本中调用。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.config import PipelineConfig          # noqa: E402
from src.core.pipeline import VisionPipeline        # noqa: E402
from src.core.result import FrameResult, Target     # noqa: E402
from src.comm.link import make_link                 # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="基于 OpenCV 的轻量化视觉识别与定位系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="单张图片路径")
    src.add_argument("--dir", help="图片目录（批量处理）")
    src.add_argument("--camera", nargs="?", const=0, type=int, default=None,
                     help="摄像头编号，默认 0")

    p.add_argument("--config", help="JSON 配置文件路径")
    p.add_argument("--preset", default="default",
                   choices=["default", "lowlight", "cluttered", "realtime"],
                   help="内置参数预设")
    p.add_argument("--colors", default=None,
                   help="要检测的颜色，逗号分隔，例如 red,green")
    p.add_argument("--target-shape", default=None,
                   help="目标形状：circle / rectangle / any")
    p.add_argument("--save-dir", default=os.path.join(ROOT, "result"),
                   help="结果保存目录")
    p.add_argument("--json", default=None, help="把检测结果导出为 JSON 文件")
    p.add_argument("--show", action="store_true", help="显示可视化窗口（按 q 退出）")
    p.add_argument("--no-save", action="store_true", help="不保存结果图片")
    p.add_argument("--serial", default=None, help="串口设备，例如 COM3 或 /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200, help="串口波特率，默认 115200")
    p.add_argument("--frames", type=int, default=0,
                   help="摄像头模式下最多处理帧数，0 表示不限")
    p.add_argument("--quiet", action="store_true", help="只输出汇总信息")
    return p


# ----------------------------------------------------------------------
def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig.load(args.config) if args.config else PipelineConfig.preset(args.preset)
    if args.colors:
        cfg.color.colors = [c.strip().lower() for c in args.colors.split(",") if c.strip()]
    if args.target_shape:
        cfg.target_shape = args.target_shape.strip().lower()
    return cfg


def print_targets(targets: List[Target], prefix: str = "") -> None:
    if not targets:
        print(f"{prefix}未检测到目标")
        return
    header = f"{'ID':>3} {'颜色':<7}{'形状':<10}{'中心坐标':<18}{'尺寸':<16}{'置信度':>7}"
    print(prefix + header)
    print(prefix + "-" * len(header))
    for t in targets:
        size = f"{t.width:.1f}x{t.height:.1f}" if t.shape != "circle" else f"d={2 * t.radius:.1f}"
        print(
            prefix
            + f"{t.id:>3} {t.color:<7}{t.shape:<10}"
            + f"({t.cx:.1f}, {t.cy:.1f})      {size:<16}{t.confidence:>7.2f}"
        )


def save_outputs(
    image: np.ndarray,
    result: FrameResult,
    save_dir: str,
    stem: str,
    link=None,
) -> Dict[str, str]:
    """保存标注图 / 掩码图 / 预处理图，并把目标列表下发到链路（若提供）。"""
    os.makedirs(save_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    annotated = VisionPipeline.annotate(image, result.targets)
    paths["annotated"] = os.path.join(save_dir, f"{stem}_result.jpg")
    cv2.imwrite(paths["annotated"], annotated)

    color_mask = result.masks.get("color")
    if color_mask is not None and np.any(color_mask):
        paths["color_mask"] = os.path.join(save_dir, f"{stem}_color_mask.jpg")
        cv2.imwrite(paths["color_mask"], color_mask)

    debug_image = result.debug.get("working_image")
    if debug_image is not None:
        paths["preprocessed"] = os.path.join(save_dir, f"{stem}_preprocessed.jpg")
        cv2.imwrite(paths["preprocessed"], debug_image)

    if link is not None:
        for t in result.targets:
            link.send_target(t)
    return paths


# ----------------------------------------------------------------------
def run_single_image(args: argparse.Namespace, cfg: PipelineConfig) -> int:
    image = cv2.imread(args.image)
    if image is None:
        print(f"[错误] 无法读取图片: {args.image}", file=sys.stderr)
        return 2

    pipeline = VisionPipeline(cfg)
    result = pipeline.run(image)
    targets = pipeline.target_targets(result)

    print(f"\n图片: {args.image}  尺寸: {result.image_size[0]}x{result.image_size[1]}")
    print_targets(targets)
    print(
        f"  耗时: 预处理 {result.timings_ms['preprocess']:.2f} ms | "
        f"颜色 {result.timings_ms['color']:.2f} ms | "
        f"形状 {result.timings_ms['shape']:.2f} ms | "
        f"合计 {result.total_ms:.2f} ms ({result.fps:.1f} FPS)"
    )

    link = make_link(args.serial, args.baud) if args.serial else None
    try:
        if not args.no_save:
            stem = os.path.splitext(os.path.basename(args.image))[0]
            paths = save_outputs(image, result, args.save_dir, stem, link)
            for key, path in paths.items():
                print(f"  [{key}] {path}")
    finally:
        if link is not None:
            link.close()

    if args.json:
        payload = {
            "source": args.image,
            "config": cfg.to_dict(),
            "frame": result.to_dict(),
            "targets": [t.to_dict() for t in targets],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  [json] {args.json}")

    if args.show:
        cv2.imshow("result", VisionPipeline.annotate(image, result.targets))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_batch(args: argparse.Namespace, cfg: PipelineConfig) -> int:
    files = sorted(
        f for f in glob.glob(os.path.join(args.dir, "*"))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        print(f"[错误] 目录中没有图片: {args.dir}", file=sys.stderr)
        return 2

    pipeline = VisionPipeline(cfg)
    link = make_link(args.serial, args.baud) if args.serial else None
    summary = {"frames": [], "total_targets": 0, "total_ms": 0.0}
    try:
        for path in files:
            image = cv2.imread(path)
            if image is None:
                continue
            result = pipeline.run(image)
            targets = pipeline.target_targets(result)
            summary["frames"].append(
                {
                    "file": os.path.basename(path),
                    **result.to_dict(with_targets=False),
                    "target_count": len(targets),
                }
            )
            summary["total_targets"] += len(targets)
            summary["total_ms"] += result.total_ms
            if not args.quiet:
                print(f"\n[{os.path.basename(path)}] {len(targets)} 个目标, "
                      f"{result.total_ms:.2f} ms")
                print_targets(targets, prefix="  ")
            if not args.no_save:
                stem = os.path.splitext(os.path.basename(path))[0]
                save_outputs(image, result, args.save_dir, stem, link)
    finally:
        if link is not None:
            link.close()

    n = len(summary["frames"])
    avg = summary["total_ms"] / n if n else 0.0
    print(f"\n共处理 {n} 张图片，检出 {summary['total_targets']} 个目标，"
          f"平均 {avg:.2f} ms/帧 ({1000.0 / avg if avg else 0:.1f} FPS)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[json] {args.json}")
    return 0


def run_camera(args: argparse.Namespace, cfg: PipelineConfig) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[错误] 摄像头 {args.camera} 打开失败", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    pipeline = VisionPipeline(cfg)
    link = make_link(args.serial, args.baud) if args.serial else None
    frame_count, fps_t0, fps_n = 0, time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[错误] 读取摄像头帧失败", file=sys.stderr)
                break
            result = pipeline.run(frame)
            targets = pipeline.target_targets(result)
            if link is not None:
                for t in targets:
                    link.send_target(t)
            fps_n += 1
            elapsed = time.time() - fps_t0
            fps = fps_n / elapsed if elapsed > 0 else 0.0
            if not args.quiet and frame_count % 15 == 0:
                print(f"帧 {frame_count:5d} | 目标 {len(targets)} | "
                      f"{result.total_ms:5.2f} ms | 循环 {fps:5.1f} FPS")
            frame_count += 1

            if not args.no_save and frame_count % 60 == 0:
                stem = f"camera_{frame_count:05d}"
                save_outputs(frame, result, args.save_dir, stem, None)

            if args.show:
                canvas = VisionPipeline.annotate(frame, result.targets)
                cv2.putText(canvas, f"{result.total_ms:.1f} ms  {fps:.1f} FPS",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("vision", canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if args.frames and frame_count >= args.frames:
                break
    finally:
        cap.release()
        if link is not None:
            link.close()
        cv2.destroyAllWindows()
    print(f"共处理 {frame_count} 帧")
    return 0


# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = build_config(args)

    if args.image:
        return run_single_image(args, cfg)
    if args.dir:
        return run_batch(args, cfg)
    return run_camera(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
