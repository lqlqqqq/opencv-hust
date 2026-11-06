# -*- coding: utf-8 -*-
"""串口协议的抗干扰实验：统计不同破坏方式下的帧漏检率与重同步能力。

关注的不是"能否发现错误"，而是"错误帧是否会被当成正确帧接受"（漏检），
以及混入噪声后还能不能重新对上后续的帧（重同步）。

用法::

    python tools/exp_protocol.py
"""

from __future__ import annotations

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.comm.protocol import FrameParser, encode_target   # noqa: E402
from src.core.result import encode_class                    # noqa: E402

N = 3000
CLS = encode_class("red", "circle")


def corrupt_bits(frame: bytes, n_bits: int, rng: random.Random) -> bytes:
    """随机翻转 n_bits 个互不相同的比特位。

    必须去重：若同一比特被翻转两次会相互抵消，等于没破坏，
    会把"未受破坏的帧被正常接收"误统计成漏检。
    """
    data = bytearray(frame)
    positions = rng.sample(range(len(data) * 8), n_bits)
    for pos in positions:
        data[pos // 8] ^= 1 << (pos % 8)
    return bytes(data)


def trial(n_bits: int, seed: int = 0) -> tuple[int, int]:
    """返回 (漏检帧数, 正确拒收帧数)。"""
    rng = random.Random(seed)
    missed = rejected = 0
    for i in range(N):
        frame = encode_target(i % 255, 100 + i % 400, 200, 0.9, CLS)
        bad = corrupt_bits(frame, n_bits, rng)
        frames = FrameParser().feed(bad)
        # 只有"解析出一帧且内容与原始一致"才算漏检通过；
        # 解析出其它内容说明校验没抓住错误。
        if any(f["x"] == (100 + i % 400) and f["y"] == 200 for f in frames):
            missed += 1
        else:
            rejected += 1
    return missed, rejected


def resync_trial(noise_bytes: int, seed: int = 1) -> tuple[int, int]:
    """在正常帧前插入随机噪声，检验能否重新同步。"""
    rng = random.Random(seed)
    ok = lost = 0
    for i in range(N):
        noise = bytes(rng.randrange(256) for _ in range(noise_bytes))
        good = encode_target(7, 320, 240, 0.8, CLS)
        frames = FrameParser().feed(noise + good)
        if any(f["id"] == 7 and f["x"] == 320 for f in frames):
            ok += 1
        else:
            lost += 1
    return ok, lost


def main() -> int:
    print(f"每组 {N} 帧，目标类别编码 {CLS}")
    print(f"{'破坏方式':<18}{'漏检帧':>8}{'正确拒收':>10}{'漏检率':>10}")
    for n_bits in (1, 2, 3, 5, 8):
        missed, rejected = trial(n_bits)
        print(f"{n_bits} bit 随机翻转{'':<6}{missed:>8}{rejected:>10}{missed / N:>10.4%}")

    print(f"\n{'帧前噪声(字节)':<18}{'重同步成功':>10}{'丢失':>8}{'成功率':>10}")
    for n in (0, 4, 16, 64):
        ok, lost = resync_trial(n)
        print(f"{n:<18}{ok:>10}{lost:>8}{ok / N:>10.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
