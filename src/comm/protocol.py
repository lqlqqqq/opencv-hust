# -*- coding: utf-8 -*-
"""串口通信协议（与 Material/串口通信协议.docx 对应）。

帧结构（小端序）::

    +--------+--------+-------------------------+--------+--------+
    | 帧头   | 长度   | 有效数据                | CRC16  | 帧尾   |
    | AA BB  | 1 byte | 7 或 10 bytes           | 2 byte | CC DD  |
    +--------+--------+-------------------------+--------+--------+

有效数据（v1，长度 7）::

    目标ID(1) | X(2) | Y(2) | 置信度(1, 0~100) | 目标类别(1)

有效数据（v2，长度 10，在 v1 之后追加）::

    目标距离(2, 毫米) | 状态标志(1)

状态标志位: bit0 = 目标锁定，bit1 = 允许发射，bit2 = 预留。

CRC 采用 CRC-16/MODBUS（初值 0xFFFF，多项式 0xA001 反射），
校验范围为「长度字节 + 有效数据」，即帧头之后、CRC 之前的部分。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

HEADER = b"\xAA\xBB"
TAIL = b"\xCC\xDD"

PAYLOAD_V1_LEN = 7
PAYLOAD_V2_LEN = 10
MAX_PAYLOAD_LEN = 32

# 状态标志
FLAG_LOCKED = 0x01
FLAG_FIRE_ENABLED = 0x02


# ----------------------------------------------------------------------
# CRC16 / MODBUS
# ----------------------------------------------------------------------
def _make_crc_table() -> Tuple[int, ...]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
        table.append(crc)
    return tuple(table)


_CRC_TABLE = _make_crc_table()


def crc16_modbus(data: bytes) -> int:
    """计算 CRC-16/MODBUS。"""
    crc = 0xFFFF
    for byte in data:
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc & 0xFFFF


# ----------------------------------------------------------------------
# 编码
# ----------------------------------------------------------------------
def _clamp(value: int, low: int, high: int, name: str) -> int:
    value = int(round(value))
    if not (low <= value <= high):
        raise ValueError(f"{name}={value} 超出范围 [{low}, {high}]")
    return value


def encode_target(
    target_id: int,
    x: float,
    y: float,
    confidence: float,
    class_code: int,
    distance_mm: Optional[float] = None,
    flags: int = 0,
) -> bytes:
    """打包一帧目标数据。

    Args:
        target_id: 目标编号 0~255
        x, y: 像素坐标，0~65535
        confidence: 置信度 0~1，编码为 0~100
        class_code: 目标类别编号（见 result.encode_class）
        distance_mm: 目标距离（毫米）。给出时自动使用 v2 帧（长度 10）
        flags: 状态标志（FLAG_LOCKED / FLAG_FIRE_ENABLED）

    Returns:
        完整的一帧字节串（14 或 17 字节）
    """
    target_id = _clamp(target_id, 0, 255, "target_id")
    x = _clamp(x, 0, 65535, "x")
    y = _clamp(y, 0, 65535, "y")
    conf = _clamp(float(confidence) * 100.0, 0, 100, "confidence")
    class_code = _clamp(class_code, 0, 99, "class_code")
    flags = _clamp(flags, 0, 255, "flags")

    payload = struct.pack("<BHHBB", target_id, x, y, conf, class_code)
    if distance_mm is not None:
        dist = _clamp(distance_mm, 0, 65535, "distance_mm")
        payload += struct.pack("<HB", dist, flags)

    body = bytes([len(payload)]) + payload
    crc = crc16_modbus(body)
    return HEADER + body + struct.pack("<H", crc) + TAIL


# ----------------------------------------------------------------------
# 解码
# ----------------------------------------------------------------------
def decode_payload(payload: bytes) -> Dict[str, object]:
    """解析有效数据为重命名字典（不校验 CRC，由 FrameParser 负责）。"""
    if len(payload) not in (PAYLOAD_V1_LEN, PAYLOAD_V2_LEN):
        raise ValueError(f"有效数据长度非法: {len(payload)}")

    target_id, x, y, conf, class_code = struct.unpack("<BHHBB", payload[:PAYLOAD_V1_LEN])
    result: Dict[str, object] = {
        "id": target_id,
        "x": x,
        "y": y,
        "confidence": round(conf / 100.0, 2),
        "class_code": class_code,
        "locked": False,
        "fire_enabled": False,
        "distance_mm": None,
    }
    if len(payload) == PAYLOAD_V2_LEN:
        dist, flags = struct.unpack("<HB", payload[PAYLOAD_V1_LEN:])
        result["distance_mm"] = dist
        result["locked"] = bool(flags & FLAG_LOCKED)
        result["fire_enabled"] = bool(flags & FLAG_FIRE_ENABLED)
        result["flags"] = flags
    return result


# ----------------------------------------------------------------------
# 流式解析
# ----------------------------------------------------------------------
@dataclass
class ParserStats:
    frames_ok: int = 0
    crc_errors: int = 0
    tail_errors: int = 0
    length_errors: int = 0
    bytes_dropped: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "frames_ok": self.frames_ok,
            "crc_errors": self.crc_errors,
            "tail_errors": self.tail_errors,
            "length_errors": self.length_errors,
            "bytes_dropped": self.bytes_dropped,
        }


class FrameParser:
    """串口字节流 → 目标字典的流式解析器。

    针对实际串口的分片特性设计：一帧可能分多次到达，一次到达也可能包含多帧。
    任何一处校验失败都只丢弃一个字节并重新同步到下一个帧头，避免整段数据作废。
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.stats = ParserStats()

    def reset(self) -> None:
        self._buffer.clear()
        self.stats = ParserStats()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, chunk: bytes) -> List[Dict[str, object]]:
        """喂入一段数据，返回本次解析出的所有目标字典。"""
        if chunk:
            self._buffer.extend(chunk)
        frames: List[Dict[str, object]] = []
        buf = self._buffer

        while True:
            start = buf.find(HEADER)
            if start < 0:
                # 没有帧头，只保留最后 1 个可能构成半个帧头的字节
                if len(buf) > 1:
                    keep = 1 if buf[-1] == HEADER[0] else 0
                    self.stats.bytes_dropped += len(buf) - keep
                    del buf[: len(buf) - keep]
                return frames
            if start > 0:
                self.stats.bytes_dropped += start
                del buf[:start]

            if len(buf) < 3:
                return frames  # 还读不到长度字节

            length = buf[2]
            if length < PAYLOAD_V1_LEN or length > MAX_PAYLOAD_LEN:
                self.stats.length_errors += 1
                self.stats.bytes_dropped += 1
                del buf[:1]
                continue

            total = 2 + 1 + length + 2 + 2
            if len(buf) < total:
                return frames  # 帧未收全，等后续数据

            body = bytes(buf[2 : 3 + length])
            crc_rx = struct.unpack("<H", bytes(buf[3 + length : 5 + length]))[0]
            tail = bytes(buf[5 + length : 7 + length])

            if tail != TAIL:
                self.stats.tail_errors += 1
                self.stats.bytes_dropped += 1
                del buf[:1]
                continue
            if crc16_modbus(body) != crc_rx:
                self.stats.crc_errors += 1
                self.stats.bytes_dropped += 1
                del buf[:1]
                continue

            try:
                frames.append(decode_payload(body[1:]))
            except ValueError:
                self.stats.length_errors += 1
                del buf[:1]
                continue

            self.stats.frames_ok += 1
            del buf[:total]


def iter_frames(data: bytes) -> Iterable[Dict[str, object]]:
    """便捷函数：一次性解析一段完整数据。"""
    return FrameParser().feed(data)
