# -*- coding: utf-8 -*-
"""串口通信：帧协议与链路封装。"""

from src.comm.link import (  # noqa: F401
    BaseLink,
    LoopbackLink,
    NullLink,
    SerialLink,
    make_link,
)
from src.comm.protocol import (  # noqa: F401
    FLAG_FIRE_ENABLED,
    FLAG_LOCKED,
    FrameParser,
    crc16_modbus,
    decode_payload,
    encode_target,
    iter_frames,
)

__all__ = [
    "BaseLink",
    "LoopbackLink",
    "NullLink",
    "SerialLink",
    "make_link",
    "FrameParser",
    "crc16_modbus",
    "decode_payload",
    "encode_target",
    "iter_frames",
    "FLAG_LOCKED",
    "FLAG_FIRE_ENABLED",
]
