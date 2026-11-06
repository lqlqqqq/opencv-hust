# -*- coding: utf-8 -*-
"""串口链路封装。

* :class:`SerialLink`   真实串口（依赖 pyserial，按需导入）
* :class:`LoopbackLink` 回环链路，用于无硬件时的联调与单元测试
* :class:`NullLink`     空实现，未接串口时保证调用方逻辑不变
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.comm.protocol import FrameParser, encode_target


class BaseLink:
    """链路基类：所有实现都提供 send / send_target / close。"""

    name = "base"

    def send(self, data: bytes) -> int:
        raise NotImplementedError

    def send_target(self, target, distance_mm: Optional[float] = None, flags: int = 0) -> int:
        """把一个 Target 对象编码并发送，返回发送字节数。"""
        frame = encode_target(
            target_id=target.id,
            x=target.cx,
            y=target.cy,
            confidence=target.confidence,
            class_code=target.class_code,
            distance_mm=distance_mm,
            flags=flags,
        )
        return self.send(frame)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class NullLink(BaseLink):
    """不做任何 IO 的空链路。"""

    name = "null"

    def __init__(self) -> None:
        self.sent_bytes = 0

    def send(self, data: bytes) -> int:
        self.sent_bytes += len(data)
        return len(data)


class LoopbackLink(BaseLink):
    """回环链路：发送的数据进入内部缓冲，可直接被 FrameParser 解析。"""

    name = "loopback"

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.parser = FrameParser()

    def send(self, data: bytes) -> int:
        self.buffer.extend(data)
        return len(data)

    def receive_parsed(self) -> List[Dict[str, object]]:
        """取出并解析缓冲中的数据（模拟下位机收帧）。"""
        chunk = bytes(self.buffer)
        self.buffer.clear()
        return self.parser.feed(chunk)

    def close(self) -> None:
        self.buffer.clear()


class SerialLink(BaseLink):
    """真实串口链路（8N1，默认 115200）。"""

    name = "serial"

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.05):
        try:
            import serial  # type: ignore
        except ImportError as exc:  # pragma: no cover - 环境缺少 pyserial 时给出明确提示
            raise RuntimeError(
                "未安装 pyserial，无法使用真实串口。请执行: pip install pyserial"
            ) from exc

        self.port = port
        self.baudrate = int(baudrate)
        self._serial = serial.Serial(
            port=port, baudrate=baudrate, bytesize=8, parity="N", stopbits=1, timeout=timeout
        )
        self.sent_frames = 0
        self.sent_bytes = 0

    def send(self, data: bytes) -> int:
        n = self._serial.write(data)
        self.sent_frames += 1
        self.sent_bytes += n
        return n

    def read_available(self) -> bytes:
        n = self._serial.in_waiting
        return self._serial.read(n) if n else b""

    def close(self) -> None:
        try:
            self._serial.close()
        except Exception:
            pass

    @staticmethod
    def available_ports() -> List[str]:
        try:
            from serial.tools import list_ports  # type: ignore
        except ImportError:
            return []
        return [p.device for p in list_ports.comports()]


def make_link(port: Optional[str] = None, baudrate: int = 115200) -> BaseLink:
    """工厂函数：给出端口就用真实串口，否则返回空链路。"""
    if port:
        return SerialLink(port, baudrate)
    return NullLink()
