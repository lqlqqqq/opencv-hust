# -*- coding: utf-8 -*-
"""串口协议单元测试：编码/解码、CRC、断帧重同步、异常输入。

运行::

    python -m unittest discover -s tests -v
"""

import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.comm.link import LoopbackLink, NullLink            # noqa: E402
from src.comm.protocol import (                              # noqa: E402
    FLAG_FIRE_ENABLED,
    FLAG_LOCKED,
    HEADER,
    TAIL,
    FrameParser,
    crc16_modbus,
    decode_payload,
    encode_target,
)
from src.core.result import Target, decode_class, encode_class  # noqa: E402


class TestCRC16(unittest.TestCase):
    def test_known_value(self):
        # CRC-16/MODBUS 的标准校验值："123456789" -> 0x4B37
        self.assertEqual(crc16_modbus(b"123456789"), 0x4B37)

    def test_empty(self):
        self.assertEqual(crc16_modbus(b""), 0xFFFF)


class TestEncodeDecode(unittest.TestCase):
    def test_roundtrip_v1(self):
        frame = encode_target(3, 320, 240, 0.87, encode_class("red", "circle"))
        self.assertEqual(frame[:2], HEADER)
        self.assertEqual(frame[-2:], TAIL)
        self.assertEqual(len(frame), 14)  # 2 + 1 + 7 + 2 + 2

        parsed = FrameParser().feed(frame)
        self.assertEqual(len(parsed), 1)
        item = parsed[0]
        self.assertEqual(item["id"], 3)
        self.assertEqual((item["x"], item["y"]), (320, 240))
        self.assertAlmostEqual(item["confidence"], 0.87, places=2)
        self.assertEqual(decode_class(item["class_code"]), ("red", "circle"))
        self.assertIsNone(item["distance_mm"])

    def test_roundtrip_v2(self):
        frame = encode_target(
            1, 100, 200, 0.5, encode_class("green", "rectangle"),
            distance_mm=1234, flags=FLAG_LOCKED | FLAG_FIRE_ENABLED,
        )
        self.assertEqual(len(frame), 17)  # 2 + 1 + 10 + 2 + 2
        item = FrameParser().feed(frame)[0]
        self.assertEqual(item["distance_mm"], 1234)
        self.assertTrue(item["locked"])
        self.assertTrue(item["fire_enabled"])

    def test_confidence_scaling(self):
        item = FrameParser().feed(encode_target(0, 0, 0, 1.0, 17))[0]
        self.assertEqual(item["confidence"], 1.0)

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            encode_target(1, -5, 0, 0.5, 17)
        with self.assertRaises(ValueError):
            encode_target(1, 0, 0, 1.5, 17)
        with self.assertRaises(ValueError):
            encode_target(300, 0, 0, 0.5, 17)


class TestStreamParser(unittest.TestCase):
    def test_split_chunks(self):
        """一帧被拆成多段到达时仍能正确解析。"""
        frame = encode_target(7, 11, 22, 0.6, encode_class("red", "circle"))
        parser = FrameParser()
        out = []
        for i in range(0, len(frame), 3):
            out.extend(parser.feed(frame[i:i + 3]))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], 7)

    def test_multiple_frames_one_chunk(self):
        data = b"".join(
            encode_target(i, i * 10, i * 5, 0.5, 17) for i in range(1, 6)
        )
        frames = FrameParser().feed(data)
        self.assertEqual([f["id"] for f in frames], [1, 2, 3, 4, 5])

    def test_crc_error_dropped_and_resync(self):
        good = encode_target(2, 30, 40, 0.9, 17)
        broken = bytearray(encode_target(9, 1, 2, 0.9, 17))
        broken[-3] ^= 0xFF  # 破坏 CRC
        parser = FrameParser()
        frames = parser.feed(bytes(broken) + good)
        self.assertEqual([f["id"] for f in frames], [2])
        self.assertGreaterEqual(parser.stats.crc_errors, 1)

    def test_noise_prefix_dropped(self):
        good = encode_target(4, 5, 6, 0.3, 17)
        parser = FrameParser()
        frames = parser.feed(b"\x00\x11\x22garbage" + good)
        self.assertEqual([f["id"] for f in frames], [4])
        self.assertGreater(parser.stats.bytes_dropped, 0)

    def test_incomplete_tail_waits(self):
        frame = encode_target(5, 7, 8, 0.2, 17)
        parser = FrameParser()
        frames = parser.feed(frame[:-2])
        self.assertEqual(frames, [])
        frames = parser.feed(frame[-2:])
        self.assertEqual(len(frames), 1)

    def test_invalid_length_resync(self):
        # 伪造一个长度非法的帧头，后面跟一帧正常数据
        bad = b"\xAA\xBB\x01\x00\x00"
        good = encode_target(8, 9, 10, 0.4, 17)
        parser = FrameParser()
        frames = parser.feed(bad + good)
        self.assertEqual([f["id"] for f in frames], [8])
        self.assertGreaterEqual(parser.stats.length_errors, 1)


class TestLinks(unittest.TestCase):
    def test_loopback(self):
        link = LoopbackLink()
        target = Target(id=1, color="red", shape="circle", center=(10.0, 20.0),
                        width=20, height=20, radius=10, area=314, confidence=0.9)
        link.send_target(target)
        parsed = link.receive_parsed()
        self.assertEqual(len(parsed), 1)
        self.assertEqual((parsed[0]["x"], parsed[0]["y"]), (10, 20))
        self.assertEqual(decode_class(parsed[0]["class_code"]), ("red", "circle"))

    def test_null_link(self):
        link = NullLink()
        n = link.send_target(Target(id=2, center=(1.0, 2.0), confidence=0.5))
        self.assertGreater(n, 0)
        self.assertEqual(link.sent_bytes, n)


class TestClassCodec(unittest.TestCase):
    def test_known_codes(self):
        self.assertEqual(encode_class("red", "circle"), 11)
        self.assertEqual(encode_class("red", "rectangle"), 12)
        self.assertEqual(encode_class("green", "circle"), 21)
        self.assertEqual(decode_class(11), ("red", "circle"))

    def test_unknown(self):
        self.assertEqual(encode_class("pink", "hexagon"), 93)
        self.assertEqual(decode_class(0), ("unknown", "unknown"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
