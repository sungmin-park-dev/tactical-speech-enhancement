"""Fixed-size, single-peer wire format: network-order header, little-endian PCM16."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 256
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
MAGIC = b"TSE1"
VERSION = 1
HEADER = struct.Struct("!4sB3x16sQ")
PACKET_BYTES = HEADER.size + FRAME_SAMPLES * 2
PCM16_LIMIT = math.floor(32768 * 10 ** (-6 / 20))


class PacketError(ValueError):
    """An invalid datagram; it must never reach audio playback."""


@dataclass(frozen=True)
class Packet:
    stream_id: bytes
    sequence: int
    audio: NDArray[np.float32]


def pcm16_limit(ceiling_dbfs: float = -6.0) -> int:
    """The wire ceiling can be lowered, but never exceed the -6 dBFS contract."""
    if not math.isfinite(ceiling_dbfs) or ceiling_dbfs > 0:
        raise ValueError("ceiling_dbfs must be finite and at most zero")
    return math.floor(32768 * 10 ** (min(-6.0, ceiling_dbfs) / 20))


def encode_packet(
    stream_id: bytes,
    sequence: int,
    audio: NDArray[np.floating],
    *,
    ceiling_dbfs: float = -6.0,
) -> bytes:
    """Encode a source sequence, silencing the entire frame if any value is invalid."""
    if not isinstance(stream_id, bytes) or len(stream_id) != 16:
        raise ValueError("stream_id must contain exactly 16 bytes")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 <= sequence < 2**64:
        raise ValueError("sequence must be an unsigned 64-bit integer")
    values = np.asarray(audio)
    if values.shape != (FRAME_SAMPLES,) or values.dtype.kind not in "fiu":
        raise ValueError("audio must be a numeric mono frame of 256 samples")
    limit = pcm16_limit(ceiling_dbfs)
    if not np.isfinite(values).all():
        pcm = np.zeros(FRAME_SAMPLES, dtype="<i2")
    else:
        # Clip before multiplication so finite, extremely large inputs cannot overflow.
        bounded = np.clip(values, -limit / 32768, limit / 32768)
        pcm = np.clip(np.rint(bounded * 32768), -limit, limit).astype("<i2")
    return HEADER.pack(MAGIC, VERSION, stream_id, sequence) + pcm.tobytes()


def decode_packet(data: bytes) -> Packet:
    """Strict length, version, reserved-field and digital-level validation."""
    if len(data) != PACKET_BYTES:
        raise PacketError("incorrect datagram size")
    magic, version, stream_id, sequence = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or data[5:8] != b"\x00\x00\x00":
        raise PacketError("unsupported packet header")
    pcm = np.frombuffer(data, dtype="<i2", offset=HEADER.size)
    # Defend playback independently of whether the sender uses our encoder.
    audio = np.clip(pcm.astype(np.float32), -PCM16_LIMIT, PCM16_LIMIT) / 32768
    return Packet(stream_id, sequence, audio)
