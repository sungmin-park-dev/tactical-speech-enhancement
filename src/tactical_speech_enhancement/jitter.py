"""Bounded receive jitter buffer, driven by the receiver's audio clock.

Only this buffer has a four-frame target and eight-frame maximum. Missing
source sequences become silence; arrival rate never determines audio speed.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

from .protocol import FRAME_SAMPLES, FRAME_SECONDS, Packet

MAX_RETIRED_STREAMS = 64
SESSION_TIMEOUT = 1.0


class JitterBuffer:
    """Not internally synchronized: the owner protects push/playout with one lock."""

    def __init__(self, target_frames: int = 4, max_frames: int = 8) -> None:
        if not 1 <= target_frames <= max_frames <= 64:
            raise ValueError("require 1 <= target_frames <= max_frames <= 64")
        self.target_frames = target_frames
        self.max_frames = max_frames
        self._frames: dict[int, NDArray[np.float32]] = {}
        self._stream_id: bytes | None = None
        self._retired: OrderedDict[bytes, None] = OrderedDict()
        self._history_truncated = False
        self._next_sequence: int | None = None
        self._deadline: float | None = None
        self._last_arrival = -math.inf
        self._played = False
        self._misses = 0
        self._receive_skips_since_playout = 0
        self._counters = dict(
            accepted=0,
            duplicate=0,
            late=0,
            retired=0,
            session_wait=0,
            restarts=0,
            lost=0,
            played=0,
            startup_silence=0,
            overflow_dropped=0,
            sequence_skipped=0,
            rebuffered=0,
            high_watermark=0,
        )

    @property
    def size(self) -> int:
        return len(self._frames)

    def _switch_stream(self, stream_id: bytes) -> None:
        if self._stream_id is not None:
            self._retired[self._stream_id] = None
            if len(self._retired) > MAX_RETIRED_STREAMS:
                self._retired.popitem(last=False)
                self._history_truncated = True
            self._counters["restarts"] += 1
        self._stream_id = stream_id
        self._frames.clear()
        self._next_sequence = None
        self._deadline = None
        self._played = False
        self._misses = 0
        self._receive_skips_since_playout = 0

    def push(self, packet: Packet, arrival: float) -> bool:
        if not math.isfinite(arrival):
            raise ValueError("arrival must be finite")
        if packet.stream_id != self._stream_id:
            if packet.stream_id in self._retired:
                self._counters["retired"] += 1
                return False
            # After the bounded retirement history fills, an evicted old session
            # cannot displace an active stream. A new session needs one second idle.
            if self._history_truncated and arrival - self._last_arrival < SESSION_TIMEOUT:
                self._counters["session_wait"] += 1
                return False
            self._switch_stream(packet.stream_id)

        sequence = packet.sequence
        if self._next_sequence is not None and sequence < self._next_sequence:
            if self._played:
                self._counters["late"] += 1
                return False
            # Reordering before the very first playout may reveal the earlier frame.
            if self._next_sequence - sequence >= self.max_frames or (
                self._frames and max(self._frames) - sequence >= self.max_frames
            ):
                self._counters["late"] += 1
                return False
            self._next_sequence = sequence
        if sequence in self._frames:
            self._counters["duplicate"] += 1
            return False

        self._last_arrival = arrival
        if self._next_sequence is None:
            self._next_sequence = sequence
        if self._deadline is None:
            self._deadline = arrival + self.target_frames * FRAME_SECONDS

        # Resynchronize after a source pause, or a sender clock that accumulates
        # more than the bounded jitter window. Never queue a long future backlog.
        floor = sequence - self.max_frames + 1
        if floor > self._next_sequence:
            receive_skips = floor - self._next_sequence
            self._counters["sequence_skipped"] += receive_skips
            self._receive_skips_since_playout += receive_skips
            self._next_sequence = floor
            for old in list(self._frames):
                if old < floor:
                    del self._frames[old]
                    self._counters["overflow_dropped"] += 1

        self._frames[sequence] = packet.audio.copy()
        self._counters["accepted"] += 1
        self._counters["high_watermark"] = max(self._counters["high_watermark"], len(self._frames))
        return True

    def playout(self, now: float) -> NDArray[np.float32]:
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        silent = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        if self._deadline is None or now + 1e-9 < self._deadline:
            self._counters["startup_silence"] += 1
            return silent
        assert self._next_sequence is not None
        # A stopped output callback must not replay old audio when it resumes.
        skipped = max(0, math.floor((now - self._deadline + 1e-9) / FRAME_SECONDS))
        if skipped:
            # Reception may already have discarded this elapsed source interval
            # while playback was paused. Advance the clock by all missed ticks,
            # but never advance the source twice for the same interval.
            source_skips = max(0, skipped - self._receive_skips_since_playout)
            self._next_sequence += source_skips
            self._deadline += skipped * FRAME_SECONDS
            self._counters["sequence_skipped"] += source_skips
            for sequence in list(self._frames):
                if sequence < self._next_sequence:
                    del self._frames[sequence]
                    self._counters["overflow_dropped"] += 1
        self._receive_skips_since_playout = 0
        result = self._frames.pop(self._next_sequence, None)
        self._next_sequence += 1
        self._deadline += FRAME_SECONDS
        self._played = True
        if result is None:
            self._misses += 1
            self._counters["lost"] += 1
            if self._misses >= self.max_frames:
                # Freeze the source watermark while waiting, so a slower sender
                # can catch up without ever replaying an already-consumed sequence.
                self._deadline = None
                self._frames.clear()
                self._misses = 0
                self._counters["rebuffered"] += 1
            return silent
        self._misses = 0
        self._counters["played"] += 1
        return result

    def summary(self) -> dict[str, int | float | None]:
        return {
            **self._counters,
            "frames": len(self._frames),
            "target_frames": self.target_frames,
            "max_frames": self.max_frames,
            "retired_streams": len(self._retired),
            "next_sequence": self._next_sequence,
            "next_deadline": self._deadline,
        }
