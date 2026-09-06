"""Local-network full-duplex UDP peer with independent, bounded send/receive work."""

from __future__ import annotations

import select
import socket
import threading
import time
import uuid
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .jitter import JitterBuffer
from .protocol import PACKET_BYTES, PacketError, decode_packet, encode_packet, pcm16_limit


class UdpPeer:
    """One IPv4 peer. submit/playout never perform network I/O.

    The operating system socket buffers are separate from the application jitter
    buffer and must be included in actual device latency measurements.
    """

    def __init__(
        self,
        bind: tuple[str, int],
        peer: tuple[str, int],
        *,
        clock: Callable[[], float] = time.monotonic,
        jitter_target_frames: int = 4,
        jitter_max_frames: int = 8,
        ceiling_dbfs: float = -6.0,
    ) -> None:
        pcm16_limit(ceiling_dbfs)
        self._clock = clock
        self._ceiling_dbfs = ceiling_dbfs
        self._jitter = JitterBuffer(jitter_target_frames, jitter_max_frames)
        self._peer = (socket.gethostbyname(peer[0]), peer[1])
        if not 1 <= peer[1] <= 65535:
            raise ValueError("peer port must be between 1 and 65535")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.bind(bind)
            self._socket.setblocking(False)
        except Exception:
            self._socket.close()
            raise
        bound = self._socket.getsockname()
        self.bound_address = (str(bound[0]), int(bound[1]))
        self._stream_id = uuid.uuid4().bytes
        self._jitter_lock = threading.Lock()
        self._tx_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._wake_tx = threading.Event()
        self._stop = threading.Event()
        self._tx_slot: bytes | None = None
        self._last_submitted = -1
        self._started = False
        self._closed = False
        self._threads: list[threading.Thread] = []
        self._counters = dict(
            tx_submitted=0,
            tx_replaced=0,
            tx_sent=0,
            tx_error=0,
            rx_received=0,
            rx_wrong_peer=0,
            rx_invalid=0,
            rx_error=0,
        )

    def _count(self, name: str) -> None:
        with self._stats_lock:
            self._counters[name] += 1

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("a closed peer cannot be restarted")
        if self._started:
            return
        self._started = True
        self._threads = [
            threading.Thread(target=self._receive, name="tse-udp-rx", daemon=True),
            threading.Thread(target=self._send, name="tse-udp-tx", daemon=True),
        ]
        try:
            for thread in self._threads:
                thread.start()
        except BaseException:
            # Includes interruption after only the receive worker has started.
            self.close()
            raise

    def submit(self, sequence: int, audio: NDArray[np.floating]) -> None:
        if self._closed:
            raise RuntimeError("peer is closed")
        data = encode_packet(self._stream_id, sequence, audio, ceiling_dbfs=self._ceiling_dbfs)
        with self._tx_lock:
            if sequence <= self._last_submitted:
                raise ValueError("source sequences must be strictly increasing")
            self._last_submitted = sequence
            replaced = self._tx_slot is not None
            self._tx_slot = data
        self._count("tx_submitted")
        if replaced:
            self._count("tx_replaced")
        self._wake_tx.set()

    def _send(self) -> None:
        while not self._stop.is_set():
            self._wake_tx.wait(0.05)
            self._wake_tx.clear()
            if self._stop.is_set():
                break
            with self._tx_lock:
                data, self._tx_slot = self._tx_slot, None
            if data is None:
                continue
            try:
                self._socket.sendto(data, self._peer)
            except OSError:
                if not self._stop.is_set():
                    self._count("tx_error")
            else:
                self._count("tx_sent")

    def _receive(self) -> None:
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self._socket], [], [], 0.02)
                if not readable:
                    continue
                # Read one extra byte: a truncated oversized datagram is still invalid.
                data, sender = self._socket.recvfrom(PACKET_BYTES + 1)
            except (OSError, ValueError):
                if not self._stop.is_set():
                    self._count("rx_error")
                continue
            self._count("rx_received")
            if sender != self._peer:
                self._count("rx_wrong_peer")
                continue
            try:
                packet = decode_packet(data)
            except PacketError:
                self._count("rx_invalid")
                continue
            arrival = self._clock()
            with self._jitter_lock:
                self._jitter.push(packet, arrival)

    def playout(self, now: float | None = None) -> NDArray[np.float32]:
        tick = self._clock() if now is None else now
        with self._jitter_lock:
            return self._jitter.playout(tick)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake_tx.set()
        try:
            for thread in self._threads:
                if thread.ident is not None:
                    thread.join(timeout=0.5)
        finally:
            self._socket.close()
            with self._tx_lock:
                self._tx_slot = None
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("UDP worker did not stop within the shutdown deadline")

    def summary(self) -> dict[str, object]:
        with self._stats_lock:
            result: dict[str, object] = dict(self._counters)
        with self._tx_lock:
            result["tx_pending"] = int(self._tx_slot is not None)
        with self._jitter_lock:
            result["jitter"] = self._jitter.summary()
        result["workers_alive"] = sum(thread.is_alive() for thread in self._threads)
        return result

    def __enter__(self) -> UdpPeer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
