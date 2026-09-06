import multiprocessing
import socket
import threading
import time

import numpy as np
import pytest

from tactical_speech_enhancement.network import UdpPeer
from tactical_speech_enhancement.protocol import FRAME_SAMPLES, encode_packet


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    assert predicate(), "network condition did not complete before timeout"


def test_latest_transmit_slot_preserves_source_sequence():
    from tactical_speech_enhancement.protocol import decode_packet

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2)
        peer = UdpPeer(("127.0.0.1", 0), receiver.getsockname())
        try:
            peer.submit(10, np.full(FRAME_SAMPLES, 0.1))
            peer.submit(99, np.full(FRAME_SAMPLES, 0.2))
            assert peer.summary()["tx_pending"] == 1
            peer.start()
            data, _ = receiver.recvfrom(4096)
            assert decode_packet(data).sequence == 99
            assert peer.summary()["tx_replaced"] == 1
        finally:
            peer.close()
        assert peer.summary()["workers_alive"] == 0


def test_receive_runs_without_transmit_and_rejects_wrong_peer_malformed_and_oversize():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.bind(("127.0.0.1", 0))
        with UdpPeer(("127.0.0.1", 0), sender.getsockname()) as peer:
            good = encode_packet(b"a" * 16, 0, np.full(FRAME_SAMPLES, 0.2))
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as stranger:
                stranger.sendto(good, peer.bound_address)
            sender.sendto(b"bad", peer.bound_address)
            sender.sendto(good + b"extra" * 500, peer.bound_address)
            sender.sendto(good, peer.bound_address)
            wait_until(lambda: peer.summary()["rx_received"] == 4)
            stats = peer.summary()
            assert stats["rx_wrong_peer"] == 1
            assert stats["rx_invalid"] == 2
            assert stats["tx_sent"] == 0
            jitter = stats["jitter"]
            assert jitter["accepted"] == 1
            assert peer.playout(jitter["next_deadline"]).any()


def test_submit_rejects_duplicate_source_sequence_and_closed_peer():
    peer = UdpPeer(("127.0.0.1", 0), ("127.0.0.1", 9))
    audio = np.zeros(FRAME_SAMPLES)
    try:
        peer.submit(2, audio)
        with pytest.raises(ValueError):
            peer.submit(2, audio)
        with pytest.raises(ValueError):
            peer.submit(1, audio)
    finally:
        peer.close()
    peer.close()
    with pytest.raises(RuntimeError):
        peer.submit(3, audio)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_thread_start_failure_closes_socket_and_started_worker(monkeypatch, error_type):
    original_start = threading.Thread.start
    started = 0

    def fail_second_start(thread):
        nonlocal started
        started += 1
        if started == 2:
            raise error_type("injected startup failure")
        original_start(thread)

    peer = UdpPeer(("127.0.0.1", 0), ("127.0.0.1", 9))
    monkeypatch.setattr(threading.Thread, "start", fail_second_start)
    with pytest.raises(error_type, match="startup failure"):
        peer.start()
    peer.close()
    assert peer._socket.fileno() == -1
    assert peer.summary()["workers_alive"] == 0


def _duplex_child(local_port, remote_port, ready, go, results):
    try:
        with UdpPeer(("127.0.0.1", local_port), ("127.0.0.1", remote_port)) as peer:
            ready.set()
            assert go.wait(5)
            begin = time.monotonic()
            received_frames = 0
            for sequence in range(80):
                scheduled = begin + sequence * 0.016
                time.sleep(max(0, scheduled - time.monotonic()))
                peer.submit(sequence, np.full(FRAME_SAMPLES, 0.1, dtype=np.float32))
                received_frames += bool(peer.playout().any())
            stats = peer.summary()
        results.put((received_frames, stats, peer.summary()["workers_alive"]))
    except Exception as error:
        results.put(repr(error))


def test_two_processes_send_and_receive_simultaneously():
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(2)]
    for item in sockets:
        item.bind(("127.0.0.1", 0))
    ports = [item.getsockname()[1] for item in sockets]
    for item in sockets:
        item.close()
    context = multiprocessing.get_context("spawn")
    ready = [context.Event(), context.Event()]
    go = context.Event()
    results = context.Queue()
    children = [
        context.Process(target=_duplex_child, args=(ports[i], ports[1 - i], ready[i], go, results))
        for i in range(2)
    ]
    try:
        for child in children:
            child.start()
        assert all(event.wait(5) for event in ready)
        go.set()
        reports = [results.get(timeout=8) for _ in children]
        for report in reports:
            assert not isinstance(report, str), report
            played, stats, workers = report
            assert played >= 60
            assert stats["tx_sent"] >= 60
            assert stats["rx_received"] >= 60
            assert stats["jitter"]["high_watermark"] <= 8
            assert workers == 0
    finally:
        for child in children:
            child.join(timeout=2)
            if child.is_alive():
                child.terminate()
                child.join(timeout=2)
        results.close()
    assert all(child.exitcode == 0 for child in children)
