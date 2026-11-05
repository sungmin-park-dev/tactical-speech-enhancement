"""Bidirectional UDP relay with Wi-Fi-like impairments (delay, jitter, stalls, loss).

A <-> [S1 port] proxy [S2 port] <-> B. A's peer address is S1, B's is S2, so
each side still sees packets from its configured peer address.

Stall model: every `stall_every` seconds the link holds all packets for
`stall_ms`, then releases them back-to-back (as Wi-Fi retransmission and
aggregation tends to do), unless `stall_drop` is set.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import socket
import threading
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="A endpoint ip:port")
    ap.add_argument("--b", required=True, help="B endpoint ip:port")
    ap.add_argument("--s1", required=True, help="proxy port facing A")
    ap.add_argument("--s2", required=True, help="proxy port facing B")
    ap.add_argument("--base-ms", type=float, default=1.0)
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="mean of exponential extra delay")
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--stall-every", type=float, default=0.0)
    ap.add_argument("--stall-ms", type=float, default=0.0)
    ap.add_argument("--stall-drop", action="store_true")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    def ep(s):
        h, p = s.rsplit(":", 1)
        return h, int(p)

    a, b = ep(args.a), ep(args.b)
    s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s1.bind(ep(args.s1))
    s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s2.bind(ep(args.s2))
    rng = random.Random(args.seed)
    t0 = time.monotonic()
    heap: list = []
    lock = threading.Condition()
    stats = {"a2b": 0, "b2a": 0, "lost": 0, "stalled": 0}
    stop = threading.Event()
    last_release = {"a2b": 0.0, "b2a": 0.0}

    def stall_end(now: float) -> float | None:
        if args.stall_every <= 0:
            return None
        rel = now - t0
        cycle = int(rel // args.stall_every)
        start = t0 + cycle * args.stall_every + args.stall_every - args.stall_ms / 1000
        end = start + args.stall_ms / 1000
        if start <= now < end:
            return end
        return None

    def rx(sock, out_sock, dest, key):
        while not stop.is_set():
            sock.settimeout(0.2)
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            now = time.monotonic()
            if rng.random() < args.loss:
                stats["lost"] += 1
                continue
            delay = args.base_ms / 1000 + (rng.expovariate(1000 / args.jitter_ms) if args.jitter_ms > 0 else 0)
            release = now + delay
            se = stall_end(now)
            if se is not None:
                stats["stalled"] += 1
                if args.stall_drop:
                    continue
                release = max(release, se)
            # FIFO per direction (Wi-Fi link layer preserves order)
            release = max(release, last_release[key])
            last_release[key] = release
            with lock:
                heapq.heappush(heap, (release, key, data, id(out_sock), out_sock, dest))
                lock.notify()

    def tx():
        while not stop.is_set():
            with lock:
                while not heap and not stop.is_set():
                    lock.wait(0.1)
                if stop.is_set():
                    return
                release = heap[0][0]
                now = time.monotonic()
                if release > now:
                    lock.wait(release - now)
                    continue
                _, key, data, _, out_sock, dest = heapq.heappop(heap)
            out_sock.sendto(data, dest)
            stats[key] += 1

    threads = [threading.Thread(target=rx, args=(s1, s2, b, "a2b"), daemon=True),
               threading.Thread(target=rx, args=(s2, s1, a, "b2a"), daemon=True),
               threading.Thread(target=tx, daemon=True)]
    for t in threads:
        t.start()
    time.sleep(args.duration)
    stop.set()
    with lock:
        lock.notify_all()
    if args.log:
        with open(args.log, "w") as f:
            json.dump({"args": vars(args), "stats": stats}, f)
    print("proxy", stats)


if __name__ == "__main__":
    main()
