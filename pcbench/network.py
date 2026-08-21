"""Local network-stack benchmark.

Measures the machine's own TCP loopback throughput and round-trip latency. This
deliberately does **not** touch the internet: it characterizes the OS network
stack, socket buffers, and scheduler — a real, reproducible property of the
machine — without sending anything off-box, contacting third-party servers, or
depending on an internet connection.

A slow loopback number is itself diagnostic: it points at an overloaded CPU, an
aggressive firewall/security agent intercepting local traffic, or a misconfigured
stack.
"""

from __future__ import annotations

import socket
import threading
import time

_HOST = "127.0.0.1"
_CHUNK = 256 * 1024        # 256 KiB per send


def _throughput(duration: float) -> float:
    """Bulk TCP loopback throughput in MB/s.

    A server thread drains bytes as fast as it can while the main thread sends
    for ``duration`` seconds.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((_HOST, 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def server():
        conn, _ = listener.accept()
        with conn:
            while conn.recv(_CHUNK):
                pass

    t = threading.Thread(target=server, daemon=True)
    t.start()

    sender = socket.create_connection((_HOST, port))
    payload = b"\x5a" * _CHUNK
    sent = 0
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < duration:
            sender.sendall(payload)
            sent += len(payload)
    finally:
        sender.close()
        listener.close()
        t.join(timeout=1.0)
    elapsed = time.perf_counter() - start
    return sent / elapsed / (1024 * 1024)


def _latency(rounds: int = 2000) -> dict:
    """Round-trip latency of a 1-byte ping/pong over loopback, in microseconds.

    TCP_NODELAY disables Nagle's algorithm, which would otherwise coalesce the
    tiny packets and hide the true per-message latency.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((_HOST, 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def echo():
        conn, _ = listener.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with conn:
            while True:
                b = conn.recv(1)
                if not b:
                    break
                conn.sendall(b)

    t = threading.Thread(target=echo, daemon=True)
    t.start()

    client = socket.create_connection((_HOST, port))
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    samples = []
    try:
        for _ in range(min(rounds, 200)):    # warm up
            client.sendall(b"\x01")
            client.recv(1)
        for _ in range(rounds):
            t0 = time.perf_counter()
            client.sendall(b"\x01")
            client.recv(1)
            samples.append((time.perf_counter() - t0) * 1e6)
    finally:
        client.close()
        listener.close()
        t.join(timeout=1.0)

    samples.sort()
    n = len(samples)
    return {
        "p50_us": round(samples[n // 2], 2),
        "p99_us": round(samples[min(n - 1, int(n * 0.99))], 2),
        "mean_us": round(sum(samples) / n, 2),
    }


def run(duration: float = 1.0) -> dict:
    """Benchmark loopback throughput and latency. Never raises."""
    try:
        mb_s = _throughput(min(duration, 2.0))
        lat = _latency()
        return {
            "loopback_mb_s": round(mb_s, 1),
            "latency": lat,
            "note": "TCP loopback (127.0.0.1); measures the OS network stack, "
                    "no external traffic",
        }
    except OSError as e:
        return {"error": f"network benchmark failed: {e}"}
