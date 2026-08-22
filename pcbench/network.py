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


# --------------------------------------------------------------------------- #
# Real network — strictly opt-in
#
# Everything above stays on 127.0.0.1. The functions below contact a host the
# user names explicitly: no default target, no telemetry, and nothing runs
# unless a --network-host or --network-url argument is supplied. Sending
# traffic off the machine should always be a deliberate choice.
# --------------------------------------------------------------------------- #
def tcp_latency(host: str, port: int = 443, attempts: int = 12) -> dict:
    """Round-trip time to open a TCP connection to ``host``.

    Connection setup is a full round trip, so this measures real network
    latency without needing ICMP privileges the way ping does.
    """
    times: list[float] = []
    errors = 0
    for _ in range(attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        start = time.perf_counter()
        try:
            sock.connect((host, port))
            times.append((time.perf_counter() - start) * 1000.0)
        except OSError:
            errors += 1
        finally:
            sock.close()
    if not times:
        return {"host": host, "error": f"could not reach {host}:{port}"}
    times.sort()
    n = len(times)
    return {
        "host": host,
        "port": port,
        "p50_ms": round(times[n // 2], 2),
        "p99_ms": round(times[min(n - 1, int(n * 0.99))], 2),
        "min_ms": round(times[0], 2),
        "attempts": attempts,
        "failed": errors,
        "loss_percent": round(errors / attempts * 100, 1),
    }


def dns_latency(hostnames: list[str] | None = None) -> dict:
    """Time to resolve names, which often dominates perceived slowness."""
    names = hostnames or ["example.com", "cloudflare.com", "wikipedia.org"]
    times, failures = [], 0
    for name in names:
        start = time.perf_counter()
        try:
            socket.getaddrinfo(name, None)
            times.append((time.perf_counter() - start) * 1000.0)
        except OSError:
            failures += 1
    if not times:
        return {"error": "no name resolved — DNS unreachable"}
    times.sort()
    return {
        "resolved": len(times),
        "failed": failures,
        "median_ms": round(times[len(times) // 2], 2),
        "max_ms": round(times[-1], 2),
    }


def download_throughput(url: str, max_seconds: float = 5.0,
                        max_bytes: int = 50 * 1024 * 1024) -> dict:
    """Download throughput from a URL the user supplied.

    Capped in both time and bytes so a mistyped URL cannot pull an unbounded
    amount of data over a metered connection.
    """
    import urllib.request

    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "pcbench"})
        start = time.perf_counter()
        received = 0
        with urllib.request.urlopen(request, timeout=10) as response:
            while received < max_bytes:
                if time.perf_counter() - start > max_seconds:
                    break
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                received += len(chunk)
        elapsed = time.perf_counter() - start
    except Exception as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"}
    if not received or elapsed <= 0:
        return {"url": url, "error": "no data received"}
    return {
        "url": url,
        "mb_per_s": round(received / elapsed / (1024 * 1024), 2),
        "mbit_per_s": round(received * 8 / elapsed / 1e6, 1),
        "downloaded_mb": round(received / (1024 * 1024), 1),
        "seconds": round(elapsed, 2),
    }


def run_external(host: str | None = None, url: str | None = None) -> dict:
    """Opt-in external tests. Returns {} when no target was given."""
    out: dict = {}
    if host:
        out["tcp"] = tcp_latency(host)
        out["dns"] = dns_latency()
    if url:
        out["download"] = download_throughput(url)
    return out


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
