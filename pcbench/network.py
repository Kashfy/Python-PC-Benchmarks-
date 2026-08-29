"""Network benchmarks: loopback, two-node, and an opt-in internet test.

The default measurement is the machine's own TCP loopback throughput and
round-trip latency. That one deliberately does **not** touch the internet: it characterizes the OS network
stack, socket buffers, and scheduler — a real, reproducible property of the
machine — without sending anything off-box, contacting third-party servers, or
depending on an internet connection.

A slow loopback number is itself diagnostic: it points at an overloaded CPU, an
aggressive firewall/security agent intercepting local traffic, or a misconfigured
stack.

Everything that does leave the machine — latency to a named host, download
from a named URL, the two-node test, and the internet speed test — is opt-in
behind its own flag and never part of a default run.
"""

from __future__ import annotations

import os
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


# --------------------------------------------------------------------------- #
# Two-node testing (iperf-style)
# --------------------------------------------------------------------------- #
# Loopback characterises the OS stack. It says nothing about the NIC, the
# cable, the switch, or the path — which is what anyone asking "is my network
# slow?" actually needs. That requires a second machine, so this provides both
# halves: a server that sinks traffic and echoes probes, and a client that
# measures against it.
#
# Deliberately *not* an iperf3 reimplementation and not wire-compatible with
# it. It exists so the measurement is available with nothing installed on
# either end beyond Python, which is frequently the situation on the machines
# that most need testing. Where iperf3 is available it remains the better tool
# and this one says so.
# --------------------------------------------------------------------------- #

DEFAULT_PORT = 51900
_MAGIC = b"PCBENCH1"


def serve(port: int = DEFAULT_PORT, bind: str = "0.0.0.0",
          seconds: float = 0.0, quiet: bool = False) -> dict:
    """Run the receiving half. Blocks until stopped or the time budget expires.

    Binds to all interfaces by default because the entire point is to be
    reached from another machine, and reports the fact plainly: this opens a
    listening socket, which is a change in the machine's exposure and should
    not happen silently.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((bind, port))
    except OSError as e:
        return {"error": f"could not bind {bind}:{port}: {e}"}
    sock.listen(8)
    sock.settimeout(1.0)

    if not quiet:
        print(f"  listening on {bind}:{port} — this port is open to the "
              f"network until you stop it (Ctrl-C)")

    deadline = time.monotonic() + seconds if seconds else None
    sessions = 0
    total_bytes = 0
    try:
        while True:
            if deadline and time.monotonic() > deadline:
                break
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            sessions += 1
            if not quiet:
                print(f"  connection from {addr[0]}:{addr[1]}", flush=True)
            total_bytes += _serve_one(conn)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    return {"sessions": sessions, "bytes_received": total_bytes,
            "port": port}


def _serve_one(conn: socket.socket) -> int:
    """Handle one client: bulk sink, then echo latency probes."""
    received = 0
    try:
        conn.settimeout(30.0)
        header = conn.recv(len(_MAGIC) + 1)
        if not header.startswith(_MAGIC):
            return 0
        mode = header[len(_MAGIC):len(_MAGIC) + 1]
        if mode == b"T":                       # throughput: sink until EOF
            while True:
                data = conn.recv(1 << 16)
                if not data:
                    break
                received += len(data)
        elif mode == b"L":                     # latency: echo each probe
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                data = conn.recv(64)
                if not data:
                    break
                conn.sendall(data)
                received += len(data)
    except (OSError, socket.timeout):
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass
    return received


def _client_socket(host: str, port: int, mode: bytes,
                   timeout: float = 10.0) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.sendall(_MAGIC + mode)
    return sock


def measure_throughput(host: str, port: int = DEFAULT_PORT,
                       seconds: float = 5.0, streams: int = 1) -> dict:
    """Bulk TCP throughput to a peer running :func:`serve`.

    Parallel streams matter: a single TCP stream is limited by the
    bandwidth-delay product and the receive window, so one stream over a
    high-latency link routinely shows a fraction of the available capacity
    while four streams saturate it. A large gap between the two is the
    signature of a window/latency limit rather than a bandwidth limit.
    """
    streams = max(1, int(streams))
    payload = b"\xa5" * (256 * 1024)
    sent = [0] * streams
    errors: list[str] = []
    stop = threading.Event()

    def worker(slot: int) -> None:
        try:
            sock = _client_socket(host, port, b"T")
        except OSError as e:
            errors.append(f"stream {slot}: {e}")
            return
        try:
            while not stop.is_set():
                sock.sendall(payload)
                sent[slot] += len(payload)
        except OSError as e:
            errors.append(f"stream {slot}: {e}")
        finally:
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            sock.close()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(streams)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(max(0.5, seconds))
    stop.set()
    for t in threads:
        t.join(timeout=15.0)
    elapsed = time.perf_counter() - start

    total = sum(sent)
    if not total:
        return {"error": "no data was transferred",
                "detail": errors[:3] or ["the peer may not be running "
                                         "'pcbench --net-server'"]}
    mb_s = total / elapsed / (1024 * 1024)
    return {
        "host": host, "port": port, "streams": streams,
        "seconds": round(elapsed, 2),
        "megabytes_per_s": round(mb_s, 2),
        "megabits_per_s": round(mb_s * 8.388608, 1),
        "bytes_sent": total,
        "errors": errors[:3] or None,
        "note": ("throughput is measured at the sender; a middlebox or the "
                 "receiver may be the limit rather than the link"),
    }


def measure_latency(host: str, port: int = DEFAULT_PORT,
                    probes: int = 500) -> dict:
    """Round-trip time and jitter to a peer running :func:`serve`.

    Jitter — the variation between consecutive round trips — is what decides
    whether interactive and real-time traffic works. A link with 40 ms RTT and
    1 ms jitter carries a call perfectly; one with 20 ms RTT and 30 ms jitter
    does not, and an average-only report cannot tell them apart.
    """
    try:
        sock = _client_socket(host, port, b"L")
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError as e:
        return {"error": f"could not connect to {host}:{port}: {e}",
                "hint": "the peer must be running 'pcbench --net-server'"}

    probe = b"p" * 64
    samples: list[float] = []
    lost = 0
    try:
        sock.settimeout(2.0)
        for _ in range(max(10, probes)):
            start = time.perf_counter()
            try:
                sock.sendall(probe)
                got = 0
                while got < len(probe):
                    chunk = sock.recv(len(probe) - got)
                    if not chunk:
                        raise OSError("peer closed the connection")
                    got += len(chunk)
            except (OSError, socket.timeout):
                lost += 1
                continue
            samples.append((time.perf_counter() - start) * 1000.0)
    finally:
        sock.close()

    if not samples:
        return {"error": "no probe completed", "lost": lost}

    ordered = sorted(samples)
    # Mean absolute consecutive difference, the same definition RFC 3550 uses
    # for RTP jitter.
    jitter = (sum(abs(samples[i] - samples[i - 1])
                  for i in range(1, len(samples))) / (len(samples) - 1)
              if len(samples) > 1 else 0.0)
    return {
        "host": host, "port": port,
        "probes": len(samples), "lost": lost,
        "rtt_ms": {
            "min": round(ordered[0], 3),
            "p50": round(ordered[len(ordered) // 2], 3),
            "p99": round(ordered[min(len(ordered) - 1,
                                     int(len(ordered) * 0.99))], 3),
            "max": round(ordered[-1], 3),
            "mean": round(sum(samples) / len(samples), 3),
        },
        "jitter_ms": round(jitter, 3),
    }


def run_peer(host: str, port: int = DEFAULT_PORT, seconds: float = 5.0,
             streams: int = 4) -> dict:
    """Full two-node measurement: latency, one stream, then parallel streams."""
    result: dict = {"host": host, "port": port}
    result["latency"] = measure_latency(host, port)
    if result["latency"].get("error"):
        return result

    result["single_stream"] = measure_throughput(host, port, seconds, 1)
    if streams > 1:
        result["parallel_streams"] = measure_throughput(host, port, seconds,
                                                        streams)
        single = result["single_stream"].get("megabits_per_s") or 0
        multi = result["parallel_streams"].get("megabits_per_s") or 0
        if single and multi:
            result["parallel_gain"] = round(multi / single, 2)
            if multi > single * 1.5:
                result["verdict"] = (
                    f"{streams} streams reach {multi:,.0f} Mb/s against "
                    f"{single:,.0f} for one — the single-stream figure is "
                    f"limited by the TCP window and latency, not by the "
                    f"link's capacity. Tune window sizes, or use parallel "
                    f"connections for bulk transfer.")
            else:
                result["verdict"] = (
                    f"parallel streams add little ({multi:,.0f} vs "
                    f"{single:,.0f} Mb/s) — the link or an endpoint is the "
                    f"limit, not the TCP window.")
    return result


def render_peer(result: dict | None) -> str:
    """Terminal block for a two-node run."""
    if not result:
        return ""
    lines = [f"  Peer                : {result.get('host')}:"
             f"{result.get('port')}"]
    lat = result.get("latency") or {}
    if lat.get("error"):
        lines.append(f"  Latency             : {lat['error']}")
        if lat.get("hint"):
            lines.append(f"                        {lat['hint']}")
        return "\n".join(lines)

    rtt = lat.get("rtt_ms", {})
    lines.append(f"  Round-trip time     : {rtt.get('p50', 0):.3f} ms median "
                 f"(min {rtt.get('min', 0):.3f}, p99 {rtt.get('p99', 0):.3f})")
    lines.append(f"  Jitter              : {lat.get('jitter_ms', 0):.3f} ms")
    if lat.get("lost"):
        lines.append(f"  Lost probes         : {lat['lost']}")

    for key, label in (("single_stream", "Throughput (1 stream)"),
                       ("parallel_streams", "Throughput (parallel)")):
        entry = result.get(key) or {}
        if entry.get("error"):
            lines.append(f"  {label:<20}: {entry['error']}")
        elif entry.get("megabits_per_s"):
            lines.append(f"  {label:<20}: "
                         f"{entry['megabits_per_s']:>10,.0f} Mb/s "
                         f"({entry['megabytes_per_s']:,.0f} MB/s"
                         f"{', ' + str(entry['streams']) + ' streams' if entry.get('streams', 1) > 1 else ''})")
    if result.get("verdict"):
        lines.append(f"      i {result['verdict']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Internet speed test (opt-in — this one leaves the machine)
# --------------------------------------------------------------------------- #
# Everything above either stays on the box (loopback) or needs a target the
# user names. "How fast is my internet?" needs neither, which is exactly why
# it has to be asked for explicitly: it sends traffic to a third party and
# spends the user's bandwidth, possibly metered. It is never part of a normal
# run, only its own mode.
#
# Deliberately not a speedtest.net client. There is no server selection, no
# multi-connection saturation, and no attempt to beat a well-provisioned CDN
# out of a shaped link — so on a fast connection this reads low against those
# tools, and says so. What it measures honestly is what a single stream from
# this machine to a nearby anycast endpoint actually achieves, which is the
# number that matters for a download, a checkout, or a container pull.
# --------------------------------------------------------------------------- #

#: Public endpoint serving both directions with no key and no account.
#: Anycast, so the path is usually short from anywhere.
DEFAULT_SPEED_SERVER = "https://speed.cloudflare.com"

#: Sent as the upload body. Incompressible, so a transparent proxy cannot
#: flatter the result by compressing a run of zeros.
_UPLOAD_CHUNK = os.urandom(256 * 1024) if hasattr(os, "urandom") else None


def _server_host(server: str) -> str:
    from urllib.parse import urlparse

    return urlparse(server).hostname or ""


def _measure_download(server: str, seconds: float, max_bytes: int) -> dict:
    """Read for up to ``seconds``, or ``max_bytes``, whichever binds."""
    import urllib.request

    url = f"{server}/__down?bytes={max_bytes}"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "pcbench"})
        received = 0
        with urllib.request.urlopen(request, timeout=15) as response:
            # Timed from the first byte, so DNS, the TCP handshake and TLS do
            # not count against the throughput they precede.
            chunk = response.read(64 * 1024)
            start = time.perf_counter()
            while chunk:
                received += len(chunk)
                if (received >= max_bytes
                        or time.perf_counter() - start > seconds):
                    break
                chunk = response.read(256 * 1024)
        elapsed = time.perf_counter() - start
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if received <= 0 or elapsed <= 0:
        return {"error": "no data received"}
    return {
        "mbit_per_s": round(received * 8 / elapsed / 1e6, 1),
        "mb_per_s": round(received / elapsed / (1024 * 1024), 2),
        "transferred_mb": round(received / (1024 * 1024), 1),
        "seconds": round(elapsed, 2),
    }


def _post(server: str, payload: bytes, timeout: float = 30.0) -> float:
    """POST ``payload`` and return the seconds it took. Raises on failure."""
    import urllib.request

    request = urllib.request.Request(
        f"{server}/__up", data=payload, method="POST",
        headers={"User-Agent": "pcbench",
                 "Content-Type": "application/octet-stream"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
    return time.perf_counter() - start


def _measure_upload(server: str, seconds: float, max_bytes: int) -> dict:
    """Probe with a small body, then send one sized to the time budget.

    An upload cannot be cut short the way a download can — the request body
    is committed before the timing starts — so the size is chosen from a
    measured rate instead of guessed. That is also what stops a slow link
    from being handed a hundred megabytes to push.
    """
    if _UPLOAD_CHUNK is None:
        return {"error": "no source of random bytes"}
    probe_bytes = 1024 * 1024
    try:
        probe_chunks = probe_bytes // len(_UPLOAD_CHUNK)
        probe = _post(server, _UPLOAD_CHUNK * probe_chunks)
        if probe <= 0:
            return {"error": "upload probe took no measurable time"}
        rate = probe_bytes / probe
        # Upload gets a quarter of the byte budget. Links are usually
        # asymmetric, and pushing hundreds of megabytes up a connection
        # someone may be paying for by the gigabyte is not a reasonable
        # default however fast it is.
        ceiling = max(probe_bytes, max_bytes // 4)
        size = int(min(max(rate * seconds, probe_bytes), ceiling))
        # Whole chunks only, so the body is a clean multiple of the buffer.
        chunks = max(1, size // len(_UPLOAD_CHUNK))
        payload = _UPLOAD_CHUNK * chunks
        elapsed = _post(server, payload)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    sent = len(payload)
    if elapsed <= 0:
        return {"error": "upload took no measurable time"}
    return {
        "mbit_per_s": round(sent * 8 / elapsed / 1e6, 1),
        "mb_per_s": round(sent / elapsed / (1024 * 1024), 2),
        "transferred_mb": round(sent / (1024 * 1024), 1),
        "seconds": round(elapsed, 2),
    }


def internet_speed(server: str = DEFAULT_SPEED_SERVER, seconds: float = 5.0,
                   max_mb: int = 200) -> dict:
    """Download, upload, latency and jitter against ``server``. Never raises.

    Both directions are capped by time *and* by bytes, so a fast link stops
    at the time budget and a slow one stops at the byte budget rather than
    running for minutes on a metered connection. Upload is capped at a
    quarter of the byte budget on top of that.
    """
    host = _server_host(server)
    if not host:
        return {"error": f"not a usable server URL: {server!r}"}
    max_bytes = max(1, max_mb) * 1024 * 1024
    out: dict = {"server": server, "host": host,
                 "note": "single stream to a public endpoint; reads lower "
                         "than multi-connection tools on a fast link"}
    out["latency"] = tcp_latency(host, 443)
    out["dns"] = dns_latency()
    out["download"] = _measure_download(server, seconds, max_bytes)
    out["upload"] = _measure_upload(server, seconds, max_bytes)
    return out


def _speed_line(label: str, result: dict) -> str:
    if not result or result.get("error"):
        return f"  {label:<12}: unavailable — {(result or {}).get('error')}"
    return (f"  {label:<12}: {result['mbit_per_s']:>8.1f} Mbit/s "
            f"({result['mb_per_s']:.1f} MB/s, "
            f"{result['transferred_mb']:.0f} MB in {result['seconds']:.1f}s)")


def render_internet(result: dict | None) -> str:
    """Human-readable internet speed report."""
    if not result:
        return "  internet test did not run"
    if result.get("error"):
        return f"  internet test failed: {result['error']}"
    lines = [_speed_line("Download", result.get("download")),
             _speed_line("Upload", result.get("upload"))]

    latency = result.get("latency") or {}
    if latency.get("p50_ms") is None:
        lines.append(f"  {'Latency':<12}: unavailable — "
                     f"{latency.get('error', 'not measured')}")
    else:
        jitter = latency.get("p99_ms", 0) - latency.get("min_ms", 0)
        lines.append(f"  {'Latency':<12}: {latency['p50_ms']:>8.1f} ms median "
                     f"(best {latency['min_ms']:.1f}, "
                     f"jitter {jitter:.1f}, "
                     f"{latency['loss_percent']:.0f}% failed)")
    # Absent and failed are different states, and neither has a number to
    # print — checking only for "error" walked straight into the missing key.
    dns = result.get("dns") or {}
    if dns.get("median_ms") is not None:
        lines.append(f"  {'DNS':<12}: {dns['median_ms']:>8.1f} ms median to "
                     f"resolve {dns.get('resolved', 0)} name(s)")
    lines.append("")
    lines.append(f"  endpoint: {result['server']}")
    lines.append(f"  {result['note']}")
    return "\n".join(lines)
