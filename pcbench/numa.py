"""NUMA topology and the local-versus-remote memory penalty.

On any multi-socket server — and on several large single-socket parts with
chiplet designs — memory is not uniformly fast. Each node owns some of the
DRAM, and a core reaching memory attached to *another* node pays an extra hop:
typically 1.5-2.2x the latency and materially less bandwidth.

This matters to benchmarking in a way that is easy to miss. A memory test that
happens to allocate on the local node reports one number; the identical test
that lands on a remote node reports a much worse one, and nothing in the output
distinguishes them. Run-to-run variance on a NUMA machine is frequently *not*
noise — it is the allocator landing differently.

It matters even more to the people running the workloads. A database or JVM
pinned to the wrong node loses a third of its memory performance for a reason
no CPU profile will ever show, and the fix is a ``numactl`` flag rather than
hardware.

So this module reports two things:

* **Topology** — how many nodes, which CPUs and how much memory each owns, and
  the firmware's own distance matrix (the ACPI SLIT table). Read from sysfs,
  so it always works on Linux and costs nothing.
* **A measured bandwidth matrix** — every (CPU node, memory node) pair, probed
  for real. This needs ``numactl`` to place the memory deliberately, because
  the kernel's default policy is to allocate locally and would otherwise never
  produce the remote case being measured.

Non-NUMA machines are the common case and are reported as such in one line,
not treated as a degenerate error.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys

MB = 1024 * 1024
_NODE_DIR = "/sys/devices/system/node"


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return ""


def parse_cpulist(text: str) -> list[int]:
    """Expand a sysfs CPU list like ``0-3,8,12-15`` into explicit ids."""
    out: list[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                lo, hi = (int(x) for x in part.split("-", 1))
            except ValueError:
                continue
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #
def topology() -> dict:
    """Nodes, their CPUs and memory, and the firmware distance matrix."""
    system = platform.system()
    if system == "Darwin":
        return {"available": True, "nodes": 1, "numa": False,
                "note": "Apple silicon presents unified memory with a single "
                        "NUMA node; there is no remote-node penalty to measure"}
    if system == "Windows":
        return _topology_windows()
    if not os.path.isdir(_NODE_DIR):
        return {"available": False,
                "note": "the kernel exposes no NUMA nodes (typical for "
                        "single-socket desktops, VMs, and containers)"}

    try:
        names = sorted(n for n in os.listdir(_NODE_DIR)
                       if re.fullmatch(r"node\d+", n))
    except OSError:
        return {"available": False}

    nodes = []
    for name in names:
        index = int(name[4:])
        cpulist = _read(os.path.join(_NODE_DIR, name, "cpulist"))
        cpus = parse_cpulist(cpulist)
        meminfo = _read(os.path.join(_NODE_DIR, name, "meminfo"))
        total = free = None
        for line in meminfo.splitlines():
            m = re.search(r"MemTotal:\s+(\d+) kB", line)
            if m:
                total = int(m.group(1)) * 1024
            m = re.search(r"MemFree:\s+(\d+) kB", line)
            if m:
                free = int(m.group(1)) * 1024
        nodes.append({
            "node": index,
            "cpulist": cpulist,
            "cpus": cpus,
            "cpu_count": len(cpus),
            "memory_bytes": total,
            "memory_free_bytes": free,
            "memory_gb": round(total / (1024 ** 3), 1) if total else None,
        })

    # The ACPI SLIT distance matrix: 10 is defined as local, and remote entries
    # are relative to it (21 means 2.1x the local cost, as the firmware
    # declares it -- not as measured).
    distances = {}
    for name in names:
        raw = _read(os.path.join(_NODE_DIR, name, "distance"))
        if raw:
            try:
                distances[int(name[4:])] = [int(x) for x in raw.split()]
            except ValueError:
                pass

    return {
        "available": True,
        "numa": len(nodes) > 1,
        "nodes": len(nodes),
        "detail": nodes,
        "distances": distances or None,
        "numactl": bool(shutil.which("numactl")),
    }


def _topology_windows() -> dict:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_ComputerSystem)."
             "NumberOfProcessors"],
            capture_output=True, text=True, timeout=8).stdout.strip()
        sockets = int(out) if out.isdigit() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        sockets = None
    return {"available": sockets is not None,
            "numa": bool(sockets and sockets > 1),
            "nodes": sockets,
            "note": "Windows NUMA detail and per-node bandwidth probing are "
                    "not implemented; socket count is reported as a proxy"}


# --------------------------------------------------------------------------- #
# Bandwidth probe
# --------------------------------------------------------------------------- #
def _probe(buf_mb: int = 64, seconds: float = 0.4) -> float:
    """Memory copy bandwidth in MB/s, for this process wherever it is bound.

    A ``bytearray`` slice assignment is a single ``memmove`` in C, so this
    measures the memory system rather than the interpreter. It is deliberately
    the same operation the main memory benchmark uses, so the figures are
    directly comparable.
    """
    import time

    size = max(1, buf_mb) * MB
    src = bytearray(os.urandom(min(size, MB))) * max(1, size // MB)
    dst = bytearray(len(src))
    n = len(src)

    # Warm both buffers so first-touch page allocation is not being timed.
    dst[:] = src
    copies = 0
    start = time.perf_counter()
    while True:
        dst[:] = src
        copies += 1
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
    return copies * n / elapsed / MB


def bandwidth_matrix(topo: dict, buf_mb: int = 64,
                     seconds: float = 0.4) -> dict:
    """Measure bandwidth for every (CPU node, memory node) pair.

    Requires ``numactl``: the kernel's default allocation policy is local-first,
    so without explicit placement the remote cases would silently be measured
    as local and the matrix would be uniform and wrong.
    """
    if not topo.get("numa"):
        return {"skipped": True,
                "reason": "single NUMA node; there is no remote case to measure"}
    if not shutil.which("numactl"):
        return {"skipped": True,
                "reason": "numactl is not installed",
                "fix": "install numactl to measure the local/remote penalty "
                       "(apt install numactl / dnf install numactl)"}

    node_ids = [d["node"] for d in topo.get("detail", [])]
    matrix: dict[str, dict[str, float]] = {}
    for cpu_node in node_ids:
        row: dict[str, float] = {}
        for mem_node in node_ids:
            rate = _run_probe(cpu_node, mem_node, buf_mb, seconds)
            if rate is not None:
                row[str(mem_node)] = round(rate, 1)
        if row:
            matrix[str(cpu_node)] = row

    if not matrix:
        return {"skipped": True,
                "reason": "no node pair could be probed"}

    return {"matrix": matrix, "unit": "MB/s",
            "buffer_mb": buf_mb, **_penalty(matrix)}


def _run_probe(cpu_node: int, mem_node: int, buf_mb: int,
               seconds: float) -> float | None:
    cmd = ["numactl", f"--cpunodebind={cpu_node}", f"--membind={mem_node}",
           sys.executable, "-m", "pcbench.numa", "--probe",
           "--mb", str(buf_mb), "--seconds", str(seconds)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=max(30.0, seconds * 20))
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip().split()[-1])
    except (ValueError, IndexError):
        return None


def _penalty(matrix: dict) -> dict:
    """Summarise the local/remote gap the matrix reveals."""
    local, remote = [], []
    for cpu_node, row in matrix.items():
        for mem_node, rate in row.items():
            (local if cpu_node == mem_node else remote).append(rate)
    if not local or not remote:
        return {}
    local_mean = sum(local) / len(local)
    remote_mean = sum(remote) / len(remote)
    penalty = (100.0 * (local_mean - remote_mean) / local_mean
               if local_mean else 0.0)
    return {
        "local_mean_mb_s": round(local_mean, 1),
        "remote_mean_mb_s": round(remote_mean, 1),
        "remote_penalty_pct": round(penalty, 1),
    }


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def run(measure: bool = True, buf_mb: int = 64) -> dict:
    """Topology, plus the measured bandwidth matrix when it is meaningful."""
    topo = topology()
    result: dict = {"topology": topo}
    if measure:
        result["bandwidth"] = bandwidth_matrix(topo, buf_mb)
    return result


def notes(result: dict) -> list[str]:
    """Conclusions a person can act on."""
    out: list[str] = []
    topo = result.get("topology") or {}
    bw = result.get("bandwidth") or {}

    if topo.get("numa"):
        out.append(
            f"{topo['nodes']} NUMA nodes — memory bandwidth and latency depend "
            f"on which node a process runs and allocates on, so results here "
            f"vary with placement rather than only with load")

    penalty = bw.get("remote_penalty_pct")
    if isinstance(penalty, (int, float)):
        if penalty >= 15:
            out.append(
                f"remote memory is {penalty:.0f}% slower than local "
                f"({bw['local_mean_mb_s']:,.0f} vs "
                f"{bw['remote_mean_mb_s']:,.0f} MB/s) — pin latency-sensitive "
                f"services with 'numactl --cpunodebind=N --membind=N', and "
                f"expect unpinned workloads to vary by about this much run to "
                f"run")
        else:
            out.append(
                f"remote memory costs only {penalty:.0f}% here — placement is "
                f"not worth tuning on this machine")

    distances = topo.get("distances")
    if distances and topo.get("numa"):
        worst = max((max(row) for row in distances.values()), default=10)
        if worst >= 30:
            out.append(
                f"firmware reports a worst-case node distance of {worst} "
                f"(local = 10), which is a wide topology; cross-node traffic "
                f"is expensive on this machine by design")
    return out


def render(result: dict, note_list: list[str] | None = None) -> str:
    """Terminal block."""
    topo = result.get("topology") or {}
    if not topo.get("available"):
        return f"  {topo.get('note', 'NUMA information is unavailable')}"
    if not topo.get("numa"):
        return f"  Single NUMA node — {topo.get('note', 'uniform memory access')}"

    lines = [f"  Nodes                     : {topo['nodes']}"]
    for node in topo.get("detail", []):
        lines.append(f"    node {node['node']}: {node['cpu_count']} CPUs "
                     f"({node['cpulist']}), {node.get('memory_gb', '?')} GB")

    bw = result.get("bandwidth") or {}
    if bw.get("skipped"):
        lines.append(f"  Bandwidth matrix          : skipped — {bw['reason']}")
        if bw.get("fix"):
            lines.append(f"                              {bw['fix']}")
    elif bw.get("matrix"):
        node_ids = sorted(bw["matrix"], key=int)
        lines.append("")
        lines.append("  Bandwidth MB/s (rows = CPU node, columns = memory node)")
        lines.append("        " + "".join(f"{'node ' + n:>12}" for n in node_ids))
        for cpu_node in node_ids:
            row = bw["matrix"][cpu_node]
            cells = "".join(f"{row.get(m, 0):>12,.0f}" for m in node_ids)
            lines.append(f"  cpu {cpu_node:<2}{cells}")
        if bw.get("remote_penalty_pct") is not None:
            lines.append(f"  Remote penalty            : "
                         f"{bw['remote_penalty_pct']:.1f}%")

    for note in note_list or []:
        lines.append(f"      i {note}")
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    """Probe entry point, invoked under numactl as a subprocess.

    It has to be a separate process: NUMA binding is applied at exec time by
    numactl, so the measurement cannot happen in the parent.
    """
    buf_mb, seconds = 64, 0.4
    i = 0
    while i < len(argv):
        if argv[i] == "--mb" and i + 1 < len(argv):
            buf_mb = int(argv[i + 1]); i += 2
        elif argv[i] == "--seconds" and i + 1 < len(argv):
            seconds = float(argv[i + 1]); i += 2
        else:
            i += 1
    print(f"{_probe(buf_mb, seconds):.3f}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--probe" in args:
        sys.exit(_main(args))
    import json as _json
    print(_json.dumps(run(), indent=2))
