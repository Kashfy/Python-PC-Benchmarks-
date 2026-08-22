"""Container, cgroup, and cloud-instance awareness.

A benchmark run inside a container or a cloud VM measures the *slice* the
scheduler hands out, not the hardware underneath it — and the two can differ by
an order of magnitude:

* A cgroup v2 ``cpu.max`` of ``50000 100000`` means half a core. ``os.cpu_count()``
  still reports 16, so a multicore test spawns 16 workers that fight over 0.5
  cores and produce a number that looks like catastrophic hardware failure.
* A container memory limit smaller than the host's RAM makes the memory-test
  sizing logic (which reads host RAM) allocate past the limit, and the kernel
  OOM-kills the process mid-run.
* Cloud instances are frequently *shared-tenant* and burstable. A t3.micro
  benchmarks like a fast machine for 30 seconds and like a slow one afterwards,
  because CPU credits ran out — a fact no amount of repetition inside the run
  can explain on its own.

Everything here is read-only detection: no network calls, no cloud metadata
endpoints (which would leak the fact that the tool ran to a third party). Cloud
identification comes from DMI strings the hypervisor already exposes locally.
"""

from __future__ import annotations

import os
import platform

_CGROUP_V2 = "/sys/fs/cgroup"
_CGROUP_V1_CPU = "/sys/fs/cgroup/cpu"
_CGROUP_V1_MEM = "/sys/fs/cgroup/memory"


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return ""


def _read_int(path: str) -> int | None:
    text = _read(path)
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Container runtime
# --------------------------------------------------------------------------- #
def container_runtime() -> str | None:
    """Name the container runtime this process is inside, if any."""
    if platform.system() != "Linux":
        # Docker Desktop on macOS/Windows runs a Linux VM, so a container there
        # still sees Linux. Native macOS/Windows processes are never in one.
        return None

    if os.path.exists("/.dockerenv"):
        return "Docker"
    if os.path.exists("/run/.containerenv"):
        return "Podman"
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "Kubernetes"

    cgroup = _read("/proc/1/cgroup")
    for needle, name in (("docker", "Docker"), ("containerd", "containerd"),
                         ("kubepods", "Kubernetes"), ("libpod", "Podman"),
                         ("lxc", "LXC"), ("garden", "Garden")):
        if needle in cgroup:
            return name

    # systemd-nspawn and LXC export this directly.
    env = _read("/proc/1/environ").replace("\x00", "\n")
    for line in env.splitlines():
        if line.startswith("container="):
            return line.split("=", 1)[1] or "container"

    # A PID 1 that is not init/systemd is a strong hint, but only a hint, so it
    # is reported as such rather than as a definite runtime.
    comm = _read("/proc/1/comm")
    if comm and comm not in ("systemd", "init", "upstart", "runit", "openrc"):
        return "container (unidentified)"
    return None


# --------------------------------------------------------------------------- #
# cgroup resource limits
# --------------------------------------------------------------------------- #
def cpu_quota() -> float | None:
    """Effective CPU allowance in cores, or None when unlimited/unknown.

    Returns a float because quotas are routinely fractional (``0.5`` cores is
    an extremely common Kubernetes request).
    """
    if platform.system() != "Linux":
        return None

    # cgroup v2: "cpu.max" holds "<quota|max> <period>".
    raw = _read(os.path.join(_CGROUP_V2, "cpu.max"))
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return quota / period
            except ValueError:
                pass
        elif parts and parts[0] == "max":
            return None

    # cgroup v1: two separate files, quota of -1 means unlimited.
    quota = _read_int(os.path.join(_CGROUP_V1_CPU, "cpu.cfs_quota_us"))
    period = _read_int(os.path.join(_CGROUP_V1_CPU, "cpu.cfs_period_us"))
    if quota and period and quota > 0 and period > 0:
        return quota / period
    return None


def cpu_affinity_count() -> int | None:
    """Cores this process is actually allowed to run on.

    Distinct from a quota: ``taskset``, CPU sets, and many CI runners pin the
    process to a subset of cores without imposing any bandwidth limit.
    """
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None


def memory_limit_bytes() -> int | None:
    """Effective memory ceiling in bytes, or None when unlimited/unknown."""
    if platform.system() != "Linux":
        return None

    raw = _read(os.path.join(_CGROUP_V2, "memory.max"))
    if raw and raw != "max":
        try:
            return int(raw)
        except ValueError:
            pass

    v1 = _read_int(os.path.join(_CGROUP_V1_MEM, "memory.limit_in_bytes"))
    # cgroup v1 spells "unlimited" as a huge sentinel close to 2**63.
    if v1 and v1 < (1 << 62):
        return v1
    return None


# --------------------------------------------------------------------------- #
# Cloud instance identity (local sources only)
# --------------------------------------------------------------------------- #
_DMI_CLOUD = [
    ("amazon ec2", "AWS EC2"), ("amazon", "AWS EC2"), ("ec2", "AWS EC2"),
    ("google", "Google Cloud"), ("googlecloud", "Google Cloud"),
    ("microsoft corporation", "Azure"), ("hetzner", "Hetzner"),
    ("digitalocean", "DigitalOcean"), ("oracle", "Oracle Cloud"),
    ("alibaba", "Alibaba Cloud"), ("scaleway", "Scaleway"),
    ("linode", "Linode"), ("vultr", "Vultr"), ("openstack", "OpenStack"),
]


def cloud_provider() -> str | None:
    """Identify the cloud provider from locally readable DMI/firmware strings.

    Deliberately does *not* query instance-metadata endpoints
    (``169.254.169.254`` and friends): that would send a request off-machine
    every time the tool runs, which a diagnostics utility has no business doing
    without being asked.
    """
    if platform.system() != "Linux":
        return None
    blob = " ".join(
        _read(f"/sys/class/dmi/id/{f}").lower()
        for f in ("sys_vendor", "product_name", "board_vendor",
                  "bios_vendor", "chassis_vendor")
    )
    if not blob.strip():
        return None
    for needle, name in _DMI_CLOUD:
        if needle in blob:
            return name
    return None


def ci_environment() -> str | None:
    """Name the CI system, if the run is happening inside one.

    CI runners are shared, virtualised, and noisy. Results from one are useful
    for *regression detection against that same runner class* and close to
    meaningless as absolute hardware figures, so the fact is recorded.
    """
    markers = [
        ("GITHUB_ACTIONS", "GitHub Actions"), ("GITLAB_CI", "GitLab CI"),
        ("CIRCLECI", "CircleCI"), ("JENKINS_URL", "Jenkins"),
        ("BUILDKITE", "Buildkite"), ("TF_BUILD", "Azure Pipelines"),
        ("TEAMCITY_VERSION", "TeamCity"), ("DRONE", "Drone"),
        ("APPVEYOR", "AppVeyor"), ("TRAVIS", "Travis CI"),
    ]
    for var, name in markers:
        if os.environ.get(var):
            return name
    if os.environ.get("CI"):
        return "CI (unidentified)"
    return None


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def detect(host_cores: int | None = None,
           host_ram_bytes: int = 0) -> dict:
    """Full confinement picture plus the *effective* resources to benchmark with.

    ``effective_cores`` and ``effective_ram_bytes`` are what callers should size
    workloads against; the host figures remain available for reporting.
    """
    runtime = container_runtime()
    quota = cpu_quota()
    affinity = cpu_affinity_count()
    mem_limit = memory_limit_bytes()
    host_cores = host_cores or os.cpu_count() or 1

    # The tightest of the three constraints wins, and a fractional quota still
    # needs at least one worker process to run at all.
    candidates = [host_cores]
    if affinity:
        candidates.append(affinity)
    if quota:
        candidates.append(max(1, int(quota)))
    effective_cores = max(1, min(candidates))

    effective_ram = host_ram_bytes
    if mem_limit and (not host_ram_bytes or mem_limit < host_ram_bytes):
        effective_ram = mem_limit

    constrained = bool(
        (quota is not None and quota < host_cores)
        or (affinity is not None and affinity < host_cores)
        or (mem_limit is not None and host_ram_bytes
            and mem_limit < host_ram_bytes)
    )

    return {
        "container": runtime,
        "cloud": cloud_provider(),
        "ci": ci_environment(),
        "cpu_quota_cores": round(quota, 3) if quota else None,
        "cpu_affinity_cores": affinity,
        "memory_limit_bytes": mem_limit,
        "host_cores": host_cores,
        "host_ram_bytes": host_ram_bytes or None,
        "effective_cores": effective_cores,
        "effective_ram_bytes": effective_ram or None,
        "constrained": constrained,
    }


def warnings(info: dict) -> list[str]:
    """Comparability warnings arising from confinement.

    These are advisory, not blocking: benchmarking inside a container is a
    completely legitimate thing to want to do — the results just must not be
    filed alongside bare-metal ones without a note.
    """
    out: list[str] = []
    quota = info.get("cpu_quota_cores")
    host = info.get("host_cores") or 1

    if quota is not None and quota < host:
        out.append(
            f"CPU quota limits this process to {quota:g} of {host} cores; "
            f"multicore results reflect the quota, not the hardware")
    affinity = info.get("cpu_affinity_cores")
    if affinity is not None and affinity < host:
        out.append(
            f"process is pinned to {affinity} of {host} cores "
            f"(taskset/cpuset); results reflect the pinned subset")
    mem = info.get("memory_limit_bytes")
    host_ram = info.get("host_ram_bytes")
    if mem and host_ram and mem < host_ram:
        out.append(
            f"memory is capped at {mem / (1024 ** 3):.1f} GB of "
            f"{host_ram / (1024 ** 3):.1f} GB; memory tests are sized to the cap")
    if info.get("ci"):
        out.append(
            f"running on {info['ci']} — shared CI hardware varies run to run; "
            f"compare against this runner's own history, not absolute figures")
    if info.get("cloud"):
        out.append(
            f"{info['cloud']} instance — burstable instance types deliver "
            f"above-baseline speed only until CPU credits are exhausted")
    return out
