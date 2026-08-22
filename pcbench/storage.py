"""Storage device enumeration and per-device benchmarking.

The single-path disk test answers "how fast is the drive I happen to be sitting
on?", which is the wrong question on most real machines. A laptop has an
internal NVMe and an external USB SSD. A workstation has a fast scratch disk and
a slow archive array. A server has a boot device and a data volume whose
performance is the only one that matters. A CI runner has an overlayfs whose
write path is nothing like the underlying disk.

This module enumerates what is actually mounted, tells the user what kind of
device each mount point is backed by, and lets the disk benchmark run against
any or all of them. It also refuses to benchmark things that would produce
meaningless numbers or cause harm:

* **tmpfs / ramfs** — writing there measures RAM and consumes it.
* **Network filesystems** (NFS, SMB, sshfs) — measures the network, and can be
  extremely slow to the point of appearing hung.
* **Read-only mounts** — nothing can be written at all.
* **Mounts without enough free space** — filling a filesystem endangers other
  processes' in-flight writes.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

MB = 1024 * 1024
GB = 1024 * MB

# Filesystems whose speed is not the storage device's speed.
_RAM_FS = {"tmpfs", "ramfs", "devtmpfs"}
_NETWORK_FS = {"nfs", "nfs4", "cifs", "smbfs", "smb", "afpfs", "fuse.sshfs",
               "sshfs", "webdav", "ftp", "9p", "afs", "glusterfs", "ceph"}
_PSEUDO_FS = {"proc", "sysfs", "devpts", "cgroup", "cgroup2", "debugfs",
              "securityfs", "pstore", "bpf", "tracefs", "configfs",
              "fusectl", "autofs", "squashfs", "overlay", "devfs"}


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout)
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# --------------------------------------------------------------------------- #
# Mount enumeration
# --------------------------------------------------------------------------- #
def _mounts_linux() -> list[dict]:
    out = []
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return out
    for line in lines:
        # mountinfo: ... <mountpoint> ... - <fstype> <source> <superopts>
        pre, _, post = line.partition(" - ")
        fields, tail = pre.split(), post.split()
        if len(fields) < 5 or len(tail) < 2:
            continue
        out.append({"mount": fields[4], "fstype": tail[0], "device": tail[1],
                    "options": fields[5] if len(fields) > 5 else ""})
    return out


def _mounts_posix() -> list[dict]:
    """macOS/BSD: parse ``mount`` output, e.g. ``/dev/disk3s5 on / (apfs, ...)``."""
    out = []
    for line in _run(["mount"]).splitlines():
        if " on " not in line:
            continue
        device, rest = line.split(" on ", 1)
        mount, _, opts = rest.partition(" (")
        opts = opts.rstrip(")")
        fstype = opts.split(",")[0].strip() if opts else "unknown"
        out.append({"mount": mount.strip(), "fstype": fstype,
                    "device": device.strip(), "options": opts})
    return out


def _mounts_windows() -> list[dict]:
    out = []
    try:
        import string
        import ctypes
        mask = ctypes.windll.kernel32.GetLogicalDrives()  # type: ignore[attr-defined]
    except Exception:
        return out
    for i, letter in enumerate(string.ascii_uppercase):
        if not (mask >> i) & 1:
            continue
        root = f"{letter}:\\"
        try:
            kind = ctypes.windll.kernel32.GetDriveTypeW(root)  # type: ignore[attr-defined]
        except Exception:
            kind = 0
        # 2=removable, 3=fixed, 4=network, 5=cdrom, 6=ramdisk
        fstype = {2: "removable", 3: "fixed", 4: "network",
                  5: "cdrom", 6: "ramdisk"}.get(kind, "unknown")
        out.append({"mount": root, "fstype": fstype, "device": root,
                    "options": ""})
    return out


def mounts() -> list[dict]:
    """Every mount point visible to this process, unfiltered."""
    system = platform.system()
    if system == "Linux":
        return _mounts_linux()
    if system == "Windows":
        return _mounts_windows()
    return _mounts_posix()


# --------------------------------------------------------------------------- #
# Device classification
# --------------------------------------------------------------------------- #
def _rotational(device: str) -> bool | None:
    """True for spinning rust, False for solid state, None when unknown.

    Rotational media is 100x slower at random I/O, so a random-read figure that
    looks alarming on an SSD is completely normal on a hard disk. Without this
    the diagnosis would be wrong.
    """
    if platform.system() != "Linux" or not device.startswith("/dev/"):
        return None
    name = os.path.basename(device)
    # Strip a partition suffix: sda1 -> sda, nvme0n1p2 -> nvme0n1.
    for base in (name.rstrip("0123456789"), name.split("p")[0], name):
        path = f"/sys/block/{base}/queue/rotational"
        try:
            with open(path, encoding="utf-8") as f:
                return f.read().strip() == "1"
        except OSError:
            continue
    return None


def classify(entry: dict) -> str:
    """Short human label for what kind of storage a mount is backed by."""
    fstype = (entry.get("fstype") or "").lower()
    if fstype in _RAM_FS or fstype == "ramdisk":
        return "RAM disk"
    if fstype in _NETWORK_FS or fstype == "network":
        return "network"
    if fstype == "overlay":
        return "overlay (container layer)"
    rot = _rotational(entry.get("device", ""))
    if rot is True:
        return "hard disk (rotational)"
    if rot is False:
        return "solid state"
    if "nvme" in (entry.get("device") or "").lower():
        return "NVMe solid state"
    return fstype or "unknown"


def usage(path: str) -> dict:
    try:
        total, used, free = shutil.disk_usage(path)
        return {"total_bytes": total, "free_bytes": free,
                "used_pct": round(100.0 * used / total, 1) if total else None}
    except OSError:
        return {}


def _writable(path: str) -> bool:
    return os.access(path, os.W_OK)


def benchmarkable(entry: dict, need_mb: int) -> tuple[bool, str | None]:
    """Whether this mount can be meaningfully and safely benchmarked."""
    fstype = (entry.get("fstype") or "").lower()
    mount = entry.get("mount", "")
    if fstype in _PSEUDO_FS and fstype != "overlay":
        return False, "pseudo filesystem"
    if fstype in _RAM_FS or fstype == "ramdisk":
        return False, "RAM-backed: would measure memory and consume it"
    if fstype in _NETWORK_FS or fstype == "network":
        return False, "network filesystem: would measure the network"
    if fstype in ("cdrom", "iso9660", "squashfs"):
        return False, "read-only medium"
    opts = entry.get("options", "")
    if opts.split(",")[0] == "ro" or ",ro," in f",{opts},":
        return False, "mounted read-only"
    if not os.path.isdir(mount):
        return False, "not a directory"
    if not _writable(mount):
        return False, "not writable by this user"
    u = usage(mount)
    free = u.get("free_bytes", 0)
    if free and free < need_mb * MB * 2:
        return False, (f"only {free / GB:.1f} GB free; "
                       f"needs at least {need_mb * 2 / 1024:.1f} GB headroom")
    return True, None


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def _uninteresting(mount: str) -> bool:
    """Mounts that exist for the OS's benefit and never for the user's.

    macOS splits the root filesystem across a dozen internal APFS volumes and
    Linux distributions mount a comparable pile of snap/flatpak loopbacks. All
    of them sit on the same physical device as a mount the user does care
    about, so listing them adds pages of noise and no information.
    """
    if mount == "/System/Volumes/Data":
        return False        # the real user data volume on macOS
    return mount.startswith((
        "/System/Volumes/", "/private/var/vm", "/snap/", "/var/snap/",
        "/var/lib/docker/", "/var/lib/snapd/", "/run/", "/sys/", "/proc/",
        "/dev/", "/boot/efi",
    ))


def inventory(need_mb: int = 256) -> dict:
    """Describe every mount, marking which ones a disk test can use.

    Duplicate mounts of the same device (bind mounts, APFS snapshots, the
    dozens of pseudo-mounts a modern Linux distribution creates) are collapsed
    so the list stays readable.
    """
    seen: set[str] = set()
    devices: list[dict] = []
    for entry in mounts():
        mount = entry.get("mount", "")
        fstype = (entry.get("fstype") or "").lower()
        if fstype in _PSEUDO_FS and fstype != "overlay":
            continue
        if _uninteresting(mount):
            continue
        key = f"{entry.get('device')}|{mount}"
        if key in seen:
            continue
        seen.add(key)
        ok, reason = benchmarkable(entry, need_mb)
        devices.append({
            "mount": mount,
            "device": entry.get("device"),
            "fstype": entry.get("fstype"),
            "kind": classify(entry),
            "benchmarkable": ok,
            "skip_reason": reason,
            **usage(mount),
        })
    devices.sort(key=lambda d: (not d["benchmarkable"], d["mount"]))
    return {"devices": devices,
            "benchmarkable_count": sum(1 for d in devices
                                       if d["benchmarkable"])}


def targets(inv: dict, requested: list[str] | None = None,
            all_devices: bool = False) -> list[dict]:
    """Choose which mounts to benchmark.

    An explicit list always wins, even for mounts the heuristics would skip:
    the user asking for a specific path knows something the heuristics do not
    (a deliberately benchmarked network share, for instance).
    """
    devices = inv.get("devices", [])
    if requested:
        chosen = []
        for want in requested:
            want_abs = os.path.abspath(os.path.expanduser(want))
            match = next((d for d in devices
                          if os.path.abspath(d["mount"]) == want_abs), None)
            chosen.append(match or {"mount": want_abs, "device": None,
                                    "fstype": None, "kind": "user-specified",
                                    "benchmarkable": True,
                                    "skip_reason": None, **usage(want_abs)})
        return chosen
    if all_devices:
        return [d for d in devices if d["benchmarkable"]]
    return []


def run(targets_list: list[dict], seconds: float, repeats: int,
        file_mb: int) -> dict:
    """Benchmark each target with the standard disk workload.

    Imported lazily so this module stays importable (for ``--list-devices``)
    on a machine where the workload module's dependencies are unavailable.
    """
    from . import workloads as wl

    out: dict = {"devices": []}
    for target in targets_list:
        mount = target["mount"]
        entry = {"mount": mount, "kind": target.get("kind"),
                 "device": target.get("device"),
                 "fstype": target.get("fstype")}
        # Write into a subdirectory so cleanup never touches user files.
        workdir = os.path.join(mount, ".pcbench")
        try:
            os.makedirs(workdir, exist_ok=True)
        except OSError as e:
            entry["skipped"] = True
            entry["reason"] = f"cannot create {workdir}: {e}"
            out["devices"].append(entry)
            continue
        try:
            entry["disk"] = wl.bench_disk(seconds, repeats, file_mb, workdir)
            entry["fsync"] = _fsync_for(workdir, seconds)
        except Exception as e:
            entry["skipped"] = True
            entry["reason"] = f"{type(e).__name__}: {e}"
        finally:
            try:
                os.rmdir(workdir)
            except OSError:
                pass
        out["devices"].append(entry)
    return out


def _fsync_for(workdir: str, seconds: float) -> dict:
    from . import apps
    return apps.bench_fsync(min(seconds, 2.0), workdir)
