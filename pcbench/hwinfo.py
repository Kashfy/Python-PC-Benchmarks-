"""Hardware facts on demand, without running a benchmark.

Most of what this tool knows about a machine is gathered in seconds and needs
no load at all — battery wear, SSD endurance, cache sizes, thermal sensors,
GPU inventory, kernel settings. Until now that information only appeared as a
side effect of a full benchmark run, which takes minutes and heats the machine
up. If the question is "how worn is this battery?" that is an absurd way to
answer it.

Every section here is read-only, takes well under a second unless noted, and
can be requested individually::

    pcbench --stats                    # everything
    pcbench --stats battery,drives     # just those two
    pcbench --list-stats               # what is available

The same functions back the benchmark report, so a figure shown here is the
same figure shown there — there is no second implementation to drift.
"""

from __future__ import annotations

import os
import platform
import shutil

MB = 1024 * 1024
GB = 1024 * MB


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def cpu_section() -> dict:
    from . import system as system_mod

    info = system_mod.inventory()
    cache_bytes, cache_source = system_mod.last_level_cache_bytes()
    return {
        "model": info.get("cpu_model"),
        "architecture": info.get("architecture"),
        "arch_family": info.get("arch_family"),
        "cores_physical": info.get("cpu_cores_physical"),
        "cores_logical": info.get("cpu_cores_logical"),
        "base_mhz": info.get("cpu_base_mhz"),
        "features": info.get("cpu_features"),
        "last_level_cache_bytes": cache_bytes,
        "last_level_cache_source": cache_source,
        "virtualization": info.get("virtualization"),
        "byte_order": info.get("byte_order"),
    }


def memory_section() -> dict:
    from . import system as system_mod

    total = system_mod.total_ram_bytes()
    out: dict = {"total_bytes": total,
                 "total_gb": round(total / GB, 2) if total else None}
    try:
        import psutil
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        out.update({
            "available_bytes": virtual.available,
            "available_gb": round(virtual.available / GB, 2),
            "used_percent": virtual.percent,
            "swap_total_gb": round(swap.total / GB, 2),
            "swap_used_percent": swap.percent,
        })
    except Exception:
        out["note"] = "install psutil for live usage and swap figures"
    return out


def battery_section() -> dict:
    """Wear, cycles, and current charge.

    Two different questions, answered from two sources: long-term health
    (cycle count against design capacity, from the platform's battery
    controller) and the state right now (charge, whether it is plugged in,
    estimated time remaining, via psutil).
    """
    from . import thermal as thermal_mod

    health = thermal_mod.battery_health() or {}
    out = dict(health)

    try:
        import psutil
        live = psutil.sensors_battery()
        if live is not None:
            out["charge_percent"] = round(live.percent, 1)
            out["plugged_in"] = bool(live.power_plugged)
            seconds = live.secsleft
            if isinstance(seconds, int) and seconds >= 0:
                out["time_left_minutes"] = seconds // 60
    except Exception:
        pass

    if not out:
        return {"present": False,
                "note": "no battery detected (desktop, server, or the "
                        "platform does not expose one)"}
    out["present"] = True

    # A cycle count is meaningless without knowing what is normal for the
    # hardware, so the interpretation is stated rather than left implied.
    cycles = out.get("cycle_count")
    health_pct = out.get("health_percent")
    notes = []
    if isinstance(health_pct, (int, float)):
        if health_pct < 80:
            notes.append(
                f"capacity is {health_pct:.0f}% of design — most vendors treat "
                f"80% as the service threshold, so this battery is at or past "
                f"the point where replacement is normally offered")
        elif health_pct < 90:
            notes.append(f"capacity is {health_pct:.0f}% of design — normal "
                         f"wear, no action needed")
    if isinstance(cycles, int) and cycles > 800:
        notes.append(f"{cycles} charge cycles — most laptop batteries are "
                     f"rated for 1000, after which capacity falls faster")
    if notes:
        out["notes"] = notes
    return out


def storage_section() -> dict:
    from . import storage as storage_mod

    inventory = storage_mod.inventory(64)
    return {
        "devices": [
            {k: v for k, v in device.items()
             if k in ("mount", "device", "fstype", "kind", "total_bytes",
                      "free_bytes", "used_pct", "benchmarkable",
                      "skip_reason")}
            for device in inventory.get("devices", [])
        ],
        "count": len(inventory.get("devices", [])),
    }


def drives_section(script_dir: str = ".") -> dict:
    """SSD endurance and wear. See :mod:`pcbench.drivelife`."""
    from . import drivelife

    return drivelife.run(script_dir)


def gpu_section() -> dict:
    from . import accel as accel_mod
    from . import gpucompute

    from . import system as system_mod
    inventory = accel_mod.inventory(system_mod.cpu_model())
    out = {
        "gpus": inventory.get("gpus", []),
        "npus": inventory.get("npus", []),
    }
    opencl = gpucompute.devices()
    if opencl:
        out["opencl_devices"] = opencl
    torch_device = gpucompute.torch_device()
    if torch_device:
        out["torch_device"] = {"device": torch_device[0],
                               "vendor": torch_device[1]}
    return out


def thermal_section(script_dir: str = ".") -> dict:
    from . import thermal as thermal_mod
    from . import system as system_mod

    temps = thermal_mod.read(script_dir) or {}
    out = dict(temps)
    pressure = system_mod.thermal_pressure()
    if pressure:
        out["thermal_pressure"] = pressure
    if not out:
        out["note"] = ("no readable temperature sensors on this platform "
                       "or configuration")
    return out


def os_section() -> dict:
    from . import provenance
    from . import system as system_mod

    info = system_mod.inventory()
    collected = provenance.collect()
    return {
        "os": info.get("os"),
        "release": info.get("os_release"),
        "platform": info.get("platform"),
        "python_version": info.get("python_version"),
        "python_implementation": info.get("python_implementation"),
        "free_threaded_build": info.get("free_threaded_build"),
        "gil_enabled": info.get("gil_enabled"),
        "provenance": collected,
        "provenance_notes": provenance.notes(collected),
    }


def environment_section() -> dict:
    from . import container
    from . import system as system_mod

    info = system_mod.inventory()
    detected = container.detect(info.get("cpu_cores_logical"),
                                info.get("ram_total_bytes", 0))
    detected["warnings"] = container.warnings(detected)
    return detected


def numa_section() -> dict:
    from . import numa

    # Topology only: the bandwidth matrix needs numactl and takes seconds.
    return numa.topology()


def packages_section() -> dict:
    from . import optional

    return optional.status()


def power_section() -> dict:
    from . import power as power_mod
    from . import system as system_mod

    info = system_mod.inventory()
    # Idle draw, deliberately: raising a load to measure it is what the
    # benchmark does, and this mode exists to avoid that.
    reading = power_mod.measure(info.get("cpu_model", ""))
    reading["on_ac_power"] = system_mod.on_ac_power()
    reading["note"] = ("idle draw; run the benchmark for a figure under load")
    return reading


#: Every section, in report order. Kept as a registry so the CLI, the
#: ``--list-stats`` output and the tests all read from one place.
SECTIONS = {
    "cpu": ("Processor", cpu_section),
    "memory": ("Memory", memory_section),
    "storage": ("Storage devices", storage_section),
    "drives": ("Drive lifetime & wear", drives_section),
    "battery": ("Battery", battery_section),
    "gpu": ("GPU / NPU", gpu_section),
    "thermal": ("Temperatures", thermal_section),
    "power": ("Power", power_section),
    "os": ("Operating system & configuration", os_section),
    "environment": ("Execution environment", environment_section),
    "numa": ("NUMA topology", numa_section),
    "packages": ("Optional packages", packages_section),
}

#: Sections needing a script directory (to build or find a native helper).
_NEEDS_DIR = {"drives", "thermal"}


def available_sections() -> list[str]:
    return list(SECTIONS)


def parse_sections(text: str) -> list[str]:
    """Turn ``"battery,drives"`` into a validated list. ``""`` means all."""
    if not text or text.strip().lower() == "all":
        return list(SECTIONS)
    wanted = [s.strip().lower() for s in text.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        raise ValueError(
            f"unknown stats section(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(SECTIONS)}")
    return wanted


def collect(sections: list[str] | None = None,
            script_dir: str = ".") -> dict:
    """Gather the requested sections. Never raises for one bad section."""
    chosen = sections or list(SECTIONS)
    out: dict = {}
    for name in chosen:
        label, func = SECTIONS[name]
        try:
            out[name] = (func(script_dir) if name in _NEEDS_DIR else func())
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_bytes(value) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "?"
    if value >= GB:
        return f"{value / GB:.1f} GB"
    return f"{value / MB:.0f} MB"


def render(data: dict, script_dir: str = ".") -> str:
    """Human-readable report for whichever sections were collected."""
    from . import drivelife
    from . import numa as numa_mod
    from . import provenance
    from . import container

    lines: list[str] = []

    def head(title: str) -> None:
        lines.append("")
        lines.append("=" * 74)
        lines.append(title)
        lines.append("=" * 74)

    def row(label: str, value) -> None:
        if value not in (None, "", [], {}):
            lines.append(f"  {label:<26}: {value}")

    for name, section in data.items():
        label = SECTIONS[name][0]
        if isinstance(section, dict) and section.get("error"):
            head(label)
            lines.append(f"  unavailable — {section['error']}")
            continue

        if name == "cpu":
            head(label)
            row("Model", section.get("model"))
            row("Architecture", f"{section.get('arch_family')} "
                                f"({section.get('architecture')})")
            row("Cores", f"{section.get('cores_physical')} physical / "
                         f"{section.get('cores_logical')} logical")
            row("Base clock", f"{section['base_mhz']:.0f} MHz"
                if section.get("base_mhz") else None)
            cache = section.get("last_level_cache_bytes")
            row("Last-level cache",
                f"{_fmt_bytes(cache)}  ({section.get('last_level_cache_source')})"
                if cache else None)
            row("Features", ", ".join(section.get("features") or []) or None)
            row("Virtualisation", section.get("virtualization"))

        elif name == "memory":
            head(label)
            row("Total", f"{section.get('total_gb')} GB")
            if section.get("available_gb") is not None:
                row("Available", f"{section['available_gb']} GB "
                                 f"({100 - section.get('used_percent', 0):.0f}% free)")
                row("Swap", f"{section.get('swap_total_gb')} GB "
                            f"({section.get('swap_used_percent')}% used)")
            row("Note", section.get("note"))

        elif name == "storage":
            head(label)
            for device in section.get("devices", []):
                free = _fmt_bytes(device.get("free_bytes"))
                total = _fmt_bytes(device.get("total_bytes"))
                lines.append(f"  {device['mount'][:30]:<30} "
                             f"{(device.get('kind') or '?')[:18]:<18} "
                             f"{free:>9} free of {total:>9}")

        elif name == "drives":
            head(label)
            lines.append(drivelife.render(section))

        elif name == "battery":
            head(label)
            if not section.get("present"):
                lines.append(f"  {section.get('note', 'no battery')}")
            else:
                if section.get("charge_percent") is not None:
                    state = ("charging / on AC" if section.get("plugged_in")
                             else "on battery")
                    row("Charge", f"{section['charge_percent']:.0f}%  ({state})")
                if section.get("time_left_minutes") is not None:
                    minutes = section["time_left_minutes"]
                    row("Estimated remaining",
                        f"{minutes // 60}h {minutes % 60:02d}m")
                if section.get("health_percent") is not None:
                    row("Health", f"{section['health_percent']:.1f}% of design "
                                  f"capacity")
                if section.get("design_capacity_mah"):
                    row("Capacity", f"{section.get('current_capacity_mah')} of "
                                    f"{section['design_capacity_mah']} mAh")
                row("Charge cycles", section.get("cycle_count"))
                row("Temperature", f"{section['celsius']} °C"
                    if section.get("celsius") else None)
                for note in section.get("notes", []):
                    lines.append(f"      i {note}")

        elif name == "gpu":
            head(label)
            for gpu in section.get("gpus", []):
                # The name must appear even when VRAM is unknown; `row` skips
                # empty values, which silently dropped the GPU entirely.
                detail = (f"{gpu['vram_mb']:,.0f} MB" if gpu.get("vram_mb")
                          else gpu.get("vram_note") or "VRAM not reported")
                if gpu.get("driver"):
                    detail += f", driver {gpu['driver']}"
                lines.append(f"  {'GPU':<26}: {gpu.get('name', '?')}  "
                             f"({detail})")
            for npu in section.get("npus", []):
                row("NPU", npu.get("name"))
            for device in section.get("opencl_devices", []):
                row(f"OpenCL: {device.get('name')}",
                    f"{device.get('class', '?')}, "
                    f"{device.get('compute_units')} CUs, "
                    f"{device.get('global_mem_mb', 0):,} MB")
            if section.get("torch_device"):
                row("PyTorch device",
                    f"{section['torch_device']['device']} "
                    f"({section['torch_device']['vendor']})")

        elif name == "thermal":
            head(label)
            for key, value in section.items():
                if isinstance(value, (int, float)):
                    row(key.replace("_", " ").title(), value)
            row("Note", section.get("note"))

        elif name == "power":
            head(label)
            row("Source", "AC" if section.get("on_ac_power") else "battery"
                if section.get("on_ac_power") is not None else None)
            if section.get("package_w"):
                row("Package draw", f"{section['package_w']:.1f} W"
                                    f"{' (estimated)' if section.get('estimated') else ''}")
            row("Source of figure", section.get("source"))
            row("Note", section.get("note"))

        elif name == "os":
            head(label)
            row("OS", f"{section.get('os')} {section.get('release')}")
            row("Python", f"{section.get('python_implementation')} "
                          f"{section.get('python_version')}"
                          f"{' (free-threaded)' if section.get('free_threaded_build') else ''}")
            body = provenance.render(section.get("provenance") or {},
                                     section.get("provenance_notes"))
            if body.strip():
                lines.append(body)

        elif name == "environment":
            head(label)
            row("Container", section.get("container") or "none")
            row("Cloud", section.get("cloud"))
            row("CI system", section.get("ci"))
            row("Effective cores", section.get("effective_cores"))
            if section.get("memory_limit_bytes"):
                row("Memory limit", _fmt_bytes(section["memory_limit_bytes"]))
            for warning in section.get("warnings", []):
                lines.append(f"      i {warning}")

        elif name == "numa":
            head(label)
            lines.append(numa_mod.render({"topology": section}, []))

        elif name == "packages":
            head(label)
            for tier, state in (section.get("tiers") or {}).items():
                installed = state.get("installed", [])
                missing = state.get("missing", [])
                mark = "complete" if not missing else f"{len(installed)} of " \
                    f"{len(installed) + len(missing)}"
                row(tier, mark + (f"  (missing: {', '.join(missing)})"
                                  if missing else ""))

    return "\n".join(lines).strip()
