"""SSD lifetime and wear: how much life the drive has left.

A benchmark says how fast storage is today. It says nothing about how long it
will keep working, and for an SSD that is the more consequential question —
flash wears out by writing, and a drive at 90% of its rated endurance is on a
clock regardless of how fast it benchmarks.

Six numbers answer it, and every one is already recorded by the drive itself:

* **Percentage used** — the controller's own wear estimate, where 100% means
  the rated endurance is exhausted. This is the headline.
* **Total written (TBW)** — how much has been written over the drive's life.
  Endurance ratings are quoted in these terms, so it is the figure to compare
  against a datasheet.
* **Total read** — no wear cost, but it establishes the read/write ratio.
* **Power-on hours** and **power cycles** — age, and how the drive is used.
* **Temperature** — sustained heat shortens flash life independently of writes.

Also collected, and arguably more urgent than any of the above: **media
errors**, **unsafe shutdowns**, and the **available spare** block pool. A
drive burning through spare blocks is failing now, not eventually.

**Where the data comes from.** Every platform stores it, and none of them make
it easy in the same way:

* **Linux** — ``/sys/class/nvme`` where present, then ``nvme smart-log``, then
  ``smartctl``. SATA SSDs report the same facts under different vendor-specific
  attribute names, which are mapped here.
* **macOS** — no command-line tool exposes it. ``system_profiler`` reports only
  a pass/fail status and the IORegistry carries none of the counters, so a
  small IOKit helper (``smart_engine.c``) reads the NVMe log page directly.
  It needs no privileges.
* **Windows** — ``Get-StorageReliabilityCounter`` via PowerShell.

Everything here is strictly read-only. Nothing writes to a drive, triggers a
self-test, or clears a log.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess

#: An NVMe "data unit" is 1000 x 512 bytes, fixed by the specification.
DATA_UNIT_BYTES = 512_000

SOURCE_NAME = "smart_engine.c"
BINARY_NAME = "smart_engine"

#: Wear at which the drive is worth planning to replace. The controller's own
#: estimate; drives normally keep working past 100% but leave the warranty and
#: their error rate climbs.
WEAR_WARN_PCT = 80
WEAR_CRITICAL_PCT = 95

#: Available spare blocks. Falling below the drive's own threshold is the
#: standard "failing now" signal, not a projection.
SPARE_WARN_PCT = 20

#: Sustained flash temperature above which endurance degrades measurably.
TEMP_WARN_C = 70


def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        # smartctl uses bit-flagged exit codes and still prints valid output on
        # a non-zero status, so stdout is returned regardless.
        return p.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# macOS — IOKit helper
# --------------------------------------------------------------------------- #
def _build_engine(script_dir: str) -> str | None:
    """Compile the IOKit SMART helper if missing or stale."""
    src = os.path.join(script_dir, SOURCE_NAME)
    if not os.path.isfile(src):
        return None
    exe = os.path.join(script_dir, BINARY_NAME)
    if (os.path.isfile(exe)
            and os.path.getmtime(exe) >= os.path.getmtime(src)):
        return exe
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        return None
    proc = subprocess.run(
        [cc, "-O2", src, "-o", exe,
         "-framework", "IOKit", "-framework", "CoreFoundation"],
        capture_output=True, text=True)
    return exe if proc.returncode == 0 else None


def _macos(script_dir: str) -> dict:
    exe = _build_engine(script_dir)
    if not exe:
        return {"available": False,
                "reason": "the SMART helper could not be built",
                "hint": "install the Xcode command line tools: "
                        "xcode-select --install"}
    raw = _run([exe, "--json"])
    if not raw:
        return {"available": False,
                "reason": "the SMART helper returned nothing"}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"available": False,
                "reason": "the SMART helper produced unreadable output"}
    if data.get("error") or not data.get("drives"):
        return {"available": False,
                "reason": data.get("error", "no drive exposed a SMART log")}
    return {"available": True, "source": "IOKit NVMe SMART log",
            "drives": [_normalise(d) for d in data["drives"]]}


# --------------------------------------------------------------------------- #
# Linux
# --------------------------------------------------------------------------- #
def _linux() -> dict:
    drives = [d for d in (_linux_sysfs() or []) if d]
    if drives:
        return {"available": True, "source": "sysfs + nvme smart-log",
                "drives": drives}

    smart = _smartctl()
    if smart:
        return {"available": True, "source": "smartctl", "drives": smart}

    return {"available": False,
            "reason": "no drive lifetime data could be read",
            "hint": ("install nvme-cli or smartmontools "
                     "(apt install nvme-cli smartmontools); reading the SMART "
                     "log usually needs root")}


def _linux_sysfs() -> list[dict]:
    """Model and capacity from sysfs, wear counters from ``nvme smart-log``."""
    base = "/sys/class/nvme"
    if not os.path.isdir(base):
        return []
    out = []
    try:
        controllers = sorted(os.listdir(base))
    except OSError:
        return []
    for name in controllers:
        entry: dict = {
            "device": f"/dev/{name}",
            "model": _read(os.path.join(base, name, "model")) or None,
            "firmware": _read(os.path.join(base, name, "firmware_rev")) or None,
            "protocol": "NVMe",
        }
        log = _nvme_smart_log(f"/dev/{name}")
        if log:
            entry.update(log)
            out.append(_normalise(entry))
    return out


def _nvme_smart_log(device: str) -> dict:
    """Parse ``nvme smart-log -o json``, which needs nvme-cli and root."""
    if not shutil.which("nvme"):
        return {}
    raw = _run(["nvme", "smart-log", device, "-o", "json"])
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    if not data:
        return {}
    kelvin = data.get("temperature")
    return {
        "critical_warning": data.get("critical_warning"),
        # nvme-cli reports temperature in Kelvin.
        "temperature_c": (round(kelvin - 273) if isinstance(kelvin, (int, float))
                          and kelvin > 200 else kelvin),
        "available_spare_pct": data.get("avail_spare"),
        "available_spare_threshold_pct": data.get("spare_thresh"),
        "percentage_used": data.get("percent_used"),
        "data_units_read": data.get("data_units_read"),
        "data_units_written": data.get("data_units_written"),
        "power_cycles": data.get("power_cycles"),
        "power_on_hours": data.get("power_on_hours"),
        "unsafe_shutdowns": data.get("unsafe_shutdowns"),
        "media_errors": data.get("media_errors"),
        "error_log_entries": data.get("num_err_log_entries"),
    }


#: SATA SSDs report the same facts under vendor-specific attribute names.
_SATA_ATTRS = {
    "power_on_hours": ("power_on_hours",),
    "power_cycles": ("power_cycle_count",),
    "percentage_used": ("percent_lifetime_remain", "ssd_life_left",
                        "media_wearout_indicator", "wear_leveling_count"),
    "reallocated_sectors": ("reallocated_sector_ct",),
    "lbas_written": ("total_lbas_written", "host_writes_32mib",
                     "lifetime_writes_gib"),
    "lbas_read": ("total_lbas_read", "host_reads_32mib"),
    "temperature_c": ("temperature_celsius", "airflow_temperature_cel"),
}


def _smartctl() -> list[dict]:
    """Read every drive smartctl can see, NVMe and SATA alike."""
    if not shutil.which("smartctl"):
        return []
    scan = _run(["smartctl", "--scan"])
    devices = [line.split()[0] for line in scan.splitlines()
               if line.startswith("/dev/")]
    drives = []
    for device in devices[:8]:
        raw = _run(["smartctl", "-a", "-j", device], timeout=15)
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            continue
        if not data:
            continue
        entry = _from_smartctl_json(data)
        if entry:
            entry["device"] = device
            drives.append(_normalise(entry))
    return drives


def _from_smartctl_json(data: dict) -> dict:
    """Map smartctl's JSON onto the common shape, NVMe or SATA."""
    entry: dict = {
        "model": data.get("model_name"),
        "firmware": data.get("firmware_version"),
        "smart_status": ("passed" if (data.get("smart_status") or {}).get("passed")
                         else "FAILED" if "smart_status" in data else None),
    }
    temp = (data.get("temperature") or {}).get("current")
    if temp is not None:
        entry["temperature_c"] = temp

    nvme = data.get("nvme_smart_health_information_log")
    if nvme:
        entry.update({
            "protocol": "NVMe",
            "critical_warning": nvme.get("critical_warning"),
            "available_spare_pct": nvme.get("available_spare"),
            "available_spare_threshold_pct": nvme.get("available_spare_threshold"),
            "percentage_used": nvme.get("percentage_used"),
            "data_units_read": nvme.get("data_units_read"),
            "data_units_written": nvme.get("data_units_written"),
            "power_cycles": nvme.get("power_cycles"),
            "power_on_hours": nvme.get("power_on_hours"),
            "unsafe_shutdowns": nvme.get("unsafe_shutdowns"),
            "media_errors": nvme.get("media_errors"),
            "error_log_entries": nvme.get("num_err_log_entries"),
        })
        return entry

    # SATA: walk the attribute table, matching on name.
    table = ((data.get("ata_smart_attributes") or {}).get("table") or [])
    if not table:
        return entry if entry.get("model") else {}
    entry["protocol"] = "SATA"
    by_name = {(a.get("name") or "").lower(): a for a in table}
    sector_size = (data.get("logical_block_size")
                   or data.get("user_capacity", {}).get("blocks") and 512
                   or 512)
    for field, names in _SATA_ATTRS.items():
        for name in names:
            attr = by_name.get(name)
            if not attr:
                continue
            value = (attr.get("raw") or {}).get("value")
            if value is None:
                continue
            if field == "percentage_used":
                # These attributes report life *remaining* as a normalised
                # value, which is the complement of what NVMe reports.
                normalised = attr.get("value")
                if isinstance(normalised, int):
                    entry["percentage_used"] = max(0, 100 - normalised)
            elif field == "lbas_written":
                entry["data_units_written"] = (
                    value * sector_size / DATA_UNIT_BYTES)
            elif field == "lbas_read":
                entry["data_units_read"] = value * sector_size / DATA_UNIT_BYTES
            else:
                entry[field] = value
            break
    return entry


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def _windows() -> dict:
    script = (
        "$out=@(); "
        "Get-PhysicalDisk | ForEach-Object { "
        "  $d=$_; $c=$null; "
        "  try { $c = $d | Get-StorageReliabilityCounter } catch {} "
        "  $out += [pscustomobject]@{ "
        "    model=$d.FriendlyName; media=$d.MediaType; health=$d.HealthStatus; "
        "    wear=$c.Wear; temperature_c=$c.Temperature; "
        "    power_on_hours=$c.PowerOnHours; "
        "    read_errors=$c.ReadErrorsTotal; write_errors=$c.WriteErrorsTotal } "
        "}; $out | ConvertTo-Json -Compress"
    )
    raw = _run(["powershell", "-NoProfile", "-Command", script], timeout=25)
    try:
        data = json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError:
        return {"available": False,
                "reason": "Get-StorageReliabilityCounter returned no usable data"}
    if not data:
        return {"available": False,
                "reason": "no physical disk reported reliability counters"}
    if isinstance(data, dict):
        data = [data]

    drives = []
    for item in data:
        entry = {
            "model": item.get("model"),
            "protocol": item.get("media"),
            "smart_status": item.get("health"),
            "percentage_used": item.get("wear"),
            "temperature_c": item.get("temperature_c"),
            "power_on_hours": item.get("power_on_hours"),
            "read_errors": item.get("read_errors"),
            "write_errors": item.get("write_errors"),
        }
        if any(v is not None for v in entry.values()):
            drives.append(_normalise(entry))
    if not drives:
        return {"available": False,
                "reason": "reliability counters were empty (many consumer "
                          "drives and USB enclosures do not expose them)"}
    return {"available": True, "source": "Get-StorageReliabilityCounter",
            "drives": drives}


# --------------------------------------------------------------------------- #
# Common shape and interpretation
# --------------------------------------------------------------------------- #
def _normalise(entry: dict) -> dict:
    """Derive the human-facing figures from whatever the platform supplied."""
    out = {k: v for k, v in entry.items() if v is not None}

    written = out.get("data_units_written")
    if isinstance(written, (int, float)):
        out["written_bytes"] = int(written * DATA_UNIT_BYTES)
        out["written_tb"] = round(written * DATA_UNIT_BYTES / 1e12, 2)
    read = out.get("data_units_read")
    if isinstance(read, (int, float)):
        out["read_bytes"] = int(read * DATA_UNIT_BYTES)
        out["read_tb"] = round(read * DATA_UNIT_BYTES / 1e12, 2)

    used = out.get("percentage_used")
    if isinstance(used, (int, float)):
        # The controller's estimate can exceed 100 once endurance is spent;
        # health is clamped so it never reads as negative.
        out["health_pct"] = max(0, 100 - int(used))

    hours = out.get("power_on_hours")
    if isinstance(hours, (int, float)) and hours > 0:
        out["power_on_days"] = round(hours / 24.0, 1)
        if isinstance(out.get("written_bytes"), int):
            # Per *power-on* day. A machine that sleeps most of the time has
            # far fewer power-on days than calendar days, so labelling this
            # "per day" would overstate the daily write load several-fold.
            out["write_rate_gb_per_power_on_day"] = round(
                out["written_bytes"] / 1e9 / (hours / 24.0), 2)
    return out


def project_lifetime(drive: dict) -> dict | None:
    """Estimate remaining life from the drive's own wear rate.

    Deliberately based on observed wear per hour rather than on a datasheet
    endurance figure, because the rating is not readable from the drive and
    varies by an order of magnitude between models. Extrapolating the drive's
    own history answers the practical question — "at the rate I actually use
    this, how long until it is spent?" — without needing to know the rating.

    Returns None when there is too little history to extrapolate honestly.
    """
    used = drive.get("percentage_used")
    hours = drive.get("power_on_hours")
    if not isinstance(used, (int, float)) or not isinstance(hours, (int, float)):
        return None
    if hours < 100:
        return {"note": "too few power-on hours to project wear reliably"}
    if used <= 0:
        return {"note": ("no measurable wear yet, so no meaningful projection "
                         "— at this rate the drive will outlast the machine")}

    hours_per_pct = hours / used
    remaining_hours = hours_per_pct * (100 - used)

    # Deliberately NOT converted to calendar years by dividing by 8760. SMART
    # counts power-on hours, not wall-clock time, and it carries no
    # manufacture date, so the drive's duty cycle is unknowable from the log.
    # A laptop that accumulates 854 power-on hours over a year of ownership
    # runs at roughly a tenth of 24/7, and dividing by 8760 would understate
    # its calendar life by that same factor. Calendar figures are given
    # instead against explicit daily-use assumptions the reader can match to
    # their own habits.
    calendar = {
        f"at_{h}h_per_day": round(remaining_hours / (h * 365.0), 1)
        for h in (4, 8, 24)
    }
    return {
        "wear_pct_per_1000_hours": round(1000.0 * used / hours, 2),
        "projected_remaining_hours": int(remaining_hours),
        "projected_remaining_years": calendar,
        "basis": (f"{used}% wear over {int(hours):,} power-on hours, "
                  f"extrapolated at the same rate"),
    }


def warnings(result: dict) -> list[str]:
    """Findings that call for action, most urgent first."""
    notes: list[str] = []
    for drive in result.get("drives", []):
        label = drive.get("model") or drive.get("device") or "drive"

        if drive.get("critical_warning"):
            notes.append(
                f"{label}: the controller has raised a CRITICAL WARNING flag "
                f"(0x{int(drive['critical_warning']):02x}) — back up now and "
                f"read the full SMART log")

        media = drive.get("media_errors")
        if isinstance(media, (int, float)) and media > 0:
            notes.append(
                f"{label}: {int(media):,} media/data-integrity error(s) — the "
                f"drive has failed to return data it stored. Back up and plan "
                f"replacement")

        spare = drive.get("available_spare_pct")
        threshold = drive.get("available_spare_threshold_pct")
        if isinstance(spare, (int, float)):
            limit = threshold if isinstance(threshold, (int, float)) else SPARE_WARN_PCT
            if spare < limit:
                notes.append(
                    f"{label}: spare blocks are down to {spare}%, below the "
                    f"drive's own {limit}% threshold — this is a failing "
                    f"drive, not a worn one")

        used = drive.get("percentage_used")
        if isinstance(used, (int, float)):
            if used >= WEAR_CRITICAL_PCT:
                notes.append(
                    f"{label}: {used}% of rated endurance used — at or past "
                    f"end of rated life; replace it")
            elif used >= WEAR_WARN_PCT:
                notes.append(
                    f"{label}: {used}% of rated endurance used — plan a "
                    f"replacement")

        temp = drive.get("temperature_c")
        if isinstance(temp, (int, float)) and temp >= TEMP_WARN_C:
            notes.append(
                f"{label}: running at {temp} °C — sustained heat shortens "
                f"flash life independently of writes; check airflow")

        unsafe = drive.get("unsafe_shutdowns")
        cycles = drive.get("power_cycles")
        if (isinstance(unsafe, (int, float)) and isinstance(cycles, (int, float))
                and cycles > 20 and unsafe > cycles * 0.25):
            notes.append(
                f"{label}: {int(unsafe):,} unsafe shutdowns out of "
                f"{int(cycles):,} power cycles — the drive is losing power "
                f"without being told to flush, which risks data loss")
    return notes


def run(script_dir: str = ".") -> dict:
    """Read drive lifetime data on whatever platform this is."""
    system = platform.system()
    if system == "Darwin":
        result = _macos(script_dir)
    elif system == "Linux":
        result = _linux()
    elif system == "Windows":
        result = _windows()
    else:
        result = {"available": False,
                  "reason": f"not implemented on {system}"}

    for drive in result.get("drives", []):
        projection = project_lifetime(drive)
        if projection:
            drive["projection"] = projection
    result["warnings"] = warnings(result)
    return result


def render(result: dict | None) -> str:
    """Terminal block, shaped like the summary a drive utility shows."""
    if not result:
        return ""
    if not result.get("available"):
        lines = [f"  unavailable — {result.get('reason', 'unknown')}"]
        if result.get("hint"):
            lines.append(f"  {result['hint']}")
        return "\n".join(lines)

    lines = []
    for drive in result.get("drives", []):
        title = drive.get("model") or drive.get("device") or "Drive"
        proto = drive.get("protocol")
        lines.append(f"  {title}{f' ({proto})' if proto else ''}")

        def row(label: str, value) -> None:
            if value is not None:
                lines.append(f"    {label:<20}: {value}")

        row("Status", drive.get("smart_status")
            or ("Verified" if not drive.get("critical_warning") else "WARNING"))
        if drive.get("written_tb") is not None:
            row("Total written", f"{drive['written_tb']:,.2f} TB")
        if drive.get("read_tb") is not None:
            row("Total read", f"{drive['read_tb']:,.2f} TB")
        if drive.get("temperature_c") is not None:
            row("Temperature", f"{drive['temperature_c']} °C")
        if drive.get("health_pct") is not None:
            row("Health", f"{drive['health_pct']}% "
                          f"({drive.get('percentage_used')}% of rated life used)")
        row("Power cycles", f"{int(drive['power_cycles']):,}"
            if isinstance(drive.get("power_cycles"), (int, float)) else None)
        if isinstance(drive.get("power_on_hours"), (int, float)):
            row("Power on hours", f"{int(drive['power_on_hours']):,} "
                                  f"({drive.get('power_on_days', 0):,.1f} days)")
        if drive.get("available_spare_pct") is not None:
            row("Spare blocks", f"{drive['available_spare_pct']}% "
                                f"(threshold "
                                f"{drive.get('available_spare_threshold_pct', '?')}%)")
        row("Unsafe shutdowns", drive.get("unsafe_shutdowns"))
        row("Media errors", drive.get("media_errors"))
        if drive.get("write_rate_gb_per_power_on_day") is not None:
            row("Write rate",
                f"{drive['write_rate_gb_per_power_on_day']:,.1f} GB "
                f"per power-on day")

        projection = drive.get("projection") or {}
        if projection.get("projected_remaining_hours") is not None:
            years = projection.get("projected_remaining_years") or {}
            lines.append(
                f"    {'Projected life left':<20}: "
                f"~{projection['projected_remaining_hours']:,} more power-on "
                f"hours")
            if years:
                lines.append(
                    f"      which is ~{years.get('at_4h_per_day')} years at "
                    f"4h/day, ~{years.get('at_8h_per_day')} at 8h/day, "
                    f"~{years.get('at_24h_per_day')} running continuously")
            lines.append(f"      ({projection['basis']})")
        elif projection.get("note"):
            lines.append(f"    {'Projection':<20}: {projection['note']}")
        lines.append("")

    for note in result.get("warnings", []):
        lines.append(f"  !  {note}")
    return "\n".join(lines).rstrip()
