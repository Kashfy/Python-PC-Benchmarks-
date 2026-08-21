# Hardware Safety

Short answer: **this tool cannot damage your hardware.** It loads components
hard — that is what a benchmark does, and CPUs, GPUs, and SSDs are designed to
run at 100% with their own protection — but it never touches the mechanisms
that could actually cause harm, and it caps the ones that could cause
*indirect* harm.

This page documents what was audited, what is guarded, and why.

## What the tool never does

These are the operations that could genuinely damage or destroy hardware or
data. **None of them appear anywhere in the codebase**, and this is verified by
inspection:

| Never done | Why it would be dangerous |
|------------|---------------------------|
| Write to a raw block device (`/dev/sda`, `\\.\PhysicalDrive0`) | Bypasses the filesystem; can destroy partitions and data |
| Format, partition, or run `mkfs`/`diskutil`/`fdisk` | Irreversible data loss |
| Modify firmware, BIOS, or SSD controller settings | Can permanently brick a device |
| Change voltage, clock multipliers, or power limits | The *only* software route to real physical damage |
| Write SMART attributes or issue vendor commands | Can corrupt drive metadata |
| Run with elevated privileges to alter system state | Not required; `sudo` is used only to *read* a power meter |

The tool writes exactly one kind of file: a scratch file created by
`tempfile.mkstemp()` inside a directory you chose, which it deletes when
finished. It never opens a device node.

## The four real risks, and how each is capped

A benchmark cannot damage hardware directly, but it *can* harm a system
indirectly — by exhausting a resource. Each is bounded.

### 1. Memory exhaustion → swap thrashing

Allocating more than physical RAM does not fail cleanly. The OS falls back to
swap, which freezes the machine and writes tens of gigabytes to the SSD as a
side effect; on Linux the OOM killer may terminate unrelated processes.

**Guard:** the memory-test buffer is clamped to **1/8 of physical RAM**. Since
the test allocates two buffers, the total stays under 1/4 of RAM. The C engine
applies the same rule to its own buffers and skips pointer-chase working sets
larger than 1/8 of RAM.

```
--mem-mb 999999 on a 16 GB machine
  → reduced to 2048 MB, with the reason printed
```

### 2. Filling the disk

A filesystem with zero free space can corrupt in-flight writes from *other*
applications and, on some filesystems, resist recovery.

**Guard:** the disk test requires **1.5× headroom** over the file it intends to
write, and skips entirely if even the 4 MB minimum will not fit. The C engine
performs the same `statvfs`/`GetDiskFreeSpaceEx` check — which matters most on
Linux, where `/tmp` is frequently **tmpfs (RAM)**, so an unguarded "disk" test
would consume memory rather than storage.

### 3. SSD write endurance

Flash has a finite total-bytes-written rating.

**Guard:** cumulative writes per run are capped at **16 GB** (file size ×
repeats), and the actual volume is always printed.

| Setting | Written per run | Runs to reach a ~300 TBW rating |
|---------|-----------------|--------------------------------|
| `--quick` | 128 MB | ~2,400,000 |
| default | 768 MB | ~410,000 |
| hard cap | 16 GB | ~19,200 |

At the default you would need to run the benchmark four hundred thousand times
to consume a typical consumer SSD's rated endurance.

### 4. Heat

Sustained 100% load is what thermal design exists for. Hardware throttles, and
if that is not enough it shuts down — both are protective, neither is damage.

**Guard:** the sustained-load test polls thermal state each window and **stops
early** if the CPU is throttled below 40% of nominal or the package exceeds
100 °C. Past that point further load measures nothing new, and on a machine
whose cooling has already failed it only invites an abrupt shutdown.

The tool also refuses to start at all when the machine is already throttled, on
battery, or under load (exit code 3) — see the machine-state guard.

## File-deletion safety

The only files removed are:

1. the scratch file the tool just created via `mkstemp`, deleted in a `finally`
   block so an interrupted run still cleans up;
2. orphaned scratch files from a previously killed run — and only those
   matching `pcbench_*.bin`, **and** older than 60 seconds so a concurrent run
   is never disturbed.

This is covered by tests asserting that a file named
`my_important_data.bin` sitting in the same directory is left untouched.

## Privilege and network

- The tool runs as a **normal user**. Nothing requires root.
- `sudo -n powermetrics` is attempted on macOS *only* to read a power meter.
  The `-n` flag means it never prompts and never blocks; without existing sudo
  rights it silently falls back to an estimate.
- The network benchmark binds **only to `127.0.0.1`** on an ephemeral port and
  sends nothing off the machine. No telemetry, no external requests.
- No `shell=True` anywhere; every subprocess call passes an argument list, so
  there is no shell-injection surface.

## Running it on hardware you are worried about

If you suspect a machine is already faulty, this tool is *safer* than most
stress tests and is genuinely useful as a diagnostic:

```bash
# Gentle: short, no sustained heat, minimal writes
python3 benchmark.py --quick --skip disk

# Validate correctness without stressing storage
python3 benchmark.py --only cpu_int,memory --seconds 30 --repeats 3
```

Every workload **validates its own output** (see
[technical.md](technical.md#validation--benchmark-as-diagnostic)). A machine
that returns a wrong answer is flagged with exit code 4 — that is how failing
RAM, an unstable overclock, or inadequate cooling reveal itself. Detecting that
is the point; the tool is not what caused it.

## Audit summary

| Area | Finding |
|------|---------|
| Raw device / format / firmware / voltage access | **None** |
| Shell injection surface | **None** (no `shell=True`) |
| Memory allocation | Capped to 1/8 RAM (Python and C) |
| Disk free space | 1.5× headroom required (Python and C) |
| Flash wear | 16 GB per run maximum, always disclosed |
| Thermal | Aborts on severe throttle or >100 °C |
| File deletion | Own `mkstemp` files only; user data verified untouched |
| Thread/process cleanup | Verified no leaks |
| Network | Loopback only, no external traffic |
