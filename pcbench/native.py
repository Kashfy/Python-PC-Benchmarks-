"""Build and run the optional native (C) engine.

The engine is optional by design: it produces compiler-optimized numbers and a
real pointer-chase latency measurement that Python cannot express, but its
absence only removes a section from the report. Any failure here is captured
and reported, never raised.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

MB = 1024 * 1024

#: STREAM requires each array to clear the last-level cache by about 4x.
#: Detected cache sizes are a floor rather than the truth — Apple silicon hides
#: its system-level cache entirely, reporting 24 MB of L2 on a part that also
#: has 48 MB of SLC behind it — so a generous minimum is applied on top of the
#: 4x rule. 256 MB arrays cover any last-level cache up to 64 MB, which spans
#: every consumer part and most server ones.
STREAM_MIN_ARRAY_BYTES = 256 * MB
STREAM_CACHE_MULTIPLE = 4

SOURCE_NAME = "native_engine.c"
BINARY_NAME = "native_engine.exe" if os.name == "nt" else "native_engine"


#: Compilers to try, in order, per platform.
#:
#: Windows puts gcc first deliberately. A bare `clang` there is usually the
#: LLVM release, which relies on a Visual Studio installation for its headers
#: and linker and fails with "unable to find a Visual Studio installation" when
#: there is none -- which is exactly what happened on a real Ryzen machine,
#: taking the native engine, STREAM, the CoreMark-style suite and the compile
#: benchmark down with it. MinGW's gcc is self-contained, so it is tried first,
#: and MSVC's cl is tried before clang because when it is on PATH the developer
#: environment is normally already set up.
_CANDIDATES_WINDOWS = ("gcc", "cl", "clang-cl", "clang", "cc")
_CANDIDATES_POSIX = ("cc", "clang", "gcc")


def compiler_candidates() -> list[str]:
    names = _CANDIDATES_WINDOWS if os.name == "nt" else _CANDIDATES_POSIX
    return [c for c in names if shutil.which(c)]


def find_compiler() -> str | None:
    """First available compiler. Availability is not the same as working."""
    candidates = compiler_candidates()
    return candidates[0] if candidates else None


def is_msvc(cc: str) -> bool:
    """MSVC takes a different flag dialect entirely from gcc and clang.

    Both separators are handled explicitly: ``os.path.basename`` follows the
    *host* convention, so a Windows path would not be split when this runs
    under a POSIX interpreter (as it does in the test suite).
    """
    name = cc.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name.split(".")[0] in ("cl", "clang-cl")


def build_command(cc: str, src: str, exe: str) -> list[str]:
    if is_msvc(cc):
        # /Fe: sets the output name; the CRT supplies the maths and threading
        # that -lm and -lpthread provide elsewhere.
        return [cc, "/nologo", "/O2", src, f"/Fe:{exe}"]
    flags = [] if os.name == "nt" else ["-lm", "-lpthread"]
    return [cc, "-O2", src, "-o", exe] + flags


def _link_flags(cc: str) -> list[str]:
    """Retained for callers outside this module; see build_command."""
    if os.name == "nt" or is_msvc(cc):
        return []
    return ["-lm", "-lpthread"]


def build(src: str, exe: str) -> tuple[bool, str]:
    """Compile the engine, trying each available compiler until one works.

    Availability on PATH does not mean a compiler can actually build anything
    -- the Windows case above is precisely that -- so every candidate is tried
    and the first that produces a binary wins. When none do, every failure is
    reported, because "no compiler found" would be wrong and unactionable when
    three were found and all failed for different reasons.
    """
    candidates = compiler_candidates()
    if not candidates:
        return False, _no_compiler_hint()

    failures = []
    for cc in candidates:
        try:
            proc = subprocess.run(build_command(cc, src, exe),
                                  capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.SubprocessError) as e:
            failures.append(f"{cc}: {e}")
            continue
        if proc.returncode == 0 and os.path.isfile(exe):
            return True, cc
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        failures.append(f"{cc}: {detail[-1] if detail else 'build failed'}")

    hint = ""
    if os.name == "nt":
        hint = ("  Install MinGW-w64 (e.g. via MSYS2 or 'winget install "
                "BrechtSanders.WinLibs.POSIX.UCRT') so gcc is on PATH, or run "
                "from a Visual Studio Developer Command Prompt so cl works.")
    return False, " | ".join(failures)[:600] + hint


def _no_compiler_hint() -> str:
    if os.name == "nt":
        return ("no C compiler found. Install MinGW-w64 so gcc is on PATH, or "
                "open a Visual Studio Developer Command Prompt for cl")
    return "no C compiler found (cc/clang/gcc)"


def stream_array_mb(cache_bytes: int | None, ram_bytes: int = 0) -> int:
    """Choose a STREAM array size that clears the cache without risking swap.

    Three arrays are allocated, so the total is 3x this. The RAM clamp keeps
    that under a quarter of memory, which matters on the small machines where
    the default would otherwise be the whole system.
    """
    target = max(STREAM_MIN_ARRAY_BYTES,
                 (cache_bytes or 0) * STREAM_CACHE_MULTIPLE)
    if ram_bytes:
        target = min(target, ram_bytes // 12)
    return max(4, int(target // MB))


def run(seconds: float, repeats: int, script_dir: str,
        threads: int | None = None, stream_mb: int | None = None,
        disk_dir: str | None = None) -> dict | None:
    """Compile if stale, then run the engine and return its parsed JSON.

    ``disk_dir`` is where the engine's disk test writes. It matters: left to
    its own devices the engine uses the system temp directory, and on most
    Linux systems /tmp is tmpfs, so the "disk" figures were memory bandwidth.
    Passing the directory the rest of the tool writes to puts it on the same
    storage the Python disk test measures, and the two then agree.

    Returns None when the source is absent; an ``{"error": ...}`` dict on any
    build or run failure.
    """
    src = os.path.join(script_dir, SOURCE_NAME)
    if not os.path.isfile(src):
        return None
    exe = os.path.join(script_dir, BINARY_NAME)

    if (not os.path.isfile(exe)
            or os.path.getmtime(exe) < os.path.getmtime(src)):
        ok, detail = build(src, exe)
        if not ok:
            return {"error": "native build failed", "detail": detail}

    cmd = [exe, "--json", "--seconds", str(seconds), "--repeats", str(repeats)]
    if threads:
        cmd += ["--threads", str(threads)]
    if stream_mb:
        cmd += ["--stream-mb", str(stream_mb)]
    if disk_dir and os.path.isdir(disk_dir):
        cmd += ["--disk-dir", os.path.abspath(disk_dir)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=seconds * repeats * 8 + 60)
    except subprocess.SubprocessError as e:
        return {"error": f"native run error: {e}"}
    if proc.returncode != 0:
        return {"error": "native run failed",
                "detail": (proc.stderr or "").strip()[:600]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"native output not valid JSON: {e}"}
