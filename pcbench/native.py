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

SOURCE_NAME = "native_engine.c"
BINARY_NAME = "native_engine.exe" if os.name == "nt" else "native_engine"


def find_compiler() -> str | None:
    for cc in ("cc", "clang", "gcc"):
        if shutil.which(cc):
            return cc
    return None


def _link_flags(cc: str) -> list[str]:
    """Math and threads. MinGW folds pthreads into its runtime."""
    if os.name == "nt":
        return []
    return ["-lm", "-lpthread"]


def build(src: str, exe: str) -> tuple[bool, str]:
    cc = find_compiler()
    if not cc:
        return False, "no C compiler found (cc/clang/gcc)"
    cmd = [cc, "-O2", src, "-o", exe] + _link_flags(cc)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[:600]
    return True, ""


def run(seconds: float, repeats: int, script_dir: str,
        threads: int | None = None) -> dict | None:
    """Compile if stale, then run the engine and return its parsed JSON.

    Returns None when the source is absent; an ``{"error": ...}`` dict on any
    build or run failure.
    """
    src = os.path.join(script_dir, SOURCE_NAME)
    if not os.path.isfile(src):
        return None
    exe = os.path.join(script_dir, BINARY_NAME)

    if (not os.path.isfile(exe)
            or os.path.getmtime(exe) < os.path.getmtime(src)):
        ok, err = build(src, exe)
        if not ok:
            return {"error": "native build failed", "detail": err}

    cmd = [exe, "--json", "--seconds", str(seconds), "--repeats", str(repeats)]
    if threads:
        cmd += ["--threads", str(threads)]
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
