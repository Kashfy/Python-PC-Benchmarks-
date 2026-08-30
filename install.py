#!/usr/bin/env python3
"""First-time setup: install the optional packages that unlock extra benchmarks.

pcbench runs on the standard library alone. This script adds the optional
tiers — real BLAS numerics, cross-platform GPU compute, hardware crypto,
better sensors and charts — which the standard library cannot provide.

By default everything goes into a project-local virtual environment
(``.venv``) rather than your system Python. That is deliberate: installing
into a system interpreter risks permission problems and leaves packages behind
that are hard to remove. Nothing is installed until you confirm.

    python3 install.py                 # interactive, all tiers, into .venv
    python3 install.py --tier compute  # just one tier
    python3 install.py --list          # show what is available and installed
    python3 install.py --here          # use the current interpreter, no venv
    python3 install.py --yes           # skip the confirmation prompt

Afterwards, run the benchmark with the environment's interpreter:

    .venv/bin/python benchmark.py            (macOS / Linux)
    .venv\\Scripts\\python.exe benchmark.py    (Windows)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pcbench import optional  # noqa: E402

VENV_DIR = ".venv"


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
def hr(title: str = "") -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}" if title else line)


def show_status(venv_dir: str = VENV_DIR) -> None:
    st = optional.status()
    hr("Optional package tiers")
    for name, tier in optional.TIERS.items():
        state = st["tiers"][name]
        mark = "✓" if state["complete"] else ("~" if state["usable"] else " ")
        print(f"\n  [{mark}] {name}  —  {tier['summary']}")
        for pkg in tier["packages"]:
            installed = state["packages"][pkg.pip_name]
            tick = "✓" if installed else "·"
            ver = f" ({optional.version_of(pkg.import_name)})" if installed \
                else f" ~{pkg.approx_mb} MB"
            target = optional.pip_target(pkg)
            label = pkg.pip_name if target == pkg.pip_name \
                else f"{pkg.pip_name} → {target}"
            print(f"        {tick} {label:<16}{ver}")
            print(f"          {pkg.purpose}")

    print("\n  Large, hardware-specific — install manually if wanted:")
    for pkg in optional.HEAVY:
        tick = "✓" if st["heavy"][pkg.pip_name] else "·"
        print(f"        {tick} {pkg.pip_name:<16} ~{pkg.approx_mb} MB — "
              f"{pkg.purpose}")

    # Probe the environment the benchmark will actually run in, which is the
    # venv when one exists — `python3 install.py --list` is otherwise reporting
    # on an interpreter nobody uses.
    print()
    runner = venv_python(venv_dir)
    report_opencl(runner if os.path.isfile(runner) else sys.executable)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def venv_python(venv_dir: str) -> str:
    """Path to the interpreter inside a virtual environment."""
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def ensure_venv(venv_dir: str) -> str | None:
    """Create the venv if absent. Returns its interpreter path, or None."""
    python = venv_python(venv_dir)
    if os.path.isfile(python):
        print(f"  Using existing environment: {venv_dir}")
        return python

    print(f"  Creating virtual environment in {venv_dir} ...")
    proc = subprocess.run([sys.executable, "-m", "venv", venv_dir],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ! could not create venv: "
              f"{(proc.stderr or proc.stdout).strip()[:300]}", file=sys.stderr)
        return None
    if not os.path.isfile(python):
        print("  ! venv created but interpreter not found", file=sys.stderr)
        return None
    return python


def pip_install(python: str, packages: list[str]) -> tuple[list[str], list[str]]:
    """Install packages one at a time. Returns (succeeded, failed).

    Installed individually rather than in one command so a single unavailable
    wheel — pyopencl has no wheel on some platforms — does not abort the whole
    batch.
    """
    ok, failed = [], []
    for name in packages:
        print(f"\n  → installing {name} ...", flush=True)
        proc = subprocess.run(
            [python, "-m", "pip", "install", "--upgrade", name],
            capture_output=True, text=True)
        if proc.returncode == 0:
            ok.append(name)
            print(f"    ✓ {name}")
        else:
            failed.append(name)
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            reason = tail[-1][:160] if tail else "unknown error"
            print(f"    ✗ {name} — {reason}")
    return ok, failed


# --------------------------------------------------------------------------- #
# OpenCL, which pip cannot install
# --------------------------------------------------------------------------- #
def opencl_platforms(python: str) -> int | None:
    """How many OpenCL platforms the target interpreter can see.

    None when pyopencl is not installed there. Zero is the interesting answer:
    the binding imports, the GPU is fine, and no vendor driver is registered
    with the loader.
    """
    probe = ("import pyopencl;"
             "print(len(pyopencl.get_platforms()))")
    proc = subprocess.run([python, "-c", probe],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        out = (proc.stderr or "")
        if "ModuleNotFoundError" in out or "ImportError" in out:
            return None
        return 0                    # imported, then failed to enumerate
    try:
        return int((proc.stdout or "0").strip())
    except ValueError:
        return 0


def report_opencl(python: str) -> None:
    """Say whether the GPU section will actually have a driver to talk to.

    `pyopencl` is only the Python binding; the driver is the vendor's ICD, a
    *system* package. A machine can have a working GPU, a current driver and
    pyopencl installed and still enumerate nothing, and the error it raises
    — PLATFORM_NOT_FOUND_KHR — reads like a hardware fault. pip cannot fix
    this, so the installer at least has to name what will.
    """
    count = opencl_platforms(python)
    if count is None:
        return
    if count:
        print(f"  OpenCL: {count} platform(s) visible — the GPU section has "
              f"a driver to talk to.")
        return
    print("  OpenCL: pyopencl is installed but no driver is registered, so "
          "GPU\n          shader throughput cannot be measured. That is a "
          "system\n          package, not a pip one.")
    hint = optional.opencl_icd_hint()
    if hint:
        print(f"\n              {hint}\n")
    elif os.name == "nt":
        print("\n          Reinstall the GPU vendor's display driver, which "
              "registers\n          the OpenCL runtime.\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Install pcbench's optional benchmark packages.")
    p.add_argument("--tier", default="",
                   help="Comma-separated tiers to install: "
                        + ", ".join(optional.TIERS) + " (default: all)")
    p.add_argument("--list", action="store_true",
                   help="Show available tiers and what is installed, then exit")
    p.add_argument("--here", action="store_true",
                   help="Install into the current interpreter instead of a venv")
    p.add_argument("--venv-dir", default=VENV_DIR,
                   help="Where to create the virtual environment")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Do not prompt for confirmation")
    args = p.parse_args(argv)

    if args.list:
        show_status(args.venv_dir)
        return 0

    tiers = ([t.strip() for t in args.tier.split(",") if t.strip()]
             if args.tier else list(optional.TIERS))
    unknown = [t for t in tiers if t not in optional.TIERS]
    if unknown:
        print(f"error: unknown tier(s): {', '.join(unknown)}. "
              f"Valid: {', '.join(optional.TIERS)}", file=sys.stderr)
        return 2

    missing = optional.missing(tiers)
    if not missing:
        print("Everything for the selected tier(s) is already installed.")
        # Still worth saying, and this is the case where it matters most: the
        # gpu tier can be complete and the GPU section still measure nothing,
        # because the driver it needs is not a pip package.
        if "gpu" in tiers:
            runner = venv_python(args.venv_dir)
            print()
            report_opencl(runner if os.path.isfile(runner) and not args.here
                          else sys.executable)
        return 0

    hr("pcbench optional package installer")
    total_mb = sum(pkg.approx_mb for pkg in missing)
    print(f"\n  Tiers selected : {', '.join(tiers)}")
    print(f"  To install     : {len(missing)} package(s), "
          f"roughly {total_mb} MB total\n")
    for pkg in missing:
        target = optional.pip_target(pkg)
        label = pkg.pip_name if target == pkg.pip_name else target
        print(f"    {label:<28} ~{pkg.approx_mb:>4} MB   {pkg.purpose}")

    target = "the current Python environment" if args.here \
        else f"a virtual environment at ./{args.venv_dir}"
    print(f"\n  Destination    : {target}")
    if args.here:
        print("  Note: --here modifies the interpreter you are running now.")
    print("\n  pcbench works without any of these; they only add extra "
          "benchmarks.")

    if not args.yes:
        try:
            reply = input("\n  Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            reply = ""
        if reply not in ("y", "yes"):
            print("  Cancelled — nothing was installed.")
            return 1

    python = sys.executable
    if not args.here:
        python = ensure_venv(args.venv_dir)
        if not python:
            return 3

    ok, failed = pip_install(python, [optional.pip_target(pkg)
                                      for pkg in missing])

    hr("Result")
    print(f"  Installed: {len(ok)}   Failed: {len(failed)}")
    if failed:
        print(f"\n  These could not be installed: {', '.join(failed)}")
        print("  That is not fatal — pcbench skips whatever is absent.")
        print("  A common cause is a package with no prebuilt wheel for this "
              "platform\n  (pyopencl often needs system OpenCL headers).")

    if "gpu" in tiers or "ai" in tiers:
        print()
        report_opencl(python)

    if not args.here:
        runner = venv_python(args.venv_dir)
        print(f"\n  Run the benchmark with that environment:\n\n"
              f"      {runner} benchmark.py\n")
    else:
        print("\n      python3 benchmark.py\n")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
