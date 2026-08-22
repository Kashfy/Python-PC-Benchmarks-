"""Configuration files and environment variables.

A benchmark is only comparable to another benchmark run the same way. The
moment two people, or two CI jobs, or two machines in a fleet type slightly
different flags, the numbers stop meaning anything relative to each other — and
nothing about the output reveals the discrepancy.

A config file fixes that by making the run definition a file that gets
committed, reviewed, and copied to every machine:

    # pcbench.toml
    [run]
    seconds = 5
    repeats = 5
    profile = "server"

    [output]
    dir = "/var/lib/pcbench"
    prometheus = "/var/lib/node_exporter/pcbench.prom"

    [gates]
    fail_under = 250
    assertions = ["disk.read_rate>=500", "sustained.droop_pct<=20"]

Precedence runs command line > environment > config file > defaults, which is
the order of increasing specificity: the file is the fleet-wide standard, the
environment is per-machine deployment detail, and a flag is a deliberate
one-off override by whoever is at the keyboard.

TOML is read with the standard library's ``tomllib`` (Python 3.11+); JSON is
accepted everywhere so nothing is lost on older interpreters.
"""

from __future__ import annotations

import json
import os

#: Names searched, in order, when no ``--config`` is given.
DEFAULT_NAMES = ("pcbench.toml", ".pcbench.toml", "pcbench.json",
                 ".pcbench.json")

#: Environment variables mapped onto the same option names the file uses.
#: Prefixed to avoid colliding with anything else in a CI environment.
ENV_PREFIX = "PCBENCH_"

#: Maps a flattened config key to the argparse destination it sets.
_KEY_MAP = {
    "run.seconds": "seconds",
    "run.repeats": "repeats",
    "run.only": "only",
    "run.profile": "profile",
    "run.skip": "skip",
    "run.quick": "quick",
    "run.disk_mb": "disk_mb",
    "run.mem_mb": "mem_mb",
    "run.force": "force",
    "run.sustained": "sustained",
    "run.soak": "soak",
    "output.dir": "output_dir",
    "output.html": "html",
    "output.spec_sheet": "spec_sheet",
    "output.json_stdout": "json_stdout",
    "output.prometheus": "prometheus",
    "output.junit": "junit",
    "output.sqlite": "sqlite",
    "output.markdown": "markdown",
    "output.no_save": "no_save",
    "gates.fail_under": "fail_under",
    "gates.assertions": "assert_",
    "accel.no_gpu": "no_gpu",
    "accel.no_npu": "no_npu",
    "accel.no_accel": "no_accel",
    "network.host": "network_host",
    "network.url": "network_url",
    "regression.threshold": "regression_threshold",
    "regression.disabled": "no_regression",
}


class ConfigError(ValueError):
    """Raised for an unreadable or malformed configuration file."""


def find(start: str | None = None) -> str | None:
    """Locate a config file in ``start`` or any parent directory.

    Walking upward is what makes a repository-level ``pcbench.toml`` apply to
    every benchmark run from anywhere inside that repository, the same way
    linters and formatters behave.
    """
    directory = os.path.abspath(start or os.getcwd())
    while True:
        for name in DEFAULT_NAMES:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def load_file(path: str) -> dict:
    """Read a TOML or JSON config file into a nested dict."""
    try:
        if path.endswith((".toml", ".tml")):
            try:
                import tomllib
            except ImportError:
                raise ConfigError(
                    f"cannot read {path}: TOML support needs Python 3.11+. "
                    f"Use a .json config file on this interpreter.")
            with open(path, "rb") as f:
                return tomllib.load(f)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e}")
    except Exception as e:              # tomllib/json decode errors
        raise ConfigError(f"invalid config in {path}: {e}")


def flatten(data: dict, prefix: str = "") -> dict:
    """Turn ``{'run': {'seconds': 5}}`` into ``{'run.seconds': 5}``."""
    out: dict = {}
    for key, value in data.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{full}."))
        else:
            out[full] = value
    return out


def from_env(environ: dict | None = None) -> dict:
    """Read ``PCBENCH_*`` variables into the same flattened key space.

    ``PCBENCH_RUN_SECONDS=5`` sets ``run.seconds``. Bare option names are also
    accepted (``PCBENCH_SECONDS``) because that is what people try first.
    """
    environ = os.environ if environ is None else environ
    aliases = {dest: key for key, dest in _KEY_MAP.items()}
    out: dict = {}
    for name, value in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        tail = name[len(ENV_PREFIX):].lower()
        if tail in aliases:                       # PCBENCH_SECONDS
            out[aliases[tail]] = value
            continue
        # PCBENCH_RUN_SECONDS -> run.seconds; only the first underscore splits
        # the section off, so multi-word options survive.
        section, _, option = tail.partition("_")
        candidate = f"{section}.{option}"
        if candidate in _KEY_MAP:
            out[candidate] = value
    return out


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(dest: str, value, defaults: dict):
    """Convert a string from a file or the environment to the flag's type."""
    reference = defaults.get(dest)
    if isinstance(reference, bool) or dest in (
            "quick", "force", "html", "spec_sheet", "json_stdout", "no_save",
            "no_gpu", "no_npu", "no_accel", "no_regression"):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ConfigError(f"{dest}: expected a boolean, got {value!r}")
    if dest == "assert_":
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, list):
            return [str(v) for v in value]
        raise ConfigError(f"{dest}: expected a list of assertions")
    if isinstance(reference, int) and not isinstance(reference, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{dest}: expected an integer, got {value!r}")
    if isinstance(reference, float) or dest in ("fail_under",):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{dest}: expected a number, got {value!r}")
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def apply(args, parser, config_path: str | None = None,
          environ: dict | None = None) -> dict:
    """Fold config-file and environment settings into parsed ``args``.

    Only options the user did *not* pass explicitly are overwritten, which is
    what preserves command-line precedence. Explicitness is determined by
    comparing against the parser's defaults rather than re-parsing ``sys.argv``,
    so it works identically when ``main()`` is called programmatically.
    """
    defaults = {a.dest: a.default for a in parser._actions}
    explicit = {dest for dest, default in defaults.items()
                if getattr(args, dest, default) != default}

    settings: dict = {}
    source = None
    path = config_path or find()
    if path:
        data = flatten(load_file(path))
        unknown = [k for k in data if k not in _KEY_MAP]
        if unknown:
            raise ConfigError(
                f"unknown setting(s) in {path}: {', '.join(sorted(unknown))}. "
                f"Valid: {', '.join(sorted(_KEY_MAP))}")
        settings.update(data)
        source = path
    settings.update(from_env(environ))

    applied: dict = {}
    for key, value in settings.items():
        dest = _KEY_MAP[key]
        if dest in explicit:
            continue                        # the command line wins
        setattr(args, dest, _coerce(dest, value, defaults))
        applied[key] = getattr(args, dest)
    return {"path": source, "applied": applied}


SAMPLE = """\
# pcbench configuration. Place at the root of a repository or deployment and
# every run from anywhere inside it uses these settings.
#
# Precedence: command-line flag > PCBENCH_* environment variable > this file.

[run]
seconds = 5          # measurement window per test, per repeat
repeats = 5          # repeats; the median is reported
profile = "server"   # quick | cpu | ai | dev | storage | laptop | server | apps
# sustained = "10m"  # also run a thermal/throttling test

[output]
dir = "results"
# prometheus = "/var/lib/node_exporter/textfile_collector/pcbench.prom"
# junit = "results/pcbench-junit.xml"
# sqlite = "results/pcbench.db"
# markdown = "results/summary.md"

[gates]
# Non-zero exit when a threshold is not met, so CI and monitoring can act.
fail_under = 250
assertions = [
    "disk.read_rate>=500",
    "sustained.droop_pct<=20",
]

[regression]
threshold = 10.0     # percent change against this machine's own history
"""


def write_sample(path: str) -> str:
    """Write a commented starter config. Never overwrites an existing file."""
    if os.path.exists(path):
        raise ConfigError(f"{path} already exists; not overwriting it")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SAMPLE)
    return path
