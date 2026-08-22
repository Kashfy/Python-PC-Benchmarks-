"""Discover and run user-supplied benchmarks from the ``plugins/`` directory.

Every benchmark so far has had to be written into the package. A plugin is a
single Python file dropped into ``plugins/`` that is picked up automatically
and treated like any built-in test: it is timed, scored against its own
baseline, printed in the report, and written to the CSV.

Minimal plugin::

    NAME = "My benchmark"
    UNIT = "ops/s"
    BASELINE = 1000.0          # the rate a score of 100 corresponds to

    def run(seconds, repeats):
        ...
        return {"rate": measured_ops_per_second}

``run`` may return extra keys, which are preserved in the JSON output.

Plugins are ordinary Python and run with full privileges, exactly like the rest
of the tool — so only add files you trust, the same rule that applies to any
script you execute. Discovery never imports anything outside ``plugins/``, and
a plugin that raises is reported and skipped rather than aborting the run.
"""

from __future__ import annotations

import importlib.util
import os
import traceback

PLUGIN_DIR = "plugins"

# Attributes a plugin must define to be usable.
_REQUIRED = ("NAME", "UNIT", "BASELINE")


def plugin_dir(root: str) -> str:
    return os.path.join(root, PLUGIN_DIR)


def discover(root: str = ".") -> list[dict]:
    """Load every valid plugin. Invalid ones are reported, never fatal."""
    directory = plugin_dir(root)
    if not os.path.isdir(directory):
        return []

    found: list[dict] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue
        path = os.path.join(directory, filename)
        key = os.path.splitext(filename)[0]
        try:
            spec = importlib.util.spec_from_file_location(
                f"pcbench_plugin_{key}", path)
            if not spec or not spec.loader:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            found.append({"key": key, "file": filename, "valid": False,
                          "error": f"{type(e).__name__}: {e}"})
            continue

        missing = [a for a in _REQUIRED if not hasattr(module, a)]
        if missing:
            found.append({"key": key, "file": filename, "valid": False,
                          "error": f"missing {', '.join(missing)}"})
            continue
        if not callable(getattr(module, "run", None)):
            found.append({"key": key, "file": filename, "valid": False,
                          "error": "no callable run(seconds, repeats)"})
            continue

        found.append({
            "key": key,
            "file": filename,
            "valid": True,
            "name": str(module.NAME),
            "unit": str(module.UNIT),
            "baseline": float(module.BASELINE),
            "module": module,
        })
    return found


def run_all(plugins: list[dict], seconds: float, repeats: int,
            on_start=None) -> dict:
    """Execute each valid plugin, capturing failures per plugin."""
    results: dict = {}
    for plugin in plugins:
        if not plugin.get("valid"):
            results[plugin["key"]] = {"error": plugin["error"],
                                      "name": plugin.get("file", "?")}
            continue
        if on_start:
            on_start(plugin["name"])
        try:
            out = plugin["module"].run(seconds, repeats)
            if not isinstance(out, dict) or "rate" not in out:
                raise ValueError("run() must return a dict containing 'rate'")
            out.setdefault("unit", plugin["unit"])
            out["name"] = plugin["name"]
            out["baseline"] = plugin["baseline"]
            results[plugin["key"]] = out
        except Exception as e:
            results[plugin["key"]] = {
                "name": plugin["name"],
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3),
            }
    return results


def scores(results: dict) -> dict:
    """Score plugin results against each plugin's own declared baseline."""
    out = {}
    for key, entry in results.items():
        rate = entry.get("rate")
        base = entry.get("baseline")
        if isinstance(rate, (int, float)) and rate > 0 and base:
            out[f"plugin_{key}"] = round(100.0 * rate / base, 1)
    return out
