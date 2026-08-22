"""Machine-readable exports for the systems that consume benchmark results.

JSON and CSV cover a human going back to look at a run. They do not cover the
three places benchmark results actually need to *arrive*:

* **Monitoring.** Prometheus/OpenMetrics text is what a node exporter's
  textfile collector reads. Dropping one file in a directory turns a scheduled
  benchmark into a dashboard and an alert rule, with no agent to install.
* **CI.** JUnit XML is the one format every CI system on earth already renders.
  Emitting it means performance regressions show up in the same test-results
  tab as failing unit tests, rather than buried in job logs nobody reads.
* **Fleets and history.** A CSV row per run stops scaling the moment there are
  several machines and dozens of metrics. SQLite gives the same file-copyable
  simplicity with real queries, and it is in the standard library.

Every writer here is total: it never raises on a partially-populated payload,
because a run that produced some results and then hit a hardware fault is
exactly the run whose output matters most.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from xml.sax.saxutils import escape, quoteattr

from .scoring import category_scores

_SAFE_LABEL = re.compile(r"[^a-zA-Z0-9_]")


def _metric_name(name: str) -> str:
    return "pcbench_" + _SAFE_LABEL.sub("_", name).strip("_").lower()


def _label_value(value) -> str:
    """Escape a Prometheus label value per the exposition format."""
    text = "" if value is None else str(value)
    return (text.replace("\\", "\\\\").replace('"', '\\"')
                .replace("\n", "\\n"))


# --------------------------------------------------------------------------- #
# Prometheus / OpenMetrics
# --------------------------------------------------------------------------- #
def prometheus_text(payload: dict) -> str:
    """Render a run as Prometheus exposition-format text.

    Machine identity goes into labels rather than metric names so a fleet's
    worth of runs aggregates cleanly: ``pcbench_score{host="a"}`` and
    ``pcbench_score{host="b"}`` are the same time series family, which is what
    makes ``min by (host)`` and alerting rules possible.
    """
    info = payload.get("system") or {}
    scores = payload.get("scores") or {}
    labels = {
        "host": info.get("hostname"),
        "os": info.get("os"),
        "arch": info.get("arch_family"),
        "cpu": info.get("cpu_model"),
        "version": payload.get("version"),
    }
    label_text = ",".join(f'{k}="{_label_value(v)}"'
                          for k, v in labels.items() if v)
    label_text = "{" + label_text + "}" if label_text else ""

    lines: list[str] = []

    def emit(name: str, value, help_text: str, kind: str = "gauge") -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        metric = _metric_name(name)
        lines.append(f"# HELP {metric} {help_text}")
        lines.append(f"# TYPE {metric} {kind}")
        # repr() gives the shortest string that round-trips the double
        # exactly. A fixed precision would either mangle a Unix timestamp into
        # scientific notation or render 308.8 as 308.80000000000001.
        lines.append(f"{metric}{label_text} {float(value)!r}")

    emit("composite_score", scores.get("composite"),
         "Composite benchmark score (baseline machine = 100)")
    for cat, value in category_scores(scores.get("subscores") or {}).items():
        emit(f"category_score_{cat}", value,
             f"{cat} category score (baseline = 100)")
    for key, value in (scores.get("subscores") or {}).items():
        emit(f"subscore_{key}", value, f"{key} subscore (baseline = 100)")

    # Raw rates carry units the scores have normalised away, and alerting on
    # "disk read fell below 400 MB/s" is more actionable than on a score.
    for key, entry in (payload.get("results") or {}).items():
        if not isinstance(entry, dict):
            continue
        for field in ("rate", "read_rate", "write_rate", "random_read_iops",
                      "peak_iops"):
            value = entry.get(field)
            if isinstance(value, (int, float)):
                unit = entry.get("unit", "")
                suffix = "" if field == "rate" else f"_{field}"
                emit(f"{key}{suffix}", value,
                     f"{key} {field} ({unit})" if unit else f"{key} {field}")

    power = payload.get("power") or {}
    emit("power_watts", power.get("watts"), "Package power draw under load")
    ppw = payload.get("perf_per_watt") or {}
    emit("perf_per_watt", ppw.get("score_per_watt"),
         "Composite score per watt")

    sustained = payload.get("sustained") or {}
    emit("sustained_droop_pct", sustained.get("droop_pct"),
         "Throughput lost from start to end of the sustained run")
    emit("sustained_peak_celsius", (sustained.get("temperature") or {}).get("max"),
         "Peak CPU temperature during the sustained run")

    emit("validation_failures",
         sum(1 for v in (payload.get("results") or {}).values()
             if isinstance(v, dict) and v.get("validation_failed")),
         "Benchmarks that produced a numerically wrong answer", "gauge")
    emit("run_timestamp_seconds", int(time.time()),
         "Unix time at which the exposition file was written")
    return "\n".join(lines) + "\n"


def save_prometheus(payload: dict, path: str) -> str:
    """Write exposition text atomically.

    The atomic rename matters: a textfile collector scrapes on its own
    schedule and will happily read a half-written file, producing parse errors
    and gaps in the series.
    """
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(prometheus_text(payload))
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# JUnit XML
# --------------------------------------------------------------------------- #
def junit_xml(payload: dict, gate_results: list[dict] | None = None) -> str:
    """Render results as a JUnit test report.

    Three things become "tests": every benchmark that ran (failing if it
    errored or produced a wrong answer), every regression the run detected, and
    every threshold gate the user asked for. That maps the tool's three failure
    modes onto the one reporting surface CI already understands.
    """
    cases: list[str] = []
    failures = errors = skipped = 0

    for name, entry in (payload.get("results") or {}).items():
        if not isinstance(entry, dict):
            continue
        case = f'<testcase classname="pcbench.benchmark" name={quoteattr(name)}'
        if entry.get("validation_failed"):
            failures += 1
            cases.append(
                case + '>\n    <failure type="ValidationError" '
                f'message={quoteattr(str(entry.get("error", "")))}>'
                f'{escape(str(entry.get("error", "")))}</failure>\n  </testcase>')
        elif entry.get("error"):
            errors += 1
            cases.append(
                case + '>\n    <error '
                f'message={quoteattr(str(entry["error"]))}>'
                f'{escape(str(entry["error"]))}</error>\n  </testcase>')
        elif entry.get("skipped"):
            skipped += 1
            cases.append(
                case + '>\n    <skipped '
                f'message={quoteattr(str(entry.get("reason", "")))}/>'
                '\n  </testcase>')
        else:
            rate = entry.get("rate", entry.get("read_rate", ""))
            unit = entry.get("unit", "")
            cases.append(
                case + '>\n    <system-out>'
                f'{escape(f"{rate} {unit}".strip())}</system-out>\n  </testcase>')

    regression = payload.get("regression") or {}
    for item in regression.get("regressions", []) or []:
        failures += 1
        msg = (f"{item.get('metric')}: {item.get('change_pct')}% vs "
               f"baseline {item.get('baseline')}")
        cases.append(
            f'<testcase classname="pcbench.regression" '
            f'name={quoteattr(str(item.get("metric", "regression")))}>\n'
            f'    <failure type="Regression" message={quoteattr(msg)}>'
            f'{escape(msg)}</failure>\n  </testcase>')

    for gate in gate_results or []:
        name = gate.get("name", "gate")
        if gate.get("passed"):
            cases.append(f'<testcase classname="pcbench.gate" '
                         f'name={quoteattr(name)}/>')
        else:
            failures += 1
            msg = gate.get("message", "gate failed")
            cases.append(
                f'<testcase classname="pcbench.gate" name={quoteattr(name)}>\n'
                f'    <failure type="ThresholdNotMet" message={quoteattr(msg)}>'
                f'{escape(msg)}</failure>\n  </testcase>')

    info = payload.get("system") or {}
    suite_name = f"pcbench.{info.get('hostname', 'machine')}"
    body = "\n  ".join(cases)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuites name="pcbench" tests="{len(cases)}" '
        f'failures="{failures}" errors="{errors}" skipped="{skipped}">\n'
        f'  <testsuite name={quoteattr(suite_name)} tests="{len(cases)}" '
        f'failures="{failures}" errors="{errors}" skipped="{skipped}" '
        f'timestamp={quoteattr(str(payload.get("timestamp_utc", "")))}>\n'
        f'  {body}\n'
        '  </testsuite>\n</testsuites>\n')


def save_junit(payload: dict, path: str,
               gate_results: list[dict] | None = None) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(junit_xml(payload, gate_results))
    return path


# --------------------------------------------------------------------------- #
# SQLite history
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    hostname      TEXT,
    os            TEXT,
    arch          TEXT,
    cpu_model     TEXT,
    cores         INTEGER,
    ram_gb        REAL,
    version       TEXT,
    composite     REAL,
    payload       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name   TEXT NOT NULL,
    value  REAL,
    unit   TEXT,
    PRIMARY KEY (run_id, name)
);
CREATE INDEX IF NOT EXISTS idx_runs_host ON runs(hostname, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
"""


def save_sqlite(payload: dict, path: str) -> str:
    """Append a run to a SQLite database, creating it if needed.

    The whole payload is stored alongside the extracted columns. Extracted
    columns make the common queries fast; the payload means a question nobody
    anticipated is still answerable years later without re-running anything.
    """
    info = payload.get("system") or {}
    scores = payload.get("scores") or {}
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        cur = conn.execute(
            "INSERT INTO runs (timestamp_utc, hostname, os, arch, cpu_model, "
            "cores, ram_gb, version, composite, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (payload.get("timestamp_utc"), info.get("hostname"),
             info.get("os"), info.get("arch_family"), info.get("cpu_model"),
             info.get("cpu_cores_logical"), info.get("ram_total_gb"),
             payload.get("version"), scores.get("composite"),
             json.dumps(payload, default=str)))
        run_id = cur.lastrowid

        rows = [(run_id, f"score.{k}", float(v), "score")
                for k, v in (scores.get("subscores") or {}).items()]
        for key, entry in (payload.get("results") or {}).items():
            if not isinstance(entry, dict):
                continue
            unit = entry.get("unit")
            for field in ("rate", "read_rate", "write_rate",
                          "random_read_iops", "peak_iops"):
                value = entry.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    name = key if field == "rate" else f"{key}.{field}"
                    rows.append((run_id, name, float(value), unit))
        conn.executemany(
            "INSERT OR REPLACE INTO metrics (run_id, name, value, unit) "
            "VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    return path


def query_sqlite(path: str, metric: str = "score.cpu_multi",
                 hostname: str | None = None, limit: int = 20) -> list[dict]:
    """Recent values of one metric, newest first — the trend query.

    Provided so the database is useful without the user having to know the
    schema; anything more elaborate is a job for ``sqlite3`` itself.
    """
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        sql = ("SELECT r.timestamp_utc, r.hostname, r.cpu_model, m.value, "
               "m.unit FROM metrics m JOIN runs r ON r.id = m.run_id "
               "WHERE m.name = ?")
        params: list = [metric]
        if hostname:
            sql += " AND r.hostname = ?"
            params.append(hostname)
        sql += " ORDER BY r.timestamp_utc DESC LIMIT ?"
        params.append(int(limit))
        return [dict(row) for row in conn.execute(sql, params)]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Markdown (issue reports, wikis, pull-request comments)
# --------------------------------------------------------------------------- #
def markdown_summary(payload: dict) -> str:
    """A compact table suited to pasting into an issue or a PR comment."""
    info = payload.get("system") or {}
    scores = payload.get("scores") or {}
    cats = category_scores(scores.get("subscores") or {})

    out = [f"### pcbench {payload.get('version', '')} — "
           f"{info.get('hostname', 'machine')}", ""]
    out.append(f"- **CPU**: {info.get('cpu_model', 'unknown')} "
               f"({info.get('cpu_cores_logical', '?')} threads, "
               f"{info.get('arch_family', '?')})")
    out.append(f"- **RAM**: {info.get('ram_total_gb', '?')} GB")
    out.append(f"- **OS**: {info.get('platform', 'unknown')}")
    out.append(f"- **Composite score**: **{scores.get('composite', 0)}** "
               f"(baseline machine = 100)")
    out.append("")
    if cats:
        out.append("| Category | Score |")
        out.append("|---|---:|")
        for name, value in sorted(cats.items(), key=lambda kv: -kv[1]):
            out.append(f"| {name} | {value} |")
        out.append("")

    warnings = payload.get("warnings") or []
    if warnings:
        out.append("> [!WARNING]")
        for w in warnings:
            out.append(f"> {w}")
        out.append("")
    return "\n".join(out)


def save_markdown(payload: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown_summary(payload))
    return path
