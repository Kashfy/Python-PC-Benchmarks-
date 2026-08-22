"""Pass/fail thresholds, so a benchmark can be an automated check.

Regression detection compares a run against *this machine's own history*, which
is the right tool for "did the last commit make things slower?". It cannot
answer the other two questions people automate:

* **Acceptance.** "Every machine we deploy to must reach a composite of 250 and
  sustain 4 GB/s of memory bandwidth." A new box either passes or goes back.
* **Fleet health.** "Alert me when any node's disk read rate falls below
  500 MB/s" catches a failing drive or a degraded RAID long before it fails
  outright, and needs no history to work on a node seen for the first time.

An assertion is written the way a person would say it — ``cpu_multi>=200``,
``disk.read_rate>=500``, ``sustained.droop_pct<=15`` — and resolves against
scores, raw metrics, or nested payload fields. Failed gates set the process
exit code, which is the only thing a CI system or a monitoring check reads.
"""

from __future__ import annotations

import re

# The path allows '-' so negative list indices work: io.jobs[-1] addresses the
# last job without the caller needing to know how many there are.
_ASSERT_RE = re.compile(r"^\s*([A-Za-z0-9_.\[\]-]+)\s*(>=|<=|>|<|==|!=)\s*"
                        r"(-?[0-9.]+(?:[eE][-+]?\d+)?)\s*$")

_INDEX_BODY = re.compile(r"-?\d+")

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class GateError(ValueError):
    """Raised for a malformed assertion, before any benchmarking happens."""


def parse(expression: str) -> tuple[str, str, float]:
    """Parse ``metric>=value`` into ``(path, operator, threshold)``."""
    m = _ASSERT_RE.match(expression)
    if not m:
        raise GateError(
            f"invalid assertion: {expression!r}. "
            f"Expected NAME OP VALUE, e.g. 'composite>=250', "
            f"'disk.read_rate>=500', 'sustained.droop_pct<=15'")
    path, op, raw = m.groups()

    # The character class permits brackets anywhere, so an unbalanced or
    # non-numeric index would parse and then quietly fail to resolve, giving
    # the user "not measured" for what is really a typo.
    if path.count("[") != path.count("]"):
        raise GateError(
            f"unbalanced brackets in assertion: {expression!r}")
    for segment in path.split("."):
        if "[" in segment:
            name, _, rest = segment.partition("[")
            if not rest.endswith("]") or not _INDEX_BODY.fullmatch(rest[:-1]):
                raise GateError(
                    f"invalid list index in assertion: {expression!r}. "
                    f"Use an integer, e.g. drives[0] or jobs[-1]")

    try:
        return path, op, float(raw)
    except ValueError:
        raise GateError(f"invalid number in assertion: {expression!r}")


def resolve(payload: dict, path: str) -> tuple:
    """Look up a dotted metric path, returning ``(value, source)``.

    Resolution order matters, and so does reporting it. A bare name like
    ``sqlite`` resolves to the *score* (baseline = 100), not the raw
    transactions per second — which is what most people mean by "cpu_multi >=
    200" but emphatically not what they mean by "sqlite >= 50000". The two
    differ by orders of magnitude, so a gate that silently picked the wrong one
    would pass or fail for reasons the user could not see. The source is
    returned so every verdict can say which number it used, and
    ``sqlite.rate`` addresses the raw figure unambiguously.
    """
    scores = payload.get("scores") or {}
    results = payload.get("results") or {}

    if path in ("composite", "score", "composite_score"):
        return scores.get("composite"), "composite score"

    # Category rollups (cpu, gpu, disk, ai, ...).
    from .scoring import category_scores
    cats = category_scores(scores.get("subscores") or {})

    if "." not in path:
        if path in (scores.get("subscores") or {}):
            return scores["subscores"][path], "score"
        if path in cats:
            return cats[path], "category score"
        entry = results.get(path)
        if isinstance(entry, dict):
            return entry.get("rate"), f"raw rate ({entry.get('unit', '')})"
        return None, None

    head, _, tail = path.partition(".")

    if head == "score":
        return (scores.get("subscores") or {}).get(tail), "score"
    if head == "category":
        return cats.get(tail), "category score"

    # Otherwise walk the payload: results first, then the top-level sections
    # (sustained, power, network, health, drive_life, ...).
    node = results.get(head, payload.get(head))
    unit = node.get("unit", "") if isinstance(node, dict) else ""
    for part in tail.split("."):
        node = _step(node, part)
        if node is None:
            return None, None
    return node, (f"raw value ({unit})" if unit else "raw value")


_INDEXED = re.compile(r"^([A-Za-z0-9_]*)\[(-?\d+)\]$")


def _step(node, part: str):
    """Take one step through the payload, honouring ``name[i]`` indexing.

    The assertion grammar has always accepted brackets, so paths like
    ``drive_life.drives[0].health_pct`` parsed cleanly and then silently failed
    to resolve. Lists are common in the payload — drives, I/O jobs, NUMA nodes
    — and gating on the first (or last) element of one is exactly what fleet
    checks need.
    """
    match = _INDEXED.match(part)
    if match:
        name, index = match.group(1), int(match.group(2))
        if name:
            if not isinstance(node, dict):
                return None
            node = node.get(name)
        if not isinstance(node, (list, tuple)):
            return None
        try:
            return node[index]
        except IndexError:
            return None
    if isinstance(node, dict):
        return node.get(part)
    return None


def evaluate(payload: dict, expressions: list[str],
             fail_under: float | None = None) -> list[dict]:
    """Check every assertion, returning one result record each.

    A metric that is missing fails rather than passing silently. Treating
    "not measured" as "met the threshold" is how acceptance checks quietly stop
    checking anything — the whole point of the gate is that a machine which
    could not produce the number does not get waved through.
    """
    out: list[dict] = []

    if fail_under is not None:
        expressions = [f"composite>={fail_under}"] + list(expressions)

    for expression in expressions:
        path, op, threshold = parse(expression)
        value, source = resolve(payload, path)
        record = {"name": expression, "metric": path, "operator": op,
                  "threshold": threshold, "value": value, "source": source}
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            record["passed"] = False
            record["message"] = (
                f"{path} was not measured in this run, so "
                f"'{expression}' cannot be satisfied")
        else:
            passed = _OPS[op](float(value), threshold)
            record["passed"] = passed
            record["message"] = (
                f"{path} = {_num(value)} [{source}] "
                f"{'meets' if passed else 'does not meet'} "
                f"{op} {_num(threshold)}")
        out.append(record)
    return out


def _num(value: float) -> str:
    """Readable number: thousands separators, never scientific notation.

    A significant-digit format is wrong here in both directions — it renders
    100000000 as ``1e+08`` and 4260.991288 with ten digits nobody wants — so
    whole numbers print whole and fractions get two decimals.
    """
    v = float(value)
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v):,}"
    return f"{v:,.2f}"


def failed(results: list[dict]) -> list[dict]:
    return [r for r in results if not r.get("passed")]


def render(results: list[dict]) -> str:
    """Terminal block summarising every gate."""
    if not results:
        return ""
    lines = []
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(f"  [{mark}] {r['message']}")
    bad = len(failed(results))
    lines.append(f"  {len(results) - bad}/{len(results)} threshold(s) met")
    return "\n".join(lines)
