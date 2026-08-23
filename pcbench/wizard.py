"""Guided, multi-step chooser for the terminal.

``--menu`` used to print one flat list of shortcuts. That works until the
thing you want is not on the list, and it cannot express the questions that
actually shape a run: how deep, which sections, how long, what to write out.
This module asks a short series of them instead — what kind of work, then
which one, then the options that apply to it, then a confirmation screen —
the way an OS installer does.

Two rules hold on every screen. ``b`` steps back and ``q`` leaves, so no
answer is a trap. And nothing starts until the confirmation screen is
accepted, which then prints the exact command line it assembled, so the
questions are a way to learn the flags rather than a substitute for them.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime

from . import __version__
from . import hwinfo
from .report import WIDTH


class _Back(Exception):
    """The user asked for the previous screen."""


class _Quit(Exception):
    """The user asked to leave without running anything."""


#: Returned by a step that does not apply to the answers given so far. The
#: step machine then keeps moving in whichever direction it was already
#: going, so a skipped screen is invisible going forward *and* going back —
#: otherwise stepping back would land on a screen that immediately returns
#: and bounce the user forward again.
_SKIP = object()

_BACK_WORDS = {"b", "back"}
_QUIT_WORDS = {"q", "quit", "exit"}
_FOOTER = "  [number] choose    [b] back    [q] quit"
_FOOTER_MULTI = "  [numbers] choose    [b] back    [q] quit"
_FOOTER_TEXT = "  [value] answer    [b] back    [q] quit"


def _cli():
    """The CLI module, imported late because it imports this one."""
    from . import cli
    return cli


# --------------------------------------------------------------------------- #
# Screen drawing and prompts
# --------------------------------------------------------------------------- #
def _fit(text: str, width: int) -> str:
    return text if len(text) <= width else text[:max(0, width - 3)] + "..."


def _header(trail: list[str]) -> None:
    """The installer-style banner: where you are, and how you got here."""
    print()
    print("=" * WIDTH)
    print("  " + _fit(" > ".join([f"pcbench {__version__}"] + list(trail)),
                      WIDTH - 4))
    print("=" * WIDTH)


def _draw(trail, question: str, lines, error: str,
          footer: str = _FOOTER) -> None:
    _header(trail)
    print(f"\n  {question}\n")
    for line in lines:
        print(line)
    if error:
        print(f"\n  !  {error}")
    print(f"\n{footer}")


def _input(label: str, default: str = "") -> str:
    """One line of input, with back/quit handled before the caller sees it."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"\n  {label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise _Quit
    lowered = answer.lower()
    if lowered in _QUIT_WORDS:
        raise _Quit
    if lowered in _BACK_WORDS:
        raise _Back
    return answer or default


def _options_block(options) -> list[str]:
    lines = []
    for number, option in enumerate(options, 1):
        label = option[0]
        detail = option[1] if len(option) > 1 else ""
        lines.append(f"    {number:>2}. {label}")
        if detail:
            lines.append("        " + _fit(detail, WIDTH - 10))
    return lines


def _choose(trail, question, options, body=(), note="") -> int:
    """Ask for one of ``options``. Returns its index."""
    error = ""
    while True:
        lines = list(body) + ([""] if body else []) + _options_block(options)
        if note:
            lines += ["", f"  {note}"]
        _draw(trail, question, lines, error)
        answer = _input("Choice")
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        error = f"{answer!r} is not one of 1-{len(options)}"


def _parse_selection(answer: str, count: int, names=None) -> list[int]:
    """Turn ``'1,4-6'``, ``'all'``, or ``'cpu_int,disk'`` into indexes."""
    lowered = answer.strip().lower()
    if lowered in ("all", "*"):
        return list(range(count))
    if lowered in ("none", "-"):
        return []
    chosen: list[int] = []
    for token in lowered.replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split("-", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            low, high = int(parts[0]), int(parts[1])
            if not 1 <= low <= high <= count:
                raise ValueError(f"{token!r} is outside 1-{count}")
            chosen += list(range(low - 1, high))
        elif token.isdigit():
            number = int(token)
            if not 1 <= number <= count:
                raise ValueError(f"{token!r} is outside 1-{count}")
            chosen.append(number - 1)
        elif names and token in names:
            chosen.append(names.index(token))
        else:
            raise ValueError(f"{token!r} is not one of the choices")
    seen, ordered = set(), []
    for index in chosen:
        if index not in seen:
            seen.add(index)
            ordered.append(index)
    return ordered


def _multi(trail, question, options, names=None, default="none",
           allow_empty=True, note="") -> list[int]:
    """Ask for any number of ``options``. Returns their indexes."""
    hint = ("Numbers (1,4), ranges (1-6), 'all', or 'none'"
            + (", or names" if names else ""))
    error = ""
    while True:
        lines = _options_block(options) + ["", f"  {note or hint}"]
        _draw(trail, question, lines, error, _FOOTER_MULTI)
        answer = _input("Choice(s)", default)
        try:
            chosen = _parse_selection(answer, len(options), names)
        except ValueError as e:
            error = str(e)
            continue
        if not chosen and not allow_empty:
            error = "choose at least one"
            continue
        return chosen


def _text(trail, question, label, default, validate=None, body=()) -> str:
    """Ask for a free-text value, re-asking until ``validate`` accepts it."""
    error = ""
    while True:
        _draw(trail, question, list(body), error, _FOOTER_TEXT)
        answer = _input(label, default)
        if validate is None:
            return answer
        try:
            validate(answer)
            return answer
        except ValueError as e:
            error = str(e)


def _confirm(trail, argv: list[str], plan: list[str]) -> None:
    """The last screen: show the command and the plan, then run or go back."""
    command = ("pcbench " + " ".join(argv)).strip()
    body = ["  This is what will run:", "", f"      {command}", "",
            "  Which means:"] + [f"    - {line}" for line in plan]
    index = _choose(trail + ["Confirm"], "Ready?", [
        ("Run it now", ""),
        ("Go back and change something", ""),
        ("Quit without running anything", ""),
    ], body=body, note="Nothing has started yet.")
    if index == 1:
        raise _Back
    if index == 2:
        raise _Quit


def _validate(argv: list[str]) -> str:
    """Reject an argv the parser would refuse, before the user commits."""
    import argparse
    import contextlib
    import io
    parser = _cli().build_parser()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return f"pcbench {' '.join(argv)} is not a valid command line"
    return ""


def _run_steps(steps, state: dict) -> list[str]:
    """Walk the screens, honouring back, until the last one commits an argv."""
    index, direction = 0, 1
    while True:
        if index < 0:
            raise _Back
        if index >= len(steps):
            return state["argv"]
        try:
            if steps[index](state) is not _SKIP:
                direction = 1
        except _Back:
            direction = -1
        index += direction


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
_BENCH_KINDS = [
    ("Quick pass",
     "The four tests that matter most, about a minute"),
    ("The full suite",
     "Every test except video; several minutes"),
    ("A profile",
     "A curated set for one kind of machine or job"),
    ("Individual tests",
     "Pick from the twenty-two tests yourself"),
    ("AI / data science",
     "LLM prefill and decode tokens/s, dataframes, batch scaling"),
    ("Storage I/O jobs",
     "Database, sequential, log-write and mixed-VM patterns"),
    ("Reference standards",
     "STREAM, LINPACK and a CoreMark-style suite"),
]

_DEPTHS = [
    ("Quick", "1 second per test, 2 repeats - noisy but fast"),
    ("Standard", "3 seconds per test, 3 repeats - the default"),
    ("Thorough", "6 seconds per test, 5 repeats - steadier medians"),
    ("Custom", "Choose the seconds and repeats yourself"),
]

#: (flags, label, detail). Each is off unless asked for, because each one
#: costs time or needs a privilege the plain run does not.
_EXTRAS = [
    (["--sustained", "5m"], "Sustained load test (5 minutes)",
     "Hold every core busy and watch for thermal throttling"),
    (["--counters"], "Hardware performance counters",
     "IPC, cache and branch misses; needs perf on Linux"),
    (["--health"], "Health checks",
     "RAM integrity test and drive SMART status"),
    (["--energy"], "Energy to solution",
     "Joules consumed by a fixed workload"),
    (["--numa"], "NUMA topology",
     "Node layout and memory affinity"),
    (["--disk-all"], "Benchmark every writable filesystem",
     "Not just the one holding the output directory"),
    (["--no-accel"], "Skip GPU and NPU benchmarks",
     "Faster, and avoids driver surprises; inventory is still reported"),
    (["--force"], "Run even in poor conditions",
     "On battery, under load, or already throttling"),
]

_REPORTS = [
    (["--html"], "Self-contained HTML report",
     "One file to open in a browser or send to someone"),
    (["--spec-sheet"], "One-page Markdown spec sheet", ""),
    (["--json-stdout"], "Print the whole payload as JSON",
     "Replaces the readable console report"),
    (["--no-save"], "Write nothing to disk",
     "Console only - no JSON, CSV or HTML"),
]


def _bench_kind(state) -> None:
    trail = ["Benchmark"]
    while True:
        index = _choose(trail, "What kind of benchmark?", _BENCH_KINDS)
        try:
            if index == 0:
                state.update(mode="suite", select=["--quick"], depth="quick",
                             label="the quick pass: integer, multicore, "
                                   "memory and disk")
            elif index == 1:
                state.update(mode="suite", select=[], depth=None,
                             label="the default suite, every test except "
                                   "video")
            elif index == 2:
                state.update(_bench_profile(trail))
            elif index == 3:
                state.update(_bench_tests(trail))
            elif index == 4:
                state.update(mode="datascience", select=[], depth=None,
                             label="the AI and data-science measurements")
            elif index == 5:
                state.update(mode="io", select=[], depth=None,
                             label="the storage I/O job suite")
            else:
                state.update(mode="standards", select=[], depth=None,
                             label="the reference standards: STREAM, LINPACK "
                                   "and CoreMark")
        except _Back:
            continue
        return None


def _bench_profile(trail) -> dict:
    cli = _cli()
    names = list(cli.PROFILES)
    options = [(name, ", ".join(cli.PROFILES[name])) for name in names]
    index = _choose(trail + ["Profile"], "Which profile?", options,
                    note="Each line lists the tests that profile runs.")
    name = names[index]
    return dict(mode="suite", select=["--profile", name], depth=None,
                label=f"the {name} profile "
                      f"({len(cli.PROFILES[name])} tests)")


def _bench_tests(trail) -> dict:
    cli = _cli()
    names = list(cli.TESTS)
    options = [(name, cli.DESCRIPTIONS.get(name, "")) for name in names]
    chosen = _multi(trail + ["Tests"], "Which tests?", options, names=names,
                    default="", allow_empty=False,
                    note="Numbers (1,4,7), ranges (1-6), names "
                         "(cpu_int,disk), or 'all'")
    picked = [names[i] for i in chosen]
    return dict(mode="suite", select=["--only", ",".join(picked)], depth=None,
                label=f"{len(picked)} test(s): {', '.join(picked)}")


def _bench_scope(state) -> object:
    """Focused runs still need one test, because a run with none is empty."""
    if state["mode"] not in ("datascience", "io", "standards"):
        return _SKIP
    index = _choose(["Benchmark", "Scope"],
                    "Run the standard test suite as well?", [
        ("No - just this, plus a one-second CPU check",
         "A run needs at least one test, so cpu_int runs briefly first"),
        ("Yes - the full default suite too", "Adds several minutes"),
    ])
    if index == 0:
        state["select"] = ["--only", "cpu_int", "--quick"]
        state["depth"] = "quick"
    else:
        state["select"] = []
        state["depth"] = None
    return None


def _bench_depth(state) -> object:
    if state.get("depth") == "quick":
        return _SKIP
    index = _choose(["Benchmark", "Depth"], "How long should each test run?",
                    _DEPTHS,
                    note="Longer runs average out noise; they do not change "
                         "what is measured.")
    if index == 3:
        seconds = _text(["Benchmark", "Depth", "Custom"],
                        "Seconds of work per test, per repeat.",
                        "Seconds", "3", _positive_float)
        repeats = _text(["Benchmark", "Depth", "Custom"],
                        "Repeats per test. The median is reported.",
                        "Repeats", "3", _positive_int)
        state["depth"] = ("custom", seconds, repeats)
    else:
        state["depth"] = ("quick", "standard", "thorough")[index]
    return None


def _positive_float(text: str) -> None:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a number")
    if value <= 0:
        raise ValueError("must be greater than zero")


def _positive_int(text: str) -> None:
    if not text.isdigit() or int(text) < 1:
        raise ValueError("must be a whole number of 1 or more")


def _bench_extras(state) -> None:
    chosen = _multi(["Benchmark", "Extras"],
                    "Anything else to measure while it runs?",
                    [(label, detail) for _, label, detail in _EXTRAS],
                    default="none",
                    note="All optional. Press Enter for none of them.")
    state["extras"] = chosen
    return None


def _bench_reports(state) -> None:
    chosen = _multi(["Benchmark", "Output"], "What should it write out?",
                    [(label, detail) for _, label, detail in _REPORTS],
                    default="none",
                    note="A JSON and a CSV row are written to results/ "
                         "either way.")
    # "Write nothing" and "write this file" cannot both be honoured, so the
    # stronger answer wins rather than producing a confusing half-result.
    if 3 in chosen:
        chosen = [3]
    state["reports"] = chosen
    return None


def _bench_confirm(state) -> None:
    argv = list(state["select"])
    depth = state.get("depth")
    if depth == "quick" and "--quick" not in argv:
        argv.append("--quick")
    elif depth == "thorough":
        argv += ["--seconds", "6", "--repeats", "5"]
    elif isinstance(depth, tuple):
        argv += ["--seconds", depth[1], "--repeats", depth[2]]
    if state["mode"] == "datascience":
        argv.append("--datascience")
    elif state["mode"] == "io":
        argv.append("--io")

    plan = [state["label"]]
    if depth == "quick":
        plan.append("1 second per test, 2 repeats")
    elif depth == "thorough":
        plan.append("6 seconds per test, 5 repeats")
    elif isinstance(depth, tuple):
        plan.append(f"{depth[1]} seconds per test, {depth[2]} repeats")
    else:
        plan.append("3 seconds per test, 3 repeats (the default)")
    if state["mode"] == "standards":
        plan.append("STREAM, LINPACK and CoreMark run as part of any "
                    "benchmark")

    for index in state.get("extras", []):
        flags, label, _ = _EXTRAS[index]
        argv += flags
        plan.append(label.lower())
    for index in state.get("reports", []):
        flags, label, _ = _REPORTS[index]
        argv += flags
        plan.append(label.lower())
    if 3 not in state.get("reports", []):
        plan.append("results saved to results/ as JSON and CSV")

    problem = _validate(argv)
    if problem:
        _choose(["Benchmark", "Confirm"], problem,
                [("Go back", "")])
        raise _Back
    _confirm(["Benchmark"], argv, plan)
    state["argv"] = argv
    return None


def _benchmark() -> list[str]:
    return _run_steps([_bench_kind, _bench_scope, _bench_depth, _bench_extras,
                       _bench_reports, _bench_confirm], {})


# --------------------------------------------------------------------------- #
# Hardware stats
# --------------------------------------------------------------------------- #
def _stats_sections(state) -> None:
    names = list(hwinfo.SECTIONS)
    options = [(f"{name:<12}  {hwinfo.SECTIONS[name][0]}", "")
               for name in names]
    chosen = _multi(["Stats"], "Which facts do you want?", options,
                    names=names, default="all", allow_empty=False,
                    note="Nothing here loads the machine or runs a "
                         "benchmark.")
    state["sections"] = [names[i] for i in chosen]
    return None


def _stats_format(state) -> None:
    index = _choose(["Stats", "Format"], "How should it be printed?", [
        ("A readable report", "Grouped, with a note where a number needs one"),
        ("JSON on stdout", "For a script or a monitoring check"),
    ])
    state["json"] = index == 1
    return None


def _stats_confirm(state) -> None:
    sections = state["sections"]
    argv = ["--stats"]
    if len(sections) != len(hwinfo.SECTIONS):
        argv.append(",".join(sections))
    if state["json"]:
        argv.append("--json-stdout")
    plan = [f"{len(sections)} section(s): {', '.join(sections)}",
            "no benchmark runs; the machine is only read, not loaded"]
    _confirm(["Stats"], argv, plan)
    state["argv"] = argv
    return None


def _stats() -> list[str]:
    return _run_steps([_stats_sections, _stats_format, _stats_confirm], {})


# --------------------------------------------------------------------------- #
# Watch or stress
# --------------------------------------------------------------------------- #
_WATCH_KINDS = [
    ("watch", "Live monitor",
     "Clocks, temperature, load and memory, sampled while you watch", "60s"),
    ("sustained", "Sustained load test",
     "Hold every core busy and show the throttling curve", "5m"),
    ("soak", "Burn-in soak",
     "Run validating work and count wrong answers", "30m"),
]

_WATCH_OPTIONS = {
    "watch": [
        (["--monitor-power"], "Sample power draw too",
         "Costs a privileged subprocess per sample on macOS"),
        (["--monitor-trace", os.path.join("results", "monitor.csv")],
         "Write every raw sample to results/monitor.csv", ""),
    ],
    "sustained": [
        (["--force"], "Run even in poor conditions",
         "On battery, under load, or already throttling"),
        ([], "Run the standard benchmark suite first",
         "Otherwise only a one-second CPU check runs before the load"),
    ],
    "soak": [
        (["--force"], "Run even in poor conditions",
         "On battery, under load, or already throttling"),
        ([], "Run the standard benchmark suite first",
         "Otherwise only a one-second CPU check runs before the soak"),
    ],
}


def _watch_kind(state) -> None:
    index = _choose(["Watch"], "What would you like to watch?",
                    [(label, detail) for _, label, detail, _ in _WATCH_KINDS])
    kind, label, _, default = _WATCH_KINDS[index]
    state.update(kind=kind, label=label, duration=default)
    return None


def _watch_duration(state) -> None:
    state["duration"] = _text(
        ["Watch", state["label"]], "How long should it run?", "Duration",
        state["duration"], _duration,
        body=["  Accepts 90, 300s, 5m or 2h. Ctrl-C stops it early and "
              "still reports."])
    return None


def _duration(text: str) -> None:
    _cli().parse_duration(text)


def _watch_options(state) -> None:
    options = _WATCH_OPTIONS[state["kind"]]
    state["options"] = _multi(
        ["Watch", state["label"], "Options"], "Any options?",
        [(label, detail) for _, label, detail in options], default="none",
        note="All optional. Press Enter for none of them.")
    return None


def _watch_confirm(state) -> None:
    kind, duration = state["kind"], state["duration"]
    options = _WATCH_OPTIONS[kind]
    chosen = state["options"]
    argv: list[str] = []
    plan: list[str] = []

    if kind == "watch":
        argv += ["--monitor", duration]
        plan.append(f"sample this machine for {duration} without loading it")
    else:
        flag = "--sustained" if kind == "sustained" else "--soak"
        argv += [flag, duration]
        if 1 in chosen:
            plan.append("the default test suite runs first")
        else:
            # A benchmark run needs at least one test; cpu_int at --quick is
            # the cheapest way to satisfy that before the real work.
            argv += ["--only", "cpu_int", "--quick"]
            plan.append("a one-second CPU check runs first")
        plan.append(f"then {duration} of "
                    + ("full load, sampling for throttling"
                       if kind == "sustained"
                       else "validated work, counting wrong answers"))
    for index in chosen:
        flags, label, _ = options[index]
        argv += flags
        if flags:
            plan.append(label.lower())

    problem = _validate(argv)
    if problem:
        _choose(["Watch", "Confirm"], problem, [("Go back", "")])
        raise _Back
    _confirm(["Watch", state["label"]], argv, plan)
    state["argv"] = argv
    return None


def _watch() -> list[str]:
    return _run_steps([_watch_kind, _watch_duration, _watch_options,
                       _watch_confirm], {})


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
#: (label, detail, argv, plan). The first four read the machine and finish
#: instantly; the last one actually exercises memory.
_HEALTH_KINDS = [
    ("Everything that can be read instantly",
     "Drive wear, battery, temperatures and power",
     ["--stats", "drives,battery,thermal,power"],
     ["read four hardware sections; nothing is loaded"]),
    ("Drive lifetime and wear",
     "Terabytes written, power-on hours, endurance remaining",
     ["--stats", "drives"],
     ["read SMART endurance data from every drive it can reach"]),
    ("Battery",
     "Cycle count and capacity against the design figure",
     ["--stats", "battery"],
     ["read the battery controller; nothing is loaded"]),
    ("Temperatures and power right now",
     "Every sensor the platform exposes",
     ["--stats", "thermal,power"],
     ["read every temperature and power sensor once"]),
    ("RAM integrity test and SMART status",
     "Writes patterns through memory and reads them back",
     None,   # built in _health_confirm: it needs the size question first
     []),
]


def _health_kind(state) -> None:
    index = _choose(["Health"], "What should be checked?",
                    [(label, detail) for label, detail, _, _ in _HEALTH_KINDS])
    state["kind"] = index
    return None


def _health_size(state) -> object:
    if state["kind"] != 4:
        return _SKIP
    state["health_mb"] = _text(
        ["Health", "RAM integrity"], "How much memory should be covered?",
        "Megabytes", "256", _positive_int,
        body=["  Larger covers more of the module but takes proportionally",
              "  longer. 256 MB takes a few seconds."])
    return None


def _health_confirm(state) -> None:
    label, _, argv, plan = _HEALTH_KINDS[state["kind"]]
    if argv is None:
        megabytes = state["health_mb"]
        argv = ["--health", "--health-mb", megabytes,
                "--only", "cpu_int", "--quick"]
        plan = [f"write and verify patterns across {megabytes} MB of memory",
                "read SMART status from every drive it can reach",
                "run a one-second CPU check, because a run needs one test"]
    problem = _validate(argv)
    if problem:
        _choose(["Health", "Confirm"], problem, [("Go back", "")])
        raise _Back
    _confirm(["Health", label], list(argv), list(plan))
    state["argv"] = list(argv)
    return None


def _health() -> list[str]:
    return _run_steps([_health_kind, _health_size, _health_confirm], {})


# --------------------------------------------------------------------------- #
# Past runs
# --------------------------------------------------------------------------- #
def _history_kind(state) -> None:
    trail = ["History"]
    while True:
        index = _choose(trail, "What would you like to see?", [
            ("A ranked table of past runs",
             "The latest run per machine, best score first"),
            ("Every run ever recorded",
             "Including repeat runs on the same machine"),
            ("Compare two saved runs statistically",
             "Says which differences are real and which are noise"),
        ])
        if index == 0:
            state.update(argv_draft=["--compare"],
                         plan=["read results/benchmarks.csv and rank it"])
            return None
        if index == 1:
            state.update(argv_draft=["--compare", "--all-runs"],
                         plan=["read results/benchmarks.csv and list "
                               "every row"])
            return None
        try:
            state.update(_history_pair(trail))
        except _Back:
            continue
        return None


def _saved_runs() -> list[str]:
    """Saved run payloads, newest first — the order people think in."""
    paths = glob.glob(os.path.join("results", "*.json"))
    return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)


def _history_pair(trail) -> dict:
    paths = _saved_runs()
    if len(paths) < 2:
        _choose(trail + ["A/B"],
                f"Only {len(paths)} saved run(s) in results/ — two are "
                f"needed to compare.", [("Go back", "")])
        raise _Back

    def label(path):
        when = datetime.fromtimestamp(os.path.getmtime(path))
        return f"{os.path.basename(path):<34} {when:%Y-%m-%d %H:%M}"

    options = [(label(p), "") for p in paths]
    first = _choose(trail + ["A/B", "Baseline"],
                    "Which run is the baseline?", options)
    remaining = [p for i, p in enumerate(paths) if i != first]
    second = _choose(trail + ["A/B", "Candidate"],
                     "Which run is the candidate?",
                     [(label(p), "") for p in remaining],
                     body=[f"  Baseline: {os.path.basename(paths[first])}"])
    return dict(argv_draft=["--compare-runs", paths[first],
                            remaining[second]],
                plan=[f"baseline: {paths[first]}",
                      f"candidate: {remaining[second]}",
                      "report which differences are statistically real",
                      "exit non-zero if the candidate regressed"])


def _history_confirm(state) -> None:
    argv = list(state["argv_draft"])
    _confirm(["History"], argv, state["plan"])
    state["argv"] = argv
    return None


def _history() -> list[str]:
    return _run_steps([_history_kind, _history_confirm], {})


# --------------------------------------------------------------------------- #
# Shortcuts — the flat list, for people who already know what they want
# --------------------------------------------------------------------------- #
#: (label, argv). Every entry is a real command line, shown before it runs,
#: so the list teaches the flags rather than hiding them.
SHORTCUTS: list[tuple[str, list[str]]] = [
    ("Quick benchmark (about a minute)", ["--quick"]),
    ("Full benchmark (default set)", []),
    ("CPU only", ["--profile", "cpu"]),
    ("Storage only", ["--profile", "storage"]),
    ("Application workloads (database, render, encode)", ["--profile", "apps"]),
    ("AI / data science", ["--datascience"]),
    ("Reference standards (STREAM, LINPACK, CoreMark)",
     ["--only", "cpu_int", "--quick"]),
    ("Hardware stats — everything, no benchmarking", ["--stats"]),
    ("Battery health", ["--stats", "battery"]),
    ("SSD lifetime and wear", ["--stats", "drives"]),
    ("Temperatures and power", ["--stats", "thermal,power"]),
    ("GPU and NPU inventory", ["--stats", "gpu"]),
    ("Live monitor (60 seconds)", ["--monitor", "60s"]),
    ("Stability soak (10 minutes)", ["--soak", "10m"]),
    ("Compare past runs", ["--compare"]),
]


def _shortcuts() -> list[str]:
    options = [(label, " ".join(argv) or "(no flags)")
               for label, argv in SHORTCUTS]
    index = _choose(["Shortcuts"], "Pick a task.", options,
                    note="The flags each one maps to are shown underneath.")
    return list(SHORTCUTS[index][1])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
_MAIN = [
    ("Run a benchmark", "Measure how fast this machine is"),
    ("Read hardware stats",
     "Battery, SSD endurance, temperatures, GPUs - nothing is loaded"),
    ("Watch or stress the machine",
     "Live monitor, thermal throttling test, burn-in soak"),
    ("Check hardware health", "RAM integrity, SMART status, drive wear"),
    ("Look at past runs", "Rank the history, or compare two saved runs"),
    ("Shortcuts", "The flat list, if you already know what you want"),
]


def _main_menu() -> list[str]:
    handlers = [_benchmark, _stats, _watch, _health, _history, _shortcuts]
    while True:
        index = _choose(["Main menu"], "What would you like to do?", _MAIN,
                        note="Every screen takes 'b' to go back and 'q' to "
                             "quit.")
        try:
            return handlers[index]()
        except _Back:
            continue


def run() -> list[str] | None:
    """Ask the questions. Returns the argv to run, or None to do nothing."""
    try:
        argv = _main_menu()
    except _Quit:
        print("\n  Nothing was run.\n")
        return None
    command = ("pcbench " + " ".join(argv)).strip()
    print(f"\n  Running: {command}")
    print("  Run that command directly next time to skip the questions.\n")
    return argv
