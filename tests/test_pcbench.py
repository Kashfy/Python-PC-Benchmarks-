"""Test suite for pcbench.

Uses stdlib ``unittest`` so the suite runs anywhere the tool itself runs, with
no test dependencies to install:

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcbench import cli, compare, core, report, scoring, system, workloads  # noqa: E402


# --------------------------------------------------------------------------- #
class TestCore(unittest.TestCase):
    def test_summarize_empty(self):
        s = core.summarize([])
        self.assertEqual(s["median"], 0.0)
        self.assertEqual(s["samples"], [])

    def test_summarize_single_sample_has_zero_stdev(self):
        s = core.summarize([42.0])
        self.assertEqual(s["median"], 42.0)
        self.assertEqual(s["stdev"], 0.0)
        self.assertEqual(s["cv"], 0.0)

    def test_summarize_statistics(self):
        s = core.summarize([10.0, 20.0, 30.0])
        self.assertEqual(s["median"], 20.0)
        self.assertEqual(s["mean"], 20.0)
        self.assertEqual(s["min"], 10.0)
        self.assertEqual(s["max"], 30.0)
        self.assertGreater(s["cv"], 0)

    def test_median_resists_outlier(self):
        # The headline figure must not be dragged by a single bad repeat.
        s = core.summarize([100.0, 101.0, 5.0])
        self.assertEqual(s["median"], 100.0)

    def test_stability_note_thresholds(self):
        self.assertEqual(core.stability_note(0.01), "excellent")
        self.assertEqual(core.stability_note(0.04), "good")
        self.assertEqual(core.stability_note(0.08), "fair")
        self.assertEqual(core.stability_note(0.5), "unstable")

    def test_timed_loop_runs_at_least_target(self):
        elapsed, count = core.timed_loop(lambda: sum(range(100)), 0.05)
        self.assertGreaterEqual(elapsed, 0.05)
        self.assertGreater(count, 0)

    def test_warmup_runs_at_least_once(self):
        calls = []
        n = core.warmup(lambda: calls.append(1), 0.1)
        self.assertGreaterEqual(n, 1)
        self.assertEqual(n, len(calls))

    def test_check_exact_raises_on_mismatch(self):
        core.check_exact("x", 5, 5)
        with self.assertRaises(core.ValidationError):
            core.check_exact("x", 5, 6)

    def test_check_close_tolerance(self):
        core.check_close("x", 1.0, 1.0 + 1e-12)
        with self.assertRaises(core.ValidationError):
            core.check_close("x", 1.0, 2.0)

    def test_check_close_handles_zero_expected(self):
        core.check_close("x", 0.0, 0.0)
        with self.assertRaises(core.ValidationError):
            core.check_close("x", 1.0, 0.0)


# --------------------------------------------------------------------------- #
class TestArchNormalization(unittest.TestCase):
    def test_x86_spellings_converge(self):
        # The same chip must not rank as two different architectures just
        # because Windows and Linux name it differently.
        for m in ("x86_64", "AMD64", "amd64", "x64"):
            self.assertEqual(system.arch_family(m), "x86-64")

    def test_arm64_spellings(self):
        for m in ("arm64", "aarch64", "ARM64", "arm64e"):
            self.assertEqual(system.arch_family(m), "ARM64")

    def test_arm32_prefix(self):
        self.assertEqual(system.arch_family("armv7l"), "ARM32")

    def test_other_families(self):
        self.assertEqual(system.arch_family("riscv64"), "RISC-V 64")
        self.assertEqual(system.arch_family("ppc64le"), "PowerPC 64")
        self.assertEqual(system.arch_family("s390x"), "IBM Z")

    def test_unknown_passthrough(self):
        self.assertEqual(system.arch_family("weird"), "weird")
        self.assertEqual(system.arch_family(""), "unknown")


class TestSystemProbes(unittest.TestCase):
    def test_inventory_has_required_keys(self):
        info = system.inventory()
        for key in ("hostname", "os", "architecture", "arch_family",
                    "arch_bits", "cpu_model", "cpu_cores_logical",
                    "ram_total_bytes", "python_version", "gil_enabled"):
            self.assertIn(key, info)

    def test_logical_cores_positive(self):
        self.assertGreaterEqual(system.inventory()["cpu_cores_logical"], 1)

    def test_arch_bits_valid(self):
        self.assertIn(system.inventory()["arch_bits"], (32, 64))

    def test_machine_state_keys(self):
        st = system.machine_state()
        for key in ("on_ac_power", "load_average", "thermal"):
            self.assertIn(key, st)

    def test_warning_on_battery(self):
        w = system.state_warnings({"on_ac_power": False})
        self.assertTrue(any("BATTERY" in x for x in w))

    def test_warning_on_high_load(self):
        w = system.state_warnings({"on_ac_power": True, "load_per_core": 0.9})
        self.assertTrue(any("busy" in x for x in w))

    def test_no_warning_when_idle_on_ac(self):
        self.assertEqual(
            system.state_warnings({"on_ac_power": True, "load_per_core": 0.05}),
            [])


# --------------------------------------------------------------------------- #
class TestWorkloadCorrectness(unittest.TestCase):
    """The workloads must compute known-correct answers.

    These double as the hardware-stability check: on healthy hardware they
    always pass, so a failure is a real signal.
    """

    def test_prime_count_matches_constant(self):
        self.assertEqual(workloads.cpu_integer_chunk(),
                         workloads.EXPECTED_PRIME_COUNT)

    def test_float_sum_matches_constant(self):
        self.assertAlmostEqual(workloads.cpu_float_chunk(),
                               workloads.EXPECTED_FLOAT_SUM, places=6)

    def test_is_prime_edges(self):
        self.assertFalse(workloads._is_prime(0))
        self.assertFalse(workloads._is_prime(1))
        self.assertTrue(workloads._is_prime(2))
        self.assertTrue(workloads._is_prime(3))
        self.assertFalse(workloads._is_prime(4))
        self.assertTrue(workloads._is_prime(7919))
        self.assertFalse(workloads._is_prime(7917))

    def test_corpus_is_deterministic_and_compressible(self):
        a, b = workloads._corpus(64 * 1024), workloads._corpus(64 * 1024)
        self.assertEqual(a, b, "corpus must be identical across machines")
        import zlib
        self.assertLess(len(zlib.compress(a, 6)), len(a) / 2,
                        "corpus must be compressible or zlib result is moot")


class TestWorkloadsRun(unittest.TestCase):
    """Very short runs, just to prove each returns a well-formed result."""

    def test_cpu_integer(self):
        r = workloads.bench_cpu_integer(0.05, 1)
        self.assertGreater(r["rate"], 0)
        self.assertEqual(r["unit"], "primes/s")
        self.assertTrue(r["validated"])

    def test_cpu_float(self):
        r = workloads.bench_cpu_float(0.05, 1)
        self.assertGreater(r["rate"], 0)

    def test_hashing(self):
        r = workloads.bench_hashing(0.05, 1)
        self.assertGreater(r["rate"], 0)
        self.assertEqual(r["unit"], "MB/s")

    def test_compression(self):
        self.assertGreater(workloads.bench_compression(0.05, 1)["rate"], 0)

    def test_json(self):
        self.assertGreater(workloads.bench_json(0.05, 1)["rate"], 0)

    def test_memory(self):
        r = workloads.bench_memory(0.05, 1, buf_mb=4)
        self.assertGreater(r["rate"], 0)

    def test_cache_sweep_is_monotonic_enough(self):
        r = workloads.bench_cache_sweep(0.3, ram_bytes=8 * 1024 ** 3)
        self.assertIn("points", r)
        self.assertGreaterEqual(len(r["points"]), 2)
        # Large working sets must not be faster than the peak.
        self.assertLessEqual(r["dram_mb_per_s"], r["peak_mb_per_s"])

    def test_disk(self):
        with tempfile.TemporaryDirectory() as d:
            r = workloads.bench_disk(0.05, 1, file_mb=8, out_dir=d)
            if not r.get("skipped"):
                self.assertGreater(r["write_rate"], 0)
                self.assertGreater(r["random_read_iops"], 0)
                self.assertIn("cache_bypassed", r)

    def test_disk_skips_when_space_insufficient(self):
        with tempfile.TemporaryDirectory() as d:
            r = workloads.bench_disk(0.05, 1, file_mb=10 ** 9, out_dir=d)
            self.assertTrue(r.get("skipped"))


# --------------------------------------------------------------------------- #
class TestScoring(unittest.TestCase):
    def test_baseline_rate_scores_100(self):
        results = {"cpu_int": {"rate": scoring.BASELINES["cpu_int"]}}
        s = scoring.compute_scores(results)
        self.assertAlmostEqual(s["subscores"]["cpu_int"], 100.0, places=4)
        self.assertAlmostEqual(s["composite"], 100.0, places=4)

    def test_double_baseline_scores_200(self):
        results = {"cpu_int": {"rate": scoring.BASELINES["cpu_int"] * 2}}
        self.assertAlmostEqual(
            scoring.compute_scores(results)["subscores"]["cpu_int"], 200.0,
            places=4)

    def test_composite_is_geometric_mean(self):
        results = {"cpu_int": {"rate": scoring.BASELINES["cpu_int"] * 4},
                   "memory": {"rate": scoring.BASELINES["memory"] * 1}}
        # geometric mean of 400 and 100 is 200, not 250
        self.assertAlmostEqual(scoring.compute_scores(results)["composite"],
                               200.0, places=1)

    def test_empty_results_score_zero(self):
        s = scoring.compute_scores({})
        self.assertEqual(s["composite"], 0.0)
        self.assertEqual(s["subscores"], {})

    def test_skipped_and_errored_entries_ignored(self):
        results = {"disk": {"skipped": True, "write_rate": 999},
                   "cpu_int": {"error": "boom"}}
        self.assertEqual(scoring.compute_scores(results)["subscores"], {})

    def test_zero_rate_ignored_not_crashing_log(self):
        # log(0) would raise; a zero rate must simply be excluded.
        self.assertEqual(
            scoring.compute_scores({"cpu_int": {"rate": 0}})["composite"], 0.0)

    def test_disk_uses_correct_fields(self):
        results = {"disk": {"write_rate": scoring.BASELINES["disk_write"],
                            "read_rate": scoring.BASELINES["disk_read"],
                            "random_read_iops": scoring.BASELINES["disk_iops"]}}
        sub = scoring.compute_scores(results)["subscores"]
        self.assertAlmostEqual(sub["disk_write"], 100.0, places=4)
        self.assertAlmostEqual(sub["disk_read"], 100.0, places=4)
        self.assertAlmostEqual(sub["disk_iops"], 100.0, places=4)

    def test_category_rollup(self):
        cats = scoring.category_scores({"cpu_int": 100, "cpu_float": 400,
                                        "memory": 50})
        self.assertAlmostEqual(cats["cpu"], 200.0, places=1)
        self.assertAlmostEqual(cats["memory"], 50.0, places=1)


# --------------------------------------------------------------------------- #
class TestCLI(unittest.TestCase):
    def test_parse_duration_units(self):
        self.assertEqual(cli.parse_duration("90"), 90.0)
        self.assertEqual(cli.parse_duration("30s"), 30.0)
        self.assertEqual(cli.parse_duration("5m"), 300.0)
        self.assertEqual(cli.parse_duration("1h"), 3600.0)

    def test_parse_duration_rejects_bad_input(self):
        for bad in ("abc", "", "-5", "0", "5x"):
            with self.assertRaises(ValueError):
                cli.parse_duration(bad)

    def test_select_tests_default_is_all(self):
        self.assertEqual(cli.select_tests("", ""), cli.DEFAULT_TESTS)

    def test_select_tests_subset(self):
        self.assertEqual(cli.select_tests("cpu_int,memory", ""),
                         ["cpu_int", "memory"])

    def test_select_tests_skip(self):
        got = cli.select_tests("", "disk,cache_sweep")
        self.assertNotIn("disk", got)
        self.assertNotIn("cache_sweep", got)
        self.assertIn("cpu_int", got)

    def test_select_tests_rejects_unknown(self):
        with self.assertRaises(ValueError):
            cli.select_tests("bogus", "")
        with self.assertRaises(ValueError):
            cli.select_tests("", "bogus")

    def test_every_selectable_test_has_a_runner(self):
        # Guards against adding a test name without wiring its implementation.
        args = cli.build_parser().parse_args([])
        info = {"cpu_cores_logical": 2, "ram_total_bytes": 8 * 1024 ** 3}
        runners = cli._runners(args, info, tempfile.gettempdir())
        for name in cli.TESTS:
            self.assertIn(name, runners)

    def test_compare_flag_exits_cleanly_without_history(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cli.main(["--compare", "--output-dir", d]), 0)


# --------------------------------------------------------------------------- #
class TestReportPersistence(unittest.TestCase):
    def _payload(self):
        return {
            "tool": "pcbench", "version": "3.0",
            "timestamp_utc": "2026-08-21T01:02:03Z",
            "config": {}, "state": {"on_ac_power": True}, "warnings": [],
            "system": {"hostname": "h1", "os": "Linux", "architecture": "x86_64",
                       "arch_family": "x86-64", "cpu_model": "Test CPU",
                       "cpu_cores_physical": 4, "cpu_cores_logical": 8,
                       "ram_total_gb": 16.0, "platform": "Linux-test",
                       "python_version": "3.12.0",
                       "python_implementation": "CPython",
                       "arch_bits": 64, "byte_order": "little"},
            "results": {"cpu_int": {"rate": 1000.0, "cv": 0.01},
                        "disk": {"write_rate": 100.0, "read_rate": 200.0,
                                 "random_read_iops": 5000.0,
                                 "cache_bypassed": True, "note": "ok"}},
            "native": None, "sustained": None,
            "scores": {"subscores": {"cpu_int": 50.0}, "composite": 50.0},
        }

    def test_json_roundtrip(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = report.save_json(self._payload(), d)
            with open(p) as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["system"]["hostname"], "h1")
            self.assertEqual(loaded["scores"]["composite"], 50.0)

    def test_csv_header_and_row(self):
        with tempfile.TemporaryDirectory() as d:
            p = report.append_csv(self._payload(), d)
            with open(p, newline='') as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["hostname"], "h1")
            self.assertEqual(float(rows[0]["disk_iops"]), 5000.0)

    def test_csv_appends_second_run(self):
        with tempfile.TemporaryDirectory() as d:
            report.append_csv(self._payload(), d)
            p = report.append_csv(self._payload(), d)
            with open(p, newline='') as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), 2)

    def test_csv_rotates_on_schema_change(self):
        # An older file with different columns must not corrupt new rows.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "benchmarks.csv")
            with open(p, "w", newline="") as f:
                f.write("old,columns\n1,2\n")
            report.append_csv(self._payload(), d)
            self.assertTrue(os.path.exists(p + ".v2.bak"))
            with open(p, newline='') as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["hostname"], "h1")

    def test_html_is_self_contained(self):
        with tempfile.TemporaryDirectory() as d:
            p = report.save_html(self._payload(), d)
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("<html", text)
            self.assertIn("Test CPU", text)
            # A strict no-external-assets rule keeps the report portable.
            for bad in ("http://", "https://", "<script"):
                self.assertNotIn(bad, text)

    def test_html_escapes_hostile_hostname(self):
        payload = self._payload()
        payload["system"]["hostname"] = "<script>alert(1)</script>"
        with tempfile.TemporaryDirectory() as d:
            with open(report.save_html(payload, d), encoding="utf-8") as fh:
                text = fh.read()
            self.assertNotIn("<script>alert", text)
            self.assertIn("&lt;script&gt;", text)


# --------------------------------------------------------------------------- #
class TestCompare(unittest.TestCase):
    ROWS = [
        {"timestamp_utc": "2026-01-01T00:00:00Z", "hostname": "a",
         "cpu_model": "CPU A", "arch_family": "x86-64",
         "composite_score": "100", "cpu_int_primes_s": "1000"},
        {"timestamp_utc": "2026-02-01T00:00:00Z", "hostname": "a",
         "cpu_model": "CPU A", "arch_family": "x86-64",
         "composite_score": "150", "cpu_int_primes_s": "1500"},
        {"timestamp_utc": "2026-01-15T00:00:00Z", "hostname": "b",
         "cpu_model": "CPU B", "arch_family": "ARM64",
         "composite_score": "200", "cpu_int_primes_s": "2000"},
    ]

    def test_latest_per_host_dedupes(self):
        latest = compare.latest_per_host(self.ROWS)
        self.assertEqual(len(latest), 2)
        by_host = {r["hostname"]: r for r in latest}
        self.assertEqual(by_host["a"]["composite_score"], "150")

    def test_latest_per_host_sorted_by_score(self):
        latest = compare.latest_per_host(self.ROWS)
        self.assertEqual(latest[0]["hostname"], "b")

    def test_render_table_contains_hosts(self):
        out = compare.render_table(self.ROWS)
        self.assertIn("CPU A", out)
        self.assertIn("CPU B", out)
        self.assertIn("(best)", out)

    def test_render_table_all_runs(self):
        out = compare.render_table(self.ROWS, all_runs=True)
        self.assertIn("3 machine(s)", out)

    def test_render_empty_history(self):
        self.assertIn("No history", compare.render_table([]))

    def test_missing_history_file(self):
        self.assertEqual(compare.load_history("/nonexistent/x.csv"), [])

    def test_malformed_score_does_not_crash(self):
        rows = [{"hostname": "x", "composite_score": "not-a-number"}]
        self.assertIsInstance(compare.render_table(rows), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
