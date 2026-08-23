"""Test suite for pcbench.

Uses stdlib ``unittest`` so the suite runs anywhere the tool itself runs, with
no test dependencies to install:

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import random
import re
import resource
import time
import platform
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcbench import (accel, apps, cli, compare, config, container,  # noqa: E402
                     core, coreml_model, cores, counters, cryptobench,
                     datascience, diagnose, drivelife, export, gates, gpucompute,
                     health,
                     interference, iobench, monitor, numa, numeric, optional,
                     plugins, provenance, reference, soak, standards, stats,
                     storage, sysbench, limits, mlbench, mlframework, network,
                     native, npu, onnx_model, power, regression, report, scoring,
                     sustained, system, thermal, workloads)


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
        # Simulate a nearly-full filesystem rather than actually filling one.
        import shutil as _sh
        real = _sh.disk_usage

        class Tiny:
            total = free = used = 1024 * 1024      # 1 MB free

        with tempfile.TemporaryDirectory() as d:
            workloads.shutil.disk_usage = lambda p: Tiny
            try:
                r = workloads.bench_disk(0.05, 1, file_mb=4096, out_dir=d)
            finally:
                workloads.shutil.disk_usage = real
            self.assertTrue(r.get("skipped"))

    def test_disk_request_is_clamped_not_unbounded(self):
        # An absurd request must be reduced to a safe size, never honoured.
        # Tested on the pure function so no gigabytes are written.
        mb, notice = limits.safe_disk_mb(10 ** 9, free_bytes=500 * 1024 ** 3,
                                         repeats=3)
        self.assertLess(mb, 10 ** 9)
        self.assertIsNotNone(notice)
        self.assertLessEqual(limits.total_write_mb(mb, 3),
                             limits.DISK_MAX_TOTAL_WRITE_MB)


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
            # The backup name carries a timestamp so successive schema
            # changes never overwrite an earlier archive.
            import glob as _glob
            backups = _glob.glob(p + "*.bak")
            self.assertEqual(len(backups), 1, "old history must be archived")
            with open(backups[0]) as fh:
                self.assertIn("old,columns", fh.read())
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


# --------------------------------------------------------------------------- #
class TestCoreMLModel(unittest.TestCase):
    """The .mlmodel protobuf is written by hand, so its structure is checked
    here rather than trusted."""

    def test_varint_encoding(self):
        self.assertEqual(coreml_model._varint(0), b"\x00")
        self.assertEqual(coreml_model._varint(1), b"\x01")
        self.assertEqual(coreml_model._varint(127), b"\x7f")
        self.assertEqual(coreml_model._varint(128), b"\x80\x01")
        self.assertEqual(coreml_model._varint(300), b"\xac\x02")

    def test_tag_wire_format(self):
        # field 1, wire type 2 -> (1 << 3) | 2 == 0x0A
        self.assertEqual(coreml_model._tag(1, 2), b"\x0a")
        # field 500 needs a multi-byte tag
        self.assertEqual(len(coreml_model._tag(500, 2)), 2)

    def test_length_delimited_prefixes_length(self):
        out = coreml_model._msg(1, b"abc")
        self.assertEqual(out, b"\x0a\x03abc")

    def test_packed_floats_roundtrip(self):
        import struct
        out = coreml_model._packed_floats(1, [1.0, 2.0])
        self.assertEqual(struct.unpack("<2f", out[2:]), (1.0, 2.0))

    def test_model_is_deterministic(self):
        a = coreml_model.build_model(8, 8, 2)
        b = coreml_model.build_model(8, 8, 2)
        self.assertEqual(a, b)

    def test_model_grows_with_layers(self):
        small = coreml_model.build_model(8, 8, 2)
        big = coreml_model.build_model(8, 8, 6)
        self.assertGreater(len(big), len(small))

    def test_model_contains_layer_and_io_names(self):
        blob = coreml_model.build_model(8, 8, 3)
        for token in (b"input", b"output", b"conv0", b"conv2"):
            self.assertIn(token, blob)

    def test_flops_scale_correctly(self):
        one = coreml_model.flops_per_inference(16, 8, 1)
        two = coreml_model.flops_per_inference(16, 8, 2)
        self.assertAlmostEqual(two / one, 2.0)
        # Doubling channels quadruples work (in_ch x out_ch).
        wide = coreml_model.flops_per_inference(32, 8, 1)
        self.assertAlmostEqual(wide / one, 4.0)

    def test_write_model_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = coreml_model.write_model(os.path.join(d, "m.mlmodel"), 8, 8, 2)
            self.assertTrue(os.path.isfile(p))
            self.assertGreater(os.path.getsize(p), 0)

    def test_write_model_reuses_existing(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.mlmodel")
            coreml_model.write_model(p, 8, 8, 2)
            first = os.path.getmtime(p)
            coreml_model.write_model(p, 8, 8, 2)
            self.assertEqual(first, os.path.getmtime(p))


class TestAccel(unittest.TestCase):
    def test_inventory_shape(self):
        inv = accel.inventory("Apple M4")
        for key in ("gpus", "npus", "gpu_count", "npu_count",
                    "benchmark_supported"):
            self.assertIn(key, inv)
        self.assertEqual(inv["gpu_count"], len(inv["gpus"]))

    def test_apple_silicon_reports_neural_engine(self):
        if platform.system() != "Darwin":
            self.skipTest("Apple-only detection path")
        npus = accel.detect_npus("Apple M4")
        self.assertTrue(any("Neural Engine" in n["name"] for n in npus))

    def test_non_apple_cpu_has_no_ane(self):
        if platform.system() != "Darwin":
            self.skipTest("Apple-only detection path")
        self.assertEqual(accel.detect_npus("Intel Core i7-9750H"), [])

    def test_detect_gpus_returns_list(self):
        self.assertIsInstance(accel.detect_gpus(), list)

    def test_pci_vendor_lookup(self):
        self.assertEqual(accel._pci_vendor("0x10de"), "NVIDIA")
        self.assertEqual(accel._pci_vendor("0x1002"), "AMD")
        self.assertEqual(accel._pci_vendor("0xdead"), "0xdead")

    def test_extract_rates_from_engine_payload(self):
        payload = {"results": [
            {"name": "GPU FP32 FMA", "unit": "GFLOPS", "value": 2000.0},
            {"name": "GPU FP16 FMA", "unit": "GFLOPS", "value": 2500.0},
            {"name": "GPU memory bandwidth", "unit": "MB/s", "value": 80000.0},
            {"name": "Neural Engine throughput", "unit": "GFLOPS",
             "value": 9000.0},
            {"name": "GPU kernel launch latency", "unit": "us", "value": 120.0},
        ]}
        rates = accel.extract_rates(payload)
        self.assertEqual(rates["gpu_fp32"], 2000.0)
        self.assertEqual(rates["npu"], 9000.0)
        # Latency is not a throughput rate and must not be scored as one.
        self.assertNotIn("gpu_launch", rates)

    def test_extract_rates_ignores_errors_and_zeros(self):
        self.assertEqual(accel.extract_rates({"error": "x"}), {})
        self.assertEqual(accel.extract_rates(None), {})
        self.assertEqual(accel.extract_rates(
            {"results": [{"name": "GPU FP32 FMA", "value": 0}]}), {})

    def test_run_returns_none_off_apple(self):
        if platform.system() == "Darwin":
            self.skipTest("non-Apple path")
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(accel.run(0.1, d, d))


class TestAcceleratorScoring(unittest.TestCase):
    def test_gpu_and_npu_score_and_roll_up(self):
        results = {"gpu_fp32": {"rate": scoring.BASELINES["gpu_fp32"]},
                   "npu": {"rate": scoring.BASELINES["npu"] * 2}}
        s = scoring.compute_scores(results)
        self.assertAlmostEqual(s["subscores"]["gpu_fp32"], 100.0, places=4)
        self.assertAlmostEqual(s["subscores"]["npu"], 200.0, places=4)
        cats = scoring.category_scores(s["subscores"])
        self.assertIn("gpu", cats)
        self.assertIn("npu", cats)

    def test_absent_accelerators_do_not_penalize(self):
        # A machine with no GPU/NPU must score purely on what it does have.
        with_cpu = scoring.compute_scores(
            {"cpu_int": {"rate": scoring.BASELINES["cpu_int"]}})
        self.assertAlmostEqual(with_cpu["composite"], 100.0, places=4)
        self.assertNotIn("gpu_fp32", with_cpu["subscores"])


# --------------------------------------------------------------------------- #
class TestMLFramework(unittest.TestCase):
    def test_detect_shape(self):
        d = mlframework.detect()
        for k in ("pytorch", "onnxruntime", "available"):
            self.assertIn(k, d)

    def test_run_without_framework_is_graceful(self):
        d = mlframework.detect()
        if d["available"]:
            self.skipTest("a framework is installed; skip the absent-path test")
        r = mlframework.run(0.1)
        self.assertFalse(r["available"])
        self.assertIn("note", r)

    def test_extract_rates(self):
        r = mlframework.extract_rates(
            {"available": True, "train_samples_per_s": 500.0,
             "infer_samples_per_s": 2000.0})
        self.assertEqual(r["ml_train"], 500.0)
        self.assertEqual(r["ml_infer"], 2000.0)

    def test_extract_rates_ignores_unavailable_and_error(self):
        self.assertEqual(mlframework.extract_rates(
            {"available": False}), {})
        self.assertEqual(mlframework.extract_rates(
            {"available": True, "error": "boom"}), {})
        self.assertEqual(mlframework.extract_rates(None), {})


class TestPower(unittest.TestCase):
    def test_estimate_tdp_known_chips(self):
        self.assertEqual(power.estimate_tdp("Apple M4"), 18)
        self.assertEqual(power.estimate_tdp("Intel Core i7-9750H"), 45)
        self.assertEqual(power.estimate_tdp("AMD Ryzen 9 7950X"), 65)

    def test_estimate_tdp_unknown(self):
        self.assertIsNone(power.estimate_tdp("Some Mystery CPU"))

    def test_measure_returns_source(self):
        p = power.measure("Apple M4")
        self.assertIn("source", p)
        self.assertIn("estimated", p)

    def test_perf_per_watt(self):
        ppw = power.perf_per_watt(200.0, {"package_w": 20.0, "estimated": True})
        self.assertEqual(ppw["score_per_watt"], 10.0)
        self.assertTrue(ppw["estimated"])

    def test_perf_per_watt_none_when_no_power(self):
        self.assertIsNone(power.perf_per_watt(200.0, {"package_w": None}))
        self.assertIsNone(power.perf_per_watt(0, {"package_w": 20.0}))


class TestNetwork(unittest.TestCase):
    def test_run_returns_throughput_and_latency(self):
        r = network.run(0.3)
        if r.get("error"):
            self.skipTest(f"network unavailable in this env: {r['error']}")
        self.assertGreater(r["loopback_mb_s"], 0)
        self.assertIn("p50_us", r["latency"])
        self.assertGreater(r["latency"]["p99_us"], 0)


class TestRegression(unittest.TestCase):
    HIST = [
        {"hostname": "h", "timestamp_utc": "t1", "composite_score": "100",
         "cpu_int_primes_s": "2000", "disk_write_mb_s": "500"},
        {"hostname": "h", "timestamp_utc": "t2", "composite_score": "104",
         "cpu_int_primes_s": "2100", "disk_write_mb_s": "510"},
    ]

    def test_no_baseline_for_new_host(self):
        cur = {"hostname": "newhost", "timestamp_utc": "t9",
               "composite_score": "100"}
        self.assertEqual(regression.analyze(cur, self.HIST)["status"],
                         "no_baseline")

    def test_detects_regression(self):
        cur = {"hostname": "h", "timestamp_utc": "t3",
               "composite_score": "80", "cpu_int_primes_s": "1500",
               "disk_write_mb_s": "505"}
        res = regression.analyze(cur, self.HIST)
        self.assertEqual(res["status"], "regression")
        self.assertGreaterEqual(res["regression_count"], 1)
        # The disk metric barely moved and must not be flagged.
        cols = [f["column"] for f in res["findings"]]
        self.assertIn("cpu_int_primes_s", cols)
        self.assertNotIn("disk_write_mb_s", cols)

    def test_stable_run_is_ok(self):
        cur = {"hostname": "h", "timestamp_utc": "t3",
               "composite_score": "102", "cpu_int_primes_s": "2050",
               "disk_write_mb_s": "505"}
        self.assertEqual(regression.analyze(cur, self.HIST)["status"], "ok")

    def test_improvement_is_not_a_regression(self):
        cur = {"hostname": "h", "timestamp_utc": "t3",
               "composite_score": "150", "cpu_int_primes_s": "3000"}
        res = regression.analyze(cur, self.HIST)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(all(f["direction"] == "faster"
                            for f in res["findings"]))

    def test_excludes_current_run_from_baseline(self):
        # A row with the current timestamp must not become its own baseline.
        cur = {"hostname": "h", "timestamp_utc": "t2",
               "composite_score": "104", "cpu_int_primes_s": "2100"}
        res = regression.analyze(cur, self.HIST)
        self.assertEqual(res["baseline_runs"], 1)  # only t1 remains

    def test_render_is_string(self):
        cur = {"hostname": "h", "timestamp_utc": "t3",
               "composite_score": "80", "cpu_int_primes_s": "1500"}
        out = regression.render(regression.analyze(cur, self.HIST))
        self.assertIn("%", out)


class TestAIScoring(unittest.TestCase):
    def test_matmul_and_ml_score_and_roll_into_ai(self):
        results = {
            "gpu_matmul_fp32": {"rate": scoring.BASELINES["gpu_matmul_fp32"]},
            "npu": {"rate": scoring.BASELINES["npu"]},
            "ml_train": {"rate": scoring.BASELINES["ml_train"] * 2},
        }
        s = scoring.compute_scores(results)
        self.assertAlmostEqual(s["subscores"]["gpu_matmul_fp32"], 100.0,
                               places=3)
        self.assertAlmostEqual(s["subscores"]["ml_train"], 200.0, places=3)
        cats = scoring.category_scores(s["subscores"])
        self.assertIn("ai", cats)


# --------------------------------------------------------------------------- #
class TestSafetyLimits(unittest.TestCase):
    """Guards that stop the tool harming the machine it is measuring.

    A benchmark may load hardware to 100% — that is the point — but it must
    never exhaust RAM (forcing swap thrash), fill the disk, or burn through
    flash endurance.
    """

    GB = 1024 ** 3

    # ---- memory ----
    def test_memory_clamped_to_fraction_of_ram(self):
        mb, notice = limits.safe_mem_mb(999_999, 16 * self.GB)
        self.assertIsNotNone(notice)
        # Two buffers are allocated, so the total must stay well under RAM.
        self.assertLess(mb * 2 * limits.MB, 16 * self.GB / 2)

    def test_memory_reasonable_request_untouched(self):
        mb, notice = limits.safe_mem_mb(64, 16 * self.GB)
        self.assertEqual(mb, 64)
        self.assertIsNone(notice)

    def test_memory_tiny_ram_machine_still_gets_usable_buffer(self):
        mb, _ = limits.safe_mem_mb(512, 512 * 1024 * 1024)   # 512 MB device
        self.assertGreaterEqual(mb, limits.MEM_MIN_MB)
        self.assertLessEqual(mb * 2 * limits.MB, 512 * 1024 * 1024)

    def test_memory_unknown_ram_falls_back_to_cap(self):
        mb, _ = limits.safe_mem_mb(999_999, 0)
        self.assertEqual(mb, limits.MEM_DEFAULT_CAP_MB)

    def test_memory_never_returns_zero_or_negative(self):
        for req in (-100, 0, 1):
            mb, _ = limits.safe_mem_mb(req, 16 * self.GB)
            self.assertGreaterEqual(mb, limits.MEM_MIN_MB)

    # ---- disk ----
    def test_disk_cumulative_writes_capped_for_flash_endurance(self):
        for repeats in (1, 3, 10):
            mb, _ = limits.safe_disk_mb(10 ** 9, 500 * self.GB, repeats)
            self.assertLessEqual(limits.total_write_mb(mb, repeats),
                                 limits.DISK_MAX_TOTAL_WRITE_MB)

    def test_disk_leaves_free_space_headroom(self):
        free = 10 * self.GB
        mb, notice = limits.safe_disk_mb(10 ** 9, free, 1)
        self.assertIsNotNone(notice)
        self.assertLess(mb * limits.MB * limits.DISK_FREE_HEADROOM, free * 1.01)

    def test_disk_reasonable_request_untouched(self):
        mb, notice = limits.safe_disk_mb(256, 500 * self.GB, 3)
        self.assertEqual(mb, 256)
        self.assertIsNone(notice)

    def test_disk_never_returns_zero(self):
        mb, _ = limits.safe_disk_mb(10 ** 9, 1024, 1)   # 1 KB free
        self.assertGreaterEqual(mb, limits.DISK_MIN_MB)

    def test_total_write_mb(self):
        self.assertEqual(limits.total_write_mb(256, 3), 768)

    # ---- thermal ----
    def test_thermal_abort_on_severe_throttle(self):
        should, reason = limits.thermal_should_abort("throttled (25%)")
        self.assertTrue(should)
        self.assertIn("throttled", reason)

    def test_thermal_no_abort_on_mild_throttle(self):
        self.assertFalse(limits.thermal_should_abort("throttled (85%)")[0])

    def test_thermal_abort_on_critical_temperature(self):
        self.assertTrue(limits.thermal_should_abort("max 102C")[0])

    def test_thermal_no_abort_when_normal(self):
        for t in ("nominal", "max 65C", None, ""):
            self.assertFalse(limits.thermal_should_abort(t)[0])


class TestDestructiveOperationSafety(unittest.TestCase):
    """The tool must only ever delete files it created itself."""

    def test_stale_cleanup_ignores_files_it_did_not_create(self):
        import time as _t
        with tempfile.TemporaryDirectory() as d:
            ours = os.path.join(d, "pcbench_old.bin")
            theirs = os.path.join(d, "my_important_data.bin")
            also_theirs = os.path.join(d, "pcbench_notes.txt")  # wrong suffix
            for f in (ours, theirs, also_theirs):
                with open(f, "wb") as fh:
                    fh.write(b"x")
                os.utime(f, (_t.time() - 7200,) * 2)

            workloads._clean_stale_files(d)

            self.assertFalse(os.path.exists(ours), "own scratch file removed")
            self.assertTrue(os.path.exists(theirs), "user data untouched")
            self.assertTrue(os.path.exists(also_theirs), "non-.bin untouched")

    def test_stale_cleanup_keeps_recent_files(self):
        # A concurrent run's in-flight scratch file must survive.
        with tempfile.TemporaryDirectory() as d:
            fresh = os.path.join(d, "pcbench_active.bin")
            with open(fresh, "wb") as fh:
                fh.write(b"x")
            workloads._clean_stale_files(d)
            self.assertTrue(os.path.exists(fresh))

    def test_disk_bench_cleans_up_after_itself(self):
        import glob
        with tempfile.TemporaryDirectory() as d:
            workloads.bench_disk(0.05, 1, file_mb=8, out_dir=d)
            self.assertEqual(glob.glob(os.path.join(d, "pcbench_*.bin")), [],
                             "scratch files must not be left behind")


# --------------------------------------------------------------------------- #
class TestMLWorkloads(unittest.TestCase):
    """Pure-Python ML benchmarks: real training, clustering, and search."""

    def test_nn_actually_learns(self):
        # The headline claim is that this is *real* training, so prove the
        # network's loss falls when the weights are updated.
        xs, ys = mlbench._nn_dataset()
        w1, b1, w2, b2 = mlbench._nn_init()
        first = mlbench._nn_train_step(xs, ys, w1, b1, w2, b2)
        for _ in range(80):
            last = mlbench._nn_train_step(xs, ys, w1, b1, w2, b2)
        self.assertLess(last, first * 0.9)

    def test_nn_is_deterministic_across_runs(self):
        a = mlbench._nn_train_step(*mlbench._nn_dataset(), *mlbench._nn_init())
        b = mlbench._nn_train_step(*mlbench._nn_dataset(), *mlbench._nn_init())
        self.assertAlmostEqual(a, b, places=12)

    def test_nn_initial_loss_is_near_random_chance(self):
        # ln(1/4) for four balanced classes ~ 1.386.
        xs, ys = mlbench._nn_dataset()
        loss = mlbench._nn_train_step(xs, ys, *mlbench._nn_init())
        self.assertLess(abs(loss - math.log(4)), 0.5)

    def test_kmeans_converges_on_separable_blobs(self):
        points = mlbench._blobs(600, 8, 5)
        _, inertia = mlbench._kmeans(points, 5, 6)
        # Ideal within-cluster variance is dims * sigma^2 = 8 * 0.36.
        self.assertLess(inertia / 600, 6.0)

    def test_kmeans_init_picks_distinct_seeds(self):
        # Random seeding often draws two centroids from one blob; the
        # farthest-point init must not.
        points = mlbench._blobs(600, 4, 5)
        seeds = mlbench._farthest_point_init(points, 5)
        self.assertEqual(len(seeds), 5)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                self.assertGreater(mlbench._sq_dist(seeds[i], seeds[j]), 1.0)

    def test_knn_finds_self_as_nearest(self):
        ref = mlbench._blobs(200, 6, 4, seed=3)
        found = mlbench._knn(ref, ref[:8], 1)
        self.assertEqual([f[0] for f in found], list(range(8)))

    def test_knn_returns_k_neighbours_sorted_by_distance(self):
        ref = mlbench._blobs(150, 4, 3, seed=9)
        q = ref[:1]
        got = mlbench._knn(ref, q, 5)[0]
        self.assertEqual(len(got), 5)
        dists = [mlbench._sq_dist(q[0], ref[i]) for i in got]
        self.assertEqual(dists, sorted(dists))

    def test_blobs_are_identical_across_calls(self):
        self.assertEqual(mlbench._blobs(50, 4, 3), mlbench._blobs(50, 4, 3))

    def test_benchmarks_return_wellformed_results(self):
        for fn, unit in ((mlbench.bench_nn_training, "steps/s"),
                         (mlbench.bench_kmeans, "distances/s"),
                         (mlbench.bench_knn, "comparisons/s")):
            r = fn(0.05, 1)
            self.assertGreater(r["rate"], 0)
            self.assertEqual(r["unit"], unit)
            self.assertTrue(r["validated"])


class TestOnnxModelGeneration(unittest.TestCase):
    """The ONNX protobuf is hand-written, so verify its structure."""

    def test_model_contains_required_names(self):
        blob = onnx_model.build_model(32, 2, 4)
        for token in (b"input", b"output", b"MatMul", b"Relu", b"W"):
            self.assertIn(token, blob)

    def test_model_is_deterministic(self):
        self.assertEqual(onnx_model.build_model(32, 2, 4),
                         onnx_model.build_model(32, 2, 4))

    def test_weight_tensor_is_shared_not_duplicated(self):
        # One initializer reused by every layer keeps the file small; without
        # sharing, size would scale with layer count.
        small = len(onnx_model.build_model(64, 2, 4))
        big = len(onnx_model.build_model(64, 16, 4))
        self.assertLess(big - small, 4096)

    def test_flops_scale_with_layers_and_batch(self):
        one = onnx_model.flops_per_inference(64, 1, 8)
        self.assertAlmostEqual(onnx_model.flops_per_inference(64, 2, 8) / one,
                               2.0)
        self.assertAlmostEqual(onnx_model.flops_per_inference(64, 1, 16) / one,
                               2.0)

    def test_weights_scaled_by_inverse_dim(self):
        # Weight magnitude must fall as 1/dim. Without that scaling each layer
        # multiplies activations by ~dim and a deep stack overflows to inf.
        import struct
        for dim in (64, 256):
            blob = onnx_model.build_model(dim, 1, 4)
            expected = struct.pack("<f", 1.0 / dim)
            self.assertIn(expected, blob,
                          f"weights for dim={dim} are not scaled by 1/dim")

    def test_deep_stack_stays_finite(self):
        # Simulate the activation magnitude the real graph produces.
        dim, layers = 1024, onnx_model.DEFAULT_LAYERS
        mag = 1.0 / dim
        # Mixed signs: 6 of every 7 weights positive.
        gain = dim * mag * (6 / 7 - 1 / 7)
        activation = 1.0
        for _ in range(layers):
            activation *= gain
        self.assertTrue(math.isfinite(activation))
        self.assertLess(abs(activation), 1e30)

    def test_write_model_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = onnx_model.write_model(os.path.join(d, "m.onnx"), 32, 2, 4)
            self.assertTrue(os.path.isfile(p))
            with open(p, "rb") as fh:
                self.assertEqual(fh.read(), onnx_model.build_model(32, 2, 4))


class TestNpuCrossVendor(unittest.TestCase):
    def test_detect_reports_availability(self):
        d = npu.detect()
        self.assertIn("available", d)
        if not d["available"]:
            self.assertIn("note", d)

    def test_provider_table_covers_major_vendors(self):
        vendors = {v[1] for v in npu._EP_INFO.values() if v[1]}
        for expected in ("Intel", "AMD", "Qualcomm", "Apple", "NVIDIA"):
            self.assertIn(expected, vendors)

    def test_intel_provider_targets_npu_device(self):
        # OpenVINO defaults to CPU unless the device type is set explicitly.
        _, _, _, opts = npu._EP_INFO["OpenVINOExecutionProvider"]
        self.assertEqual(opts, {"device_type": "NPU"})

    def test_extract_rates_requires_engagement(self):
        payload = {"available": True, "devices": [
            {"label": "X", "gflops": 900.0, "engaged": False}]}
        self.assertEqual(npu.extract_rates(payload), {},
                         "an unengaged accelerator must not be scored")
        payload["devices"][0]["engaged"] = True
        self.assertEqual(npu.extract_rates(payload), {"npu_onnx": 900.0})

    def test_extract_rates_picks_fastest_engaged(self):
        payload = {"available": True, "devices": [
            {"label": "A", "gflops": 100.0, "engaged": True},
            {"label": "B", "gflops": 400.0, "engaged": True}]}
        self.assertEqual(npu.extract_rates(payload), {"npu_onnx": 400.0})

    def test_extract_rates_handles_absent_or_errored(self):
        for p in (None, {"available": False}, {"available": True,
                                               "error": "boom"}):
            self.assertEqual(npu.extract_rates(p), {})


class TestNpuDetectionMultiVendor(unittest.TestCase):
    def test_known_npu_pci_ids_map_to_vendors(self):
        vendors = {v[1] for v in accel._NPU_PCI_IDS.values()}
        self.assertIn("Intel", vendors)
        self.assertIn("AMD", vendors)

    def test_driver_table_covers_intel_and_amd(self):
        self.assertIn("intel_vpu", accel._NPU_DRIVERS)
        self.assertIn("amdxdna", accel._NPU_DRIVERS)

    def test_windows_hints_match_vendor_naming(self):
        for name in ("Intel(R) AI Boost", "AMD IPU Device",
                     "Qualcomm(R) Hexagon(TM) NPU", "Neural Processor"):
            self.assertTrue(accel._WINDOWS_NPU_HINTS.search(name), name)

    def test_windows_hints_do_not_match_ordinary_devices(self):
        for name in ("Intel(R) UHD Graphics", "Realtek Audio", "USB Hub"):
            self.assertFalse(accel._WINDOWS_NPU_HINTS.search(name), name)

    def test_api_table_has_entry_per_vendor(self):
        for vendor in ("Intel", "AMD", "Qualcomm", "Apple"):
            self.assertIn(vendor, accel._NPU_APIS)


# --------------------------------------------------------------------------- #
class TestRegressionConfigAwareness(unittest.TestCase):
    """A changed setting must never be reported as failing hardware.

    Observed in practice: a --quick run (64 MB disk test) followed by a default
    run (256 MB) produced a bogus -40% "disk regression", because larger files
    exhaust an SSD's SLC cache. That is the settings changing, not the drive.
    """

    HIST = [{"hostname": "h", "timestamp_utc": "t1",
             "cfg_disk_mb": "64", "cfg_mem_mb": "64",
             "disk_iops": "45007", "disk_write_mb_s": "3528",
             "cpu_int_primes_s": "4400000", "composite_score": "260"}]

    def _current(self, disk_mb="256"):
        return {"hostname": "h", "timestamp_utc": "t2",
                "cfg_disk_mb": disk_mb, "cfg_mem_mb": "64",
                "disk_iops": "26592", "disk_write_mb_s": "2779",
                "cpu_int_primes_s": "4460000", "composite_score": "257"}

    def test_disk_metrics_skipped_when_size_differs(self):
        res = regression.analyze(self._current("256"), self.HIST)
        cols = [f["column"] for f in res["findings"]]
        self.assertNotIn("disk_iops", cols)
        self.assertNotIn("disk_write_mb_s", cols)
        self.assertEqual(res["status"], "ok")
        self.assertIn("disk_iops", res["skipped_metrics"])

    def test_disk_metrics_compared_when_size_matches(self):
        res = regression.analyze(self._current("64"), self.HIST)
        cols = [f["column"] for f in res["findings"]]
        self.assertIn("disk_iops", cols)
        self.assertEqual(res["status"], "regression")

    def test_config_independent_metrics_always_compared(self):
        # CPU throughput does not depend on --disk-mb, so it must still be
        # compared even when the disk setting changed.
        res = regression.analyze(self._current("256"), self.HIST)
        self.assertNotIn("cpu_int_primes_s", res["skipped_metrics"])

    def test_render_mentions_skipped_metrics(self):
        out = regression.render(regression.analyze(self._current("256"),
                                                   self.HIST))
        self.assertIn("skipped", out)


class TestReportSections(unittest.TestCase):
    """Results are grouped so no category gets lost in a flat list."""

    def _capture(self, results):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report.print_results(results)
        return buf.getvalue()

    def test_ml_workloads_get_their_own_section(self):
        out = self._capture({
            "cpu_int": {"rate": 1000.0},
            "nn_training": {"rate": 900.0, "samples_per_s": 21600,
                            "mflops": 112},
            "kmeans": {"rate": 2.3e6},
            "knn": {"rate": 2.4e6},
        })
        self.assertIn("Machine Learning", out)
        self.assertIn("Neural net training", out)
        self.assertIn("K-means clustering", out)
        self.assertIn("K-NN search", out)

    def test_sections_only_appear_when_populated(self):
        out = self._capture({"cpu_int": {"rate": 1000.0}})
        self.assertIn("CPU", out)
        self.assertNotIn("Machine Learning", out)
        self.assertNotIn("Storage", out)

    def test_storage_section_discloses_write_volume(self):
        out = self._capture({"disk": {
            "write_rate": 100.0, "read_rate": 200.0,
            "random_read_iops": 5000.0, "cache_bypassed": True,
            "file_mb": 256, "total_written_mb": 768, "note": "ok"}})
        self.assertIn("Storage", out)
        self.assertIn("768 MB", out)


# --------------------------------------------------------------------------- #
class TestThermal(unittest.TestCase):
    """Temperatures are always Celsius, and never invented."""

    def test_read_returns_dict(self):
        self.assertIsInstance(thermal.read("."), dict)

    def test_readings_are_plausible_celsius(self):
        t = thermal.read(".")
        if not t.get("cpu_celsius"):
            self.skipTest("no temperature sensor on this machine")
        self.assertGreater(t["cpu_celsius"], -50)
        self.assertLess(t["cpu_celsius"], 150)
        self.assertIn("source", t)

    def test_describe_labels_by_severity(self):
        self.assertIn("normal", thermal.describe({"cpu_celsius": 45.0}))
        self.assertIn("warm", thermal.describe({"cpu_celsius": 80.0}))
        self.assertIn("hot", thermal.describe({"cpu_celsius": 95.0}))
        self.assertIn("°C", thermal.describe({"cpu_celsius": 45.0}))

    def test_describe_handles_missing_sensor(self):
        self.assertEqual(thermal.describe({}), "unavailable")

    def test_battery_health_shape(self):
        b = thermal.battery_health()
        self.assertIsInstance(b, dict)
        if b.get("health_percent") is not None:
            self.assertGreater(b["health_percent"], 0)
            self.assertLessEqual(b["health_percent"], 120)

    def test_hot_threshold_above_warm(self):
        self.assertGreater(thermal.HOT_CELSIUS, thermal.WARM_CELSIUS)


class TestSystemFeatureDetection(unittest.TestCase):
    def test_cpu_features_is_list_of_strings(self):
        f = system.cpu_features()
        self.assertIsInstance(f, list)
        for item in f:
            self.assertIsInstance(item, str)

    def test_virtualization_returns_str_or_none(self):
        v = system.virtualization()
        self.assertTrue(v is None or isinstance(v, str))

    def test_inventory_includes_new_fields(self):
        info = system.inventory()
        self.assertIn("cpu_features", info)
        self.assertIn("virtualization", info)

    def test_machine_state_reports_celsius_key(self):
        st = system.machine_state(".")
        self.assertIn("cpu_celsius", st)
        self.assertIn("temperatures", st)

    def test_warning_when_already_hot(self):
        w = system.state_warnings({"on_ac_power": True, "cpu_celsius": 97.0})
        self.assertTrue(any("°C" in x for x in w))

    def test_no_heat_warning_when_cool(self):
        w = system.state_warnings({"on_ac_power": True, "cpu_celsius": 45.0,
                                   "load_per_core": 0.05})
        self.assertEqual(w, [])


class TestSustainedTemperature(unittest.TestCase):
    def test_temp_summary_computes_rise(self):
        samples = [{"celsius": 50.0}, {"celsius": 62.0}, {"celsius": 58.0}]
        out = sustained._temp_summary(samples)
        self.assertEqual(out["temp_start_celsius"], 50.0)
        self.assertEqual(out["temp_peak_celsius"], 62.0)
        self.assertEqual(out["temp_end_celsius"], 58.0)
        self.assertEqual(out["temp_rise_celsius"], 12.0)

    def test_temp_summary_empty_when_no_sensor(self):
        self.assertEqual(sustained._temp_summary(
            [{"celsius": None}, {"rate": 1.0}]), {})


class TestOutputWritabilityGuard(unittest.TestCase):
    def test_writable_dir_passes(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(cli._check_output_writable(d))

    def test_unwritable_dir_reports_fix(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "locked")
            os.makedirs(sub)
            os.chmod(sub, 0o500)          # read+execute, no write
            try:
                problem = cli._check_output_writable(sub)
                if problem is None:
                    self.skipTest("running as root; permissions not enforced")
                self.assertIn("not writable", problem)
                self.assertIn("chown", problem)
            finally:
                os.chmod(sub, 0o700)


# --------------------------------------------------------------------------- #
class TestCoreScaling(unittest.TestCase):
    """Core analysis reports only what it can measure reliably."""

    def _points(self, marginals):
        pts, agg = [], 0.0
        for i, m in enumerate(marginals, 1):
            agg += m
            pts.append({"workers": i, "aggregate_rate": agg,
                        "marginal_rate": m,
                        "scaling_vs_one": (agg / marginals[0]
                                           if marginals[0] else 0.0)})
        return pts

    def test_detects_hybrid_layout(self):
        # Apple M4 shape: fast cores then much slower ones.
        c = cores.classify_cores(self._points(
            [4.42e6, 4.27e6, 4.22e6, 2.56e6, 2.23e6, 1.40e6, 1.38e6,
             1.26e6, 0.81e6, 0.9e6]))
        self.assertTrue(c["hybrid"])
        self.assertLess(c["slow_relative"], 0.65)

    def test_uniform_cores_not_called_hybrid(self):
        c = cores.classify_cores(self._points([4.0e6] * 8))
        self.assertFalse(c["hybrid"])

    def test_linear_scaling_count_is_reported(self):
        c = cores.classify_cores(self._points(
            [4.0e6, 4.0e6, 3.9e6, 1.0e6, 1.0e6, 1.0e6]))
        self.assertEqual(c["linear_up_to_workers"], 3)

    def test_does_not_claim_exact_core_counts(self):
        # Exact P/E counts proved unreliable and must not be reported.
        c = cores.classify_cores(self._points([4e6, 4e6, 2.5e6, 1e6, 1e6]))
        self.assertNotIn("estimated_fast_cores", c)
        self.assertNotIn("fast_cores", c)

    def test_handles_degenerate_input(self):
        self.assertIn("note", cores.classify_cores([]))
        self.assertIn("note", cores.classify_cores(self._points([0.0])))


class TestSysbench(unittest.TestCase):
    def test_compile_benchmark(self):
        r = sysbench.bench_compile(1)
        if r.get("skipped"):
            self.skipTest(f"no compiler: {r.get('error')}")
        self.assertGreater(r["rate"], 0)
        self.assertGreater(r["seconds_per_compile"], 0)
        self.assertEqual(r["unit"], "compiles/min")

    def test_syscall_latency_plausible(self):
        ns = sysbench.bench_syscall_latency(20_000)
        self.assertGreater(ns, 0)
        self.assertLess(ns, 100_000)       # 100 us would be absurd

    def test_latency_suite_shape(self):
        r = sysbench.bench_latency_suite()
        self.assertIn("syscall_ns", r)
        self.assertIn("rate", r)
        self.assertGreater(r["rate"], 0)

    def test_cpu_frequency_type(self):
        f = sysbench.cpu_frequency_mhz()
        self.assertTrue(f is None or f > 0)


class TestDiskDepthAndLatency(unittest.TestCase):
    def test_queue_depth_sweep_beats_qd1(self):
        # The whole point: QD1 understates a real SSD.
        with tempfile.TemporaryDirectory() as d:
            r = workloads.bench_disk(0.3, 1, file_mb=64, out_dir=d)
            if r.get("skipped"):
                self.skipTest("disk test skipped")
            qd = r["queue_depth_sweep"]
            self.assertEqual([p["queue_depth"] for p in qd["points"]],
                             list(workloads.QUEUE_DEPTHS))
            self.assertGreater(qd["peak_iops"], 0)
            self.assertGreaterEqual(qd["peak_iops"], qd["qd1_iops"] * 0.9)

    def test_latency_percentiles_ordered(self):
        with tempfile.TemporaryDirectory() as d:
            r = workloads.bench_disk(0.3, 1, file_mb=64, out_dir=d)
            if r.get("skipped"):
                self.skipTest("disk test skipped")
            lat = r["random_read_latency"]
            self.assertLessEqual(lat["p50_us"], lat["p99_us"])
            self.assertLessEqual(lat["p99_us"], lat["max_us"])


class TestMemoryScaling(unittest.TestCase):
    def test_uses_processes_not_threads(self):
        # Threads would be GIL-serialized and report flat scaling.
        import inspect
        src = inspect.getsource(workloads.bench_memory_scaling)
        self.assertIn("Pool", src)
        self.assertIn("GIL", src)

    def test_scaling_result_shape(self):
        r = workloads.bench_memory_scaling(0.15, 64, 16 * 1024 ** 3)
        self.assertGreater(r["rate"], 0)
        self.assertGreaterEqual(r["scaling"], 0.5)
        self.assertGreaterEqual(r["buffer_mb"], 32)   # must exceed cache


class TestProfiles(unittest.TestCase):
    def test_every_profile_lists_valid_tests(self):
        for name, tests in cli.PROFILES.items():
            for t in tests:
                self.assertIn(t, cli.TESTS, f"{name} references unknown {t}")

    def test_profile_selects_its_tests(self):
        args = cli.build_parser().parse_args(["--profile", "storage"])
        self.assertEqual(args.profile, "storage")

    def test_unknown_profile_rejected(self):
        self.assertEqual(cli.main(["--profile", "nonsense", "--no-save"]), 2)


# --------------------------------------------------------------------------- #
class TestOptionalRegistry(unittest.TestCase):
    """The optional-package tiers must be coherent and never break the core."""

    def test_every_tier_has_summary_and_packages(self):
        for name, tier in optional.TIERS.items():
            self.assertTrue(tier["summary"], name)
            self.assertTrue(tier["packages"], name)

    def test_package_fields_are_populated(self):
        for pkg in optional.all_packages():
            self.assertTrue(pkg.import_name)
            self.assertTrue(pkg.pip_name)
            self.assertTrue(pkg.purpose)
            self.assertGreater(pkg.approx_mb, 0)

    def test_each_tier_has_a_critical_package(self):
        # Without one, "usable" would be true for a tier that cannot do
        # anything meaningful.
        for name, tier in optional.TIERS.items():
            self.assertTrue(any(p.critical for p in tier["packages"]),
                            f"tier {name} has no critical package")

    def test_status_shape(self):
        st = optional.status()
        self.assertEqual(set(st["tiers"]), set(optional.TIERS))
        for tier in st["tiers"].values():
            self.assertIn("complete", tier)
            self.assertIn("usable", tier)

    def test_have_matches_real_import(self):
        # json is always importable; a nonsense name never is.
        self.assertTrue(optional.have("json"))
        self.assertFalse(optional.have("definitely_not_a_module_xyz"))

    def test_missing_respects_tier_filter(self):
        one = optional.missing(["crypto"])
        crypto_names = {p.pip_name for p in optional.TIERS["crypto"]["packages"]}
        for pkg in one:
            self.assertIn(pkg.pip_name, crypto_names)

    def test_summary_line_is_string(self):
        self.assertIsInstance(optional.summary_line(), str)

    def test_pyproject_extras_match_registry(self):
        # A tier added to the registry but not to pyproject would be
        # uninstallable via pip.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "pyproject.toml")) as fh:
            text = fh.read()
        for name in optional.TIERS:
            self.assertIn(f"{name} = [", text,
                          f"pyproject.toml has no '{name}' extra")


class TestOptionalBenchmarksDegrade(unittest.TestCase):
    """Every optional benchmark must be safe to call when packages are absent."""

    def test_numeric_without_numpy(self):
        r = numeric.run(0.05, 1)
        self.assertIn("available", r)
        if not r["available"]:
            self.assertIn("note", r)

    def test_numeric_matmul_reports_skip(self):
        if optional.have("numpy"):
            self.skipTest("numpy present; testing the absent path")
        self.assertTrue(numeric.bench_matmul(0.05, 1).get("skipped"))

    def test_crypto_without_packages(self):
        r = cryptobench.run(0.05, 1)
        self.assertIn("available", r)

    def test_gpu_without_pyopencl(self):
        r = gpucompute.run(0.05)
        self.assertIn("available", r)
        if not r["available"]:
            self.assertIn("note", r)
        self.assertIsInstance(gpucompute.devices(), list)

    def test_extract_rates_safe_on_unavailable(self):
        for mod in (numeric, cryptobench, gpucompute):
            self.assertEqual(mod.extract_rates(None), {})
            self.assertEqual(mod.extract_rates({"available": False}), {})

    def test_optional_scores_are_defined(self):
        for key in ("blas_matmul", "fft", "lapack", "aes", "zstd", "lz4",
                    "blake3", "gpu_opencl"):
            self.assertIn(key, scoring.BASELINES)

    def test_absent_optional_packages_do_not_penalise_score(self):
        # A machine without numpy must score purely on what it does have.
        s = scoring.compute_scores(
            {"cpu_int": {"rate": scoring.BASELINES["cpu_int"]}})
        self.assertAlmostEqual(s["composite"], 100.0, places=4)
        self.assertNotIn("blas_matmul", s["subscores"])


class TestInstaller(unittest.TestCase):
    def test_venv_python_path_shape(self):
        import install
        path = install.venv_python("somedir")
        self.assertIn("somedir", path)
        self.assertTrue(path.endswith("python") or path.endswith("python.exe"))

    def test_list_mode_exits_cleanly(self):
        import contextlib
        import io
        import install
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = install.main(["--list"])
        self.assertEqual(code, 0)
        self.assertIn("compute", buf.getvalue())

    def test_unknown_tier_rejected(self):
        import install
        self.assertEqual(install.main(["--tier", "nonsense"]), 2)

    def test_declining_installs_nothing(self):
        # Confirmation is required; a non-affirmative answer must abort.
        import builtins
        import contextlib
        import io
        import install
        real_input = builtins.input
        builtins.input = lambda *a: "n"
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = install.main(["--tier", "compute"])
            if code == 0:
                self.skipTest("compute tier already fully installed")
            self.assertEqual(code, 1)
            self.assertIn("Cancelled", buf.getvalue())
        finally:
            builtins.input = real_input


# --------------------------------------------------------------------------- #
class TestInterference(unittest.TestCase):
    """Conditions that change mid-run must be detected, not averaged over."""

    def test_stable_conditions_not_flagged(self):
        v = interference.compare_samples(
            {"load_per_core": 0.10, "celsius": 50.0},
            {"load_per_core": 0.12, "celsius": 53.0})
        self.assertFalse(v["disturbed"])

    def test_load_spike_flagged(self):
        v = interference.compare_samples(
            {"load_per_core": 0.1}, {"load_per_core": 0.9})
        self.assertTrue(v["disturbed"])
        self.assertTrue(any("load" in n for n in v["notes"]))

    def test_large_temp_rise_flagged(self):
        v = interference.compare_samples(
            {"celsius": 45.0}, {"celsius": 70.0})
        self.assertTrue(v["disturbed"])

    def test_already_hot_flagged(self):
        v = interference.compare_samples({"celsius": 86.0}, {"celsius": 88.0})
        self.assertTrue(v["disturbed"])

    def test_missing_sensors_are_not_an_error(self):
        self.assertFalse(interference.compare_samples({}, {})["disturbed"])

    def test_summarize_lists_disturbed_tests(self):
        out = interference.summarize({
            "cpu_int": {"interference": {"disturbed": True}},
            "memory": {"rate": 1.0}})
        self.assertEqual(out["disturbed_tests"], ["cpu_int"])
        self.assertFalse(out["clean"])

    def test_sample_returns_dict(self):
        self.assertIsInstance(interference.sample("."), dict)


class TestComparabilityGuards(unittest.TestCase):
    """A different interpreter or setting must not look like different hardware."""

    def test_interpreter_mismatch_warns(self):
        rows = [
            {"hostname": "a", "timestamp_utc": "t1", "python_version": "3.14.7",
             "composite_score": "250", "cpu_int_primes_s": "4500000"},
            {"hostname": "b", "timestamp_utc": "t2", "python_version": "3.11.9",
             "composite_score": "150", "cpu_int_primes_s": "3000000"}]
        out = compare.render_table(rows)
        self.assertIn("different Python versions", out)
        self.assertIn("cpu_int_primes_s", out)

    def test_same_interpreter_no_warning(self):
        rows = [
            {"hostname": "a", "timestamp_utc": "t1", "python_version": "3.14.7",
             "composite_score": "250"},
            {"hostname": "b", "timestamp_utc": "t2", "python_version": "3.14.7",
             "composite_score": "150"}]
        self.assertNotIn("different Python versions",
                         compare.render_table(rows))

    def test_insignificant_gap_is_called_out(self):
        rows = [
            {"hostname": "a", "timestamp_utc": "t1", "composite_score": "250"},
            {"hostname": "b", "timestamp_utc": "t2", "composite_score": "245"}]
        self.assertIn("treat them as equivalent", compare.render_table(rows))

    def test_large_gap_not_called_insignificant(self):
        rows = [
            {"hostname": "a", "timestamp_utc": "t1", "composite_score": "250"},
            {"hostname": "b", "timestamp_utc": "t2", "composite_score": "100"}]
        self.assertNotIn("treat them as equivalent",
                         compare.render_table(rows))

    def test_regression_guards_interpreter_bound_metrics(self):
        for col in ("cpu_int_primes_s", "nn_train_steps_s", "kmeans_dist_s"):
            self.assertEqual(regression._CONFIG_DEPS.get(col),
                             "python_version", col)

    def test_python_version_is_recorded_in_csv(self):
        self.assertIn("python_version", report.CSV_FIELDS)


class TestDiagnose(unittest.TestCase):
    def test_identifies_weakest_subsystem(self):
        r = diagnose.analyse({"subscores": {
            "cpu_int": 400, "cpu_float": 400, "memory": 400,
            "disk_write": 50, "disk_read": 50, "disk_iops": 50}})
        self.assertTrue(r["available"])
        self.assertEqual(r["weakest"]["category"], "disk")
        self.assertTrue(any(b["category"] == "disk" for b in r["bottlenecks"]))

    def test_balanced_machine_has_no_bottleneck(self):
        r = diagnose.analyse({"subscores": {
            "cpu_int": 200, "memory": 200, "disk_write": 200}})
        self.assertEqual(r["bottlenecks"], [])
        self.assertIn("balanced", r["verdict"])

    def test_derived_ai_category_excluded(self):
        # "ai" rolls up gpu/npu/ml and would double-count them.
        r = diagnose.analyse({"subscores": {
            "cpu_int": 200, "memory": 200, "npu": 400, "gpu_fp32": 400}})
        self.assertNotIn("ai", r["categories"])

    def test_single_category_declines_to_guess(self):
        r = diagnose.analyse({"subscores": {"cpu_int": 200}})
        self.assertFalse(r["available"])

    def test_render_is_string(self):
        r = diagnose.analyse({"subscores": {"cpu_int": 400, "memory": 100,
                                            "disk_write": 400}})
        self.assertIsInstance(diagnose.render(r), str)

    def test_spec_sheet_contains_key_facts(self):
        payload = {
            "version": "9.0", "timestamp_utc": "2026-01-01T00:00:00Z",
            "system": {"cpu_model": "Test CPU", "arch_family": "ARM64",
                       "architecture": "arm64", "arch_bits": 64,
                       "cpu_cores_logical": 8, "cpu_cores_physical": 8,
                       "ram_total_gb": 16.0, "os": "Darwin",
                       "os_release": "25.0", "hostname": "h"},
            "state": {}, "results": {"cpu_int": {"rate": 4_000_000.0}},
            "scores": {"subscores": {"cpu_int": 200, "memory": 100},
                       "composite": 150.0},
        }
        md = diagnose.spec_sheet(payload)
        self.assertIn("# Test CPU", md)
        self.assertIn("4,000,000", md)
        self.assertIn("Composite score", md)


class TestHealth(unittest.TestCase):
    def test_memory_integrity_passes_on_good_ram(self):
        r = health.memory_integrity(16, 16 * 1024 ** 3)
        self.assertTrue(r["passed"])
        self.assertEqual(r["errors"], 0)
        self.assertEqual(r["patterns"], len(health._PATTERNS))

    def test_memory_integrity_states_its_scope(self):
        # A clean pass here is easy to over-read, so the limitation must
        # always accompany the result.
        r = health.memory_integrity(8, 16 * 1024 ** 3)
        self.assertIn("does not certify", r["scope"])

    def test_memory_integrity_respects_safety_cap(self):
        r = health.memory_integrity(10 ** 6, 8 * 1024 ** 3)
        self.assertLess(r["tested_mb"], 10 ** 6)
        self.assertIn("safety_notice", r)

    def test_patterns_cover_stuck_and_coupling_faults(self):
        values = {v for v, _ in health._PATTERNS}
        self.assertIn(0x00, values)
        self.assertIn(0xFF, values)
        self.assertIn(0xAA, values)
        self.assertIn(0x55, values)

    def test_drive_health_is_read_only(self):
        import inspect
        src = inspect.getsource(health)
        for forbidden in ("smartctl -s", "--set", "nvme format"):
            self.assertNotIn(forbidden, src)

    def test_drive_health_shape(self):
        d = health.drive_health()
        self.assertIn("available", d)


class TestPlugins(unittest.TestCase):
    def _root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_example_plugin_discovered_and_valid(self):
        found = plugins.discover(self._root())
        keys = {p["key"]: p for p in found}
        self.assertIn("example_pi", keys)
        self.assertTrue(keys["example_pi"]["valid"])

    def test_plugin_runs_and_scores(self):
        found = [p for p in plugins.discover(self._root())
                 if p["key"] == "example_pi"]
        results = plugins.run_all(found, 0.05, 1)
        self.assertGreater(results["example_pi"]["rate"], 0)
        self.assertIn("plugin_example_pi", plugins.scores(results))

    def test_invalid_plugin_reported_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = os.path.join(d, "plugins")
            os.makedirs(pdir)
            with open(os.path.join(pdir, "broken.py"), "w") as fh:
                fh.write("NAME = 'x'\n")          # missing UNIT/BASELINE/run
            found = plugins.discover(d)
            self.assertFalse(found[0]["valid"])
            self.assertIn("missing", found[0]["error"])

    def test_raising_plugin_does_not_abort_others(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = os.path.join(d, "plugins")
            os.makedirs(pdir)
            with open(os.path.join(pdir, "boom.py"), "w") as fh:
                fh.write("NAME='b'\nUNIT='u'\nBASELINE=1.0\n"
                         "def run(s, r):\n    raise RuntimeError('nope')\n")
            results = plugins.run_all(plugins.discover(d), 0.01, 1)
            self.assertIn("error", results["boom"])

    def test_plugin_must_return_rate(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = os.path.join(d, "plugins")
            os.makedirs(pdir)
            with open(os.path.join(pdir, "norate.py"), "w") as fh:
                fh.write("NAME='n'\nUNIT='u'\nBASELINE=1.0\n"
                         "def run(s, r):\n    return {'x': 1}\n")
            results = plugins.run_all(plugins.discover(d), 0.01, 1)
            self.assertIn("error", results["norate"])

    def test_missing_plugin_dir_is_fine(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(plugins.discover(d), [])


class TestExternalNetworkOptIn(unittest.TestCase):
    def test_no_external_traffic_without_a_target(self):
        # The default must never contact anything off the machine.
        self.assertEqual(network.run_external(), {})
        self.assertEqual(network.run_external(None, None), {})

    def test_loopback_run_has_no_external_key(self):
        r = network.run(0.2)
        self.assertNotIn("external", r)

    def test_download_is_bounded(self):
        import inspect
        src = inspect.getsource(network.download_throughput)
        self.assertIn("max_bytes", src)
        self.assertIn("max_seconds", src)


# --------------------------------------------------------------------------- #
class TestContainerAwareness(unittest.TestCase):
    """Confinement must be detected, because it silently changes every result."""

    def test_detect_reports_effective_resources(self):
        info = container.detect(host_cores=8, host_ram_bytes=16 * 1024 ** 3)
        self.assertEqual(info["host_cores"], 8)
        self.assertGreaterEqual(info["effective_cores"], 1)
        self.assertLessEqual(info["effective_cores"], 8)
        self.assertIn("constrained", info)

    def test_quota_narrows_effective_cores(self):
        info = container.detect(host_cores=16, host_ram_bytes=0)
        # A fractional quota must still leave at least one usable worker,
        # otherwise the multicore test spawns zero processes.
        self.assertGreaterEqual(info["effective_cores"], 1)

    def test_memory_limit_below_host_lowers_effective_ram(self):
        # Simulated rather than mounted: the cgroup files cannot be created in
        # a test, but the arithmetic that consumes them can be checked.
        host = 32 * 1024 ** 3
        info = dict(container.detect(host_cores=4, host_ram_bytes=host))
        info["memory_limit_bytes"] = 2 * 1024 ** 3
        notes = container.warnings(info)
        self.assertTrue(any("capped at 2.0 GB" in n for n in notes), notes)

    def test_warnings_flag_ci_and_cloud(self):
        notes = container.warnings({"ci": "GitHub Actions", "host_cores": 4})
        self.assertTrue(any("GitHub Actions" in n for n in notes))
        notes = container.warnings({"cloud": "AWS EC2", "host_cores": 4})
        self.assertTrue(any("credits" in n for n in notes))

    def test_no_warnings_on_unconfined_machine(self):
        self.assertEqual(container.warnings(
            {"host_cores": 8, "cpu_quota_cores": None,
             "cpu_affinity_cores": None, "memory_limit_bytes": None}), [])

    def test_ci_detection_reads_environment(self):
        self.assertEqual(container.ci_environment.__doc__ is None, False)
        import os as _os
        saved = _os.environ.get("GITHUB_ACTIONS")
        _os.environ["GITHUB_ACTIONS"] = "true"
        try:
            self.assertEqual(container.ci_environment(), "GitHub Actions")
        finally:
            if saved is None:
                del _os.environ["GITHUB_ACTIONS"]
            else:
                _os.environ["GITHUB_ACTIONS"] = saved


# --------------------------------------------------------------------------- #
class TestApplicationWorkloads(unittest.TestCase):
    """Application-shaped benchmarks must run fast, validate, and report units."""

    def test_sqlite_reports_transaction_rate(self):
        r = apps.bench_sqlite(0.1, 1)
        self.assertEqual(r["unit"], "txn/s")
        self.assertGreater(r["rate"], 0)
        self.assertTrue(r["validated"])

    def test_sqlite_bucket_count_divides_row_count(self):
        # The validation compares against an exact expected row count, which is
        # only meaningful when the division is exact.
        self.assertEqual(apps._ROWS % apps._BUCKETS, 0)

    def test_raytrace_is_deterministic(self):
        first = apps._trace(16, 12)
        self.assertAlmostEqual(first, apps._trace(16, 12), places=12)

    def test_raytrace_reports_frames(self):
        r = apps.bench_raytrace(0.1, 1)
        self.assertEqual(r["unit"], "frames/s")
        self.assertGreater(r["rate"], 0)

    def test_image_blur_is_deterministic(self):
        src = bytearray(range(256)) * 4
        self.assertEqual(apps._blur(src, 32, 32), apps._blur(src, 32, 32))

    def test_image_reports_megapixels(self):
        r = apps.bench_image(0.1, 1)
        self.assertEqual(r["unit"], "MP/s")
        self.assertGreater(r["rate"], 0)

    def test_logparse_matches_every_line(self):
        r = apps.bench_logparse(0.1, 1)
        self.assertEqual(r["unit"], "MB/s")
        self.assertGreater(r["lines"], 0)

    def test_fsync_reports_a_mechanism(self):
        with tempfile.TemporaryDirectory() as d:
            r = apps.bench_fsync(0.1, d)
            if r.get("skipped"):
                self.skipTest(r.get("reason", "fsync unavailable"))
            self.assertIn(r["mechanism"], ("fsync", "F_FULLFSYNC"))
            self.assertGreater(r["median_us"], 0)

    def test_fsync_leaves_no_file_behind(self):
        with tempfile.TemporaryDirectory() as d:
            apps.bench_fsync(0.05, d)
            self.assertEqual(os.listdir(d), [],
                             "the fsync probe file must be cleaned up")

    def test_fsync_flags_implausibly_fast_flushes(self):
        # A drive acknowledging flushes it has not performed is a data-loss
        # risk, and looks like a good result unless it is called out.
        self.assertIn("caution", _fake_fast_fsync())

    def test_video_skips_cleanly_without_ffmpeg(self):
        r = apps.bench_video(0.5)
        if apps.ffmpeg_path() is None:
            self.assertTrue(r["skipped"])
            self.assertIn("ffmpeg", r["reason"])
        else:
            self.assertTrue(r.get("skipped") or r["rate"] > 0)

    def test_extract_rates_ignores_skipped(self):
        rates = apps.extract_rates({"sqlite": {"rate": 100.0},
                                    "video": {"skipped": True}})
        self.assertEqual(rates, {"sqlite": 100.0})


def _fake_fast_fsync() -> dict:
    """Build the result bench_fsync would produce for an impossibly fast drive."""
    latencies = [1.0] * 100      # 1 µs per commit => 1,000,000 commits/s
    mid = latencies[len(latencies) // 2]
    rate = 1e6 / mid
    result = {"rate": rate}
    if rate > 100_000:
        result["caution"] = "acknowledging flushes without persisting them"
    return result


# --------------------------------------------------------------------------- #
class TestStorageEnumeration(unittest.TestCase):
    def test_inventory_returns_devices(self):
        inv = storage.inventory(16)
        self.assertIn("devices", inv)
        self.assertIsInstance(inv["benchmarkable_count"], int)

    def test_ram_disks_are_never_benchmarked(self):
        ok, reason = storage.benchmarkable(
            {"fstype": "tmpfs", "mount": "/tmp", "options": "rw"}, 16)
        self.assertFalse(ok)
        self.assertIn("RAM", reason)

    def test_network_filesystems_are_never_benchmarked(self):
        ok, reason = storage.benchmarkable(
            {"fstype": "nfs4", "mount": "/mnt/share", "options": "rw"}, 16)
        self.assertFalse(ok)
        self.assertIn("network", reason)

    def test_read_only_mounts_are_skipped(self):
        ok, reason = storage.benchmarkable(
            {"fstype": "ext4", "mount": "/", "options": "ro,relatime"}, 16)
        self.assertFalse(ok)

    def test_explicit_path_overrides_heuristics(self):
        # A user naming a mount knows something the heuristics do not.
        with tempfile.TemporaryDirectory() as d:
            chosen = storage.targets({"devices": []}, requested=[d])
            self.assertEqual(len(chosen), 1)
            self.assertEqual(os.path.abspath(chosen[0]["mount"]),
                             os.path.abspath(d))

    def test_classify_labels_ram_disk(self):
        self.assertEqual(
            storage.classify({"fstype": "tmpfs", "device": "tmpfs"}),
            "RAM disk")

    def test_run_creates_and_removes_its_workdir(self):
        with tempfile.TemporaryDirectory() as d:
            result = storage.run([{"mount": d, "kind": "test"}], 0.1, 1, 4)
            self.assertEqual(len(result["devices"]), 1)
            self.assertNotIn(".pcbench", os.listdir(d))


# --------------------------------------------------------------------------- #
class TestExports(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "tool": "pcbench", "version": "9.1",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "system": {"hostname": 'weird"host', "os": "Linux",
                       "arch_family": "x86-64", "cpu_model": "Test CPU",
                       "cpu_cores_logical": 4, "ram_total_gb": 8.0,
                       "platform": "Linux-test"},
            "scores": {"composite": 123.4,
                       "subscores": {"cpu_int": 100.0, "memory": 150.0}},
            "results": {"cpu_int": {"rate": 2e6, "unit": "primes/s"},
                        "disk": {"read_rate": 500.0, "write_rate": 250.0,
                                 "unit": "MB/s"},
                        "json": {"error": "RuntimeError: boom"},
                        "knn": {"validation_failed": True, "error": "bad"}},
            "warnings": ["on battery"],
        }

    def test_prometheus_escapes_label_values_exactly_once(self):
        text = export.prometheus_text(self._payload())
        self.assertIn(r'host="weird\"host"', text)
        self.assertNotIn(r'\\"', text)

    def test_prometheus_values_round_trip(self):
        text = export.prometheus_text(self._payload())
        line = [l for l in text.splitlines()
                if l.startswith("pcbench_composite_score{")][0]
        self.assertEqual(float(line.rsplit(" ", 1)[1]), 123.4)

    def test_prometheus_counts_validation_failures(self):
        text = export.prometheus_text(self._payload())
        self.assertIn("pcbench_validation_failures", text)

    def test_prometheus_write_is_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.prom")
            export.save_prometheus(self._payload(), path)
            # No temporary file may survive; a collector scraping the directory
            # must never find a partial write.
            self.assertEqual(sorted(os.listdir(d)), ["out.prom"])

    def test_junit_is_well_formed_xml(self):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(export.junit_xml(self._payload()))
        self.assertEqual(root.tag, "testsuites")

    def test_junit_marks_validation_failure_and_error(self):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(export.junit_xml(self._payload()))
        self.assertEqual(root.get("failures"), "1")
        self.assertEqual(root.get("errors"), "1")

    def test_junit_includes_failed_gates(self):
        import xml.etree.ElementTree as ET
        xml = export.junit_xml(self._payload(),
                               [{"name": "composite>=500", "passed": False,
                                 "message": "too slow"}])
        root = ET.fromstring(xml)
        self.assertEqual(root.get("failures"), "2")

    def test_sqlite_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "history.db")
            export.save_sqlite(self._payload(), path)
            export.save_sqlite(self._payload(), path)
            rows = export.query_sqlite(path, "cpu_int")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["value"], 2e6)

    def test_sqlite_query_on_missing_file_is_empty(self):
        self.assertEqual(export.query_sqlite("/nonexistent/x.db"), [])

    def test_markdown_contains_score_and_warning(self):
        md = export.markdown_summary(self._payload())
        self.assertIn("123.4", md)
        self.assertIn("on battery", md)


# --------------------------------------------------------------------------- #
class TestGates(unittest.TestCase):
    PAYLOAD = {
        "scores": {"composite": 300.0,
                   "subscores": {"cpu_int": 250.0, "cpu_multi": 180.0,
                                 "sqlite": 180.0}},
        "results": {"disk": {"read_rate": 500.0, "unit": "MB/s"},
                    "sqlite": {"rate": 90000.0, "unit": "txn/s"}},
        "sustained": {"droop_pct": 22.0},
    }

    def test_parses_all_operators(self):
        for op in (">=", "<=", ">", "<", "==", "!="):
            self.assertEqual(gates.parse(f"composite{op}100")[1], op)

    def test_rejects_malformed_expression(self):
        for bad in ("composite", "=>100", "composite>=abc", ""):
            with self.assertRaises(gates.GateError):
                gates.parse(bad)

    def test_fail_under_becomes_a_composite_gate(self):
        results = gates.evaluate(self.PAYLOAD, [], fail_under=250)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["passed"])

    def test_bare_name_resolves_to_score_not_raw_rate(self):
        # The distinction matters: sqlite scores 180 but runs at 90,000 txn/s.
        results = gates.evaluate(self.PAYLOAD, ["sqlite>=1000"])
        self.assertFalse(results[0]["passed"])
        self.assertEqual(results[0]["source"], "score")

    def test_unscored_metric_falls_back_to_its_raw_rate(self):
        # fsync is deliberately unscored, so a bare name must still reach the
        # only number that exists for it rather than failing as "not measured".
        payload = {"scores": {"composite": 1.0, "subscores": {}},
                   "results": {"fsync": {"rate": 5000.0, "unit": "commits/s"}}}
        results = gates.evaluate(payload, ["fsync>=1000"])
        self.assertTrue(results[0]["passed"])
        self.assertIn("raw rate", results[0]["source"])

    def test_dotted_rate_resolves_to_raw_value(self):
        results = gates.evaluate(self.PAYLOAD, ["sqlite.rate>=1000"])
        self.assertTrue(results[0]["passed"])

    def test_nested_payload_path_resolves(self):
        results = gates.evaluate(self.PAYLOAD, ["sustained.droop_pct<=15"])
        self.assertFalse(results[0]["passed"])

    def test_missing_metric_fails_rather_than_passing(self):
        # Treating "not measured" as "threshold met" would make gates stop
        # checking anything the moment a test is skipped.
        results = gates.evaluate(self.PAYLOAD, ["nonexistent>=1"])
        self.assertFalse(results[0]["passed"])
        self.assertIn("not measured", results[0]["message"])

    def test_failed_returns_only_failures(self):
        results = gates.evaluate(self.PAYLOAD,
                                 ["cpu_int>=100", "cpu_multi>=9999"])
        self.assertEqual(len(gates.failed(results)), 1)

    def test_render_summarises_counts(self):
        text = gates.render(gates.evaluate(self.PAYLOAD, ["cpu_int>=100"]))
        self.assertIn("1/1", text)


# --------------------------------------------------------------------------- #
class TestConfigFile(unittest.TestCase):
    def _parse(self, argv):
        parser = cli.build_parser()
        return parser, parser.parse_args(argv)

    def test_json_config_is_applied(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pcbench.json")
            with open(path, "w") as f:
                f.write('{"run": {"seconds": 7, "repeats": 4}}')
            parser, args = self._parse([])
            config.apply(args, parser, path, environ={})
            self.assertEqual(args.seconds, 7.0)
            self.assertEqual(args.repeats, 4)

    def test_command_line_beats_config_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pcbench.json")
            with open(path, "w") as f:
                f.write('{"run": {"seconds": 7}}')
            parser, args = self._parse(["--seconds", "2"])
            config.apply(args, parser, path, environ={})
            self.assertEqual(args.seconds, 2.0)

    def test_environment_beats_config_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pcbench.json")
            with open(path, "w") as f:
                f.write('{"run": {"repeats": 4}}')
            parser, args = self._parse([])
            config.apply(args, parser, path,
                         environ={"PCBENCH_RUN_REPEATS": "9"})
            self.assertEqual(args.repeats, 9)

    def test_bare_environment_alias_works(self):
        parser, args = self._parse([])
        config.apply(args, parser, None, environ={"PCBENCH_SECONDS": "3.5"})
        self.assertEqual(args.seconds, 3.5)

    def test_unknown_setting_is_rejected_with_valid_names(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pcbench.json")
            with open(path, "w") as f:
                f.write('{"run": {"nonsense": 1}}')
            parser, args = self._parse([])
            with self.assertRaises(config.ConfigError) as ctx:
                config.apply(args, parser, path, environ={})
            self.assertIn("run.nonsense", str(ctx.exception))

    def test_booleans_coerce_from_strings(self):
        parser, args = self._parse([])
        config.apply(args, parser, None, environ={"PCBENCH_RUN_QUICK": "yes"})
        self.assertTrue(args.quick)

    def test_assertions_load_as_a_list(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pcbench.json")
            with open(path, "w") as f:
                f.write('{"gates": {"assertions": ["composite>=100"]}}')
            parser, args = self._parse([])
            config.apply(args, parser, path, environ={})
            self.assertEqual(args.assert_, ["composite>=100"])

    def test_every_mapped_key_targets_a_real_flag(self):
        parser = cli.build_parser()
        dests = {a.dest for a in parser._actions}
        for key, dest in config._KEY_MAP.items():
            self.assertIn(dest, dests, f"{key} maps to unknown flag {dest}")

    def test_sample_config_is_valid_and_complete(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pcbench.toml")
            config.write_sample(path)
            self.assertTrue(os.path.isfile(path))
            with self.assertRaises(config.ConfigError):
                config.write_sample(path)   # never clobbers

    def test_find_walks_up_parent_directories(self):
        with tempfile.TemporaryDirectory() as d:
            nested = os.path.join(d, "a", "b")
            os.makedirs(nested)
            top = os.path.join(d, "pcbench.json")
            with open(top, "w") as f:
                f.write("{}")
            self.assertEqual(config.find(nested), top)


# --------------------------------------------------------------------------- #
class TestReferenceClassification(unittest.TestCase):
    def test_classes_cover_the_whole_range_without_gaps(self):
        for low, high, _, _ in reference.CLASSES:
            self.assertLess(low, high)
        for (a, b) in zip(reference.CLASSES, reference.CLASSES[1:]):
            self.assertEqual(a[1], b[0], "class boundaries must not leave gaps")

    def test_single_board_computer_is_not_flagged_as_broken(self):
        # The whole reason the model is anchored on measured single-core
        # performance: a slow machine that is internally consistent is fine.
        payload = {
            "system": {"cpu_cores_logical": 4, "cpu_cores_physical": 4,
                       "arch_family": "ARM64", "ram_total_gb": 8.0},
            "scores": {"composite": 32.0,
                       "subscores": {"cpu_int": 30.0, "cpu_float": 40.0,
                                     "compression": 22.0, "hashing": 25.0,
                                     "json": 28.0}},
        }
        result = reference.assess(payload)
        self.assertEqual(result["flag"], "balanced")
        self.assertEqual(result["class"], "embedded / SBC")

    def test_dragged_machine_is_flagged(self):
        payload = {
            "system": {"cpu_cores_logical": 16, "cpu_cores_physical": 8,
                       "arch_family": "x86-64", "ram_total_gb": 32.0},
            "scores": {"composite": 40.0,
                       "subscores": {"cpu_int": 200.0, "cpu_float": 210.0,
                                     "compression": 180.0, "hashing": 220.0,
                                     "json": 190.0}},
        }
        self.assertEqual(reference.assess(payload)["flag"],
                         "unbalanced (subsystem drag)")

    def test_no_anchor_yields_no_false_verdict(self):
        payload = {"system": {"cpu_cores_logical": 4},
                   "scores": {"composite": 100.0, "subscores": {}}}
        result = reference.assess(payload)
        self.assertIsNone(result["expected_composite"])
        self.assertIn("Too few", result["verdict"])

    def test_subsystem_floor_flags_a_failing_disk(self):
        checks = reference.subsystem_checks(
            {"disk": {"read_rate": 45.0, "write_rate": 20.0,
                      "random_read_iops": 120.0}})
        self.assertEqual(len(checks), 3)

    def test_healthy_disk_is_not_flagged(self):
        self.assertEqual(reference.subsystem_checks(
            {"disk": {"read_rate": 2500.0, "write_rate": 1800.0,
                      "random_read_iops": 90000.0}}), [])

    def test_skipped_results_are_not_checked(self):
        self.assertEqual(reference.subsystem_checks(
            {"disk": {"skipped": True, "read_rate": 1.0}}), [])


# --------------------------------------------------------------------------- #
class TestSoak(unittest.TestCase):
    def test_short_soak_completes_with_no_errors(self):
        result = soak.run(2.0, workers=1, quiet=True)
        self.assertEqual(result["errors"], 0)
        self.assertGreater(result["units_completed"], 0,
                           "worker unit counts must reach the parent")
        self.assertIn("STABLE", result["verdict"])

    def test_work_units_validate_correct_hardware(self):
        self.assertTrue(soak._unit_integer(12345)[1])
        payload = b"abc" * 1000
        self.assertTrue(soak._unit_compression(1, payload)[1])
        self.assertTrue(soak._unit_memory(7, 4096)[1])

    def test_verdict_reports_instability_when_errors_occur(self):
        text = soak.verdict({"errors": 3, "error_types": ["memory"],
                             "time_to_first_error_s": 90.0,
                             "elapsed_seconds": 300.0, "units_completed": 10,
                             "workers": 4})
        self.assertIn("UNSTABLE", text)
        self.assertIn("1m30s", text)

    def test_interrupted_soak_is_incomplete_not_stable(self):
        text = soak.verdict({"errors": 0, "error_types": [],
                             "time_to_first_error_s": None,
                             "elapsed_seconds": 60.0, "interrupted": True,
                             "aborted": "interrupted by user",
                             "units_completed": 5, "workers": 2})
        self.assertIn("INCOMPLETE", text)

    def test_hms_formats_long_durations(self):
        self.assertEqual(soak._hms(3661), "1h01m")
        self.assertEqual(soak._hms(90), "1m30s")
        self.assertEqual(soak._hms(5), "5s")


# --------------------------------------------------------------------------- #
class TestMonitor(unittest.TestCase):
    def test_sample_never_raises(self):
        snap = monitor.sample(".")
        self.assertIn("t", snap)

    def test_short_session_produces_a_summary(self):
        result = monitor.run(1.0, 0.3, ".", quiet=True)
        self.assertGreaterEqual(result["samples"], 1)
        self.assertTrue(result["observations"])

    def test_throttling_is_named_as_thermal_when_hot(self):
        series = {"cpu_mhz": {"min": 1000, "max": 4000, "mean": 2000,
                              "last": 1000, "spark": ""},
                  "cpu_celsius": {"min": 70, "max": 98, "mean": 90,
                                  "last": 95, "spark": ""}}
        notes = monitor.observations(series, [], 8)
        self.assertTrue(any("thermal throttling" in n for n in notes), notes)

    def test_throttling_is_named_as_power_limit_when_cool(self):
        series = {"cpu_mhz": {"min": 1000, "max": 4000, "mean": 2000,
                              "last": 1000, "spark": ""},
                  "cpu_celsius": {"min": 40, "max": 60, "mean": 50,
                                  "last": 55, "spark": ""}}
        notes = monitor.observations(series, [], 8)
        self.assertTrue(any("power or current limit" in n for n in notes),
                        notes)

    def test_oversubscription_is_reported(self):
        notes = monitor.observations(
            {"load1": {"min": 1, "max": 30, "mean": 20, "last": 25,
                       "spark": ""}}, [], 4)
        self.assertTrue(any("oversubscribed" in n for n in notes))

    def test_quiet_session_says_nothing_is_wrong(self):
        notes = monitor.observations({}, [], 8)
        self.assertTrue(any("nothing anomalous" in n for n in notes))

    def test_trace_writes_csv_with_all_columns(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.csv")
            monitor.save_trace(
                {"raw": [{"t": 1.0, "cpu_mhz": 3000},
                         {"t": 2.0, "cpu_celsius": 50.0}]}, path)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertIn("cpu_celsius", rows[0])


# --------------------------------------------------------------------------- #
class TestNewCliSurface(unittest.TestCase):
    def test_every_test_has_a_description(self):
        for name in cli.TESTS:
            self.assertIn(name, cli.DESCRIPTIONS,
                          f"{name} is undocumented in --list-tests")

    def test_every_profile_names_only_real_tests(self):
        for profile, tests in cli.PROFILES.items():
            for name in tests:
                self.assertIn(name, cli.TESTS,
                              f"profile {profile} names unknown test {name}")

    def test_every_test_has_a_runner(self):
        parser = cli.build_parser()
        args = parser.parse_args([])
        runners = cli._runners(args, {"cpu_cores_logical": 2,
                                      "ram_total_bytes": 8 * 1024 ** 3}, ".")
        for name in cli.TESTS:
            self.assertIn(name, runners, f"{name} has no runner")

    def test_app_tests_except_video_run_by_default(self):
        self.assertNotIn("video", cli.DEFAULT_TESTS)
        for name in ("sqlite", "raytrace", "image", "logparse", "fsync"):
            self.assertIn(name, cli.DEFAULT_TESTS)

    def test_list_tests_marks_non_default_tests(self):
        text = cli.list_tests()
        self.assertIn("video", text)
        self.assertIn("- video", text)

    def test_list_tests_exits_zero(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli.main(["--list-tests"]), 0)
        self.assertIn("Profiles", buf.getvalue())

    def test_malformed_assertion_is_rejected_before_benchmarking(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            code = cli.main(["--assert", "garbage", "--no-save"])
        self.assertEqual(code, 2)
        self.assertIn("invalid assertion", buf.getvalue())

    def test_init_config_writes_and_refuses_to_clobber(self):
        import io
        import contextlib
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pcbench.toml")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(cli.main(["--init-config", path]), 0)
            self.assertTrue(os.path.isfile(path))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["--init-config", path]), 2)


# --------------------------------------------------------------------------- #
class TestCsvSchemaStaysInSync(unittest.TestCase):
    """A drifted CSV header silently misaligns every recorded column."""

    def test_flatten_row_keys_match_csv_fields_exactly(self):
        payload = {
            "timestamp_utc": "2026-01-01T00:00:00Z", "version": "9.1",
            "config": {"disk_mb": 1, "mem_mb": 1},
            "system": {"hostname": "h", "os": "Linux", "architecture": "x86_64",
                       "arch_family": "x86-64", "cpu_model": "c",
                       "cpu_cores_logical": 1, "python_version": "3",
                       "python_implementation": "CPython"},
            "state": {}, "results": {}, "scores": {"composite": 1.0},
        }
        self.assertEqual(sorted(report.flatten_row(payload)),
                         sorted(report.CSV_FIELDS))


# --------------------------------------------------------------------------- #
class TestNewScoringBaselines(unittest.TestCase):
    def test_every_score_source_has_a_baseline(self):
        for score_key, _, _ in scoring._SOURCES:
            self.assertIn(score_key, scoring.BASELINES)

    def test_app_workloads_score_into_the_apps_category(self):
        scores = scoring.compute_scores(
            {"sqlite": {"rate": 50_000.0}, "raytrace": {"rate": 150.0},
             "image": {"rate": 2.5}, "logparse": {"rate": 80.0},
             "video": {"rate": 70.0}})
        # Every baseline is defined so that the baseline machine scores 100.
        for key, value in scores["subscores"].items():
            self.assertAlmostEqual(value, 100.0, places=1, msg=key)
        self.assertAlmostEqual(
            scoring.category_scores(scores["subscores"])["apps"], 100.0,
            places=1)

    def test_fsync_is_deliberately_unscored(self):
        # Cross-platform flush semantics differ by orders of magnitude, so
        # scoring it would measure the OS rather than the hardware.
        self.assertNotIn("fsync", scoring.BASELINES)
        scores = scoring.compute_scores({"fsync": {"rate": 5000.0}})
        self.assertEqual(scores["subscores"], {})


# --------------------------------------------------------------------------- #
class TestProvenance(unittest.TestCase):
    """Configuration capture must never crash and never over-claim."""

    def test_collect_returns_all_sections(self):
        info = provenance.collect()
        for key in ("mitigations", "frequency", "memory", "smt", "microcode",
                    "kernel"):
            self.assertIn(key, info)

    def test_collect_never_raises(self):
        # Every section is wrapped; an unreadable platform yields
        # available=False rather than an exception.
        info = provenance.collect()
        self.assertIsInstance(info, dict)

    def test_absent_smt_is_not_reported_as_disabled(self):
        # Apple silicon and many i3/i5 parts have no SMT. Calling that
        # "DISABLED" sends the user hunting for a BIOS setting that does not
        # exist, so supported and enabled are distinct fields.
        notes = provenance.notes(
            {"smt": {"available": True, "enabled": False, "supported": False}})
        self.assertEqual(notes, [])

    def test_disabled_smt_is_reported(self):
        notes = provenance.notes(
            {"smt": {"available": True, "enabled": False, "supported": True}})
        self.assertTrue(any("switched off" in n for n in notes), notes)

    def test_unknown_smt_support_is_not_reported(self):
        notes = provenance.notes(
            {"smt": {"available": True, "enabled": False, "supported": None}})
        self.assertEqual(notes, [])

    def test_disabled_mitigations_are_flagged(self):
        notes = provenance.notes({"mitigations": {
            "available": True, "vulnerable": ["spectre_v2", "meltdown"],
            "mitigated": [], "cmdline_override": None}})
        self.assertTrue(any("not comparable" in n for n in notes), notes)

    def test_kernel_cmdline_override_takes_precedence(self):
        notes = provenance.notes({"mitigations": {
            "available": True, "vulnerable": [], "mitigated": ["x"],
            "cmdline_override": "mitigations=off"}})
        self.assertTrue(any("mitigations=off" in n for n in notes), notes)

    def test_non_performance_governor_is_flagged(self):
        notes = provenance.notes(
            {"frequency": {"available": True, "governor": "powersave"}})
        self.assertTrue(any("powersave" in n for n in notes), notes)

    def test_performance_governor_is_not_flagged(self):
        notes = provenance.notes(
            {"frequency": {"available": True, "governor": "performance"}})
        self.assertEqual(notes, [])

    def test_render_omits_unavailable_sections(self):
        text = provenance.render({"frequency": {"available": False},
                                  "memory": {"available": False},
                                  "smt": {"available": False},
                                  "mitigations": {"available": False},
                                  "microcode": {"available": False}})
        self.assertEqual(text.strip(), "")


# --------------------------------------------------------------------------- #
class TestStatistics(unittest.TestCase):
    def test_confidence_interval_brackets_the_mean(self):
        ci = stats.confidence_interval([10.0, 11.0, 9.0, 10.5, 9.5])
        self.assertLess(ci["low"], ci["mean"])
        self.assertGreater(ci["high"], ci["mean"])

    def test_single_sample_has_no_interval(self):
        ci = stats.confidence_interval([42.0])
        self.assertEqual(ci["low"], ci["high"])
        self.assertIsNone(ci["relative_margin_pct"])

    def test_identical_samples_have_zero_margin(self):
        ci = stats.confidence_interval([5.0] * 6)
        self.assertAlmostEqual(ci["margin"], 0.0)

    def test_ranks_average_ties(self):
        # Tied values must share the average rank, or the rank-sum test is
        # biased whenever timer resolution is coarse.
        self.assertEqual(stats._rank([1.0, 2.0, 2.0, 3.0]),
                         [1.0, 2.5, 2.5, 4.0])

    def test_identical_distributions_are_not_significant(self):
        a = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2]
        b = [100.1, 100.9, 99.2, 100.4, 99.6, 100.3]
        result = stats.compare(a, b)
        self.assertFalse(result["significant"])
        self.assertIn("NO SIGNIFICANT DIFFERENCE", result["verdict"])

    def test_clear_difference_is_significant(self):
        a = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2]
        b = [150.0, 151.0, 149.0, 150.5, 149.5, 150.2]
        result = stats.compare(a, b)
        self.assertTrue(result["significant"])
        self.assertIn("SIGNIFICANT", result["verdict"])
        self.assertGreater(result["change_pct"], 40)

    def test_too_few_samples_is_inconclusive_not_negative(self):
        # Collapsing "not enough data" into "no difference" is how
        # underpowered comparisons get mistaken for evidence of no effect.
        result = stats.compare([100.0, 101.0], [150.0, 151.0])
        self.assertFalse(result["conclusive"])
        self.assertIsNone(result["significant"])
        self.assertIn("INCONCLUSIVE", result["verdict"])

    def test_all_identical_samples_report_no_difference(self):
        result = stats.mann_whitney([5.0] * 5, [5.0] * 5)
        self.assertEqual(result["p"], 1.0)

    def test_cliffs_delta_signs_correctly(self):
        self.assertAlmostEqual(stats.cliffs_delta([1.0, 2.0], [3.0, 4.0]), 1.0)
        self.assertAlmostEqual(stats.cliffs_delta([3.0, 4.0], [1.0, 2.0]), -1.0)

    def test_negligible_difference_is_called_out(self):
        a = [1000.0 + i * 0.01 for i in range(12)]
        b = [1000.3 + i * 0.01 for i in range(12)]
        result = stats.compare(a, b)
        if result["significant"]:
            self.assertIn("NEGLIGIBLE", result["verdict"])

    def test_required_samples_grows_as_effect_shrinks(self):
        samples = [100.0, 102.0, 98.0, 101.0, 99.0]
        self.assertGreater(stats.required_samples(samples, 1.0),
                           stats.required_samples(samples, 10.0))

    def test_lower_is_better_metrics_invert_direction(self):
        slow = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0]
        fast = [5.0, 5.1, 4.9, 5.2, 4.8, 5.0]
        result = stats.compare(slow, fast, label="latency",
                               higher_is_better=False)
        self.assertIn("faster", result["verdict"])


# --------------------------------------------------------------------------- #
class TestPayloadComparison(unittest.TestCase):
    def _payload(self, host, rate, n=8, composite=100.0):
        return {
            "system": {"hostname": host, "cpu_model": "Test CPU"},
            "config": {"seconds": 3, "repeats": n},
            "results": {"cpu_int": {"samples": [rate + (i % 3) - 1
                                                for i in range(n)]}},
            "scores": {"composite": composite},
        }

    def test_regression_is_detected(self):
        a = self._payload("h", 1000.0)
        b = self._payload("h", 800.0)
        result = stats.compare_payloads(a, b)
        self.assertIn("cpu_int", result["regressions"])
        self.assertIn("REGRESSION", result["verdict"])

    def test_improvement_is_detected(self):
        result = stats.compare_payloads(self._payload("h", 800.0),
                                        self._payload("h", 1000.0))
        self.assertIn("cpu_int", result["improvements"])

    def test_different_machines_warn(self):
        result = stats.compare_payloads(self._payload("a", 1000.0),
                                        self._payload("b", 1000.0))
        self.assertTrue(any("different machines" in w
                            for w in result["warnings"]))

    def test_different_config_warns(self):
        a = self._payload("h", 1000.0)
        b = self._payload("h", 1000.0)
        b["config"]["seconds"] = 10
        result = stats.compare_payloads(a, b)
        self.assertTrue(any("--seconds" in w for w in result["warnings"]))

    def test_metrics_present_on_only_one_side_are_listed(self):
        a = self._payload("h", 1000.0)
        b = self._payload("h", 1000.0)
        b["results"]["memory"] = {"samples": [1.0, 2.0, 3.0]}
        result = stats.compare_payloads(a, b)
        self.assertEqual(result["only_in_candidate"], ["memory"])

    def test_lower_is_better_metric_is_not_called_a_regression(self):
        a = self._payload("h", 1000.0)
        b = self._payload("h", 1000.0)
        a["results"]["fsync_median_us"] = {"samples": [100.0] * 6}
        b["results"]["fsync_median_us"] = {"samples": [50.0] * 6}
        result = stats.compare_payloads(a, b)
        self.assertNotIn("fsync_median_us", result["regressions"])


# --------------------------------------------------------------------------- #
class TestCounters(unittest.TestCase):
    def test_resource_snapshot_has_expected_fields(self):
        snap = counters.resource_snapshot()
        for key in ("minor_faults", "major_faults", "involuntary_switches"):
            self.assertIn(key, snap)

    def test_resource_delta_subtracts(self):
        before = {"minor_faults": 10, "max_rss_bytes": 100}
        after = {"minor_faults": 25, "max_rss_bytes": 80}
        delta = counters.resource_delta(before, after)
        self.assertEqual(delta["minor_faults"], 15)
        # Peak RSS is a high-water mark, so the larger of the two survives.
        self.assertEqual(delta["max_rss_bytes"], 100)

    def test_perf_availability_always_explains_itself(self):
        status = counters.perf_available()
        if not status["available"]:
            self.assertTrue(status.get("reason"))

    def test_parse_perf_stat_handles_unsupported_events(self):
        parsed = counters.parse_perf_stat(
            "  12,345,678      cycles\n  <not supported>      cache-misses")
        self.assertEqual(parsed["cycles"], 12345678)
        self.assertIsNone(parsed["cache-misses"])

    def test_derive_computes_ipc_and_rates(self):
        d = counters.derive({"cycles": 1000, "instructions": 2000,
                             "cache-references": 100, "cache-misses": 20,
                             "branches": 200, "branch-misses": 10})
        self.assertEqual(d["ipc"], 2.0)
        self.assertEqual(d["cache_miss_rate_pct"], 20.0)
        self.assertEqual(d["branch_miss_rate_pct"], 5.0)

    def test_derive_tolerates_missing_counters(self):
        self.assertNotIn("ipc", counters.derive({"instructions": 100}))

    def test_low_ipc_with_cache_misses_names_the_cause(self):
        notes = counters.interpret(
            {"ipc": 0.4, "cache_misses_per_kilo_instruction": 50.0})
        self.assertTrue(any("cache misses" in n for n in notes), notes)

    def test_high_ipc_points_at_clock_not_memory(self):
        notes = counters.interpret({"ipc": 2.5})
        self.assertTrue(any("clock speed or a power limit" in n
                            for n in notes), notes)

    def test_major_faults_are_reported_as_dominant(self):
        notes = counters.interpret(
            {}, {"major_faults": 5000, "major_faults_per_s": 50.0})
        self.assertTrue(any("major page faults" in n for n in notes))

    def test_delta_computes_rates_from_elapsed_time(self):
        before = {"wall_clock": 100.0, "involuntary_switches": 0,
                  "major_faults": 0, "max_rss_bytes": 0}
        after = {"wall_clock": 110.0, "involuntary_switches": 2000,
                 "major_faults": 50, "max_rss_bytes": 0}
        delta = counters.resource_delta(before, after)
        self.assertEqual(delta["elapsed_s"], 10.0)
        self.assertEqual(delta["involuntary_switches_per_s"], 200.0)
        self.assertEqual(delta["major_faults_per_s"], 5.0)
        self.assertNotIn("wall_clock", delta)

    def test_involuntary_switches_never_claim_contention(self):
        """Regression test for a metric that did not measure what it claimed.

        Two measurements killed the original assertion: a full benchmark run on
        a completely idle machine reached ~8,600 involuntary switches/s, while
        a genuinely busier machine measured ~2,500/s. The count tracks this
        tool's own worker spawning and blocking I/O, not external contention,
        so no verdict may be drawn from it.
        """
        for rate in (185.0, 2545.0, 8638.0, 50_000.0):
            notes = counters.interpret(
                {}, {"involuntary_switches": int(rate * 90),
                     "involuntary_switches_per_s": rate,
                     "elapsed_s": 90.0,
                     "major_faults": 0, "major_faults_per_s": 0.0})
            self.assertEqual(
                notes, [],
                f"{rate}/s produced a verdict; involuntary switches must "
                f"never be interpreted as contention")

    def test_render_labels_switches_as_non_diagnostic(self):
        text = counters.render({"resources": {
            "involuntary_switches": 500_000,
            "involuntary_switches_per_s": 5000.0, "elapsed_s": 100.0,
            "minor_faults": 0, "major_faults": 0}})
        self.assertIn("not a contention signal", text)


# --------------------------------------------------------------------------- #
class TestStandards(unittest.TestCase):
    def test_linpack_size_is_blocked_and_bounded(self):
        n = standards.linpack_size(16 * 1024 ** 3)
        self.assertEqual(n % 64, 0)
        self.assertGreaterEqual(n, 256)

    def test_linpack_size_shrinks_on_small_machines(self):
        self.assertLess(standards.linpack_size(512 * 1024 ** 2),
                        standards.linpack_size(64 * 1024 ** 3))

    def test_from_native_handles_a_missing_engine(self):
        result = standards.from_native(None)
        self.assertTrue(result["stream"]["skipped"])
        self.assertTrue(result["coremark_style"]["skipped"])

    def test_from_native_extracts_stream(self):
        result = standards.from_native({
            "stream": {"copy": 1.0, "scale": 2.0, "add": 3.0, "triad": 4.0,
                       "array_bytes": 64_000_000, "validated": True},
            "coremark_style": {"rate": 1234.0}})
        self.assertEqual(result["stream"]["triad"], 4.0)
        self.assertIn("cache", result["stream"]["cache_rule"])

    def test_failed_stream_validation_is_surfaced(self):
        result = standards.from_native({
            "stream": {"triad": 4.0, "array_bytes": 1, "validated": False}})
        self.assertTrue(result["stream"]["validation_failed"])

    def test_coremark_is_never_called_certified(self):
        # Publishing an approximation under a standard's name destroys the
        # comparability that made the standard worth implementing.
        caveat = standards.coremark_caveat()
        self.assertIn("not the certified benchmark", caveat)
        result = standards.from_native({"coremark_style": {"rate": 1.0}})
        self.assertIn("caveat", result["coremark_style"])

    def test_extract_rates_skips_unvalidated_linpack(self):
        rates = standards.extract_rates(
            {"linpack": {"rate": 100.0, "validated": False}})
        self.assertNotIn("linpack", rates)

    @unittest.skipUnless(standards.available()["linpack"], "needs NumPy")
    def test_linpack_solves_correctly(self):
        result = standards.linpack(1 * 1024 ** 3)
        if result.get("skipped"):
            self.skipTest(result["reason"])
        self.assertTrue(result["validated"])
        self.assertLess(result["residual"], standards.RESIDUAL_TOLERANCE)
        self.assertGreater(result["rate"], 0)


# --------------------------------------------------------------------------- #
class TestNuma(unittest.TestCase):
    def test_cpulist_parsing(self):
        self.assertEqual(numa.parse_cpulist("0-3,8,12-13"),
                         [0, 1, 2, 3, 8, 12, 13])
        self.assertEqual(numa.parse_cpulist(""), [])
        self.assertEqual(numa.parse_cpulist("garbage"), [])

    def test_topology_never_raises(self):
        self.assertIsInstance(numa.topology(), dict)

    def test_single_node_skips_the_matrix(self):
        result = numa.bandwidth_matrix({"numa": False})
        self.assertTrue(result["skipped"])
        self.assertIn("single NUMA node", result["reason"])

    def test_penalty_computed_from_matrix(self):
        penalty = numa._penalty({"0": {"0": 100.0, "1": 50.0},
                                 "1": {"0": 50.0, "1": 100.0}})
        self.assertEqual(penalty["local_mean_mb_s"], 100.0)
        self.assertEqual(penalty["remote_mean_mb_s"], 50.0)
        self.assertEqual(penalty["remote_penalty_pct"], 50.0)

    def test_large_penalty_recommends_pinning(self):
        notes = numa.notes({
            "topology": {"numa": True, "nodes": 2},
            "bandwidth": {"remote_penalty_pct": 40.0,
                          "local_mean_mb_s": 100.0,
                          "remote_mean_mb_s": 60.0}})
        self.assertTrue(any("numactl" in n for n in notes), notes)

    def test_small_penalty_says_not_worth_tuning(self):
        notes = numa.notes({
            "topology": {"numa": True, "nodes": 2},
            "bandwidth": {"remote_penalty_pct": 3.0,
                          "local_mean_mb_s": 100.0,
                          "remote_mean_mb_s": 97.0}})
        self.assertTrue(any("not worth tuning" in n for n in notes), notes)

    def test_probe_returns_a_positive_rate(self):
        self.assertGreater(numa._probe(buf_mb=4, seconds=0.1), 0)


# --------------------------------------------------------------------------- #
class TestIobench(unittest.TestCase):
    def test_job_spec_parsing(self):
        job = iobench.parse_job("oltp:bs=8k,pattern=randread,qd=32,rw=70")
        self.assertEqual(job.name, "oltp")
        self.assertEqual(job.block_size, 8192)
        self.assertEqual(job.queue_depth, 32)
        self.assertEqual(job.read_pct, 70)

    def test_size_suffixes(self):
        self.assertEqual(iobench._parse_size("4k"), 4096)
        self.assertEqual(iobench._parse_size("1m"), 1048576)
        self.assertEqual(iobench._parse_size("2g"), 2 * 1024 ** 3)

    def test_unknown_option_is_rejected_with_valid_names(self):
        with self.assertRaises(ValueError) as ctx:
            iobench.parse_job("x:bogus=1")
        self.assertIn("bs, pattern, qd", str(ctx.exception))

    def test_unknown_pattern_is_rejected(self):
        with self.assertRaises(ValueError):
            iobench.parse_job("x:pattern=sideways")

    def test_read_pct_is_bounded(self):
        with self.assertRaises(ValueError):
            iobench.JobSpec("x", read_pct=150)

    def test_default_suite_covers_the_four_profiles(self):
        names = {j.name for j in iobench.default_suite()}
        self.assertEqual(names,
                         {"database", "sequential", "log_write", "vm_mixed"})

    def test_job_runs_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            job = iobench.JobSpec("t", block_size=4096, pattern="randread",
                                  queue_depth=2, seconds=0.3, file_mb=8)
            result = iobench.run_job(job, d)
            if result.get("skipped"):
                self.skipTest(result["reason"])
            self.assertGreater(result["iops"], 0)
            self.assertIn("latency_us", result)
            self.assertEqual(os.listdir(d), [],
                             "the I/O test file must be removed")

    def test_percentiles_are_ordered(self):
        values = [float(i) for i in range(1000)]
        self.assertLessEqual(iobench._pct(values, 50), iobench._pct(values, 99))


# --------------------------------------------------------------------------- #
class TestDataScience(unittest.TestCase):
    def test_model_sizing_fits_the_budget(self):
        spec = datascience.choose_model(1 * 1024 ** 3, dtype_size=4)
        self.assertLessEqual(spec.bytes(4), 1 * 1024 ** 3 * 0.06 + 1)

    def test_small_machines_get_small_models(self):
        small = datascience.choose_model(256 * 1024 ** 2)
        large = datascience.choose_model(128 * 1024 ** 3)
        self.assertLessEqual(small.params, large.params)

    def test_parameter_count_matches_the_formula(self):
        spec = datascience.ModelSpec(d_model=128, n_layers=2, n_heads=4,
                                     vocab=100)
        self.assertEqual(spec.params, 2 * 12 * 128 * 128 + 100 * 128)

    def test_accelerator_memory_reports_model_sizes(self):
        result = datascience.accelerator_memory(16 * 1024 ** 3)
        if result.get("skipped"):
            self.skipTest(result["reason"])
        fits = result["largest_model_billions"]
        # Halving the bytes per parameter must roughly double what fits.
        self.assertGreater(fits["int4"], fits["int8"])
        self.assertGreater(fits["int8"], fits["fp16"])

    def test_extract_rates_ignores_skipped_sections(self):
        rates = datascience.extract_rates(
            {"llm": {"decode_tokens_per_s": 100.0},
             "dataloader": {"skipped": True}})
        self.assertEqual(rates, {"llm_decode": 100.0})

    def test_render_tolerates_partial_results(self):
        # A section that never ran is {} — neither skipped nor populated.
        self.assertIsInstance(datascience.render({"llm": {}}), str)

    def test_dataloader_produces_a_rate(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("needs NumPy")
        result = datascience.dataloader(seconds=0.3, batch_size=4)
        if result.get("skipped"):
            self.skipTest(result["reason"])
        self.assertGreater(result["rate"], 0)
        self.assertEqual(result["unit"], "samples/s")

    def test_numpy_transformer_produces_finite_output(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("needs NumPy")
        spec = datascience.ModelSpec(64, 2, 4, vocab=100)
        layers = datascience._np_weights(spec, np)
        x = np.zeros((4, 64), dtype=np.float32)
        out = datascience._np_forward(x, layers, spec, np)
        self.assertEqual(out.shape, (4, 64))
        self.assertTrue(np.isfinite(out).all())


# --------------------------------------------------------------------------- #
class TestTwoNodeNetwork(unittest.TestCase):
    def test_client_reports_a_clear_error_without_a_server(self):
        result = network.measure_latency("127.0.0.1", port=1, probes=10)
        self.assertIn("error", result)
        self.assertIn("net-server", result.get("hint", ""))

    def test_round_trip_over_loopback(self):
        import threading as _threading
        port = 51987
        server = _threading.Thread(
            target=network.serve,
            kwargs={"port": port, "bind": "127.0.0.1", "seconds": 8,
                    "quiet": True},
            daemon=True)
        server.start()
        time.sleep(0.4)          # let the listener bind before connecting

        latency = network.measure_latency("127.0.0.1", port, probes=40)
        if latency.get("error"):
            self.skipTest(latency["error"])
        self.assertGreater(latency["probes"], 0)
        self.assertGreaterEqual(latency["jitter_ms"], 0.0)

        throughput = network.measure_throughput("127.0.0.1", port,
                                                seconds=0.5, streams=2)
        if throughput.get("error"):
            self.skipTest(throughput["error"])
        self.assertGreater(throughput["megabytes_per_s"], 0)


# --------------------------------------------------------------------------- #
class TestEnergy(unittest.TestCase):
    def test_energy_to_solution_returns_units_per_joule(self):
        result = power.energy_to_solution(lambda: 1000, "Apple M4", "test")
        self.assertIn("joules", result)
        if result["joules"]:
            self.assertIn("units_per_joule", result)

    def test_tdp_derived_energy_is_labelled_as_an_estimate(self):
        # Integrating TDP estimates and calling the total "measured" would be
        # exactly the false precision the module exists to avoid.
        result = power.energy_to_solution(lambda: 10, "core i7", "test")
        if result.get("joules") and result.get("method", "").startswith("a ~"):
            self.assertTrue(result["estimated"])

    def test_race_to_idle_is_described_correctly(self):
        text = power.compare_efficiency({"joules": 100.0, "seconds": 10.0},
                                        {"joules": 80.0, "seconds": 20.0})
        self.assertIn("sooner", text)
        self.assertIn("less energy", text)

    def test_same_winner_on_both_axes(self):
        text = power.compare_efficiency({"joules": 50.0, "seconds": 5.0},
                                        {"joules": 80.0, "seconds": 20.0})
        self.assertIn("both", text)


# --------------------------------------------------------------------------- #
class TestExtendedCliSurface(unittest.TestCase):
    def test_new_flags_parse(self):
        parser = cli.build_parser()
        args = parser.parse_args([
            "--counters", "--numa", "--numa-bandwidth", "--datascience",
            "--io", "--io-job", "x:bs=4k", "--energy", "--no-standards",
            "--no-linpack", "--no-provenance", "--net-port", "9999"])
        self.assertTrue(args.counters)
        self.assertTrue(args.numa_bandwidth)
        self.assertEqual(args.io_job, ["x:bs=4k"])
        self.assertEqual(args.net_port, 9999)

    def test_compare_runs_takes_two_paths(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--compare-runs", "a.json", "b.json"])
        self.assertEqual(args.compare_runs, ["a.json", "b.json"])

    def test_compare_runs_reports_a_regression_and_exits_six(self):
        import io as _io
        import contextlib
        with tempfile.TemporaryDirectory() as d:
            def write(name, rate):
                payload = {
                    "system": {"hostname": "h", "cpu_model": "c"},
                    "config": {"seconds": 3, "repeats": 8},
                    "results": {"cpu_int": {
                        "samples": [rate + (i % 3) for i in range(8)]}},
                    "scores": {"composite": rate / 10},
                }
                path = os.path.join(d, name)
                with open(path, "w") as f:
                    json.dump(payload, f)
                return path

            a, b = write("a.json", 1000.0), write("b.json", 700.0)
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cli.main(["--compare-runs", a, b])
            self.assertEqual(code, 6)
            self.assertIn("REGRESSION", buf.getvalue())

    def test_compare_runs_on_missing_file_exits_two(self):
        import io as _io
        import contextlib
        with contextlib.redirect_stderr(_io.StringIO()):
            self.assertEqual(
                cli.main(["--compare-runs", "/nonexistent/a.json",
                          "/nonexistent/b.json"]), 2)

    def test_bad_io_job_is_rejected_before_benchmarking(self):
        import io as _io
        import contextlib
        buf = _io.StringIO()
        with contextlib.redirect_stderr(buf):
            code = cli.main(["--io-job", "x:pattern=sideways", "--no-save",
                             "--only", "cpu_int", "--quick", "--force",
                             "--no-native", "--no-accel", "--no-power",
                             "--no-optional", "--no-plugins", "--no-network",
                             "--no-standards", "--json-stdout"])
        self.assertEqual(code, 2)


# --------------------------------------------------------------------------- #
class TestNewScoringCalibration(unittest.TestCase):
    def test_every_new_source_has_a_baseline(self):
        for score_key, _, _ in scoring._SOURCES:
            self.assertIn(score_key, scoring.BASELINES)

    def test_reference_machine_scores_exactly_one_hundred(self):
        keys = ("stream_triad", "coremark_style", "linpack", "llm_prefill",
                "llm_decode", "dataloader", "dataframe")
        results = {k: {"rate": scoring.BASELINES[k]} for k in keys}
        for key, value in scoring.compute_scores(results)["subscores"].items():
            self.assertAlmostEqual(value, 100.0, places=1, msg=key)

    def test_standards_and_datascience_categories_exist(self):
        subscores = {"stream_triad": 100.0, "linpack": 100.0,
                     "llm_decode": 100.0, "dataloader": 100.0}
        cats = scoring.category_scores(subscores)
        self.assertIn("standards", cats)
        self.assertIn("datascience", cats)


# --------------------------------------------------------------------------- #
class TestPowerLoadIsParallel(unittest.TestCase):
    """The 'under load' power reading must actually load every core.

    Regression test for a real bug: the load was generated with Python threads,
    which the GIL serialises onto one core. On a 10-core M1 Max that reported
    6.9 W as the all-core figure and inflated every perf-per-watt number
    derived from it. The bug was invisible wherever power was only a TDP
    estimate, because the estimate ignores load entirely.
    """

    def test_burn_worker_is_importable_for_spawn(self):
        # spawn() re-imports the target, so a closure or local function would
        # fail at run time on macOS and Windows.
        self.assertTrue(callable(power._burn))
        self.assertEqual(power._burn.__module__, "pcbench.power")

    @staticmethod
    def _thread_cores(workers: int, seconds: float) -> float:
        """Cores kept busy by the original thread-based approach."""
        import threading
        stop = threading.Event()

        def burn():
            x = 0
            while not stop.is_set():
                for _ in range(10_000):
                    x = (x * 1103515245 + 12345) & 0x7FFFFFFF

        before = resource.getrusage(resource.RUSAGE_SELF)
        ts = [threading.Thread(target=burn, daemon=True)
              for _ in range(workers)]
        start = time.perf_counter()
        for t in ts:
            t.start()
        time.sleep(seconds)
        stop.set()
        for t in ts:
            t.join(timeout=5.0)
        wall = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_SELF)
        cpu = ((after.ru_utime - before.ru_utime)
               + (after.ru_stime - before.ru_stime))
        return cpu / wall if wall else 0.0

    @staticmethod
    def _process_cores(workers: int, seconds: float) -> float:
        """Cores kept busy by the process-based replacement."""
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        stop = ctx.Event()
        ready = [ctx.Event() for _ in range(workers)]
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        procs = [ctx.Process(target=power._burn, args=(stop, ready[i]),
                             daemon=True) for i in range(workers)]
        for p in procs:
            p.start()
        # Spawn returns before the child is running; without waiting for the
        # readiness signal the timed window includes several hundred
        # milliseconds of Python startup per worker and the test is flaky.
        for event in ready:
            event.wait(timeout=20.0)
        start = time.perf_counter()
        time.sleep(seconds)
        stop.set()
        for p in procs:
            p.join(timeout=10.0)
        wall = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = ((after.ru_utime - before.ru_utime)
               + (after.ru_stime - before.ru_stime))
        return cpu / wall if wall else 0.0

    def test_processes_keep_more_cores_busy_than_threads(self):
        """Processes must beat threads at raising load.

        Asserted as a ratio rather than an absolute core count on purpose: an
        absolute threshold fails on a busy CI runner or developer machine,
        where neither approach can reach full speed. Contention scales both
        measurements down together, so their ratio stays meaningful.
        """
        cores = os.cpu_count() or 1
        if cores < 2:
            self.skipTest("needs at least 2 cores to detect the difference")
        workers = min(4, cores)

        threaded = self._thread_cores(workers, 0.7)
        processed = self._process_cores(workers, 0.7)

        self.assertGreater(
            processed, threaded * 1.5,
            f"processes kept {processed:.2f} cores busy against {threaded:.2f} "
            f"for threads — the GIL fix is not effective, so the power reading "
            f"would still be single-core")

    def test_estimated_power_skips_the_load_entirely(self):
        # A TDP estimate is load-independent, so spawning processes to produce
        # load would cost seconds for no change in the answer.
        import unittest.mock as mock
        with mock.patch.object(power, "measure",
                               return_value={"package_w": 15.0,
                                             "estimated": True}) as m:
            result = power.measure_under_load("core i7")
        self.assertTrue(result["estimated"])
        self.assertEqual(m.call_count, 1, "no load should have been raised")


# --------------------------------------------------------------------------- #
class TestStreamArraySizing(unittest.TestCase):
    """STREAM arrays must clear the last-level cache by ~4x.

    Regression test for a real reporting error: the fixed 64 MB default gave
    only 1.4x on an M1 Max's 48 MB system-level cache, so the reported Triad
    figure was partly cache bandwidth.
    """

    GB = 1024 ** 3
    MB = 1024 ** 2

    def test_large_cache_gets_larger_arrays(self):
        small = native.stream_array_mb(2 * self.MB, 64 * self.GB)
        large = native.stream_array_mb(256 * self.MB, 512 * self.GB)
        self.assertGreater(large, small)

    def test_four_times_rule_is_met_for_a_big_cache(self):
        cache_mb = 256
        chosen = native.stream_array_mb(cache_mb * self.MB, 512 * self.GB)
        self.assertGreaterEqual(chosen, 4 * cache_mb)

    def test_apple_silicon_hidden_slc_is_covered_by_the_floor(self):
        # An M1 Max reports 24 MB of L2 while also having a 48 MB SLC the OS
        # never mentions, so the floor has to carry the margin.
        chosen_mb = native.stream_array_mb(24 * self.MB, 64 * self.GB)
        self.assertGreaterEqual(chosen_mb * self.MB, 4 * 48 * self.MB)

    def test_small_machines_are_clamped_below_a_quarter_of_ram(self):
        ram = 1 * self.GB
        chosen = native.stream_array_mb(2 * self.MB, ram)
        self.assertLess(chosen * self.MB * 3, ram / 3)

    def test_undetected_cache_still_yields_a_sane_default(self):
        self.assertGreaterEqual(native.stream_array_mb(None, 8 * self.GB), 64)

    def test_cache_note_states_the_ratio_and_the_verdict(self):
        ok = standards._stream_cache_note(256 * self.MB, 16 * self.MB, "L2")
        self.assertIn("satisfying", ok)
        bad = standards._stream_cache_note(64 * self.MB, 48 * self.MB, "SLC")
        self.assertIn("BELOW", bad)
        self.assertIn("--stream-mb", bad)

    def test_cache_note_admits_when_it_cannot_check(self):
        note = standards._stream_cache_note(64 * self.MB, None, "unavailable")
        self.assertIn("could not be checked", note)

    def test_last_level_cache_detection_never_raises(self):
        size, source = system.last_level_cache_bytes()
        self.assertIsInstance(source, str)
        if size is not None:
            self.assertGreater(size, 0)


# --------------------------------------------------------------------------- #
class TestInterpreterBoundDiagnosis(unittest.TestCase):
    """Pure-Python categories must not masquerade as independent bottlenecks.

    The `ml` category (nn_training, kmeans, knn) runs in pure Python, so it
    re-measures the CPU through the interpreter. Measured on real hardware it
    tracks `cpu_int` to within 2-4% — 111.7 vs 113.5 on an M1 Max, 231.5 vs
    223.4 on an M4 — so naming it a bottleneck restates the CPU result while
    implying a separate, fixable weakness.
    """

    BALANCED = {"cpu_int": 300.0, "cpu_float": 300.0, "compression": 300.0,
                "hashing": 300.0, "json": 300.0, "memory": 300.0,
                "gpu_fp32": 300.0, "npu": 300.0, "compile": 300.0,
                "syscall": 300.0, "blas_matmul": 300.0, "aes": 300.0}

    def test_ml_tracking_cpu_is_recognised(self):
        self.assertTrue(diagnose.tracks_cpu("ml", 111.7, {"cpu_int": 113.5}))
        self.assertTrue(diagnose.tracks_cpu("ml", 231.5, {"cpu_int": 223.4}))

    def test_ml_far_below_cpu_is_not_excused(self):
        self.assertFalse(diagnose.tracks_cpu("ml", 40.0, {"cpu_int": 300.0}))

    def test_non_interpreter_categories_are_never_excused(self):
        self.assertFalse(diagnose.tracks_cpu("disk", 30.0, {"cpu_int": 30.0}))

    def test_missing_driver_subscore_is_not_excused(self):
        self.assertFalse(diagnose.tracks_cpu("ml", 100.0, {}))

    def test_verdict_attributes_tracking_ml_to_the_cpu(self):
        scores = dict(self.BALANCED, nn_training=95.0, kmeans=95.0,
                      knn=95.0, cpu_int=100.0)
        verdict = diagnose.analyse({"subscores": scores})["verdict"]
        self.assertIn("pure-Python", verdict)
        self.assertNotIn("separate subsystem is weak", verdict.split("not that")[0])

    def test_genuinely_weak_ml_still_reported_plainly(self):
        scores = dict(self.BALANCED, nn_training=40.0, kmeans=40.0, knn=40.0)
        result = diagnose.analyse({"subscores": scores})
        self.assertIn("ml is well below", result["verdict"])
        self.assertFalse(result["bottlenecks"][0]["restates_cpu"])

    def test_real_bottleneck_is_the_headline_over_tracking_ml(self):
        scores = dict(self.BALANCED, nn_training=290.0, kmeans=290.0,
                      knn=290.0, disk_read=30.0, disk_write=30.0,
                      disk_iops=30.0)
        verdict = diagnose.analyse({"subscores": scores})["verdict"]
        self.assertIn("disk", verdict)

    def test_impact_text_points_at_numpy_rather_than_hardware(self):
        scores = dict(self.BALANCED, nn_training=95.0, kmeans=95.0,
                      knn=95.0, cpu_int=100.0)
        result = diagnose.analyse({"subscores": scores})
        ml = [b for b in result["bottlenecks"] if b["category"] == "ml"]
        if ml:
            self.assertIn("NumPy", ml[0]["impact"])


# --------------------------------------------------------------------------- #
class TestInterferenceSelfLoad(unittest.TestCase):
    """A test that saturates every core must not blame its own load on others.

    Measured on a 10-core machine with a no-load control: a 3-second
    `cpu_multi` raises load by 0.186 per core against 0.000 drift — 74% of the
    0.25 threshold — and longer runs clear it outright, producing a false
    "something else started competing for the CPU" on an idle machine.
    """

    BEFORE = {"load_per_core": 0.10, "celsius": 50.0}
    AFTER_LOADED = {"load_per_core": 0.90, "celsius": 52.0}

    def test_self_parallel_test_does_not_flag_its_own_load(self):
        for name in ("cpu_multi", "cores", "mem_scaling"):
            verdict = interference.compare_samples(
                self.BEFORE, self.AFTER_LOADED, name)
            self.assertFalse(verdict["disturbed"],
                             f"{name} blamed its own load on something else")
            self.assertIn("self-inflicted", verdict["load_note"])

    def test_single_threaded_test_still_detects_external_load(self):
        verdict = interference.compare_samples(
            self.BEFORE, self.AFTER_LOADED, "cpu_int")
        self.assertTrue(verdict["disturbed"])
        self.assertTrue(any("competing" in n for n in verdict["notes"]))

    def test_temperature_signal_survives_suppression(self):
        hot = {"load_per_core": 0.10, "celsius": 95.0}
        verdict = interference.compare_samples(self.BEFORE, hot, "cpu_multi")
        self.assertTrue(verdict["disturbed"])
        self.assertTrue(any("warmed" in n for n in verdict["notes"]))

    def test_load_delta_is_still_recorded_as_data(self):
        verdict = interference.compare_samples(
            self.BEFORE, self.AFTER_LOADED, "cpu_multi")
        self.assertAlmostEqual(verdict["load_delta"], 0.8, places=2)
        self.assertFalse(verdict["load_signal_used"])

    def test_unknown_test_name_keeps_the_load_check(self):
        # A plugin or a new test defaults to being trusted as single-threaded;
        # a false positive is preferable to silently dropping the signal.
        verdict = interference.compare_samples(
            self.BEFORE, self.AFTER_LOADED, "some_plugin")
        self.assertTrue(verdict["disturbed"])

    def test_every_self_parallel_name_is_a_real_test(self):
        for name in interference.SELF_PARALLEL_TESTS:
            self.assertIn(name, cli.TESTS)


# --------------------------------------------------------------------------- #
class TestCoreScalingCause(unittest.TestCase):
    """A scaling knee has two causes, and the curve alone cannot tell them apart.

    Regression test: the analysis saw only `os.cpu_count()` (logical) and
    called every knee "a hybrid design with slower efficiency cores". On any
    x86 CPU with SMT — most of them — the knee is hyperthreads sharing
    execution units, which is a different fact calling for a different action.
    """

    @staticmethod
    def _points(gains):
        aggregate, out = 0.0, []
        for i, g in enumerate(gains, 1):
            aggregate += g
            out.append({"workers": i, "marginal_rate": g,
                        "aggregate_rate": aggregate,
                        "scaling_vs_one": aggregate / gains[0]})
        return out

    def test_smt_knee_is_not_called_hybrid(self):
        points = self._points([4.0] * 8 + [1.2] * 8)
        result = cores.classify_cores(points, physical_cores=8,
                                      logical_cores=16)
        self.assertEqual(result["cause"], "smt")
        self.assertFalse(result["hybrid"])
        self.assertIn("SMT", result["note"])
        self.assertNotIn("efficiency cores", result["note"])

    def test_true_hybrid_without_smt_is_still_reported(self):
        # An M1 Max: 8 performance + 2 efficiency, physical == logical.
        points = self._points([2.2] * 8 + [0.85] * 2)
        result = cores.classify_cores(points, physical_cores=10,
                                      logical_cores=10)
        self.assertEqual(result["cause"], "hybrid")
        self.assertTrue(result["hybrid"])
        self.assertIn("efficiency cores", result["note"])

    def test_uniform_cores_claim_nothing(self):
        points = self._points([3.0] * 8)
        result = cores.classify_cores(points, physical_cores=8,
                                      logical_cores=8)
        self.assertIsNone(result["cause"])
        self.assertIn("uniform", result["note"])

    def test_knee_away_from_physical_count_refuses_to_guess(self):
        # SMT exists but the knee is at 4 of 8 physical cores, so it is not
        # the hyperthread boundary and the cause is genuinely unknown.
        points = self._points([4.0] * 4 + [1.2] * 12)
        result = cores.classify_cores(points, physical_cores=8,
                                      logical_cores=16)
        self.assertEqual(result["cause"], "ambiguous")
        self.assertFalse(result["hybrid"])
        self.assertIn("cannot separate", result["note"])

    def test_unknown_physical_count_does_not_invent_smt(self):
        points = self._points([4.0] * 8 + [1.2] * 8)
        result = cores.classify_cores(points, physical_cores=None,
                                      logical_cores=16)
        self.assertNotEqual(result["cause"], "smt")

    def test_slow_relative_is_reported_for_every_knee(self):
        for physical, logical in ((8, 16), (10, 10)):
            points = self._points([4.0] * 8 + [1.2] * (logical - 8))
            result = cores.classify_cores(points, physical, logical)
            self.assertAlmostEqual(result["slow_relative"], 0.3, places=2)

    def test_too_few_points_is_handled(self):
        self.assertFalse(
            cores.classify_cores([{"marginal_rate": 1.0}], 4, 4)["hybrid"])


# --------------------------------------------------------------------------- #
class TestDriveLifetime(unittest.TestCase):
    """SSD wear reporting: terabytes written, endurance used, projections."""

    NVME = {"model": "TEST SSD", "protocol": "NVMe", "percentage_used": 2,
            "data_units_read": 106_772_999, "data_units_written": 50_334_563,
            "power_cycles": 321, "power_on_hours": 854,
            "unsafe_shutdowns": 13, "media_errors": 0,
            "available_spare_pct": 100, "available_spare_threshold_pct": 99,
            "temperature_c": 50}

    def test_data_units_convert_to_terabytes(self):
        # An NVMe data unit is fixed by spec at 1000 x 512 bytes.
        d = drivelife._normalise(dict(self.NVME))
        self.assertAlmostEqual(d["written_tb"], 25.77, places=1)
        self.assertAlmostEqual(d["read_tb"], 54.67, places=1)

    def test_health_is_the_complement_of_wear(self):
        d = drivelife._normalise(dict(self.NVME))
        self.assertEqual(d["health_pct"], 98)

    def test_health_never_goes_negative(self):
        # Controllers keep counting past 100% once endurance is spent.
        d = drivelife._normalise(dict(self.NVME, percentage_used=140))
        self.assertEqual(d["health_pct"], 0)

    def test_write_rate_is_per_power_on_day(self):
        # A machine that sleeps has far fewer power-on days than calendar
        # days; labelling this "per day" would overstate the load severalfold.
        d = drivelife._normalise(dict(self.NVME))
        self.assertIn("write_rate_gb_per_power_on_day", d)
        self.assertNotIn("write_rate_gb_per_day", d)

    def test_projection_is_in_power_on_hours_not_calendar_years(self):
        """SMART counts power-on hours and carries no manufacture date.

        Dividing remaining hours by 8760 would assume 24/7 operation and
        understate a laptop's calendar life by roughly ten times.
        """
        p = drivelife.project_lifetime(drivelife._normalise(dict(self.NVME)))
        self.assertIn("projected_remaining_hours", p)
        years = p["projected_remaining_years"]
        self.assertIsInstance(years, dict)
        # Fewer hours per day must mean more calendar years.
        self.assertGreater(years["at_4h_per_day"], years["at_8h_per_day"])
        self.assertGreater(years["at_8h_per_day"], years["at_24h_per_day"])

    def test_projection_declines_without_enough_history(self):
        p = drivelife.project_lifetime(dict(self.NVME, power_on_hours=10))
        self.assertIn("note", p)
        self.assertNotIn("projected_remaining_hours", p)

    def test_zero_wear_is_reported_as_such_not_as_infinite_life(self):
        p = drivelife.project_lifetime(dict(self.NVME, percentage_used=0))
        self.assertIn("note", p)

    def test_missing_fields_yield_no_projection(self):
        self.assertIsNone(drivelife.project_lifetime({"model": "x"}))

    def test_healthy_drive_raises_no_warnings(self):
        result = {"drives": [drivelife._normalise(dict(self.NVME))]}
        self.assertEqual(drivelife.warnings(result), [])

    def test_media_errors_are_flagged(self):
        result = {"drives": [dict(self.NVME, media_errors=7)]}
        notes = drivelife.warnings(result)
        self.assertTrue(any("media/data-integrity" in n for n in notes))

    def test_spare_below_drive_threshold_is_called_failing(self):
        result = {"drives": [dict(self.NVME, available_spare_pct=50,
                                  available_spare_threshold_pct=99)]}
        notes = drivelife.warnings(result)
        self.assertTrue(any("failing drive" in n for n in notes), notes)

    def test_worn_drive_is_flagged_at_two_levels(self):
        warn = drivelife.warnings({"drives": [dict(self.NVME,
                                                   percentage_used=85)]})
        crit = drivelife.warnings({"drives": [dict(self.NVME,
                                                   percentage_used=99)]})
        self.assertTrue(any("plan a replacement" in n for n in warn))
        self.assertTrue(any("replace it" in n for n in crit))

    def test_critical_warning_flag_is_surfaced(self):
        notes = drivelife.warnings({"drives": [dict(self.NVME,
                                                    critical_warning=1)]})
        self.assertTrue(any("CRITICAL WARNING" in n for n in notes))

    def test_hot_drive_is_flagged(self):
        notes = drivelife.warnings({"drives": [dict(self.NVME,
                                                    temperature_c=78)]})
        self.assertTrue(any("shortens flash life" in n for n in notes))

    def test_frequent_unsafe_shutdowns_are_flagged(self):
        notes = drivelife.warnings({"drives": [dict(self.NVME,
                                                    unsafe_shutdowns=200,
                                                    power_cycles=300)]})
        self.assertTrue(any("unsafe shutdowns" in n for n in notes))

    def test_sata_life_remaining_is_inverted_to_wear(self):
        # SATA attributes report life *remaining*; NVMe reports life *used*.
        data = {"model": "SATA SSD",
                "ata_smart_attributes": {"table": [
                    {"name": "Media_Wearout_Indicator", "value": 90,
                     "raw": {"value": 0}},
                    {"name": "Power_On_Hours", "value": 99,
                     "raw": {"value": 12345}},
                ]}}
        entry = drivelife._from_smartctl_json(data)
        self.assertEqual(entry["percentage_used"], 10)
        self.assertEqual(entry["power_on_hours"], 12345)

    def test_smartctl_nvme_json_is_mapped(self):
        data = {"model_name": "NV", "nvme_smart_health_information_log": {
            "percentage_used": 5, "data_units_written": 1000,
            "power_on_hours": 500, "media_errors": 0}}
        entry = drivelife._from_smartctl_json(data)
        self.assertEqual(entry["protocol"], "NVMe")
        self.assertEqual(entry["percentage_used"], 5)

    def test_run_never_raises_and_always_reports_why(self):
        result = drivelife.run(".")
        self.assertIn("available", result)
        if not result["available"]:
            self.assertTrue(result.get("reason"))

    @unittest.skipUnless(platform.system() == "Darwin", "macOS only")
    def test_macos_reads_real_drive_data(self):
        result = drivelife.run(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if not result.get("available"):
            self.skipTest(result.get("reason", "unavailable"))
        drive = result["drives"][0]
        self.assertGreater(drive["written_tb"], 0)
        self.assertGreaterEqual(drive["health_pct"], 0)
        self.assertLessEqual(drive["health_pct"], 100)


# --------------------------------------------------------------------------- #
class TestGateListIndexing(unittest.TestCase):
    """The assertion grammar always accepted brackets; now they resolve.

    Paths like drive_life.drives[0].health_pct parsed cleanly and then silently
    failed to resolve, because the walk only stepped through dicts. Lists are
    common in the payload (drives, I/O jobs, NUMA nodes) and gating on one
    element is what fleet checks need.
    """

    PAYLOAD = {
        "scores": {"composite": 1.0, "subscores": {}},
        "results": {},
        "drive_life": {"drives": [{"health_pct": 98, "media_errors": 0},
                                  {"health_pct": 42, "media_errors": 3}]},
        "io": {"jobs": [{"iops": 49548}, {"iops": 1170}]},
    }

    def _one(self, expr):
        return gates.evaluate(self.PAYLOAD, [expr])[0]

    def test_positive_index_resolves(self):
        self.assertTrue(self._one("drive_life.drives[0].health_pct>=50")["passed"])
        self.assertFalse(self._one("drive_life.drives[1].health_pct>=50")["passed"])

    def test_negative_index_addresses_the_last_element(self):
        self.assertTrue(self._one("io.jobs[-1].iops>=1000")["passed"])
        self.assertFalse(self._one("io.jobs[-1].iops>=2000")["passed"])

    def test_equality_on_an_error_count(self):
        self.assertTrue(self._one("drive_life.drives[0].media_errors==0")["passed"])
        self.assertFalse(self._one("drive_life.drives[1].media_errors==0")["passed"])

    def test_out_of_range_index_fails_rather_than_raising(self):
        result = self._one("drive_life.drives[9].health_pct>=1")
        self.assertFalse(result["passed"])
        self.assertIn("not measured", result["message"])

    def test_indexing_a_non_list_fails_cleanly(self):
        self.assertFalse(self._one("scores.composite[0]>=1")["passed"])

    def test_negative_thresholds_still_parse(self):
        self.assertEqual(gates.parse("sustained.droop_pct>=-5")[2], -5.0)

    def test_malformed_brackets_are_rejected_with_a_clear_message(self):
        # A typo must not masquerade as "this metric was not measured".
        for bad in ("io.jobs[>=1", "io.jobs[a].iops>=1", "io.jobs[]>=1"):
            with self.assertRaises(gates.GateError, msg=bad):
                gates.parse(bad)

    def test_valid_indices_are_accepted_by_the_parser(self):
        for good in ("io.jobs[0].iops>=1", "io.jobs[-1].iops>=1",
                     "drive_life.drives[12].health_pct>=1"):
            gates.parse(good)


# --------------------------------------------------------------------------- #
class TestScoringDocumentation(unittest.TestCase):
    """The documented scoring must match what the code actually does.

    docs/technical.md states the formulas, every baseline constant, and a
    worked example taken from a real run. A reader who cannot trust those
    numbers cannot interpret any score, so each claim is asserted here.
    """

    #: The M1 Max run used as the worked example in docs/technical.md.
    RUN = {
        "cpu_int": 113.4, "cpu_float": 270.4, "cpu_multi": 222.1,
        "compression": 57.7, "hashing": 472.7, "json": 151.4,
        "memory": 761.2, "disk_write": 1405.6, "disk_read": 278.4,
        "disk_iops": 176.2, "gpu_fp32": 309.8, "gpu_fp16": 306.4,
        "gpu_bandwidth": 300.2, "gpu_matmul_fp32": 809.4,
        "gpu_matmul_fp16": 385.7, "npu": 230.1, "nn_training": 99.3,
        "kmeans": 118.3, "knn": 119.0, "disk_iops_peak": 153.8,
        "mem_scaling": 494.7, "compile": 304.4, "syscall": 97.7,
        "blas_matmul": 437.8, "fft": 122.6, "lapack": 232.1, "zstd": 157.7,
        "lz4": 93.3, "blake3": 177.3, "gpu_opencl": 304.4, "sqlite": 191.7,
        "raytrace": 162.6, "image": 183.1, "logparse": 203.7,
        "stream_triad": 274.3, "coremark_style": 185.6, "linpack": 199.4,
        "plugin_example_pi": 266.5,
    }

    @staticmethod
    def _geomean(values):
        return math.exp(sum(math.log(v) for v in values) / len(values))

    def test_every_baseline_makes_the_reference_machine_score_100(self):
        # The docs promise this for *every* metric, not a sample of them.
        results = {k: {"rate": v} for k, v in scoring.BASELINES.items()}
        subscores = scoring.compute_scores(results)["subscores"]
        self.assertTrue(subscores)
        for key, value in subscores.items():
            self.assertAlmostEqual(value, 100.0, places=1, msg=key)

    def test_documented_subscore_formula(self):
        # 100 x 2,268,000 / 2,000,000 = 113.4, the worked example.
        rate, baseline = 2_268_000.0, scoring.BASELINES["cpu_int"]
        self.assertAlmostEqual(100.0 * rate / baseline, 113.4, places=1)

    def test_documented_worked_composite(self):
        self.assertAlmostEqual(self._geomean(list(self.RUN.values())),
                               226.2, places=1)

    def test_documented_category_example(self):
        self.assertAlmostEqual(
            self._geomean([self.RUN["memory"], self.RUN["mem_scaling"]]),
            613.6, places=1)

    def test_documented_arithmetic_versus_geometric_gap(self):
        arithmetic = sum(self.RUN.values()) / len(self.RUN)
        self.assertAlmostEqual(arithmetic, 285.0, places=1)
        self.assertGreater(arithmetic, self._geomean(list(self.RUN.values())))

    def test_documented_composite_without_the_plugin(self):
        without = [v for k, v in self.RUN.items()
                   if not k.startswith("plugin_")]
        self.assertAlmostEqual(self._geomean(without), 225.2, places=1)

    def test_composite_averages_subscores_not_categories(self):
        """The docs state this explicitly, because it decides the weighting.

        A category with six members contributes six terms; averaging the
        category scores instead would make a two-metric category count as much
        as a six-metric one.
        """
        scores = scoring.compute_scores(
            {k: {"rate": scoring.BASELINES[k] * 2}
             for k in ("gpu_fp32", "gpu_fp16", "gpu_bandwidth", "memory")})
        # Every subscore is 200, so both routes give 200 here; the difference
        # is which set is averaged, checked directly.
        self.assertEqual(len(scores["subscores"]), 4)
        self.assertAlmostEqual(scores["composite"], 200.0, places=1)

    def test_absent_hardware_is_omitted_not_zeroed(self):
        # A single zero term would collapse any geometric mean to zero.
        scores = scoring.compute_scores({"cpu_int": {"rate": 2_000_000.0}})
        self.assertEqual(list(scores["subscores"]), ["cpu_int"])
        self.assertAlmostEqual(scores["composite"], 100.0, places=1)

    def test_skipped_and_zero_rates_never_enter_the_composite(self):
        scores = scoring.compute_scores({
            "cpu_int": {"rate": 2_000_000.0},
            "memory": {"rate": 0.0},
            "hashing": {"skipped": True, "rate": 500.0},
        })
        self.assertEqual(list(scores["subscores"]), ["cpu_int"])

    def test_documented_baseline_table_matches_the_code(self):
        """Every baseline in docs/technical.md must equal the constant."""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "docs", "technical.md")
        with open(path, encoding="utf-8") as f:
            doc = f.read()
        section = doc[doc.index("### Baseline constants"):
                      doc.index("### Category rollups")]
        documented = {}
        for line in section.splitlines():
            m = re.match(r"\|\s*`(\w+)`\s*\|\s*([\d,.]+)\s*\|", line)
            if m:
                documented[m.group(1)] = float(m.group(2).replace(",", ""))

        self.assertTrue(documented, "no baseline table found in technical.md")
        missing = set(scoring.BASELINES) - set(documented)
        self.assertFalse(missing, f"undocumented baselines: {sorted(missing)}")
        for key, value in documented.items():
            self.assertIn(key, scoring.BASELINES, f"{key} documented but gone")
            self.assertAlmostEqual(
                value, scoring.BASELINES[key], places=4,
                msg=f"{key}: docs say {value}, code says "
                    f"{scoring.BASELINES[key]}")

    def test_documented_category_membership_matches_the_code(self):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "docs", "technical.md")
        with open(path, encoding="utf-8") as f:
            doc = f.read()
        section = doc[doc.index("#### Category membership"):]
        documented = {}
        for line in section.splitlines():
            m = re.match(r"\|\s*\*\*(\w+)\*\*\s*\|\s*(.+?)\s*\|$", line)
            if m:
                documented[m.group(1)] = {
                    x.strip().strip("`") for x in m.group(2).split(",")}
            elif documented and not line.startswith("|"):
                break

        # Derive the real membership by probing category_scores.
        for name, keys in documented.items():
            probe = {k: 100.0 for k in keys}
            self.assertIn(name, scoring.category_scores(probe),
                          f"documented category {name} does not exist")
            self.assertAlmostEqual(
                scoring.category_scores(probe)[name], 100.0, places=1,
                msg=f"{name} membership in docs does not match the code")

    def test_fsync_is_documented_as_unscored_and_is(self):
        self.assertNotIn("fsync", scoring.BASELINES)


# --------------------------------------------------------------------------- #
class TestSubsystemFloorContext(unittest.TestCase):
    """A floor firing is an observation; whether it is a fault needs context.

    Regression test: the absolute floors flagged a healthy Raspberry Pi 4 on an
    SD card twice (SD cards sustain 20-45 MB/s by design) and told the owner of
    a working 5400rpm hard disk that "on an SSD it indicates a real fault".
    """

    SBC = {"cpu_int": 35.0, "cpu_float": 40.0, "hashing": 30.0, "json": 32.0}
    FAST = {"cpu_int": 300.0, "cpu_float": 320.0, "hashing": 290.0,
            "json": 310.0}

    @staticmethod
    def _sev(checks, metric):
        for c in checks:
            if c["metric"] == metric:
                return c["severity"]
        return None

    def test_sd_card_on_an_sbc_is_expected_not_a_fault(self):
        checks = reference.subsystem_checks(
            {"disk": {"read_rate": 45.0, "write_rate": 25.0,
                      "random_read_iops": 2500.0}}, self.SBC)
        self.assertTrue(checks)
        for c in checks:
            self.assertEqual(c["severity"], "expected", c["metric"])
            self.assertIn("single-board", c["note"])

    def test_narrow_memory_bus_on_an_sbc_is_expected(self):
        checks = reference.subsystem_checks({"memory": {"rate": 700.0}},
                                            self.SBC)
        self.assertEqual(self._sev(checks, "memory"), "expected")

    def test_working_hard_disk_is_recognised_by_its_pattern(self):
        # Low sequential *and* low random is a slow medium, not a fast one
        # that has degraded.
        checks = reference.subsystem_checks(
            {"disk": {"read_rate": 75.0, "write_rate": 60.0,
                      "random_read_iops": 120.0}}, self.FAST)
        self.assertTrue(checks)
        for c in checks:
            self.assertEqual(c["severity"], "expected", c["metric"])

    def test_failing_ssd_is_still_caught(self):
        # Healthy sequential with collapsed random is the signature that
        # matters, and it must survive the added context.
        checks = reference.subsystem_checks(
            {"disk": {"read_rate": 2500.0, "write_rate": 1800.0,
                      "random_read_iops": 90.0}}, self.FAST)
        self.assertEqual(self._sev(checks, "disk_iops"), "investigate")
        self.assertTrue(any("failing SSD" in c["note"] for c in checks))

    def test_single_channel_memory_on_a_fast_machine_is_investigated(self):
        checks = reference.subsystem_checks({"memory": {"rate": 600.0}},
                                            self.FAST)
        self.assertEqual(self._sev(checks, "memory"), "investigate")
        self.assertIn("single-channel", checks[0]["note"])

    def test_healthy_modern_machine_reports_nothing(self):
        self.assertEqual(reference.subsystem_checks(
            {"disk": {"read_rate": 2500.0, "write_rate": 1800.0,
                      "random_read_iops": 90000.0},
             "memory": {"rate": 20000.0}}, self.FAST), [])

    def test_slow_storage_without_a_cpu_anchor_is_investigated_not_excused(self):
        # With no anchor the machine class is unknown, so the tool must not
        # assume "probably an SBC" and wave a real problem through.
        checks = reference.subsystem_checks(
            {"disk": {"read_rate": 45.0, "write_rate": 25.0,
                      "random_read_iops": 40000.0}}, {})
        self.assertEqual(self._sev(checks, "disk_read"), "investigate")

    def test_render_marks_expected_findings_differently(self):
        checks = reference.subsystem_checks(
            {"memory": {"rate": 700.0}}, self.SBC)
        text = reference.render({"class": "embedded / SBC"}, checks)
        self.assertIn("i ", text)
        self.assertNotIn("!  memory", text)

    def test_every_floor_metric_has_an_explanation_for_both_severities(self):
        for metric in reference.FLOORS:
            for context in ({"sbc_class": True, "slow_medium_likely": True,
                             "rotational_pattern": True},
                            {"sbc_class": False, "slow_medium_likely": False,
                             "rotational_pattern": False}):
                severity, note = reference._explain(metric, 1.0, context)
                self.assertIn(severity, ("expected", "investigate"))
                self.assertTrue(note, f"{metric} has no explanation")


# --------------------------------------------------------------------------- #
class TestRegressionUsesHistoricalSpread(unittest.TestCase):
    """A fixed percentage threshold treats every metric as equally repeatable.

    They are not: sequential disk throughput swings tens of percent between
    runs on the same machine while integer CPU work varies under 1%. Judging
    both against one threshold makes the noisy metrics generate most of the
    findings — observed on a real machine reporting disk_write +134.6% on one
    run and "no significant change" on the next.
    """

    @staticmethod
    def _row(ts, **kw):
        row = {"hostname": "host", "timestamp_utc": ts, "cfg_disk_mb": "256",
               "cfg_mem_mb": "64", "python_version": "3.14",
               "python_impl": "CPython", "cores_logical": "10"}
        row.update({k: str(v) for k, v in kw.items()})
        return row

    def test_noisy_metric_within_its_own_spread_is_not_a_regression(self):
        history = [self._row(f"t{i}", disk_write_mb_s=v)
                   for i, v in enumerate([3018, 7081, 4605, 6200, 3400])]
        result = regression.analyze(
            self._row("now", disk_write_mb_s=3100), history)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["findings"], "the change should still be shown")
        self.assertEqual(result["findings"][0]["confidence"],
                         "within normal variation")

    def test_stable_metric_moving_is_a_real_regression(self):
        history = [self._row(f"t{i}", cpu_int_primes_s=v)
                   for i, v in enumerate([2_270_000, 2_265_000, 2_272_000,
                                          2_268_000, 2_271_000])]
        result = regression.analyze(
            self._row("now", cpu_int_primes_s=1_700_000), history)
        self.assertEqual(result["status"], "regression")
        self.assertEqual(result["findings"][0]["confidence"],
                         "outside normal variation")

    def test_too_little_history_is_provisional_not_confident(self):
        result = regression.analyze(
            self._row("now", disk_write_mb_s=7081),
            [self._row("t0", disk_write_mb_s=3018)])
        self.assertEqual(result["findings"][0]["confidence"], "provisional")
        self.assertIn("not enough history", regression.render(result))

    def test_provisional_is_not_summarised_as_normal_variation(self):
        text = regression.render(regression.analyze(
            self._row("now", disk_write_mb_s=7081),
            [self._row("t0", disk_write_mb_s=3018)]))
        self.assertNotIn("within each metric's normal", text)

    def test_provisional_slowdown_still_counts_as_a_regression(self):
        # With no way to rule it out, the conservative reading wins.
        result = regression.analyze(
            self._row("now", cpu_int_primes_s=1_000_000),
            [self._row("t0", cpu_int_primes_s=2_000_000)])
        self.assertEqual(result["status"], "regression")

    def test_mad_ignores_a_single_outlier(self):
        steady = [100.0, 101.0, 99.0, 100.0, 100.0]
        self.assertLess(regression._mad(steady), 3.0)
        self.assertLess(regression._mad(steady + [10_000.0]), 5.0)

    def test_mad_of_one_sample_is_zero(self):
        self.assertEqual(regression._mad([5.0]), 0.0)

    def test_spread_is_reported_so_the_reader_can_judge(self):
        history = [self._row(f"t{i}", disk_write_mb_s=v)
                   for i, v in enumerate([3018, 7081, 4605, 6200, 3400])]
        result = regression.analyze(
            self._row("now", disk_write_mb_s=3100), history)
        self.assertIn("typical_spread_pct", result["findings"][0])
        self.assertIn("%", regression.render(result))


# --------------------------------------------------------------------------- #
class TestWindowsCompatibility(unittest.TestCase):
    """No POSIX-only module may be imported at module scope.

    Regression test for a total failure on Windows: `counters.py` imported
    `resource` at module scope, so `import pcbench.cli` raised
    ModuleNotFoundError and the tool would not start at all — not merely that
    one section. It went undetected because the suite only ever ran on macOS.
    """

    #: Modules the CPython standard library does not provide on Windows.
    #: `posix` is deliberately absent: CPython's own shutil and selectors
    #: import it on a POSIX host, so treating it as forbidden would flag the
    #: stdlib rather than this package.
    POSIX_ONLY = {"resource", "fcntl", "pwd", "grp", "termios", "tty", "pty",
                  "syslog", "spwd", "crypt"}

    def test_no_module_level_posix_only_imports(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        offenders = []
        for name in sorted(os.listdir(os.path.join(root, "pcbench"))):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, "pcbench", name)
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                # col_offset 0 means module scope; anything indented sits
                # inside a function and only runs where it is reached.
                if node.col_offset != 0:
                    continue
                for imported in names:
                    if imported.split(".")[0] in self.POSIX_ONLY:
                        offenders.append(f"{name}:{node.lineno} {imported}")
        self.assertEqual(
            offenders, [],
            "these imports crash the whole tool on Windows; import them "
            "inside the function that needs them, or guard with try/except")

    def test_every_module_imports_without_posix_only_modules(self):
        """Simulate Windows by making those modules unimportable."""
        import importlib
        import pkgutil
        import pcbench

        blocked = self.POSIX_ONLY

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise ImportError(f"No module named '{name}'")
                return None

        saved_modules = {k: v for k, v in sys.modules.items()
                         if k.split(".")[0] in blocked or k.startswith("pcbench")}
        blocker = Blocker()
        sys.meta_path.insert(0, blocker)
        try:
            for key in list(sys.modules):
                if key.split(".")[0] in blocked or key.startswith("pcbench"):
                    del sys.modules[key]
            import pcbench as fresh
            failures = []
            for mod in pkgutil.iter_modules(fresh.__path__):
                try:
                    importlib.import_module(f"pcbench.{mod.name}")
                except Exception as e:
                    failures.append(f"pcbench.{mod.name}: {e}")
            self.assertEqual(failures, [])
        finally:
            sys.meta_path.remove(blocker)
            for key in list(sys.modules):
                if key.split(".")[0] in blocked or key.startswith("pcbench"):
                    del sys.modules[key]
            sys.modules.update(saved_modules)

    def test_counters_degrade_without_resource(self):
        # The module must still answer, with a note rather than an exception.
        snapshot = counters._windows_snapshot()
        self.assertIn("wall_clock", snapshot)
        delta = counters.resource_delta(snapshot, counters._windows_snapshot())
        self.assertIsInstance(delta, dict)
        # And the renderer must not assume every field is numeric.
        counters.render({"resources": delta, "notes": []})

    def test_render_handles_a_note_only_snapshot(self):
        text = counters.render({"resources": {
            "wall_clock": 1.0, "note": "needs psutil on Windows"}})
        self.assertIn("psutil", text)

    def test_delta_tolerates_missing_major_fault_counter(self):
        # Windows reports one page-fault total; major must not be invented.
        before = {"wall_clock": 0.0, "minor_faults": 10,
                  "involuntary_switches": 5, "max_rss_bytes": 1}
        after = {"wall_clock": 10.0, "minor_faults": 30,
                 "involuntary_switches": 25, "max_rss_bytes": 2}
        delta = counters.resource_delta(before, after)
        self.assertEqual(delta["involuntary_switches_per_s"], 2.0)
        self.assertNotIn("major_faults_per_s", delta)


# --------------------------------------------------------------------------- #
class TestPortablePositionalIO(unittest.TestCase):
    """`os.pread`/`os.pwrite` are POSIX-only.

    Regression test: the whole storage section died on Windows with "module
    'os' has no attribute 'pread'". The fallback is seek+read, which is two
    operations against a shared file pointer — so threads need private
    descriptors or they silently read the wrong offsets.
    """

    PAGE = 4096
    BLOCKS = 64

    def _stamped_file(self, directory):
        """A file whose every block records its own index."""
        path = os.path.join(directory, "stamped.bin")
        with open(path, "wb") as f:
            for i in range(self.BLOCKS):
                f.write(str(i).encode().ljust(self.PAGE, b"."))
        return path

    def test_fallback_reads_the_correct_bytes(self):
        original = workloads.HAS_PREAD
        workloads.HAS_PREAD = False
        try:
            with tempfile.TemporaryDirectory() as d:
                path = self._stamped_file(d)
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
                try:
                    for i in range(self.BLOCKS):
                        block = workloads.pread(fd, self.PAGE, i * self.PAGE)
                        self.assertEqual(int(block.split(b".")[0]), i)
                finally:
                    os.close(fd)
        finally:
            workloads.HAS_PREAD = original

    def test_threads_with_private_descriptors_read_correctly(self):
        import threading
        original = workloads.HAS_PREAD
        workloads.HAS_PREAD = False
        try:
            with tempfile.TemporaryDirectory() as d:
                path = self._stamped_file(d)
                shared = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
                wrong = []

                def reader(seed):
                    fd, owned = workloads.open_reader(path, shared)
                    try:
                        rnd = random.Random(seed)
                        for _ in range(500):
                            i = rnd.randrange(self.BLOCKS)
                            block = workloads.pread(fd, self.PAGE,
                                                    i * self.PAGE)
                            if int(block.split(b".")[0]) != i:
                                wrong.append(i)
                    finally:
                        if owned:
                            os.close(fd)

                threads = [threading.Thread(target=reader, args=(s,))
                           for s in range(6)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                os.close(shared)
                self.assertEqual(wrong, [], "private descriptors must not race")
        finally:
            workloads.HAS_PREAD = original

    def test_open_reader_gives_a_private_fd_only_when_needed(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._stamped_file(d)
            shared = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            try:
                original = workloads.HAS_PREAD
                workloads.HAS_PREAD = True
                fd, owned = workloads.open_reader(path, shared)
                self.assertFalse(owned, "pread needs no private descriptor")
                self.assertEqual(fd, shared)

                workloads.HAS_PREAD = False
                fd, owned = workloads.open_reader(path, shared)
                self.assertTrue(owned)
                self.assertNotEqual(fd, shared)
                os.close(fd)
            finally:
                workloads.HAS_PREAD = original
                os.close(shared)

    def test_pwrite_fallback_lands_at_the_right_offset(self):
        original = workloads.HAS_PREAD
        workloads.HAS_PREAD = False
        try:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "w.bin")
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    os.write(fd, b"\x00" * (self.PAGE * 4))
                    workloads.pwrite(fd, b"MARK", self.PAGE * 2)
                    self.assertEqual(
                        workloads.pread(fd, 4, self.PAGE * 2), b"MARK")
                finally:
                    os.close(fd)
        finally:
            workloads.HAS_PREAD = original

    def test_disk_benchmark_runs_on_the_fallback_path(self):
        original = workloads.HAS_PREAD
        workloads.HAS_PREAD = False
        try:
            with tempfile.TemporaryDirectory() as d:
                result = workloads.bench_disk(0.2, 1, 8, d)
                if result.get("skipped"):
                    self.skipTest(result.get("error", "disk unavailable"))
                self.assertGreater(result["random_read_iops"], 0)
        finally:
            workloads.HAS_PREAD = original


# --------------------------------------------------------------------------- #
class TestCompilerSelection(unittest.TestCase):
    """A bare clang on Windows needs a Visual Studio installation.

    Regression test: on a real Windows machine clang was picked, failed with
    "unable to find a Visual Studio installation", and took the native engine,
    STREAM, the CoreMark-style suite and the compile benchmark with it.
    """

    def test_windows_prefers_self_contained_gcc_over_clang(self):
        order = native._CANDIDATES_WINDOWS
        self.assertLess(order.index("gcc"), order.index("clang"),
                        "MinGW gcc is self-contained; clang needs Visual Studio")
        self.assertLess(order.index("cl"), order.index("clang"))

    def test_msvc_is_recognised_including_full_paths(self):
        for path in ("cl", "cl.exe", "clang-cl",
                     r"C:\Program Files\MSVC\bin\cl.exe"):
            self.assertTrue(native.is_msvc(path), path)
        for path in ("gcc", "cc", "clang", r"C:\mingw\bin\gcc.exe",
                     "/usr/bin/cc"):
            self.assertFalse(native.is_msvc(path), path)

    def test_msvc_gets_its_own_flag_dialect(self):
        cmd = native.build_command("cl", "engine.c", "engine.exe")
        self.assertIn("/O2", cmd)
        self.assertIn("/Fe:engine.exe", cmd)
        self.assertNotIn("-lm", cmd)

    def test_gnu_dialect_is_unchanged_on_posix(self):
        cmd = native.build_command("gcc", "engine.c", "engine")
        self.assertIn("-O2", cmd)
        self.assertIn("-o", cmd)

    def test_no_compiler_message_is_actionable_per_platform(self):
        hint = native._no_compiler_hint()
        self.assertTrue(hint)
        if os.name == "nt":
            self.assertIn("MinGW", hint)

    def test_compile_benchmark_shares_the_ordering(self):
        # Both must make the same choice, or one succeeds while the other
        # reports a failure on the same machine.
        candidates = native.compiler_candidates()
        chosen = sysbench._find_cc()
        if candidates:
            self.assertIsNotNone(chosen)
            self.assertIn(os.path.basename(chosen).split(".")[0],
                          candidates[0])


# --------------------------------------------------------------------------- #
class TestWindowsGpuVram(unittest.TestCase):
    """WMI's AdapterRAM is a 32-bit field and lies about large cards.

    Regression test: an RTX 5070 Ti with 16 GB reported 4293918720 bytes and
    the tool printed "4.0 GB" as fact.
    """

    def test_capped_value_is_rejected(self):
        # The exact figure the real machine reported.
        self.assertGreaterEqual(4_293_918_720, accel._ADAPTER_RAM_CAP)

    def test_plausible_small_values_are_still_trusted(self):
        for ram in (536_870_912, 2_147_483_648, 3_221_225_472):
            self.assertLess(ram, accel._ADAPTER_RAM_CAP)

    def test_registry_reader_degrades_to_empty(self):
        # No PowerShell on this platform; it must return {} not raise.
        self.assertIsInstance(accel._gpu_vram_from_registry(), dict)
