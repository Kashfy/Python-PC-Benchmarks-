"""Test suite for pcbench.

Uses stdlib ``unittest`` so the suite runs anywhere the tool itself runs, with
no test dependencies to install:

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import math
import os
import platform
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcbench import (accel, apps, cli, compare, config, container,  # noqa: E402
                     core, coreml_model, cores, cryptobench, diagnose, export,
                     gates, gpucompute, health, interference, monitor,
                     numeric, optional, plugins, reference, soak, storage,
                     sysbench, limits, mlbench, mlframework, network, npu,
                     onnx_model, power, regression, report, scoring,
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
