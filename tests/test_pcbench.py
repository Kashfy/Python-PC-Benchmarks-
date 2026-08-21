"""Test suite for pcbench.

Uses stdlib ``unittest`` so the suite runs anywhere the tool itself runs, with
no test dependencies to install:

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import os
import platform
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcbench import (accel, cli, compare, core, coreml_model,  # noqa: E402
                     limits, mlframework, network, power, regression, report,
                     scoring, system, workloads)


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
