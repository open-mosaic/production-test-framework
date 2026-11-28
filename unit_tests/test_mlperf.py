# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

import json
import subprocess
from unittest.mock import patch

import pytest

from production_test_framework.agentx.power import PowerSummary
from production_test_framework.reporting import mlperf as report_mlperf
from production_test_framework.reporting.formatting import ReportFormat
from production_test_framework.reporting.report import Reporter
from production_test_framework.ssh import CommandResult
from production_test_framework.workload.harness import HarnessInstallError
from production_test_framework.workload.mlperf_workload import (
    ROUGE_TARGETS,
    SUT_SCRIPT,
    MlperfHarness,
    MlperfInstallError,
    MlperfResult,
    MlperfWorkload,
    accuracy_problems,
    parse_detail_log,
    parse_rouge,
)
from production_test_framework.workload.workload import WorkloadResult, WorkloadStatus

#: Entries as LoadGen writes them to mlperf_log_detail.txt, from a Server run on 8 GPUs.
DETAIL = {
    "result_validity": "VALID",
    "result_completed_tokens_per_second": 1929.14,
    "result_first_token_mean_latency_ns": 97292305,
    "result_first_token_99.00_percentile_latency_ns": 646000000,
    "result_min_duration_met": True,
    "early_stopping_met": True,
}

SUT_STATS = {"requests": 200, "errors": 2, "output_tokens": 25600, "error_messages": ["sample 7: Server disconnected"]}


def _write_detail(path, entries):
    lines = ["not an MLLOG line", ":::MLLOG {broken"]
    lines += [f":::MLLOG {json.dumps({'key': key, 'value': value})}" for key, value in entries.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def harness(tmp_path):
    return MlperfHarness(root=tmp_path / "home")


@pytest.fixture
def workload(harness, tmp_path):
    return MlperfWorkload(
        harness=harness,
        api_server="http://localhost:8080",
        model="meta-llama/Llama-3.1-8B-Instruct",
        scenario="Server",
        mode="performance",
        run_dir=tmp_path / "run",
        sut_options={"max_concurrency": 256},
        loadgen_settings={"target_qps": 15},
    )


@pytest.fixture
def report(tmp_path):
    return Reporter(tmp_path / "report.html", fmt=ReportFormat.HTML)


def _rendered(reporter):
    reporter.flush()
    return reporter.path.read_text(encoding="utf-8")


class TestMlperfHarness:
    def test_not_installed_without_a_marker(self, harness):
        assert harness.is_installed() is False

    def test_a_current_install_is_not_repeated(self, harness):
        harness.python.parent.mkdir(parents=True)
        harness.python.touch()
        harness.marker.write_text(json.dumps(harness._wanted()))
        with patch("production_test_framework.workload.harness.subprocess.run") as run:
            assert harness.install() is False
        run.assert_not_called()

    def test_a_failed_step_names_itself_and_its_output(self, harness):
        failed = subprocess.CompletedProcess([], 1, "", "fatal: repository not found")
        with (
            patch("production_test_framework.workload.harness.shutil.which", return_value="/usr/bin/uv"),
            patch("production_test_framework.workload.harness.subprocess.run", return_value=failed),
            pytest.raises(MlperfInstallError, match=r"(?s)MLPerf install could not clone .*repository not found"),
        ):
            harness.install()

    def test_its_install_error_is_the_shared_one(self):
        assert issubclass(MlperfInstallError, HarnessInstallError)


class TestMlperfWorkload:
    def test_command_runs_the_sut_from_the_harness_venv(self, workload):
        argv = workload.build_command()
        assert argv[:2] == [str(workload.harness.python), str(SUT_SCRIPT)]
        assert SUT_SCRIPT.is_file()
        assert argv[argv.index("--scenario") + 1] == "Server"
        assert argv[argv.index("--max-concurrency") + 1] == "256"
        assert (workload.run_dir / "benchmark_command.txt").read_text().strip() == " ".join(argv)

    def test_loadgen_settings_go_into_user_conf(self, workload):
        workload.build_command()
        assert (workload.run_dir / "user.conf").read_text() == "llama3_1-8b.Server.target_qps = 15\n"

    def test_parse_reads_loadgen_logs_and_sut_stats(self, workload):
        workload.run_dir.mkdir(parents=True)
        _write_detail(workload.run_dir / "mlperf_log_detail.txt", DETAIL)
        (workload.run_dir / "sut_stats.json").write_text(json.dumps(SUT_STATS))
        result = workload.parse_output(CommandResult(0, "LoadGen output", ""))
        assert result.validity == "VALID"
        assert result.error_rate == pytest.approx(0.01)
        assert workload.log_path.read_text() == "LoadGen output"

    def test_parse_without_logs_fails(self, workload):
        with pytest.raises(RuntimeError, match="no readable logs"):
            workload.parse_output(CommandResult(0, "", ""))

    def test_a_failed_run_saves_its_log(self, workload):
        with patch(
            "production_test_framework.workload.command_workload.run_cancellable_command",
            return_value=CommandResult(1, "", "boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                workload._run_command_sync()
        assert "boom" in workload.log_path.read_text()


class TestMlperfParsing:
    def test_detail_log_keeps_the_last_value_of_each_key(self, tmp_path):
        path = tmp_path / "detail.txt"
        _write_detail(path, {"result_validity": "INVALID"})
        with path.open("a") as out:
            out.write(f":::MLLOG {json.dumps({'key': 'result_validity', 'value': 'VALID'})}\n")
        assert parse_detail_log(path) == {"result_validity": "VALID"}

    def test_rouge_reads_numpy_values(self):
        output = "{'rouge1': '38.79', 'rougeL': 24.5, 'gen_len': np.int64(8167644), 'gen_num': 13368}"
        assert parse_rouge(output) == {"rouge1": 38.79, "rougeL": 24.5, "gen_len": 8167644.0, "gen_num": 13368.0}

    def test_reference_scores_meet_the_targets(self):
        assert accuracy_problems({**ROUGE_TARGETS, "gen_len": 8167644}) == []

    def test_low_scores_and_length_are_named(self):
        problems = accuracy_problems({**ROUGE_TARGETS, "rouge1": 30.0, "gen_len": 1})
        assert [problem.split()[0] for problem in problems] == ["rouge1", "gen_len"]

    def test_invalid_reason_names_the_checks_not_met(self, tmp_path):
        result = MlperfResult(tmp_path, "performance", {"result_min_duration_met": False}, "", {})
        assert result.invalid_reason == "not met: result_min_duration_met"


class TestReportMlperf:
    def test_configuration_shows_the_flags_and_user_conf(self, report, workload):
        report.start_test("t.py::a")
        report_mlperf.report_mlperf_configuration(report, workload, [["profile", "fornax"]])
        html = _rendered(report)
        assert "--max-concurrency 256" in html
        assert "llama3_1-8b.Server.target_qps = 15" in html
        assert "fornax" in html

    def test_result_shows_latencies_in_ms_failed_requests_and_power(self, report, workload, tmp_path):
        result = MlperfResult(tmp_path, "performance", DETAIL, "summary text", SUT_STATS)
        power = PowerSummary(samples=3, gpu_count=8, gpu_avg_w=200.0)
        report.start_test("t.py::a")
        report_mlperf.report_mlperf_result(
            report, workload, WorkloadResult(0.0, 60.0, result, WorkloadStatus.COMPLETED), power, False
        )
        html = _rendered(report)
        assert "97.2923" in html
        assert "646" in html
        assert "1,929.1" in html
        assert "Request failed: sample 7: Server disconnected" in html
        assert "Measured power" in html
        assert "summary text" in html

    def test_a_failed_run_reports_its_output(self, report, workload):
        report.start_test("t.py::a")
        report_mlperf.report_mlperf_result(
            report, workload, WorkloadResult(0.0, 1.0, "LoadGen crashed", WorkloadStatus.ERROR), None, False
        )
        html = _rendered(report)
        assert "Workload output -- MLPerf (error)" in html
        assert "LoadGen crashed" in html
