# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Unit tests for the AgentX workload and its report."""

import json
import subprocess
from unittest.mock import patch

import pytest

from production_test_framework.reporting import agentx as report_agentx
from production_test_framework.reporting.formatting import ReportFormat
from production_test_framework.reporting.report import Reporter
from production_test_framework.ssh import CommandResult
from production_test_framework.workload.agentx_workload import (
    AGGREGATE_FILE,
    RESULT_FILENAME,
    AgentxHarness,
    AgentxResult,
    AgentxWorkload,
    find_artifacts,
)
from production_test_framework.workload.workload import WorkloadResult, WorkloadStatus

AGGREGATE = {
    "aiperf_version": "0.12.0",
    "request_count": {"unit": "requests", "avg": 95.0},
    "error_request_count": {"unit": "requests", "avg": 5.0},
    "time_to_first_token": {"unit": "ms", "avg": 900.0, "p50": 800.0, "p90": 1500.0, "p99": 2500.0},
    "inter_token_latency": {"unit": "ms", "avg": 20.0, "p50": 19.0, "p90": 25.0, "p99": 40.0},
    "goodput": None,
    "metadata": {"dataset": {"loader": "semianalysis_cc_traces_weka_062126_256k"}},
    "run_info": {"scenario_outcome": {"submission_valid": False}},
    "error_summary": [{"error_details": {"code": 400, "type": "BadRequest", "message": "too long"}, "count": 5}],
}

SCORED = {
    "num_gpus": 2,
    "kv_cache_pool_tokens": 123456,
    "request_accounting": {"records_profiled": 90, "records_warmup_dropped": 5},
    "request_metrics": {
        "qps": {"mean": 1.5, "p95": 3.0},
        "latency": {"ttft": {"mean": 0.9, "p50": 0.8}, "intvty": {"mean": 50.0}},
        "tokens": {"input": {"mean": 30000.0}},
        "throughput": {"total": {"tokens_per_second": 40000.0}, "per_gpu": {"total_tput_tps": 20000.0}},
        "cache": {"theoretical_cache_hit_rate": 0.9},
    },
    "server_metrics": {"cache": {"gpu_cache_hit_rate": 0.8}, "kv_cache": {"gpu_usage_pct": 0.5}},
    "warnings": ["server_metrics_export.json missing or empty"],
}


@pytest.fixture
def harness(tmp_path):
    return AgentxHarness(root=tmp_path / "home")


@pytest.fixture
def workload(harness, tmp_path):
    return AgentxWorkload(
        harness=harness,
        aiperf_options={"scenario": "agentx", "concurrency": 8, "unsafe_override": True, "ui_type": None},
        run_dir=tmp_path / "run",
        scoring_env={"FRAMEWORK": "vllm", "TP": "2"},
        plots=False,
    )


def _write_aggregate(directory, aggregate=AGGREGATE):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / AGGREGATE_FILE).write_text(json.dumps(aggregate), encoding="utf-8")


def _scoring_succeeds(workload):
    def run(argv, **kwargs):
        (workload.run_dir / f"{RESULT_FILENAME}.json").write_text(json.dumps(SCORED), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "scored", "")

    return patch("production_test_framework.workload.agentx_workload.subprocess.run", side_effect=run)


class TestAgentxHarness:
    def test_not_installed_without_a_marker(self, harness):
        assert harness.is_installed() is False

    def test_a_current_install_is_not_repeated(self, harness):
        harness.aiperf.parent.mkdir(parents=True)
        harness.aiperf.touch()
        harness.marker.write_text(
            json.dumps({"harness": harness.harness_ref, "inferencex": harness.inferencex_ref, "python": "3.11"})
        )
        with patch("production_test_framework.workload.agentx_workload.subprocess.run") as run:
            assert harness.install() is False
        run.assert_not_called()

    def test_a_marker_for_other_refs_is_stale(self, harness):
        harness.aiperf.parent.mkdir(parents=True)
        harness.aiperf.touch()
        harness.marker.write_text(json.dumps({"harness": "old", "inferencex": "old", "python": "3.11"}))
        assert harness.is_installed() is False


class TestAgentxWorkload:
    def test_command_is_aiperf_profile_with_generic_flags(self, workload):
        argv = workload.build_command()
        assert argv[:2] == [str(workload.harness.aiperf), "profile"]
        assert ["--scenario", "agentx"] == argv[2:4]
        assert "--unsafe-override" in argv
        assert "--ui-type" not in argv
        assert argv[argv.index("--output-artifact-dir") + 1] == str(workload.artifact_dir)

    def test_artifact_dir_cannot_be_overridden(self, harness, tmp_path):
        workload = AgentxWorkload(
            harness=harness, aiperf_options={"output_artifact_dir": "/elsewhere"}, run_dir=tmp_path / "run"
        )
        assert workload.aiperf_options["output_artifact_dir"] == str(tmp_path / "run" / "aiperf_artifacts")

    def test_parse_reads_aggregate_and_scores(self, workload):
        _write_aggregate(workload.artifact_dir)
        with _scoring_succeeds(workload) as run:
            result = workload.parse_output(CommandResult(0, "log", ""))

        assert result.scored == SCORED
        assert result.scoring_error == ""
        env = run.call_args.kwargs["env"]
        assert env["CONC"] == "8"
        assert env["TP"] == "2"
        assert env["KV_OFFLOADING"] == "none"
        assert workload.log_path.read_text() == "log"

    @pytest.mark.parametrize(
        ("logs", "files"),
        [
            (["one server"], ["server.log"]),
            (["prefill", "decode"], ["watchtower-0.out", "watchtower-1.out"]),
        ],
    )
    def test_server_logs_are_where_scoring_reads_them_before_scoring(self, harness, tmp_path, logs, files):
        run_dir = tmp_path / "run"
        workload = AgentxWorkload(
            harness=harness, aiperf_options={}, run_dir=run_dir, plots=False, server_log=lambda: logs
        )
        _write_aggregate(workload.artifact_dir)
        seen = []

        def score(argv, **kwargs):
            seen.extend((run_dir / name).read_text() for name in files)
            return subprocess.CompletedProcess(argv, 1, "", "no result")

        with patch("production_test_framework.workload.agentx_workload.subprocess.run", side_effect=score):
            workload.parse_output(CommandResult(0, "log", ""))

        assert seen == logs

    def test_scoring_failure_is_recorded_not_raised(self, workload):
        _write_aggregate(workload.artifact_dir)
        failed = subprocess.CompletedProcess([], 1, "", "SystemExit: Missing required environment variable")
        with patch("production_test_framework.workload.agentx_workload.subprocess.run", return_value=failed):
            result = workload.parse_output(CommandResult(0, "", ""))

        assert result.scored is None
        assert "Missing required environment variable" in result.scoring_error

    def test_missing_aggregate_raises(self, workload):
        with pytest.raises(RuntimeError, match=AGGREGATE_FILE):
            workload.parse_output(CommandResult(0, "", ""))

    def test_failed_exit_with_a_result_keeps_the_result(self, workload):
        _write_aggregate(workload.artifact_dir)
        failed = CommandResult(1, "", "request error rate exceeded")
        with (
            patch("production_test_framework.workload.command_workload.run_cancellable_command", return_value=failed),
            _scoring_succeeds(workload),
        ):
            result = workload._run_command_sync()

        assert isinstance(result, AgentxResult)
        assert "exit status 1" in result.exit_error

    def test_failed_exit_without_a_result_raises(self, workload):
        failed = CommandResult(1, "", "connection refused")
        with patch("production_test_framework.workload.command_workload.run_cancellable_command", return_value=failed):
            with pytest.raises(RuntimeError, match="connection refused"):
                workload._run_command_sync()

    def test_artifacts_found_one_level_down(self, tmp_path):
        _write_aggregate(tmp_path / "artifacts" / "run-1")
        assert find_artifacts(tmp_path / "artifacts") == tmp_path / "artifacts" / "run-1"


class TestAgentxResult:
    @staticmethod
    def _result(tmp_path, **kwargs):
        return AgentxResult(run_dir=tmp_path, artifact_dir=tmp_path, aggregate=AGGREGATE, **kwargs)

    def test_request_counts_and_error_rate(self, tmp_path):
        result = self._result(tmp_path)
        assert (result.successful_requests, result.error_requests, result.completed_requests) == (95, 5, 100)
        assert result.error_rate == pytest.approx(0.05)

    def test_metrics_leave_out_what_aiperf_did_not_fill(self, tmp_path):
        assert set(self._result(tmp_path).metrics) == {
            "request_count",
            "error_request_count",
            "time_to_first_token",
            "inter_token_latency",
        }

    def test_submission_valid_is_found_wherever_it_is_nested(self, tmp_path):
        assert self._result(tmp_path).submission_valid is False


@pytest.fixture
def report(tmp_path):
    return Reporter(tmp_path / "report.html", fmt=ReportFormat.HTML)


def _rendered(reporter):
    reporter.flush()
    return reporter.path.read_text(encoding="utf-8")


class TestReportAgentx:
    def test_configuration_shows_each_flag(self, report, workload):
        report.start_test("t.py::a")
        report_agentx.report_agentx_configuration(report, workload, [["profile", "two-gpu"]])
        html = _rendered(report)
        assert "Workload configuration -- AgentX" in html
        assert "--concurrency 8" in html
        assert workload.harness.harness_ref in html
        assert "two-gpu" in html

    def test_configuration_marks_options_queried_from_the_server(self, report, workload):
        report.start_test("t.py::a")
        report_agentx.report_agentx_configuration(report, workload, system_queried={"concurrency"})
        html = _rendered(report)
        assert "8 (system query)" in html
        assert "--concurrency 8" in html

    def test_result_tables_come_before_figures(self, report, workload, tmp_path):
        png = tmp_path / "ttft_over_time.png"
        png.write_bytes(b"\x89PNG fake")
        result = AgentxResult(run_dir=tmp_path, artifact_dir=tmp_path, aggregate=AGGREGATE, scored=SCORED, plots=[png])
        report.start_test("t.py::a")
        report_agentx.report_agentx_result(
            report, workload, WorkloadResult(0.0, 60.0, result, WorkloadStatus.COMPLETED)
        )
        html = _rendered(report)

        positions = [
            html.index(title)
            for title in (
                "Workload result -- AgentX",
                "AgentX latency and interactivity",
                "AgentX server metrics",
                "aiperf metrics",
                "aiperf errors",
                "Latency distribution -- AgentX",
                "aiperf plots -- AgentX",
            )
        ]
        assert positions == sorted(positions)
        assert "data:image/png;base64," in html
        assert html.count("Data: <a href='#s0-aiperf-metrics'>aiperf metrics</a>") == 1
        plots = html.index("aiperf plots -- AgentX")
        assert "Data:" not in html[plots : plots + 200]
        assert "server_metrics_export.json missing or empty" in html

    def test_a_failed_run_reports_its_output(self, report, workload):
        report.start_test("t.py::a")
        report_agentx.report_agentx_result(
            report, workload, WorkloadResult(0.0, 1.0, "aiperf failed\nconnection refused", WorkloadStatus.ERROR)
        )
        html = _rendered(report)
        assert "Workload output -- AgentX (error)" in html
        assert "connection refused" in html


class TestAgentxRows:
    def test_summary_rows_read_the_scored_aggregate(self, tmp_path):
        result = AgentxResult(run_dir=tmp_path, artifact_dir=tmp_path, aggregate=AGGREGATE, scored=SCORED)
        rows = {row[0]: row[1] for row in report_agentx.agentx_summary_rows(result)}
        assert rows["error rate"] == "5.00%"
        assert rows["throughput per GPU (input+output)"] == "20,000.0"
        assert rows["theoretical prefix cache hit rate"] == "90.00%"
        assert rows["submission valid"] == "false"

    def test_token_rows_end_with_the_in_out_ratio_of_the_means(self):
        """The vela run's figures: 32,013 tokens in and 567 out per request on average."""
        tokens = {
            "input": {"mean": 32013.1, "p50": 32197.5},
            "output_actual": {"mean": 567.4, "p50": 369},
            "output_expected": {"mean": 440.9},
        }
        rows = report_agentx.agentx_token_rows(tokens)
        assert [row[0] for row in rows] == ["input", "output_actual", "output_expected", "tokens in/out ratio"]
        assert rows[-1] == ["tokens in/out ratio", "ratio", "56:1", "-", "-", "-", "-", "-"]

    @pytest.mark.parametrize(
        ("tokens", "ratio"),
        [
            ({"input": {"mean": 100}, "output_actual": {"mean": 400}}, "0.25:1"),
            ({"input": {"mean": 100}, "output_actual": {"mean": 0}}, "-"),
            ({"input": {"mean": 100}}, "-"),
        ],
    )
    def test_token_ratio_below_one_or_without_output(self, tokens, ratio):
        assert report_agentx.agentx_token_rows(tokens)[-1][2] == ratio

    def test_summary_rows_survive_no_scoring(self, tmp_path):
        result = AgentxResult(run_dir=tmp_path, artifact_dir=tmp_path, aggregate=AGGREGATE)
        rows = {row[0]: row[1] for row in report_agentx.agentx_summary_rows(result)}
        assert rows["QPS mean"] == "-"

    def test_stat_rows_skip_empty_families(self):
        rows = report_agentx.agentx_stat_rows({"ttft": {"mean": 1.0}, "tpot": {}}, {"ttft": "s"})
        assert rows == [["ttft", "s", "1", "-", "-", "-", "-", "-"]]

    def test_flattened_rows_use_dotted_keys(self):
        rows = report_agentx.flattened_rows({"cache": {"hit": 0.5, "missing": None}, "adapter": "vllm"})
        assert rows == [["cache.hit", "0.5"], ["adapter", "vllm"]]

    def test_latency_figure_maps_aiperf_metrics(self):
        figure = report_agentx.agentx_latency_figure(AGGREGATE)
        assert "TTFT" in figure
        assert "ITL" in figure

    def test_unreadable_png_yields_no_figure(self, tmp_path):
        assert report_agentx.png_figure(tmp_path / "missing.png") == ""


class TestFormatForUnit:
    @pytest.mark.parametrize(
        ("value", "unit", "shown"),
        [
            (7.19, "tok/s", "7.2"),
            (330.3039, "tok/s/GPU", "330.3"),
            (0.125, "$", "0.13"),
            (7.25, "tok/s", "7.3"),
            (604952.1978, "tok/s/MW", "604,952.2"),
            (43.29, "tokens/sec/user", "43.3"),
            (0.29, "$", "0.29"),
            (3.7006, "$", "3.70"),
            (0.0699, "$/GPU/hr", "0.07"),
            (0.17, "$/M tokens", "0.17"),
        ],
    )
    def test_token_rates_keep_one_place_and_dollars_two(self, value, unit, shown):
        assert report_agentx.format_for_unit(value, unit) == shown

    def test_other_units_are_unchanged(self):
        assert report_agentx.format_for_unit(1.6530, "J/tok") == "1.653"
        assert report_agentx.format_for_unit(6994670.82, "tok/$") == "6,994,670.82"


class TestExitSummary:
    def test_picks_aiperf_error_line_without_its_source(self, tmp_path):
        log = (
            "19:43:35.459 NOTICE   Phase profiling complete\n"
            "19:43:37.559 ERROR    Profiling metric coverage below the required 95.0% (results.py:211)\n"
            "19:43:37.589 ERROR    Received fatal profile-results validation error\n"
        )
        result = AgentxResult(run_dir=tmp_path, artifact_dir=tmp_path, aggregate={}, exit_error=log)
        assert result.exit_summary == "Profiling metric coverage below the required 95.0%"

    def test_falls_back_to_the_first_line(self, tmp_path):
        result = AgentxResult(
            run_dir=tmp_path, artifact_dir=tmp_path, aggregate={}, exit_error="agentx command failed\nmore"
        )
        assert result.exit_summary == "agentx command failed"
