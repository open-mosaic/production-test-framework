# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.
"""Report an MLPerf run: the options it ran with, LoadGen's result, the accuracy scores and the power."""

from typing import Any

from ..agentx.power import PowerSummary
from ..workload.mlperf_workload import (
    BENCHMARK,
    INFERENCE_REF,
    LOADGEN_VERSION,
    REFERENCE_DIR,
    MlperfResult,
    MlperfWorkload,
    accuracy_problems,
)
from ..workload.workload import WorkloadResult
from .environment import benchmark_option_rows, display_path
from .formatting import format_cell, format_number
from .power import report_power

__all__ = ["report_mlperf_configuration", "report_mlperf_result"]

#: (measure, LoadGen detail key, unit); nanosecond latencies are shown in milliseconds.
HEADLINE = (
    ("samples per second", "result_samples_per_second", "samples/s"),
    ("completed samples per second", "result_completed_samples_per_sec", "samples/s"),
    ("tokens per second", "result_tokens_per_second", "tok/s"),
    ("completed tokens per second", "result_completed_tokens_per_second", "tok/s"),
    ("scheduled samples per second", "result_scheduled_samples_per_sec", "samples/s"),
    ("TTFT mean", "result_first_token_mean_latency_ns", "ms"),
    ("TPOT mean", "result_time_to_output_token_mean", "ms"),
    ("TTFT p99", "result_first_token_99.00_percentile_latency_ns", "ms"),
    ("TPOT p99", "result_time_per_output_token_99.00_percentile_ns", "ms"),
    ("latency mean", "result_mean_latency_ns", "ms"),
    ("latency max", "result_max_latency_ns", "ms"),
    ("queries", "result_query_count", "queries"),
    ("min duration met", "result_min_duration_met", ""),
    ("min queries met", "result_min_queries_met", ""),
    ("performance constraints met", "result_perf_constraints_met", ""),
    ("early stopping met", "early_stopping_met", ""),
    ("LoadGen errors", "num_errors", "errors"),
)


def _cell(value: Any, unit: str = "") -> str:
    """LoadGen's nanosecond latencies in milliseconds, anything else as :func:`format_cell` shows it."""
    if unit == "ms" and isinstance(value, int | float) and not isinstance(value, bool):
        value = value / 1e6
    return format_cell(value, unit)


def report_mlperf_configuration(reporter, workload: MlperfWorkload, extra_rows: list[list[str]]) -> None:
    """The options and harness a run uses, recorded before it starts so a failed run still has them."""
    reporter.table(
        ["option", "value", "mlperf_sut_openai.py flag"],
        benchmark_option_rows(workload.sut_options),
        title="Workload configuration -- MLPerf",
        left={2},
    )
    harness = workload.harness
    reporter.table(
        ["property", "value"],
        [
            ["benchmark", BENCHMARK],
            ["scenario", workload.scenario],
            ["mode", workload.mode],
            ["mlcommons/inference commit", INFERENCE_REF],
            ["LoadGen", LOADGEN_VERSION],
            ["reference", REFERENCE_DIR],
            ["dataset", display_path(harness.dataset_path)],
            ["user.conf", "; ".join(workload.user_conf_lines) or "none -- mlperf.conf only"],
            *extra_rows,
        ],
        title="MLPerf harness",
        left={1},
    )


def _report_tables(reporter, result: MlperfResult, check_accuracy: bool) -> None:
    rows = [
        ["result", result.validity or "-", ""],
        ["invalid reason", result.invalid_reason if result.validity == "INVALID" else "-", ""],
        ["requests sent", _cell(result.requests), "requests"],
        ["requests failed", _cell(result.errors), "requests"],
        ["output tokens", _cell(result.sut_stats.get("output_tokens")), "tokens"],
        *[[name, _cell(result.detail.get(key), unit), unit] for name, key, unit in HEADLINE],
    ]
    reporter.table(["measure", "value", "unit"], rows, title="Workload result -- MLPerf", left={2})
    loadgen = [[key, _cell(value)] for key, value in sorted(result.detail.items()) if key.startswith("result_")]
    if loadgen:
        reporter.table(["LoadGen key", "value"], loadgen, title="LoadGen results", left={1})
    for message in result.sut_stats.get("error_messages") or []:
        reporter.note(f"Request failed: {message}")
    if result.mode == "accuracy":
        rows = [[key, format_number(value)] for key, value in result.accuracy.items()]
        reporter.table(["score", "value"], rows, title="MLPerf accuracy (evaluation.py)", left={1})
        if result.accuracy_error:
            reporter.output(result.accuracy_error, title="MLPerf accuracy problem")
        elif check_accuracy:
            for problem in accuracy_problems(result.accuracy):
                reporter.note(f"Below MLCommons' accuracy target: {problem}")


def report_mlperf_result(
    reporter,
    workload: MlperfWorkload,
    result: WorkloadResult,
    power: PowerSummary | None,
    check_accuracy: bool,
) -> None:
    """Write *result* into *reporter*: LoadGen's result, the accuracy scores, the power, and LoadGen's summary."""
    match result.result:
        case MlperfResult() as mlperf:
            reporter.note(f"MLPerf artifacts: {display_path(mlperf.run_dir)}")
            _report_tables(reporter, mlperf, check_accuracy)
            report_power(reporter, power)
            reporter.output(mlperf.summary, title="LoadGen summary (mlperf_log_summary.txt)")
        case str() as text:
            reporter.output(text, title=f"Workload output -- MLPerf ({result.status.value})")
            reporter.note(f"Full log: {display_path(workload.log_path)}")
        case None:
            reporter.note("Workload result -- MLPerf: no result (stopped before LoadGen finished)")
