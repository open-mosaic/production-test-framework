# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.
"""
Report an AgentX run: the options it ran with, every metric it produced, and its plots.
"""

import base64
import html
from collections.abc import Collection, Iterator, Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from ..agentx.gpu_economics import ECONOMICS_METRICS, SYSTEM_QUERY, X_AXIS_METRICS, EconomicsInputs, economics_metrics
from ..agentx.power import PowerSummary
from ..workload.agentx_workload import AgentxResult, AgentxWorkload
from ..workload.workload import WorkloadResult
from .charts import latency_percentile_figure, scatter_figure
from .environment import benchmark_option_rows, display_path
from .formatting import format_duration, format_number
from .workload import _token_ratio

__all__ = [
    "agentx_economics",
    "agentx_economics_input_rows",
    "agentx_latency_figure",
    "agentx_stat_rows",
    "agentx_token_rows",
    "agentx_summary_rows",
    "aiperf_metric_rows",
    "flattened_rows",
    "format_for_unit",
    "png_figure",
    "power_summary_rows",
    "report_agentx_configuration",
    "report_agentx_result",
]

STAT_COLUMNS = ("mean", "p50", "p75", "p90", "p95", "std")

AIPERF_METRICS_TITLE = "aiperf metrics"
ECONOMICS_TITLE = "AgentX economics"

AIPERF_STAT_COLUMNS = ("avg", "p50", "p90", "p95", "p99", "min", "max", "std")

#: InferenceX reports latencies in seconds and interactivity in tokens per second per user.
LATENCY_UNITS = {
    "ttft": "s",
    "e2el": "s",
    "itl": "s",
    "tpot": "s",
    "full_response_itl": "s",
    "intvty": "tok/s/user",
    "e2e_norm_intvty": "tok/s/user",
    "full_response_intvty": "tok/s/user",
}

#: aiperf metric -> the family name :func:`latency_percentile_figure` draws.
_FIGURE_FAMILIES = {"time_to_first_token": "ttft", "inter_token_latency": "itl", "request_latency": "e2el"}
_FIGURE_STATS = {"avg": "mean", "p50": "median", "p90": "p90", "p99": "p99"}


def _get(data: Mapping[str, Any] | None, *keys: str) -> Any:
    """``data[k1][k2]...``, or None when any level is missing."""
    for key in keys:
        if not isinstance(data, Mapping):
            return None
        data = data.get(key)
    return data


def _percent(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.2f}%"


def _unit_decimals(unit: str) -> int | None:
    """Decimal places this suite shows for *unit*: 1 for a token rate, 2 for a dollar amount."""
    if unit.startswith("$"):
        return 2
    if "tok/s" in unit or "tokens/sec" in unit:
        return 1
    return None


def format_for_unit(value: float, unit: str = "", significant: int | None = None) -> str:
    """*value* rounded, halves up, to its unit's decimal places, or as :func:`format_number` shows it."""
    places = _unit_decimals(unit)
    if places is None:
        return format_number(value, significant=significant)
    rounded = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{rounded:,.{places}f}"


def _axis_value(value: float, unit: str) -> str:
    return format_for_unit(value, unit, significant=2)


def _cell(value: Any, unit: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return format_for_unit(value, unit)
    if isinstance(value, list):
        return ", ".join(_cell(item) for item in value) or "-"
    return str(value)


def agentx_summary_rows(result: AgentxResult) -> list[list[str]]:
    """The headline measures of a run, as (measure, value, unit) rows."""
    scored = result.scored or {}
    requests = _get(scored, "request_metrics") or {}
    throughput = requests.get("throughput") or {}
    accounting = scored.get("request_accounting") or {}
    server = scored.get("server_metrics") or {}

    return [
        ["requests completed (aiperf)", format_number(result.completed_requests, ",d"), "requests"],
        ["requests successful (aiperf)", format_number(result.successful_requests, ",d"), "requests"],
        ["requests failed (aiperf)", format_number(result.error_requests, ",d"), "requests"],
        ["error rate", _percent(result.error_rate), ""],
        ["requests profiled (scored)", _cell(accounting.get("records_profiled")), "requests"],
        ["warmup requests dropped", _cell(accounting.get("records_warmup_dropped")), "requests"],
        ["profiling window", format_duration(throughput.get("duration_seconds")), ""],
        ["QPS mean", _cell(_get(requests, "qps", "mean")), "req/s"],
        ["QPS p95", _cell(_get(requests, "qps", "p95")), "req/s"],
        ["throughput (input+output)", _cell(_get(throughput, "total", "tokens_per_second"), "tok/s"), "tok/s"],
        ["throughput (input)", _cell(_get(throughput, "input", "tokens_per_second"), "tok/s"), "tok/s"],
        ["throughput (output)", _cell(_get(throughput, "output", "tokens_per_second"), "tok/s"), "tok/s"],
        [
            "throughput per GPU (input+output)",
            _cell(_get(throughput, "per_gpu", "total_tput_tps"), "tok/s/GPU"),
            "tok/s/GPU",
        ],
        [
            "throughput per GPU (output)",
            _cell(_get(throughput, "per_gpu", "output_tput_tps"), "tok/s/GPU"),
            "tok/s/GPU",
        ],
        ["TTFT mean", _cell(_get(requests, "latency", "ttft", "mean")), "s"],
        ["TTFT p95", _cell(_get(requests, "latency", "ttft", "p95")), "s"],
        ["interactivity mean", _cell(_get(requests, "latency", "intvty", "mean"), "tok/s/user"), "tok/s/user"],
        ["E2E latency mean", _cell(_get(requests, "latency", "e2el", "mean")), "s"],
        ["theoretical prefix cache hit rate", _percent(_get(requests, "cache", "theoretical_cache_hit_rate")), ""],
        ["GPU prefix cache hit rate (server)", _percent(_get(server, "cache", "gpu_cache_hit_rate")), ""],
        ["GPU KV cache usage (server)", _percent(_get(server, "kv_cache", "gpu_usage_pct")), ""],
        ["GPU KV cache capacity (server)", _cell(scored.get("kv_cache_pool_tokens")), "tokens"],
        ["GPUs scored", _cell(scored.get("num_gpus")), ""],
        ["submission valid", _cell(result.submission_valid), ""],
    ]


def agentx_stat_rows(families: Mapping[str, Any], units: Mapping[str, str] | None = None) -> list[list[str]]:
    """One row per distribution in *families*: its name, unit and :data:`STAT_COLUMNS`."""
    rows = []
    for name, stats in families.items():
        if not isinstance(stats, Mapping) or not stats:
            continue
        unit = (units or {}).get(name, "")
        rows.append([name, unit, *(_cell(stats.get(column), unit) for column in STAT_COLUMNS)])
    return rows


def agentx_token_rows(tokens: Mapping[str, Any]) -> list[list[str]]:
    """
    The tokens-per-request distributions, then the tokens in/out ratio as the InferenceX report
    gives it: mean input over mean actual output, which is total in over total out since every
    request counts in both. Only the mean column: a ratio of percentiles is no percentile.
    """
    rows = agentx_stat_rows(tokens, dict.fromkeys(tokens, "tokens"))
    ratio = _token_ratio(_get(tokens, "input", "mean"), _get(tokens, "output_actual", "mean"))
    return [*rows, ["tokens in/out ratio", "ratio", ratio, *("-" for _ in STAT_COLUMNS[1:])]]


def aiperf_metric_rows(metrics: Mapping[str, Mapping[str, Any]]) -> list[list[str]]:
    """Every aiperf metric with its unit and :data:`AIPERF_STAT_COLUMNS`."""
    return [
        [
            name,
            str(metric.get("unit", "")),
            *(_cell(metric.get(column), str(metric.get("unit", ""))) for column in AIPERF_STAT_COLUMNS),
        ]
        for name, metric in metrics.items()
    ]


def _flatten(data: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(data, Mapping):
        for key, value in data.items():
            yield from _flatten(value, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, data


def flattened_rows(data: Mapping[str, Any]) -> list[list[str]]:
    """A nested document as (dotted key, value) rows, leaving out what was not reported."""
    return [[key, _cell(value)] for key, value in _flatten(data) if value not in (None, {}, [])]


def agentx_latency_figure(aggregate: Mapping[str, Any]) -> str:
    """aiperf's TTFT, ITL and end-to-end latency percentiles, drawn by :func:`latency_percentile_figure`."""
    latency = {}
    for metric, family in _FIGURE_FAMILIES.items():
        for stat, label in _FIGURE_STATS.items():
            if (value := _get(aggregate, metric, stat)) is not None:
                latency[f"{label}_{family}"] = float(value)
    return latency_percentile_figure(latency)


def png_figure(path: Path, caption: str = "") -> str:
    """A PNG embedded in the report file itself, or "" when it cannot be read."""
    try:
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError:
        return ""
    caption = caption or Path(path).stem.replace("_", " ")
    return (
        "<figure class='chart'>"
        f"<figcaption class='above'>{html.escape(caption)}</figcaption>"
        f"<img src='data:image/png;base64,{encoded}' alt='{html.escape(caption)}' style='width:100%;height:auto'>"
        "</figure>"
    )


def report_agentx_configuration(
    reporter,
    workload: AgentxWorkload,
    extra_rows: list[list[str]] | None = None,
    system_queried: Collection[str] = (),
) -> None:
    """
    The options and harness a run uses, recorded before it starts so a failed run still has them.

    The values of the options in *system_queried*, probed from the server rather than
    configured, are marked as such.
    """
    options = workload.aiperf_options
    options["output_artifact_dir"] = display_path(Path(options["output_artifact_dir"]))
    rows = benchmark_option_rows(options)
    for row in rows:
        if row[0] in system_queried:
            row[1] = f"{row[1]} ({SYSTEM_QUERY})"
    reporter.table(
        ["option", "value", "aiperf flag"],
        rows,
        title="Workload configuration -- AgentX",
        left={2},
    )
    harness = workload.harness
    reporter.table(
        ["property", "value"],
        [
            ["agentx-harness commit", harness.harness_ref],
            ["InferenceX scoring commit", harness.inferencex_ref],
            *(extra_rows or []),
        ],
        title="AgentX harness",
        left={1},
    )


def _report_agentx_tables(reporter, result: AgentxResult, runtime: float) -> None:
    dataset = result.dataset
    reporter.table(
        ["measure", "value", "unit"],
        [
            *agentx_summary_rows(result),
            ["dataset", _cell(dataset.get("loader")), ""],
            ["aiperf version", _cell(result.aiperf_version), ""],
            ["wall time (incl. dataset load and warmup)", format_duration(runtime), ""],
        ],
        title="Workload result -- AgentX",
        left={2},
    )

    scored = result.scored or {}
    headers = ["distribution", "unit", *STAT_COLUMNS]
    if latency := _get(scored, "request_metrics", "latency"):
        reporter.table(
            headers, agentx_stat_rows(latency, LATENCY_UNITS), title="AgentX latency and interactivity", left={1}
        )
    if tokens := _get(scored, "request_metrics", "tokens"):
        reporter.table(headers, agentx_token_rows(tokens), title="AgentX tokens per request", left={1})
    if server := scored.get("server_metrics"):
        reporter.table(["metric", "value"], flattened_rows(server), title="AgentX server metrics", left={1})

    reporter.table(
        ["metric", "unit", *AIPERF_STAT_COLUMNS],
        aiperf_metric_rows(result.metrics),
        title=AIPERF_METRICS_TITLE,
        left={1},
    )
    if errors := result.aggregate.get("error_summary"):
        reporter.table(
            ["count", "code", "type", "message"],
            [
                [
                    _cell(entry.get("count")),
                    _cell(_get(entry, "error_details", "code")),
                    _cell(_get(entry, "error_details", "type")),
                    _cell(_get(entry, "error_details", "message")),
                ]
                for entry in errors
            ],
            title="aiperf errors",
            left={2, 3},
        )

    for warning in scored.get("warnings") or []:
        reporter.note(f"InferenceX scoring warning: {warning}")
    for problem in (result.exit_error, result.scoring_error, result.plot_error):
        if problem:
            reporter.output(problem, title="AgentX problem")


#: Short y-axis titles for the economics charts; the full name is in each panel's caption.
ECONOMICS_AXIS_TITLES = {
    "tpPerGpu": "Throughput/chip",
    "inputTputPerGpu": "Input tput/chip",
    "outputTputPerGpu": "Output tput/chip",
    "tpPerMw": "Throughput/MW",
    "inputTputPerMw": "Input tput/MW",
    "outputTputPerMw": "Output tput/MW",
    "tokenRevenuePerGpuHour": "Revenue/GPU-hr",
    "tokensPerDollar": "Tokens per $1",
    "outputTokensPerDollar": "Output tok per $1",
    "inputTokensPerDollar": "Input tok per $1",
    "cost": "Cost/M tokens",
    "costOutput": "Cost/M output tok",
    "costInput": "Cost/M input tok",
    "jTotal": "J/token",
    "jOutput": "J/output token",
    "jInput": "J/input token",
}

#: InferenceX's default percentile for the x-axis.
DEFAULT_PERCENTILE = "p90"


def agentx_economics(
    scored: Mapping[str, Any], inputs: EconomicsInputs, percentile: str = DEFAULT_PERCENTILE
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """``(y, x)``: the economics metrics of a scored AgentX run, and its x-axis values at *percentile*."""
    per_gpu = _get(scored, "request_metrics", "throughput", "per_gpu") or {}
    y = economics_metrics(
        per_gpu.get("total_tput_tps"), per_gpu.get("input_tput_tps"), per_gpu.get("output_tput_tps"), inputs
    )
    latency = _get(scored, "request_metrics", "latency") or {}
    x = {key: _get(latency, family, percentile) for key, _, _, family in X_AXIS_METRICS}
    return y, x


#: (field, label, unit) for each value the all-in power can be computed from, in formula order.
POWER_INPUT_ROWS = (
    ("tdp_w", "GPU TDP", "W/GPU"),
    ("host_watts_per_gpu", "host power share", "W/GPU"),
    ("server_watts", "server wall power", "W"),
    ("gpus_per_server", "GPUs sharing server power", ""),
    ("pue", "PUE", ""),
)


def _sourced(inputs: EconomicsInputs, field: str, value: Any, unit: str) -> str:
    """*value* as a cell, marked when it was probed from the system rather than configured."""
    text = _cell(value, unit)
    return f"{text} ({SYSTEM_QUERY})" if inputs.sources.get(field) == SYSTEM_QUERY else text


def agentx_economics_input_rows(inputs: EconomicsInputs) -> list[list[str]]:
    """
    The economics inputs as (input, value, unit, source) rows, including every value the
    all-in power was computed from.
    """
    not_set = "not configured"
    rows = [
        ["GPU defaults", inputs.gpu or "no match -- set economics.gpu", "", inputs.sources.get("gpu", "")],
        ["cost basis", inputs.cost_basis, "", inputs.sources.get("cost_basis", "")],
    ]
    for field, label, unit in POWER_INPUT_ROWS:
        if field in inputs.power_inputs:
            value = inputs.power_inputs[field]
            rows.append([label, _sourced(inputs, field, value, unit), unit, inputs.sources.get(field, "")])
    for field, label, unit in (
        ("all_in_watts_per_gpu", "all-in provisioned power", "W/GPU"),
        ("cost_per_gpu_hour", "TCO", "$/GPU/hr"),
        ("input_price_per_million", "input token price", "$/M tokens"),
        ("output_price_per_million", "output token price", "$/M tokens"),
    ):
        value = getattr(inputs, field)
        text = not_set if value is None else _sourced(inputs, field, value, unit)
        rows.append([label, text, unit, inputs.sources.get(field, "")])
    return rows


def power_summary_rows(power: PowerSummary) -> list[list[str]]:
    """What a power sampler measured during the run, as (measure, value, unit) rows."""
    not_measured = "not measured"
    no_server = power.server_unavailable or not_measured
    return [
        ["samples", _cell(power.samples), ""],
        ["GPUs sampled", _cell(power.gpu_count), ""],
        ["GPU power limit", _cell(power.gpu_limit_w), "W/GPU"],
        ["GPU power mean", _cell(power.gpu_avg_w) if power.gpu_avg_w is not None else not_measured, "W/GPU"],
        ["GPU power max", _cell(power.gpu_max_w) if power.gpu_max_w is not None else not_measured, "W/GPU"],
        ["server wall power mean", _cell(power.server_avg_w) if power.server_avg_w is not None else no_server, "W"],
        ["server wall power max", _cell(power.server_max_w) if power.server_max_w is not None else no_server, "W"],
    ]


def _report_agentx_economics_tables(reporter, scored: Mapping[str, Any], inputs: EconomicsInputs) -> None:
    reporter.table(
        ["input", "value", "unit", "source"],
        agentx_economics_input_rows(inputs),
        title="AgentX economics inputs",
        left={1, 2, 3},
    )
    y, x = agentx_economics(scored, inputs)
    rows = [[label, _cell(y[key], unit), unit] for key, label, unit in ECONOMICS_METRICS]
    rows += [[f"{label} ({DEFAULT_PERCENTILE})", _cell(x[key], unit), unit] for key, label, unit, _ in X_AXIS_METRICS]
    reporter.table(["metric", "value", "unit"], rows, title=ECONOMICS_TITLE, left={2})


def _report_agentx_economics_figures(
    reporter,
    scored: Mapping[str, Any],
    inputs: EconomicsInputs,
    history: Sequence[tuple[str, Mapping[str, Any]]],
    run_label: str,
) -> None:
    runs = [(label, *agentx_economics(other, inputs), False) for label, other in history]
    runs.append((run_label, *agentx_economics(scored, inputs), True))
    for key, label, unit in ECONOMICS_METRICS:
        panels = [
            (
                f"{label} vs {x_label}",
                x_unit,
                unit,
                [(x[x_key], y[key], run, current) for run, y, x, current in runs if None not in (x[x_key], y[key])],
                f"{x_label} ({x_unit})",
                f"{ECONOMICS_AXIS_TITLES[key]} ({unit})",
            )
            for x_key, x_label, x_unit, _ in X_AXIS_METRICS
        ]
        reporter.figure(
            scatter_figure(panels, ("this run", "earlier runs of this profile"), formatter=_axis_value),
            title=f"{label} -- AgentX",
            data_tables=(ECONOMICS_TITLE,),
        )


def _report_agentx_figures(reporter, result: AgentxResult) -> None:
    reporter.figure(
        agentx_latency_figure(result.aggregate),
        title="Latency distribution -- AgentX",
        data_tables=(AIPERF_METRICS_TITLE,),
    )
    if plots := [figure for path in result.plots if (figure := png_figure(path))]:
        full_width = "<div class='charts' style='grid-template-columns:1fr'>"
        reporter.figure(f"{full_width}{''.join(plots)}</div>", title="aiperf plots -- AgentX")


def report_agentx_result(
    reporter,
    workload: AgentxWorkload,
    result: WorkloadResult,
    economics: EconomicsInputs | None = None,
    history: Sequence[tuple[str, Mapping[str, Any]]] = (),
    power: PowerSummary | None = None,
    run_label: str = "this run",
) -> None:
    """
    Write *result* into *reporter*: the result tables, then the figures.

    With *economics*, adds the InferenceX cost and power metrics, charted against each x-axis
    with one point per *history* run, ``(label, scored aggregate)``, besides this one, labelled
    *run_label*.
    """
    match result.result:
        case AgentxResult() as agentx:
            reporter.note(f"AgentX artifacts: {display_path(agentx.run_dir)}")
            _report_agentx_tables(reporter, agentx, result.runtime)
            if power is not None:
                reporter.table(
                    ["measure", "value", "unit"], power_summary_rows(power), title="Measured power", left={2}
                )
                for error in power.errors:
                    reporter.note(f"Power reading unavailable: {error}")
            if economics is not None and agentx.scored:
                _report_agentx_economics_tables(reporter, agentx.scored, economics)
            _report_agentx_figures(reporter, agentx)
            if economics is not None and agentx.scored:
                _report_agentx_economics_figures(reporter, agentx.scored, economics, history, run_label)
        case str() as text:
            reporter.output(text, title=f"Workload output -- AgentX ({result.status.value})")
            reporter.note(f"Full aiperf log: {display_path(workload.log_path)}")
        case None:
            reporter.note("Workload result -- AgentX: no result (stopped before aiperf finished)")
