# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.
"""
Report the power a :class:`~production_test_framework.agentx.power.PowerSampler` measured during a run.
"""

from ..agentx.power import PowerSummary
from .formatting import format_cell

__all__ = ["power_summary_rows", "report_power"]


def power_summary_rows(power: PowerSummary) -> list[list[str]]:
    """What a power sampler measured during the run, as (measure, value, unit) rows."""
    not_measured = "not measured"
    no_server = power.server_unavailable or not_measured
    return [
        ["samples", format_cell(power.samples), ""],
        ["GPUs sampled", format_cell(power.gpu_count), ""],
        ["GPU power limit", format_cell(power.gpu_limit_w), "W/GPU"],
        ["GPU power mean", format_cell(power.gpu_avg_w) if power.gpu_avg_w is not None else not_measured, "W/GPU"],
        ["GPU power max", format_cell(power.gpu_max_w) if power.gpu_max_w is not None else not_measured, "W/GPU"],
        [
            "server wall power mean",
            format_cell(power.server_avg_w) if power.server_avg_w is not None else no_server,
            "W",
        ],
        [
            "server wall power max",
            format_cell(power.server_max_w) if power.server_max_w is not None else no_server,
            "W",
        ],
    ]


def report_power(reporter, power: PowerSummary | None) -> None:
    """The "Measured power" table, and why any reading was unavailable; nothing without *power*."""
    if power is None:
        return
    reporter.table(["measure", "value", "unit"], power_summary_rows(power), title="Measured power", left={2})
    for error in power.errors:
        reporter.note(f"Power reading unavailable: {error}")
