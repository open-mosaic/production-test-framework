# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.
"""
One result report per run, in HTML or Markdown.

A suite turns the report on from its root ``conftest.py``::

    pytest_plugins = ["production_test_framework.reporting.report_plugin"]

and a run writes one with ``--report-file build/report.html``. Tests take the ``reporter``
fixture and add tables to it; everything else -- the header, the environment, the per-category
results and the failure summary -- is assembled from what pytest already knows.

The submodules are layered so a caller only imports what it needs:

* :mod:`~.formatting` -- numbers, tables and the status vocabulary. Knows nothing of pytest.
* :mod:`~.report` -- the :class:`Reporter`, which assembles and writes the report.
* :mod:`~.report_plugin` -- the pytest plugin: the report's options, the ``reporter``
  fixture, and the hooks that feed pytest's results into the :class:`Reporter`.
* :mod:`~.environment` -- rows describing the runner, the checkout, the GPUs and the profile.
* :mod:`~.workload` -- the configuration and result tables for a finished workload.
* :mod:`~.agentx` -- the configuration, result tables and plots of an AgentX run.
* :mod:`~.assets` -- the header logo, embedded into the report file itself.
"""

from .agentx import report_agentx_configuration, report_agentx_result
from .assets import Logo, load_logo
from .environment import (
    benchmark_option_rows,
    ci_rows,
    command_output,
    display_path,
    git_rows,
    gpu_rows,
    profile_host,
    profile_rows,
    run_rows,
    runner_rows,
    selection_rows,
)
from .formatting import (
    COMPACT_DECIMALS,
    SIGNIFICANT_DECIMALS,
    ReportFormat,
    Table,
    format_delta,
    format_duration,
    format_number,
    format_value,
    render_table,
    render_table_html,
    render_table_markdown,
)
from .report import DEFAULT_TITLE, Reporter
from .report_plugin import ReportPlugin
from .workload import report_workload_result, traffic_shape, workload_label

__all__ = [
    "COMPACT_DECIMALS",
    "DEFAULT_TITLE",
    "SIGNIFICANT_DECIMALS",
    "Logo",
    "ReportFormat",
    "ReportPlugin",
    "Reporter",
    "Table",
    "benchmark_option_rows",
    "ci_rows",
    "command_output",
    "display_path",
    "format_delta",
    "format_duration",
    "format_number",
    "format_value",
    "git_rows",
    "gpu_rows",
    "load_logo",
    "profile_host",
    "profile_rows",
    "render_table",
    "render_table_html",
    "render_table_markdown",
    "report_agentx_configuration",
    "report_agentx_result",
    "report_workload_result",
    "run_rows",
    "runner_rows",
    "selection_rows",
    "traffic_shape",
    "workload_label",
]
