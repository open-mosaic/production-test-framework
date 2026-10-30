# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for the run report: grouping, failures, the header and the logo."""

import logging

import pytest

from production_test_framework.reporting.assets import Logo, load_logo
from production_test_framework.reporting.charts import MIN_BARS, latency_percentile_figure
from production_test_framework.reporting.environment import (
    benchmark_option_rows,
    command_output,
    display_path,
    git_rows,
    pool_summary,
    profile_host,
    profile_rows,
    run_rows,
    runner_rows,
    selection_rows,
)
from production_test_framework.reporting.formatting import ReportFormat
from production_test_framework.reporting.report import (
    DEFAULT_TITLE,
    UNCATEGORIZED,
    Reporter,
    failure_snippet,
    summary_line,
)
from production_test_framework.reporting.workload import (
    _token_ratio,
    report_workload_result,
    traffic_shape,
    workload_label,
)
from production_test_framework.workload.workload import WorkloadResult, WorkloadStatus

TRACEBACK = """self = <TestEndpoints object at 0x7f3a>

    def test_health(self, service):
>       assert resp.status_code == 200
E       assert 503 == 200
E        +  where 503 = <Response [503]>.status_code

tests/test_endpoints.py:88: AssertionError"""


@pytest.fixture
def report(tmp_path):
    """A reporter writing HTML to a temporary file, with three categories configured."""
    return Reporter(
        tmp_path / "report.html",
        fmt=ReportFormat.HTML,
        category_order=["lgtm", "api", "disruptive"],
    )


def rendered(reporter):
    reporter.flush()
    return reporter.path.read_text(encoding="utf-8")


class TestSummaryLine:
    """The one line of a traceback shown in the failure table."""

    def test_takes_the_first_e_line_not_the_last(self):
        """The assertion itself, not the trailing explanation of where a value came from."""
        assert summary_line(TRACEBACK) == "assert 503 == 200"

    def test_falls_back_to_the_exception_when_there_are_no_e_lines(self):
        text = 'File "x.py", line 3\n    raise RuntimeError("boom")\nRuntimeError: boom'
        assert summary_line(text) == "RuntimeError: boom"

    def test_empty_text_yields_empty_string(self):
        assert summary_line("   \n  ") == ""


class TestFailureSnippet:
    """Tracebacks are truncated from the front, keeping the error at the end."""

    def test_short_text_is_untouched(self):
        assert failure_snippet("boom", max_chars=100) == "boom"

    def test_long_text_keeps_the_tail(self):
        text = "A" * 50 + "THE ERROR"
        snippet = failure_snippet(text, max_chars=20)
        assert snippet.endswith("THE ERROR")
        assert "truncated" in snippet


class TestCategories:
    """Results are grouped into sections, ordered as the run configured them."""

    def test_sections_follow_the_configured_order_not_the_run_order(self, report):
        report.record_outcome("t.py::c", "passed", 1.0, category="disruptive")
        report.record_outcome("t.py::a", "passed", 1.0, category="lgtm")
        report.record_outcome("t.py::b", "passed", 1.0, category="api")

        html = rendered(report)
        assert html.index(">lgtm<") < html.index(">api<") < html.index(">disruptive<")

    def test_an_unconfigured_category_still_gets_a_section_after_the_configured_ones(self, report):
        report.record_outcome("t.py::a", "passed", 1.0, category="lgtm")
        report.record_outcome("t.py::b", "passed", 1.0, category="surprise")

        html = rendered(report)
        assert html.index(">lgtm<") < html.index(">surprise<")

    def test_the_catch_all_sorts_last_of_all(self, report):
        report.record_outcome("t.py::a", "passed", 1.0)
        report.record_outcome("t.py::b", "passed", 1.0, category="zzz_unconfigured")

        html = rendered(report)
        assert html.index(">zzz_unconfigured<") < html.index(f">{UNCATEGORIZED}<")

    def test_counts_are_reported_per_category(self, report):
        report.record_outcome("t.py::a", "passed", 1.0, category="lgtm")
        report.record_outcome("t.py::b", "failed", 1.0, TRACEBACK, "lgtm")

        assert "1 failed, 1 passed" in rendered(report)


class TestFailures:
    """Every failure is tabulated; only the first few carry a traceback."""

    def test_table_lists_every_failure_while_snippets_are_capped(self, tmp_path):
        reporter = Reporter(tmp_path / "r.html", max_failure_snippets=2)
        for index in range(5):
            reporter.record_outcome(f"t.py::test_{index}", "failed", 1.0, TRACEBACK, "lgtm")

        html = rendered(reporter)
        assert "Failures (5)" in html
        # Each failure contributes one table row; only two get a <pre> traceback.
        for index in range(5):
            assert f"test_{index}" in html
        assert html.count("pre class='failure'") == 2
        assert "3 further failure(s)" in html

    def test_an_error_counts_as_a_failure(self, report):
        report.record_outcome("t.py::a", "error", 1.0, TRACEBACK, "lgtm")
        assert "Failures (1)" in rendered(report)

    def test_no_failure_section_when_everything_passed(self, report):
        report.record_outcome("t.py::a", "passed", 1.0, category="lgtm")
        assert "Failures" not in rendered(report)

    def test_a_traceback_appears_once_not_in_the_details_as_well(self, report):
        report.start_test("t.py::a")
        report.table(["k", "v"], [["x", "1"]], title="Some detail")
        report.record_outcome("t.py::a", "failed", 1.0, TRACEBACK, "lgtm")

        assert rendered(report).count("pre class='failure'") == 1


class TestHeader:
    """Title and logo."""

    def test_default_title(self, tmp_path):
        assert DEFAULT_TITLE in rendered(Reporter(tmp_path / "r.html"))

    def test_title_is_configurable(self, tmp_path):
        assert "Nightly run" in rendered(Reporter(tmp_path / "r.html", title="Nightly run"))

    def test_no_logo_by_default(self, tmp_path):
        assert "class='logo'" not in rendered(Reporter(tmp_path / "r.html"))

    def test_a_logo_is_inlined_and_theme_aware(self, tmp_path):
        path = tmp_path / "logo.svg"
        path.write_text('<svg viewBox="0 0 10 10"><rect fill="currentColor"/></svg>')
        html = rendered(Reporter(tmp_path / "r.html", logo=load_logo(path)))
        # Inlined rather than linked, so the report survives being copied anywhere, and on
        # currentColor so the mark follows the theme.
        assert "<svg" in html
        assert "currentColor" in html
        assert "src=" not in html.split("</header>")[0]

    def test_an_empty_logo_renders_no_image(self, tmp_path):
        html = rendered(Reporter(tmp_path / "r.html", logo=Logo("")))
        assert "class='logo'" not in html

    def test_markdown_carries_the_title_but_no_logo(self, tmp_path):
        reporter = Reporter(tmp_path / "r.md", fmt=ReportFormat.MD, title="Nightly run")
        text = rendered(reporter)
        assert text.startswith("# Nightly run")
        assert "<svg" not in text


class TestLoadLogo:
    """A logo the report cannot read must never break the run."""

    def test_a_missing_file_degrades_to_no_logo(self, tmp_path):
        assert load_logo(tmp_path / "absent.svg").is_empty

    def test_an_unrecognised_type_degrades_to_no_logo(self, tmp_path):
        path = tmp_path / "logo.txt"
        path.write_text("not an image")
        assert load_logo(path).is_empty

    def test_a_raster_logo_is_embedded_as_a_data_uri(self, tmp_path):
        path = tmp_path / "logo.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
        assert "data:image/png;base64," in load_logo(path).markup

    def test_an_svg_loses_its_xml_prolog_so_it_can_be_inlined(self, tmp_path):
        path = tmp_path / "logo.svg"
        path.write_text('<?xml version="1.0"?><svg viewBox="0 0 10 10"><rect/></svg>')
        markup = load_logo(path).markup
        assert markup.startswith("<svg")
        assert "<?xml" not in markup

    def test_width_and_height_give_way_to_a_viewbox(self, tmp_path):
        """Height comes from CSS; an SVG with no viewBox could not scale at all."""
        path = tmp_path / "logo.svg"
        path.write_text('<svg width="200" height="50"><rect/></svg>')
        markup = load_logo(path).markup
        assert 'viewBox="0 0 200 50"' in markup
        assert "width=" not in markup.split(">")[0]


class TestTrafficShape:
    """The label that distinguishes one InferenceX run from another."""

    def test_a_finite_request_rate_is_fixed_traffic(self):
        assert traffic_shape({"request_rate": 8}) == "fixed traffic"

    def test_an_infinite_request_rate_is_not(self):
        assert traffic_shape({"request_rate": "inf"}) == "burst traffic"

    def test_concurrency_without_a_rate_is_fixed_concurrency(self):
        assert traffic_shape({"max_concurrency": 32}) == "fixed concurrency"

    def test_neither_is_burst(self):
        assert traffic_shape({"num_prompts": 100}) == "burst traffic"


class TestWorkloadLabel:
    """The label a report's workload sections are headed with."""

    class _Workload:
        workload_name = "Inferencex"
        benchmark_options = {"max_concurrency": 8}

    def test_an_explicit_shape_wins_over_what_the_flags_imply(self):
        """A fixed/agentic choice is not recoverable from the flags it produces."""
        assert workload_label(self._Workload(), "agentic workload") == "Inferencex, agentic workload"

    def test_without_one_the_shape_is_inferred(self):
        assert workload_label(self._Workload()) == "Inferencex, fixed concurrency"

    def test_a_workload_with_no_options_is_named_alone(self):
        class Bare:
            workload_name = "Nccl"

        assert workload_label(Bare()) == "Nccl"


class TestTokenRatio:
    """Prompt tokens per generated token, the row that says which way a run leans."""

    def test_prefill_heavy_run(self):
        assert _token_ratio(262144, 32768) == "8:1"

    def test_agentic_run(self):
        assert _token_ratio(294912, 2048) == "144:1"

    def test_it_rounds_to_a_whole_number(self):
        assert _token_ratio(1700, 200) == "8:1"  # 8.5 -> 8
        assert _token_ratio(1760, 200) == "9:1"  # 8.8 -> 9

    def test_below_one_to_one_it_keeps_its_decimals(self):
        """A decode-heavy run rounds to 0:1, which says nothing at all."""
        assert _token_ratio(1024, 4096) == "0.25:1"

    def test_large_ratios_are_grouped(self):
        assert _token_ratio(10_000_000, 1000) == "10,000:1"

    def test_nothing_generated_is_not_an_enormous_ratio(self):
        """A ratio against zero is unbounded; printing a big number would read as a measure."""
        assert _token_ratio(1024, 0) == "-"

    @pytest.mark.parametrize("pair", [(None, 100), (100, None), (None, None)])
    def test_a_missing_total_yields_a_dash(self, pair):
        assert _token_ratio(*pair) == "-"


class TestBenchmarkOptionRows:
    """Options are shown with the flag each becomes, so a run can be reproduced by hand."""

    def test_load_shaping_options_come_first(self):
        rows = benchmark_option_rows({"backend": "vllm", "num_prompts": 10})
        assert [row[0] for row in rows] == ["num_prompts", "backend"]

    def test_none_is_shown_as_an_omitted_flag(self):
        assert benchmark_option_rows({"host": None}) == [["host", "unset", "(omitted)"]]

    def test_a_true_boolean_becomes_a_bare_flag(self):
        assert benchmark_option_rows({"ignore_eos": True}) == [["ignore_eos", "true", "--ignore-eos"]]

    def test_a_false_boolean_emits_nothing(self):
        assert benchmark_option_rows({"ignore_eos": False}) == [["ignore_eos", "false", "(omitted)"]]

    def test_container_local_result_paths_are_left_out(self):
        """Fixed by the workload, identical every run, and not part of the load offered."""
        rows = benchmark_option_rows(
            {"num_prompts": 10, "result_dir": ".", "result_filename": "inferencex-result.json"}
        )
        assert [row[0] for row in rows] == ["num_prompts"]


class TestCommandOutput:
    """A probe that cannot run says why, and never raises."""

    def test_stdout_on_success(self):
        assert command_output(["sh", "-c", "echo hello"]) == ("hello", "")

    def test_a_failure_reports_the_first_line_of_stderr(self):
        assert command_output(["sh", "-c", "echo one >&2; echo two >&2; exit 3"]) == (None, "one")

    def test_a_silent_failure_reports_its_exit_status(self):
        assert command_output(["sh", "-c", "exit 3"]) == (None, "exit status 3")

    def test_no_output_is_not_a_result(self):
        assert command_output(["true"]) == (None, "command produced no output")

    def test_a_missing_binary(self):
        output, reason = command_output(["no-such-binary-xyz"])
        assert output is None
        assert "not found" in reason

    def test_a_timeout(self):
        output, reason = command_output(["sleep", "5"], timeout=0.2)
        assert output is None
        assert "timed out" in reason

    def test_a_file_that_cannot_be_executed(self, tmp_path):
        script = tmp_path / "not-executable"
        script.write_text("#!/bin/sh\necho hi\n")
        output, reason = command_output([str(script)])
        assert output is None
        assert reason


class TestRunnerRows:
    """The machine the tests ran on."""

    def test_python_is_a_version_not_an_interpreter_path(self):
        rows = dict((label, value) for label, value in runner_rows())
        assert rows["python"].count(".") == 2, rows["python"]
        assert "/" not in rows["python"]


class TestGitRows:
    """The checkout under test, and never an exception."""

    def test_a_directory_that_is_not_a_checkout_says_so(self, tmp_path):
        ((label, value),) = git_rows(tmp_path)
        assert label == "code under test"
        assert "not collected" in value


class TestProfileRows:
    """The workload profile, read as the plain mapping either suite's loader hands over."""

    PROFILE = {
        "name": "example",
        "path": "workload_profiles/example.yaml",
        "description": "Example nightly",
        "hardware": {"machines": 2, "gpus_per_machine": 4, "sku": "rtx-pro-2000-blackwell"},
        "serving": {"mode": "aggregated", "model": "Qwen/Qwen3-8B", "tensor_parallel": 4},
        "endpoint": {"host": "gpu-node", "port": 8000},
        "coverage": {"hosts": 2, "gpus": 8, "communicators": 1},
        "timeouts": {"workload": 600, "metrics_available": 120, "quiesce": 30},
        "expected_metrics": "nccl",
    }

    def rows(self, profile=None, **kwargs):
        return dict(profile_rows(profile or self.PROFILE, "http://prom:9090", "http://graf:3000", **kwargs))

    def test_total_gpus_is_derived(self):
        assert self.rows()["GPUs total"] == "8"

    def test_order_is_hardware_then_serving_then_addresses(self):
        labels = list(self.rows())
        assert labels.index("machines") < labels.index("model") < labels.index("endpoint") < labels.index("Prometheus")
        assert labels[-1] == "timeouts"

    def test_a_missing_field_omits_its_row(self):
        rows = self.rows({"name": "bare", "path": "bare.yaml"})
        assert "machines" not in rows
        assert "timeouts" not in rows
        assert rows["Grafana"] == "http://graf:3000"

    def test_endpoint_and_deployment_can_be_supplied(self):
        rows = self.rows(endpoint="gpu-node:9999  (overridden: VLLM_PORT=9999)", deployment="external")
        assert rows["endpoint"].startswith("gpu-node:9999")
        assert list(rows).index("deployment") == list(rows).index("endpoint") + 1

    def test_disaggregated_pools(self):
        profile = {
            **self.PROFILE,
            "serving": {
                "mode": "disaggregated",
                "model": "m",
                "prefill": {"nodes": 1, "tensor_parallel": 2},
                "decode": {"workers": 2, "tensor_parallel": 4, "expert_parallel": 4, "spans_nodes": True},
                "kv_transfer": "nixl",
            },
        }
        rows = self.rows(profile)
        assert rows["prefill pool"] == "1 node(s), TP=2, in-node"
        assert rows["decode pool"] == "2 worker(s), TP=4, EP=4, spans nodes"
        assert rows["KV transfer"] == "nixl"

    def test_pool_summary_omits_unset_fields(self):
        assert pool_summary({}) == "in-node"


class TestProfileHost:
    """Where to probe GPU versions."""

    def test_host(self):
        assert profile_host({"endpoint": {"host": "gpu-node", "port": 8000}}) == "gpu-node"

    def test_base_url(self):
        assert profile_host({"endpoint": {"base_url": "http://frontend.local:8000/v1"}}) == "frontend.local"

    def test_no_endpoint(self):
        assert profile_host({}) is None


class _Config:
    """Just the parts of a pytest config the rows read."""

    def __init__(self, rootpath, markexpr="", keyword=""):
        self.rootpath = rootpath
        self._options = {"markexpr": markexpr, "keyword": keyword}

    def getoption(self, name, default=None):
        return self._options.get(name, default)


class TestSelectionRows:
    """What the run selected on."""

    def test_nothing_selected(self, tmp_path):
        assert selection_rows(_Config(tmp_path)) == [["selection", "all collected tests"]]

    def test_markers_and_keywords(self, tmp_path):
        rows = selection_rows(_Config(tmp_path, markexpr="lgtm or k3s", keyword="grafana"))
        assert rows == [["selection", "-m lgtm or k3s and grafana"]]


class TestRunRows:
    """The rows every suite's report shares about the run."""

    def test_order(self, tmp_path):
        labels = [label for label, _ in run_rows(_Config(tmp_path))]
        assert labels[:4] == ["selection", "code under test", "test runner", "python"]

    def test_checkout_defaults_to_the_rootdir(self, tmp_path):
        rows = dict(run_rows(_Config(tmp_path)))
        assert "not collected" in rows["code under test"]

    def test_ci_rows_follow(self, tmp_path, monkeypatch):
        for name in ("CI_JOB_URL", "CI_COMMIT_REF_NAME", "CI_RUNNER_DESCRIPTION"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("CI_PIPELINE_URL", "https://gitlab.example/p/1")
        rows = run_rows(_Config(tmp_path))
        assert rows[-1] == ["CI pipeline", "https://gitlab.example/p/1"]


class TestStatusClasses:
    """A status column is coloured by its text, from pytest's outcomes and the table's own map."""

    def cell_class(self, report, value, **kwargs):
        report.table(["name", "status"], [["a", value]], status_column=1, **kwargs)
        html = rendered(report)
        return html.split(f">{value}</td>")[0].rsplit("<td", 1)[1]

    def test_pytest_outcomes_are_known(self, report):
        assert "bad" in self.cell_class(report, "failed")

    def test_an_unknown_status_is_left_plain(self, report):
        # Only the numeric alignment every right-aligned cell carries.
        assert self.cell_class(report, "wobbly") == " class='num'"

    def test_a_table_names_its_own_statuses(self, report):
        assert "warn" in self.cell_class(report, "wobbly", status_classes={"wobbly": "warn"})


class TestNoFileMode:
    """Without a path the tables are logged instead, and nothing is written."""

    def test_no_file_is_written(self, caplog):
        reporter = Reporter(None)
        assert reporter.writes_file is False
        with caplog.at_level(logging.INFO, logger="production_test_framework.reporting.report"):
            reporter.table(["a"], [["1"]], title="A table")
        assert "A table" in caplog.text


class TestReportWorkloadResult:
    """What a workload contributes to the report, on each way it can end."""

    class _Workload:
        workload_name = "Inferencex"
        benchmark_options = {"num_prompts": 256, "host": "localhost", "port": 8080}

    @staticmethod
    def _result(payload, status=WorkloadStatus.COMPLETED):
        return WorkloadResult(start_time=0.0, end_time=30.0, result=payload, status=status)

    def test_a_failed_run_still_reports_its_configuration(self, report):
        """The settings are the usual answer to why a run failed, so they must survive it."""
        report.start_test("t.py::a")
        report_workload_result(report, self._Workload(), self._result("boom", WorkloadStatus.ERROR))

        html = rendered(report)
        assert "Workload configuration -- Inferencex, burst traffic" in html
        assert "--num-prompts 256" in html

    def test_command_output_is_preformatted_not_a_paragraph(self, report):
        """stderr arrives as many lines; a note would collapse them into one."""
        report.start_test("t.py::a")
        output = "Request failed: ConnectionError\nFAIL: request failure rate 100.0%"
        report_workload_result(report, self._Workload(), self._result(output, WorkloadStatus.ERROR))

        html = rendered(report)
        assert "pre class='output'" in html
        assert "FAIL: request failure rate 100.0%" in html
        assert "Workload output -- Inferencex, burst traffic (error)" in html

    def test_long_output_keeps_its_tail(self, report):
        """The line that says what went wrong is the last one."""
        report.start_test("t.py::a")
        noise = "\n".join("Request failed" for _ in range(2000))
        report_workload_result(
            report, self._Workload(), self._result(f"{noise}\nFAIL: nothing served", WorkloadStatus.ERROR)
        )

        html = rendered(report)
        assert "FAIL: nothing served" in html
        assert "truncated" in html

    def test_a_stopped_run_with_no_output_says_so(self, report):
        report.start_test("t.py::a")
        report_workload_result(report, self._Workload(), self._result(None, WorkloadStatus.STOPPED))
        assert "no result (stopped, or no parseable output)" in rendered(report)


class TestLatencyFigure:
    """The latency small multiples: one panel per family, each on its own scale."""

    FULL = {
        f"{p}_{fam}": value
        for fam, base in (("ttft", 300.0), ("tpot", 30.0), ("itl", 31.0), ("e2el", 4300.0))
        for p, value in (
            ("mean", base),
            ("median", base),
            ("p90", base * 1.2),
            ("p99", base * 1.4),
            ("p99_9", base * 1.5),
        )
    }

    def test_one_panel_per_family(self):
        svg = latency_percentile_figure(self.FULL)
        for name in ("TTFT", "TPOT", "ITL", "E2EL"):
            assert f"<b>{name}</b>" in svg
        assert svg.count("<figure class='chart'>") == 4

    def test_a_family_with_one_value_is_not_charted(self):
        """A one-bar bar chart is a number; the result table already states it."""
        assert latency_percentile_figure({"mean_ttft": 12.0}) == ""

    def test_two_values_are_enough(self):
        svg = latency_percentile_figure({"mean_ttft": 10.0, "p99_ttft": 20.0})
        assert svg.count("<figure class='chart'>") == 1
        assert MIN_BARS == 2

    def test_nothing_to_draw_yields_nothing(self):
        assert latency_percentile_figure({}) == ""
        assert latency_percentile_figure({"request_throughput": 5.0}) == ""

    def test_std_is_not_a_percentile(self):
        """A standard deviation is not a point on the latency scale."""
        svg = latency_percentile_figure({"mean_ttft": 10.0, "p99_ttft": 20.0, "std_ttft": 3.0})
        assert ">std<" not in svg

    def test_each_family_is_scaled_independently(self):
        """TPOT and E2EL differ by two orders of magnitude; a shared axis would erase TPOT."""
        import re

        svg = latency_percentile_figure(self.FULL)
        # Each panel divides by its own maximum, so the longest bar of every panel reaches the
        # same x -- whatever that family's numbers are. (H stops a corner radius short of the
        # bar's end, which is why this is not the full span.)
        per_panel = [
            max(float(m) for m in re.findall(r"M46,[\d.]+ H([\d.]+)", panel))
            for panel in svg.split("<figure class='chart'>")[1:]
        ]
        assert len(per_panel) == 4
        assert len(set(per_panel)) == 1, per_panel

    def test_every_bar_carries_its_value_and_a_tooltip(self):
        svg = latency_percentile_figure({"mean_ttft": 10.0, "p99_ttft": 20.0})
        assert "10 ms" in svg and "20 ms" in svg
        assert "<title>TTFT mean: 10 ms</title>" in svg

    def test_the_figure_reaches_the_report_only_in_html(self, tmp_path):
        from production_test_framework.reporting.formatting import ReportFormat

        for fmt, ext, present in ((ReportFormat.HTML, "html", True), (ReportFormat.MD, "md", False)):
            reporter = Reporter(tmp_path / f"r.{ext}", fmt=fmt)
            reporter.start_test("t.py::a")
            reporter.figure(latency_percentile_figure(self.FULL), title="Latency distribution")
            body = rendered(reporter)
            assert ("<figure class='chart'>" in body) is present


class TestFigureDataLinks:
    """A figure links to the tables in its own section that hold its numbers."""

    def test_links_to_a_table_in_the_same_section(self, report):
        report.start_test("t.py::a")
        report.table(["m"], [["1"]], title="Raw numbers")
        report.figure("<svg></svg>", title="Chart", data_tables=("Raw numbers",))
        html = rendered(report)
        assert "<h4 id='s0-raw-numbers'>Raw numbers</h4>" in html
        assert "<a href='#s0-raw-numbers'>Raw numbers</a>" in html

    def test_no_link_when_the_table_is_absent(self, report):
        report.start_test("t.py::a")
        report.figure("<svg></svg>", title="Chart", data_tables=("Raw numbers",))
        assert "Data:" not in rendered(report)

    def test_same_titles_in_two_sections_get_distinct_anchors(self, report):
        for nodeid in ("t.py::a", "t.py::b"):
            report.start_test(nodeid)
            report.table(["m"], [["1"]], title="Raw numbers")
            report.figure("<svg></svg>", title="Chart", data_tables=("Raw numbers",))
        html = rendered(report)
        assert "href='#s0-raw-numbers'" in html
        assert "href='#s1-raw-numbers'" in html


class TestDisplayPath:
    """Paths in a shared report never carry the private prefix."""

    @pytest.fixture
    def checkout(self, tmp_path, monkeypatch):
        repo = tmp_path / "home" / "someone" / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "suite" / "build").mkdir(parents=True)
        return repo

    def test_relative_to_the_working_directory_inside_the_checkout(self, checkout, monkeypatch):
        monkeypatch.chdir(checkout / "suite")
        assert display_path(checkout / "suite" / "build" / "run") == "build/run"

    def test_repo_relative_when_outside_the_working_directory(self, checkout, monkeypatch):
        monkeypatch.chdir(checkout / "suite" / "build")
        assert display_path(checkout / "suite" / "x.yaml") == "repo/suite/x.yaml"

    def test_repo_relative_when_run_from_outside_the_checkout(self, checkout, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert display_path(checkout / "suite" / "build") == "repo/suite/build"
