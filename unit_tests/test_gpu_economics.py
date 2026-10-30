# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Unit tests for the GPU economics defaults and derived metrics."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from production_test_framework.agentx import gpu_economics
from production_test_framework.agentx.gpu_economics import (
    ECONOMICS_METRICS,
    GPU_ECONOMICS,
    SYSTEM_QUERY,
    economics_metrics,
    estimate_all_in_watts,
    fetch_inferencex_gpus,
    lookup_gpu,
    resolve_economics,
    update_gpu_economics,
)
from production_test_framework.reporting import agentx as report_agentx
from production_test_framework.reporting.charts import scatter_figure
from production_test_framework.reporting.formatting import ReportFormat
from production_test_framework.reporting.report import Reporter


class TestLookupGpu:
    @pytest.mark.parametrize(
        ("name", "key"),
        [
            ("NVIDIA RTX PRO 2000 Blackwell", "rtx-pro-2000-blackwell"),
            ("rtx-pro-2000-blackwell", "rtx-pro-2000-blackwell"),
            ("NVIDIA H100 80GB HBM3", "h100"),
            ("AMD Instinct MI300X", "mi300x"),
            ("GB200 NVL72", "gb200"),
        ],
    )
    def test_names_from_smi_and_profiles_resolve(self, name, key):
        assert lookup_gpu(name)[0] == key

    def test_unknown_gpu_is_none(self):
        assert lookup_gpu("Voodoo 2") is None
        assert lookup_gpu(None) is None

    def test_every_entry_has_a_positive_tdp_and_all_in_above_it(self):
        for gpu in GPU_ECONOMICS.values():
            assert 0 < gpu.tdp_w < gpu.all_in_w


class TestAllInEstimate:
    def test_default_reproduces_the_published_h100_figure(self):
        assert estimate_all_in_watts(700) == pytest.approx(GPU_ECONOMICS["h100"].all_in_w, rel=0.01)


class TestResolveEconomics:
    def test_defaults_come_from_the_dictionary(self):
        inputs = resolve_economics("B200", "Qwen/Qwen3-8B")
        assert (inputs.gpu, inputs.cost_basis) == ("b200", "owning")
        assert inputs.all_in_watts_per_gpu == 1710
        assert inputs.cost_per_gpu_hour == 1.73
        assert inputs.input_price_per_million == 0.05
        assert inputs.sources["cost_per_gpu_hour"] == "InferenceX (owning)"

    def test_rental_basis_uses_the_rental_rate(self):
        assert resolve_economics("B200", None, {"cost_basis": "rental"}).cost_per_gpu_hour == 3.70

    def test_owning_falls_back_to_rental_when_unpublished(self):
        inputs = resolve_economics("rtx-pro-2000-blackwell", None)
        assert (inputs.cost_basis, inputs.cost_per_gpu_hour) == ("rental", 0.21)
        assert inputs.sources["cost_basis"] == "default owning, which RTX PRO 2000 Blackwell has no rate for"

    def test_overrides_win(self):
        inputs = resolve_economics(
            "b200",
            "Qwen/Qwen3-8B",
            {"all_in_watts_per_gpu": 1500, "cost_per_gpu_hour": 2.5, "output_price_per_million": 1.0},
        )
        assert (inputs.all_in_watts_per_gpu, inputs.cost_per_gpu_hour) == (1500, 2.5)
        assert (inputs.input_price_per_million, inputs.output_price_per_million) == (0.05, 1.0)
        assert inputs.sources["all_in_watts_per_gpu"] == "override"

    def test_host_share_and_pue_re_estimate_all_in(self):
        inputs = resolve_economics("rtx-pro-2000-blackwell", None, {"host_watts_per_gpu": 200, "pue": 1.2})
        assert inputs.all_in_watts_per_gpu == pytest.approx((70 + 200) * 1.2)

    def test_measured_server_power_is_shared_over_its_gpus(self):
        inputs = resolve_economics(
            "rtx-pro-2000-blackwell", None, {"server_watts": 600, "gpus_per_server": 2, "pue": 1.2}
        )
        assert inputs.all_in_watts_per_gpu == pytest.approx(600 * 1.2 / 2)
        assert inputs.sources["all_in_watts_per_gpu"] == "server_watts x pue / gpus_per_server"
        assert inputs.power_inputs == {"server_watts": 600, "gpus_per_server": 2, "pue": 1.2}

    def test_origins_name_where_each_override_came_from(self):
        inputs = resolve_economics(
            "rtx-pro-2000-blackwell",
            None,
            {"server_watts": 612, "gpus_per_server": 4, "cost_basis": "rental"},
            {"server_watts": SYSTEM_QUERY, "gpus_per_server": "machine profile", "cost_basis": "AgentX profile"},
        )
        assert inputs.sources["server_watts"] == SYSTEM_QUERY
        assert inputs.sources["gpus_per_server"] == "machine profile"
        assert inputs.sources["cost_basis"] == "AgentX profile"
        assert (inputs.power_inputs["pue"], inputs.sources["pue"]) == (1.3, "default")

    def test_an_estimated_default_records_what_it_was_estimated_from(self):
        inputs = resolve_economics("rtx-pro-2000-blackwell", None)
        assert inputs.all_in_watts_per_gpu == GPU_ECONOMICS["rtx-pro-2000-blackwell"].all_in_w
        assert inputs.power_inputs == {"tdp_w": 70, "host_watts_per_gpu": 350, "pue": 1.3}
        assert inputs.sources["host_watts_per_gpu"] == "default"

    def test_a_published_default_has_no_power_inputs(self):
        inputs = resolve_economics("b200", None)
        assert inputs.power_inputs == {}
        assert inputs.sources["cost_basis"] == "default"

    def test_measured_server_power_needs_a_gpu_count(self):
        with pytest.raises(ValueError, match="gpus_per_server"):
            resolve_economics("b200", None, {"server_watts": 600})

    def test_gpu_override_selects_the_defaults(self):
        assert resolve_economics("unknown card", None, {"gpu": "h200"}).gpu == "h200"

    def test_unknown_gpu_and_model_leave_inputs_unset(self):
        inputs = resolve_economics("unknown card", "some/model")
        assert inputs.all_in_watts_per_gpu is None
        assert inputs.cost_per_gpu_hour is None
        assert inputs.input_price_per_million is None

    def test_bad_cost_basis_is_rejected(self):
        with pytest.raises(ValueError, match="cost_basis"):
            resolve_economics("b200", None, {"cost_basis": "leased"})


#: A stand-in for inferencex.semianalysis.com: an H100 priced up, with its page, and a new chip
#: with neither a page nor an entry here.
INFERENCEX_PAGES = {
    "/api/v1/views/options": {
        "hardware": [
            {"key": "h100", "label": "H100", "vendor": "NVIDIA", "tdpW": 700, "costPerHour": {"h": 1.5, "r": 2.5}},
            {"key": "newchip", "label": "New Chip", "vendor": "Acme", "tdpW": 500, "costPerHour": {"h": 1, "r": 2}},
        ]
    },
    "/chips": "<a href='/chips/h100'>H100</a> <a href='/chips/h100-vs-b200'>vs</a>",
    "/chips/h100": "<dt>TDP per chip</dt><dd>700 W</dd><dt>All-in power per chip</dt><dd>1.42 kW</dd>",
}


def _inferencex_get(url, timeout):
    page = INFERENCEX_PAGES[url.removeprefix(gpu_economics.INFERENCEX_URL)]
    response = MagicMock()
    response.json.return_value = page
    response.text = page if isinstance(page, str) else ""
    return response


@pytest.fixture
def table(monkeypatch):
    """GPU_ECONOMICS as a copy, so an update does not outlive the test."""
    copy = dict(GPU_ECONOMICS)
    monkeypatch.setattr(gpu_economics, "GPU_ECONOMICS", copy)
    return copy


class TestInferencexRefresh:
    def test_fetch_reads_costs_and_the_chip_page_power(self):
        with patch.object(gpu_economics.requests, "get", side_effect=_inferencex_get):
            chips = fetch_inferencex_gpus()

        h100 = chips["h100"]
        assert (h100.tdp_w, h100.all_in_w, h100.owning_cost_per_hour, h100.rental_cost_per_hour) == (
            700,
            1420,
            1.5,
            2.5,
        )
        assert h100.power_source.startswith("InferenceX, fetched ")
        assert chips["newchip"].all_in_w == estimate_all_in_watts(500)
        assert chips["newchip"].power_source == gpu_economics.ESTIMATED_POWER

    def test_update_replaces_known_chips_and_adds_new_ones(self, table):
        with patch.object(gpu_economics.requests, "get", side_effect=_inferencex_get):
            updated = update_gpu_economics(fetch_inferencex_gpus())

        assert updated == ["h100", "newchip"]
        assert (table["h100"].owning_cost_per_hour, table["h100"].all_in_w) == (1.5, 1420)
        assert table["h100"].aliases == GPU_ECONOMICS["h100"].aliases
        assert lookup_gpu("New Chip")[0] == "newchip"
        assert resolve_economics("h100", None).sources["cost_per_gpu_hour"].startswith("InferenceX, fetched ")

    def test_update_keeps_an_all_in_figure_inferencex_does_not_publish(self, table):
        fetched = {"vr200": gpu_economics.GpuEconomics("Vera Rubin NVL72", "NVIDIA", 1800, 1.0, 4.0, 9.0, "x", "y")}
        fetched["vr200"] = gpu_economics.dataclasses.replace(
            fetched["vr200"], power_source=gpu_economics.ESTIMATED_POWER
        )
        update_gpu_economics(fetched)
        assert table["vr200"].all_in_w == GPU_ECONOMICS["vr200"].all_in_w
        assert table["vr200"].owning_cost_per_hour == 4.0

    def test_an_unreachable_site_raises(self):
        with (
            patch.object(gpu_economics.requests, "get", side_effect=requests.ConnectionError("down")),
            pytest.raises(requests.ConnectionError),
        ):
            fetch_inferencex_gpus()


class TestEconomicsMetrics:
    def test_matches_inferencex_published_b200_cost(self):
        """InferenceX: B200 at 6,505 tok/s/GPU costs $0.074/M total tokens at owning pricing."""
        metrics = economics_metrics(6505, None, None, resolve_economics("b200", None))
        assert metrics["cost"] == pytest.approx(0.074, abs=0.0005)

    def test_every_metric_from_known_inputs(self):
        inputs = resolve_economics(
            None,
            None,
            {
                "all_in_watts_per_gpu": 1000,
                "cost_per_gpu_hour": 2.0,
                "input_price_per_million": 1.0,
                "output_price_per_million": 4.0,
            },
        )
        metrics = economics_metrics(1000, 900, 100, inputs)

        assert set(metrics) == {key for key, _, _ in ECONOMICS_METRICS}
        assert metrics["tpPerMw"] == pytest.approx(1e6)
        assert metrics["tokenRevenuePerGpuHour"] == pytest.approx((900 * 1 + 100 * 4) * 3600 / 1e6)
        assert metrics["tokensPerDollar"] == pytest.approx(1000 * 3600 / 2)
        assert metrics["costOutput"] == pytest.approx(2 / (100 * 3600) * 1e6)
        assert metrics["jInput"] == pytest.approx(1000 / 900)

    def test_missing_inputs_give_none(self):
        metrics = economics_metrics(1000, 900, 100, resolve_economics(None, None))
        assert metrics["tpPerGpu"] == 1000
        assert metrics["cost"] is None
        assert metrics["jTotal"] is None
        assert metrics["tokenRevenuePerGpuHour"] is None


SCORED = {
    "request_metrics": {
        "throughput": {"per_gpu": {"total_tput_tps": 330.0, "input_tput_tps": 317.0, "output_tput_tps": 13.0}},
        "latency": {
            "ttft": {"p90": 3.3},
            "e2el": {"p90": 40.9},
            "intvty": {"p90": 21.0},
            "e2e_norm_intvty": {"p90": 18.0},
        },
    }
}


class TestAgentxEconomicsReport:
    def test_x_values_use_the_percentile(self):
        _, x = report_agentx.agentx_economics(SCORED, resolve_economics(None, None))
        assert x == {"e2e-normalized-interactivity": 18.0, "interactivity": 21.0, "e2e": 40.9, "ttft": 3.3}

    def test_input_rows_say_what_is_not_configured(self):
        rows = report_agentx.agentx_economics_input_rows(resolve_economics("unknown", None))
        assert ["TCO", "not configured", "$/GPU/hr", ""] in rows

    def test_input_rows_show_the_power_inputs_and_mark_system_queries(self):
        inputs = resolve_economics(
            "rtx-pro-2000-blackwell",
            None,
            {"server_watts": 612, "gpus_per_server": 4},
            {"server_watts": SYSTEM_QUERY, "gpus_per_server": "machine profile"},
        )
        rows = {row[0]: row for row in report_agentx.agentx_economics_input_rows(inputs)}
        assert rows["server wall power"][1].endswith(" (system query)")
        assert rows["server wall power"][3] == SYSTEM_QUERY
        assert rows["GPUs sharing server power"][1:] == ["4", "", "machine profile"]
        assert rows["PUE"][3] == "default"
        assert not rows["all-in provisioned power"][1].endswith("(system query)")
        assert "GPU TDP" not in rows

    def test_report_has_a_chart_per_metric(self, tmp_path):
        reporter = Reporter(tmp_path / "r.html", fmt=ReportFormat.HTML)
        reporter.start_test("t.py::a")
        inputs = resolve_economics("rtx-pro-2000-blackwell", "Qwen/Qwen3-8B")
        report_agentx._report_agentx_economics_tables(reporter, SCORED, inputs)
        report_agentx._report_agentx_economics_figures(reporter, SCORED, inputs, [("earlier", SCORED)], "baseline")
        reporter.flush()
        html = reporter.path.read_text()

        assert "AgentX economics inputs" in html
        for _, label, _ in ECONOMICS_METRICS:
            assert f"{label} -- AgentX" in html
        assert "Cost per Million Total Tokens vs TTFT" in html
        assert "<title>baseline: " in html
        assert "<title>earlier: " in html
        assert html.count("Data: <a href='#s0-agentx-economics'>AgentX economics</a>") == len(ECONOMICS_METRICS)


class TestScatterFigure:
    def test_empty_panels_are_dropped(self):
        assert scatter_figure([("a", "s", "tok", [])]) == ""

    def test_current_run_is_emphasised(self):
        figure = scatter_figure([("a", "s", "tok", [(1.0, 2.0, "old", False), (2.0, 3.0, "this run", True)])])
        assert figure.count("<circle") == 4
        assert "cb4" in figure

    def test_legend_names_both_kinds_of_point(self):
        figure = scatter_figure(
            [("a", "s", "tok", [(1.0, 2.0, "old", False), (2.0, 3.0, "this run", True)])], ("now", "before")
        )
        assert "aria-label='legend'" in figure
        assert ">now<" in figure and ">before<" in figure

    def test_long_y_title_puts_its_unit_on_a_second_line(self):
        figure = scatter_figure([("a", "s", "tok/s/MW", [(1.0, 2.0, "run", True)], "x", "Throughput/MW (tok/s/MW)")])
        assert "<tspan x='0' dy='0'>Throughput/MW</tspan><tspan x='0' dy='1.15em'>(tok/s/MW)</tspan>" in figure

    def test_short_y_title_stays_on_one_line(self):
        figure = scatter_figure([("a", "s", "J/tok", [(1.0, 2.0, "run", True)], "x", "J/token (J/tok)")])
        assert "<tspan x='0' dy='0'>J/token (J/tok)</tspan></text>" in figure

    def test_point_names_show_only_on_hover(self):
        figure = scatter_figure([("a", "s", "tok", [(1.0, 2.0, "old", False), (2.0, 3.0, "mine", True)])])
        assert ">mine</text>" not in figure
        assert "<title>mine: " in figure
        assert "<title>old: " in figure

    def test_legend_leaves_out_kinds_not_drawn(self):
        figure = scatter_figure([("a", "s", "tok", [(2.0, 3.0, "this run", True)])])
        assert ">this run<" in figure
        assert "other runs" not in figure
