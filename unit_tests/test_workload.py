# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for workload base and concrete workload classes."""

import json
import shlex
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from production_test_framework.docker import (
    CONTAINER_LABEL,
    docker_argv,
    force_remove_container,
    label_args,
    list_containers_by_label,
    remove_containers_by_label,
    unique_container_name,
)
from production_test_framework.ssh import CommandResult
from production_test_framework.vllm import DEFAULT_MODEL, InferenceResult
from production_test_framework.workload.command_workload import CommandWorkload
from production_test_framework.workload.docker_mixin import DockerContainerMixin
from production_test_framework.workload.inferencex_workload import (
    DEFAULT_BENCHMARK_OPTIONS,
    JSON_RESULT_INDICATOR,
    BenchmarkResultMissing,
    InferencexWorkload,
    benchmark_option_argv,
    parse_benchmark_result_json,
)
from production_test_framework.workload.nccl_workload import (
    NcclTest,
    NcclWorkload,
    parse_nccl_output,
)
from production_test_framework.workload.prompt_workload import BACKEND_TYPE, PromptWorkload
from production_test_framework.workload.workload import Workload, WorkloadStatus


@pytest.fixture
def mock_inferencex_run():
    """
    Stand in for the docker run subprocess, returning a successful benchmark result.

    The stdout has to be a parseable run -- indicator and JSON -- because parse_output now
    raises on anything else, which would fail these lifecycle tests for the wrong reason.
    """
    with patch(
        "production_test_framework.workload.command_workload.run_cancellable_command",
    ) as m:
        m.return_value = CommandResult(returncode=0, stdout=BENCHMARK_SERVING_OUTPUT, stderr="")
        yield m


@pytest.fixture(autouse=True)
def no_real_docker_cleanup():
    """Stop container cleanup from invoking the real docker CLI.

    _cleanup_after_run runs after every command and again on stop, so without this the unit
    tests would shell out to docker and depend on whether a daemon happens to be running.

    One patch target covers every containerised workload now that cleanup lives on
    DockerContainerMixin rather than being repeated per workload.
    """
    with patch("production_test_framework.workload.docker_mixin.force_remove_container") as remove:
        yield remove


NCCL_ALL_REDUCE_OUTPUT = """\
# nThread 1 nGpus 8 minBytes 8 maxBytes 268435456 step: 2(factor) warmup iters: 5 iters: 20 validation: 1
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    23.45    0.00    0.00      0    22.11    0.00    0.00      0
   268435456      67108864     float     sum      -1  1234.50  217.45  380.54      0  1230.10  218.22  381.89      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 190.715
#
"""

NCCL_ALL_GATHER_OUTPUT = """\
#       size         count      type     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
   134217728       2097152     float   500.10  268.38  234.83    N/A   498.00  269.51  235.82    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 235.325
"""

# A real run's JSON, as `--save-result` writes it, preceded by the summary block the script
# still prints and the indicator the workload echoes. Note "request_goodput:" carries a trailing
# colon in the key upstream; that is not a typo in this fixture.
BENCHMARK_RESULT_JSON = json.dumps(
    {
        "date": "20260828-132824",
        "backend": "vllm",
        "model_id": "Qwen/Qwen3-32B",
        "tokenizer_id": "Qwen/Qwen3-32B",
        "best_of": 1,
        "num_prompts": 512,
        "request_rate": "inf",
        "burstiness": 1.0,
        "max_concurrency": 64,
        "duration": 34.77475159900007,
        "completed": 512,
        "total_input_tokens": 262144,
        "total_output_tokens": 65536,
        "request_throughput": 14.723325874590067,
        "request_goodput:": None,
        "output_throughput": 1884.5857119475286,
        "total_token_throughput": 9422.928559737642,
        "mean_ttft_ms": 341.84747221679635,
        "median_ttft_ms": 340.8126045000017,
        "std_ttft_ms": 63.713206116146715,
        "p90_ttft_ms": 429.54680829999467,
        "p99_ttft_ms": 475.53700237995713,
        "p99.9_ttft_ms": 480.6516839620159,
        "mean_tpot_ms": 31.38726279315329,
        "p99_tpot_ms": 32.370740385826686,
        "mean_itl_ms": 31.387265713751834,
        "p99_itl_ms": 45.6792856800723,
        "mean_e2el_ms": 4328.029846947265,
        "p99_e2el_ms": 4456.193843490014,
    }
)

BENCHMARK_SERVING_OUTPUT = f"""\
Starting main benchmark run...
Traffic request rate: inf
============ Serving Benchmark Result ============
Successful requests:                     512
==================================================
{JSON_RESULT_INDICATOR}
{BENCHMARK_RESULT_JSON}
"""


class TestWorkloadStatus:
    """Tests for WorkloadStatus enum."""

    def test_member_values(self):
        assert WorkloadStatus.RUNNING.value == "running"
        assert WorkloadStatus.STOPPED.value == "stopped"
        assert WorkloadStatus.COMPLETED.value == "completed"
        assert WorkloadStatus.ERROR.value == "error"


class TestWorkload:
    """Tests for abstract Workload base class."""

    def test_cannot_instantiate_abstract_workload(self):
        with pytest.raises(TypeError, match="abstract"):
            Workload()

    def test_wait_for_completion_returns_true_when_predicate_succeeds(self):
        class CompletingWorkload(Workload):
            def __init__(self):
                super().__init__()

            def start(self):
                self._workload_status = WorkloadStatus.RUNNING

                def finish():
                    self._workload_status = WorkloadStatus.COMPLETED

                self.submit_background(finish)

            def stop(self):
                self._workload_status = WorkloadStatus.STOPPED

            def get_result(self) -> str:
                return "done"

        with patch(
            "production_test_framework.workload.workload.wait_for",
            return_value=True,
        ) as mock_wait:
            wl = CompletingWorkload()
            assert wl.wait_for_completion(timeout=10.0, poll_interval=1.0) is True

        mock_wait.assert_called_once()
        args, kwargs = mock_wait.call_args
        assert kwargs == {}
        pred, timeout_arg, poll_arg = args
        assert timeout_arg == 10.0
        assert poll_arg == 1.0
        wl._workload_status = WorkloadStatus.COMPLETED
        assert pred() is True

    def test_wait_for_completion_returns_false_when_wait_times_out(self):
        class NeverCompletingWorkload(Workload):
            def __init__(self):
                super().__init__()
                self._workload_status = WorkloadStatus.RUNNING

            def start(self):
                pass

            def stop(self):
                pass

            def get_result(self) -> str:
                return ""

        with patch(
            "production_test_framework.workload.workload.wait_for",
            return_value=False,
        ) as mock_wait:
            wl = NeverCompletingWorkload()
            assert wl.wait_for_completion(timeout=5.0, poll_interval=1.0) is False

        mock_wait.assert_called_once()

    def test_status_property_reads_workload_status(self):
        class DummyWorkload(Workload):
            def start(self):
                pass

            def stop(self):
                pass

            def get_result(self) -> str:
                return ""

        w = DummyWorkload()
        assert w.status == WorkloadStatus.STOPPED
        w._workload_status = WorkloadStatus.RUNNING
        assert w.status == WorkloadStatus.RUNNING

    def test_submit_background_runs_callable(self):
        class RunnableWorkload(Workload):
            def __init__(self):
                super().__init__()
                self.seen = []

            def start(self):
                pass

            def stop(self):
                pass

            def get_result(self) -> str:
                return ""

            def capture(self, x):
                self.seen.append(x)

        w = RunnableWorkload()
        try:
            fut = w.submit_background(w.capture, 42)
            assert fut.result(timeout=5.0) is None
            assert w.seen == [42]
        finally:
            w.shutdown_executor(wait=True)


class TestInferencexWorkload:
    """Tests for Inferencex workload."""

    @pytest.fixture
    def mock_inferencex_run(self):
        with patch(
            "production_test_framework.workload.command_workload.run_cancellable_command",
        ) as m:
            # Parseable output -- indicator and JSON. parse_output raises on anything else, which
            # would fail these lifecycle tests for a reason they are not about.
            m.return_value = CommandResult(
                returncode=0,
                stdout=BENCHMARK_SERVING_OUTPUT,
                stderr="",
            )
            yield m

    def test_is_workload_subclass(self):
        assert issubclass(InferencexWorkload, Workload)

    def test_can_instantiate(self):
        w = InferencexWorkload()
        assert isinstance(w, Workload)

    def test_initial_status_is_stopped(self):
        w = InferencexWorkload()
        assert w.status == WorkloadStatus.STOPPED

    def test_start_transitions_to_running(self, mock_inferencex_run):
        w = InferencexWorkload()
        w.start()
        assert w.status in (WorkloadStatus.RUNNING, WorkloadStatus.COMPLETED)
        w.shutdown_executor(wait=True, cancel_futures=True)

    def test_stop_returns_to_stopped(self, mock_inferencex_run):
        w = InferencexWorkload()
        w.start()
        w.stop()
        assert w.status == WorkloadStatus.STOPPED
        _args, kwargs = mock_inferencex_run.call_args
        assert kwargs["cancel_event"].is_set()
        w.shutdown_executor(wait=True)

    def test_get_result_after_completion(self, mock_inferencex_run):
        w = InferencexWorkload()
        w.start()
        fut = w._completion_fut
        assert fut is not None
        fut.result(timeout=10.0)
        assert w.status == WorkloadStatus.COMPLETED
        assert w.get_result().result.raw_output == BENCHMARK_SERVING_OUTPUT
        assert w.get_result().status == WorkloadStatus.COMPLETED
        assert w.get_result().start_time is not None
        assert w.get_result().end_time is not None
        assert w.get_result().runtime is not None
        w.shutdown_executor(wait=True)

    def test_docker_exec_argv_includes_container_host_port(self, mock_inferencex_run):
        w = InferencexWorkload(
            container_name="mycontainer",
            benchmark_options={"host": "vllm.svc", "port": 9090},
        )
        w.start()
        w._completion_fut.result(timeout=10.0)
        cmd = mock_inferencex_run.call_args[0][0]
        assert cmd[:2] == ["docker", "run"]
        inner = benchmark_argv(cmd)
        assert inner[inner.index("--host") + 1] == "vllm.svc"
        assert inner[inner.index("--port") + 1] == "9090"
        assert "--base-url" not in cmd
        w.shutdown_executor(wait=True)

    def test_stop_while_running_sets_cancel_on_mock(self, mock_inferencex_run):
        def run_until_cancel(cmd, *, timeout, cancel_event, **kwargs):
            for _ in range(500):
                if cancel_event.is_set():
                    return CommandResult(returncode=-1, stdout="", stderr="cancelled")
                time.sleep(0.01)
            return CommandResult(returncode=0, stdout="done", stderr="")

        mock_inferencex_run.side_effect = run_until_cancel
        w = InferencexWorkload()
        w.start()
        time.sleep(0.05)
        w.stop()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if w.status == WorkloadStatus.STOPPED and w.get_result().result is None:
                break
            time.sleep(0.02)
        assert w.status == WorkloadStatus.STOPPED
        assert w.get_result().result is None
        assert w.get_result().status == WorkloadStatus.STOPPED
        assert w.get_result().start_time is not None
        assert w.get_result().end_time is not None
        assert w.get_result().runtime is not None
        w.shutdown_executor(wait=True)

    def test_second_start_raises_when_already_running(self, mock_inferencex_run):
        block = threading.Event()

        def slow_run(*_args, **_kwargs):
            block.wait(timeout=60.0)
            # Parseable, so the run this test unblocks at the end finishes cleanly rather than
            # raising over output this test does not care about.
            return CommandResult(returncode=0, stdout=BENCHMARK_SERVING_OUTPUT, stderr="")

        mock_inferencex_run.side_effect = slow_run
        w = InferencexWorkload()
        w.start()
        assert w.status == WorkloadStatus.RUNNING
        with pytest.raises(
            RuntimeError,
            match="Inferencex workload already running",
        ):
            w.start()
        block.set()
        w._completion_fut.result(timeout=10.0)
        w.shutdown_executor(wait=True)


def benchmark_argv(cmd: list[str]) -> list[str]:
    """
    The benchmark's own argv, unwrapped from the ``sh -c`` script the workload builds.

    The flags are no longer separate argv elements -- they live inside a shell script so the
    JSON result can be catted onto stdout -- so a test that wants to reason about them has to
    unwrap it. Splits at the ``&&`` that separates the benchmark from the result-echoing tail.
    """
    return shlex.split(cmd[-1].split("&&")[0])


def expected_inferencex_command(name: str) -> list[str]:
    """The full argv a default InferencexWorkload emits, for the container called *name*.

    Frozen on purpose: consumers construct this workload with nothing but a timeout, so drift
    here changes what they run. The container name and its label are the only parts that vary
    between instances -- they are generated per run so a container left behind by an earlier
    run cannot block the next one, and so cleanup has something to remove.

    The benchmark itself is wrapped in `sh -c` so the JSON written by --save-result can be
    echoed onto stdout: that is the only channel out of a --rm container.
    """
    return [
        "docker",
        "run",
        "--rm",
        "-t",
        "--network",
        "host",
        "--name",
        name,
        "--label",
        f"{CONTAINER_LABEL}={name}",
        "openmosaic/inferencex:latest",
        "sh",
        "-c",
        "python3 /workspace/InferenceX/utils/bench_serving/benchmark_serving.py "
        "--host localhost --port 8080 --model Qwen/Qwen3-8B --backend vllm "
        "--dataset-name random --save-result --result-dir . "
        "--result-filename inferencex-result.json "
        f"&& echo {JSON_RESULT_INDICATOR} && cat ./inferencex-result.json",
    ]


class TestInferencexCommand:
    """Tests for the InferenceX argv construction."""

    def test_default_command_matches_expected_argv(self):
        workload = InferencexWorkload(docker_exec_timeout=1200)
        assert workload.build_command() == expected_inferencex_command(workload.container_name)

    def test_empty_options_emit_no_extra_flags(self):
        workload = InferencexWorkload(benchmark_options={})
        assert workload.build_command() == expected_inferencex_command(workload.container_name)

    def test_options_layer_over_the_defaults(self):
        cmd = benchmark_argv(InferencexWorkload(benchmark_options={"model": "other/model"}).build_command())
        assert cmd[cmd.index("--model") + 1] == "other/model"
        assert cmd.count("--model") == 1
        # Untouched defaults survive.
        assert cmd[cmd.index("--host") + 1] == "localhost"

    def test_none_drops_a_default(self):
        # This is how a disagg target replaces host/port with a single frontend URL.
        cmd = benchmark_argv(
            InferencexWorkload(
                benchmark_options={"base_url": "http://frontend.svc:8000", "host": None, "port": None}
            ).build_command()
        )
        assert cmd[cmd.index("--base-url") + 1] == "http://frontend.svc:8000"
        assert "--host" not in cmd
        assert "--port" not in cmd

    def test_benchmark_options_property_reports_effective_options(self):
        w = InferencexWorkload(benchmark_options={"num_prompts": 64})
        assert w.benchmark_options["num_prompts"] == 64
        assert w.benchmark_options["model"] == "Qwen/Qwen3-8B"
        # A copy, so mutating it cannot change what the workload will run.
        w.benchmark_options["num_prompts"] = 1
        assert w.benchmark_options["num_prompts"] == 64

    def test_defaults_constant_is_not_mutated_by_construction(self):
        before = dict(DEFAULT_BENCHMARK_OPTIONS)
        InferencexWorkload(benchmark_options={"model": "other/model", "num_prompts": 8})
        assert dict(DEFAULT_BENCHMARK_OPTIONS) == before

    def test_container_name_defaults_to_a_unique_name(self):
        # Not a fixed name: a container an earlier run left behind would make this one fail
        # with "name already in use". Not anonymous either, or nothing could remove it.
        first, second = InferencexWorkload(), InferencexWorkload()
        assert first.container_name != second.container_name
        assert first.container_name.startswith("inferencex-")
        assert first.build_command()[first.build_command().index("--name") + 1] == first.container_name

    def test_explicit_container_name_is_used_verbatim(self):
        workload = InferencexWorkload(container_name="mine")
        assert workload.container_name == "mine"
        assert workload.build_command()[workload.build_command().index("--name") + 1] == "mine"

    def test_container_is_labelled_for_orphan_sweeps(self):
        # Lets a caller find containers this framework started without knowing their names:
        # docker ps -aq --filter label=production-test-framework.workload
        workload = InferencexWorkload()
        cmd = workload.build_command()
        assert cmd[cmd.index("--label") + 1] == f"{CONTAINER_LABEL}={workload.container_name}"

    def test_env_and_docker_extra_args_precede_the_image(self):
        cmd = InferencexWorkload(
            env={"HF_TOKEN": "secret"},
            docker_extra_args=("-v", "/tmp/out:/out"),
        ).build_command()
        image_index = cmd.index("openmosaic/inferencex:latest")
        assert cmd[image_index - 4 : image_index] == [
            "-e",
            "HF_TOKEN=secret",
            "-v",
            "/tmp/out:/out",
        ]

    def test_extra_args_come_last_so_they_can_override(self):
        cmd = benchmark_argv(
            InferencexWorkload(
                benchmark_options={"num_prompts": 64},
                benchmark_extra_args=("--num-prompts", "128"),
            ).build_command()
        )
        assert cmd[-2:] == ["--num-prompts", "128"]


class TestBenchmarkOptionArgv:
    """Tests for the generic option -> flag conversion."""

    def test_scalars_become_kebab_case_flags(self):
        assert benchmark_option_argv({"num_prompts": 64, "random_input_len": 512}) == [
            "--num-prompts",
            "64",
            "--random-input-len",
            "512",
        ]

    def test_true_emits_a_bare_flag(self):
        assert benchmark_option_argv({"ignore_eos": True}) == ["--ignore-eos"]

    def test_none_and_false_emit_nothing(self):
        assert benchmark_option_argv({"seed": None, "disable_tqdm": False}) == []

    def test_sequence_emits_repeated_values(self):
        assert benchmark_option_argv({"goodput": ["ttft:200", "tpot:50"]}) == [
            "--goodput",
            "ttft:200",
            "tpot:50",
        ]

    def test_mapping_emits_key_value_pairs(self):
        assert benchmark_option_argv({"metadata": {"tp": 8, "sku": "rtx6000pro"}}) == [
            "--metadata",
            "tp=8",
            "sku=rtx6000pro",
        ]

    def test_string_value_is_not_treated_as_a_sequence(self):
        assert benchmark_option_argv({"endpoint": "/v1/chat/completions"}) == [
            "--endpoint",
            "/v1/chat/completions",
        ]

    def test_zero_is_emitted_rather_than_skipped(self):
        # 0 is falsy but meaningful (--seed 0, --random-prefix-len 0); only None/False are skipped.
        assert benchmark_option_argv({"seed": 0}) == ["--seed", "0"]

    def test_explicit_flag_key_passes_through(self):
        # An escape hatch for any flag whose name does not round-trip through snake_case.
        assert benchmark_option_argv({"--some-new-flag": "v"}) == ["--some-new-flag", "v"]

    def test_unknown_flag_is_passed_through_untouched(self):
        # The whole point: an option added upstream needs no change here.
        assert benchmark_option_argv({"invented_upstream_flag": 3}) == [
            "--invented-upstream-flag",
            "3",
        ]


class TestBenchmarkResultJsonParsing:
    """Tests for reading the --save-result JSON out of a run's stdout."""

    def test_parses_summary_fields(self):
        r = parse_benchmark_result_json(BENCHMARK_SERVING_OUTPUT)
        # Several of these are spelled differently in the JSON than the property that reads
        # them -- completed, duration, total_output_tokens, output_throughput -- so this is the
        # regression guard on that mapping.
        assert r.successful_requests == 512
        assert r.duration_seconds == 34.77475159900007
        assert r.total_input_tokens == 262144
        assert r.total_generated_tokens == 65536
        assert r.request_throughput == 14.723325874590067
        assert r.output_token_throughput == 1884.5857119475286
        assert r.total_token_throughput == 9422.928559737642
        assert r.raw_output == BENCHMARK_SERVING_OUTPUT
        assert r.passed is True

    def test_parses_percentile_latencies(self):
        latency = parse_benchmark_result_json(BENCHMARK_SERVING_OUTPUT).latency_ms
        assert latency["mean_ttft"] == 341.84747221679635
        assert latency["median_ttft"] == 340.8126045000017
        assert latency["p99_ttft"] == 475.53700237995713
        # A fractional percentile keeps its precision in the key rather than colliding with p99.
        assert latency["p99_9_ttft"] == 480.6516839620159
        assert latency["mean_tpot"] == 31.38726279315329

    def test_carries_families_the_printed_block_never_had(self):
        latency = parse_benchmark_result_json(BENCHMARK_SERVING_OUTPUT).latency_ms
        assert latency["std_ttft"] == 63.713206116146715
        assert latency["mean_itl"] == 31.387265713751834
        assert latency["p99_e2el"] == 4456.193843490014

    def test_metrics_dict_exposes_normalised_json_keys(self):
        m = parse_benchmark_result_json(BENCHMARK_SERVING_OUTPUT).metrics
        assert m["completed"] == 512
        assert m["duration"] == 34.77475159900007
        assert m["total_token_throughput"] == 9422.928559737642
        assert m["p99_9_ttft_ms"] == 480.6516839620159
        assert m["max_concurrency"] == 64

    def test_unknown_field_is_captured_generically(self):
        # A field added upstream lands in .metrics with no change to the parser.
        data = json.loads(BENCHMARK_RESULT_JSON)
        data["Invented Upstream Metric (widgets)"] = 12.5
        r = parse_benchmark_result_json(f"{JSON_RESULT_INDICATOR}\n{json.dumps(data)}")
        assert r.metrics["invented_upstream_metric_widgets"] == 12.5
        assert r.total_input_tokens == 262144

    def test_non_numeric_fields_are_not_metrics(self):
        m = parse_benchmark_result_json(BENCHMARK_SERVING_OUTPUT).metrics
        for key in ("date", "backend", "model_id", "tokenizer_id", "request_rate"):
            assert key not in m
        # ...but they remain visible for debugging.
        assert "Qwen/Qwen3-32B" in parse_benchmark_result_json(BENCHMARK_SERVING_OUTPUT).raw_output

    def test_goodput_absent_when_not_requested(self):
        # Null in the JSON, under a key that carries a trailing colon upstream.
        assert parse_benchmark_result_json(BENCHMARK_SERVING_OUTPUT).request_goodput is None

    def test_goodput_read_through_the_trailing_colon_key(self):
        data = json.loads(BENCHMARK_RESULT_JSON)
        data["request_goodput:"] = 12.25
        r = parse_benchmark_result_json(f"{JSON_RESULT_INDICATOR}\n{json.dumps(data)}")
        assert r.request_goodput == 12.25

    def test_summary_block_above_the_indicator_is_ignored(self):
        # The printed table is still in stdout and must not be mistaken for the result.
        r = parse_benchmark_result_json(BENCHMARK_SERVING_OUTPUT)
        assert r.successful_requests == 512
        assert "successful_requests" not in r.metrics

    def test_missing_indicator_raises(self):
        with pytest.raises(BenchmarkResultMissing, match=JSON_RESULT_INDICATOR):
            parse_benchmark_result_json("Traceback (most recent call last):\n  boom\n")

    def test_malformed_json_raises(self):
        with pytest.raises(BenchmarkResultMissing, match="not valid JSON"):
            parse_benchmark_result_json(f"{JSON_RESULT_INDICATOR}\n{{not json at all")

    def test_non_object_json_raises(self):
        with pytest.raises(BenchmarkResultMissing, match="expected a JSON object"):
            parse_benchmark_result_json(f"{JSON_RESULT_INDICATOR}\n[1, 2, 3]")

    def test_empty_output_raises(self):
        with pytest.raises(BenchmarkResultMissing):
            parse_benchmark_result_json("")

    def test_zero_completed_requests_is_not_passed(self):
        data = json.loads(BENCHMARK_RESULT_JSON)
        data["completed"] = 0
        r = parse_benchmark_result_json(f"{JSON_RESULT_INDICATOR}\n{json.dumps(data)}")
        assert r.successful_requests == 0
        assert r.passed is False


class TestNcclOutputParsing:
    """Tests for the nccl-tests output parser."""

    def test_parses_samples_and_summary(self):
        result = parse_nccl_output(NCCL_ALL_REDUCE_OUTPUT, test="all_reduce_perf")
        assert result.test == "all_reduce_perf"
        assert result.avg_bus_bandwidth_gbps == 190.715
        assert result.out_of_bounds_errors == 0
        # two data rows x (out-of-place, in-place)
        assert len(result.samples) == 4
        largest = [s for s in result.samples if s.size_bytes == 268435456]
        assert {s.in_place for s in largest} == {True, False}
        out_of_place = next(s for s in largest if not s.in_place)
        assert out_of_place.count == 67108864
        assert out_of_place.dtype == "float"
        assert out_of_place.time_us == 1234.50
        assert out_of_place.algbw_gbps == 217.45
        assert out_of_place.busbw_gbps == 380.54
        assert out_of_place.wrong == 0

    def test_max_busbw_and_passed(self):
        result = parse_nccl_output(NCCL_ALL_REDUCE_OUTPUT)
        assert result.max_busbw_gbps == 381.89
        assert result.passed is True

    def test_parses_rows_without_redop_and_root_columns(self):
        result = parse_nccl_output(NCCL_ALL_GATHER_OUTPUT, test="all_gather_perf")
        assert len(result.samples) == 2
        assert result.avg_bus_bandwidth_gbps == 235.325
        assert result.max_busbw_gbps == 235.82
        # validation disabled -> "N/A" is not an error
        assert all(s.wrong is None for s in result.samples)
        assert result.passed is True

    def test_out_of_bounds_errors_fail_the_run(self):
        output = NCCL_ALL_REDUCE_OUTPUT.replace("# Out of bounds values : 0 OK", "# Out of bounds values : 2 FAILED")
        result = parse_nccl_output(output)
        assert result.out_of_bounds_errors == 2
        assert result.passed is False

    def test_empty_output_has_no_samples_and_does_not_pass(self):
        result = parse_nccl_output("")
        assert result.samples == []
        assert result.avg_bus_bandwidth_gbps is None
        assert result.passed is False


class TestNcclWorkload:
    """Tests for NCCL test workload."""

    @pytest.fixture
    def mock_nccl_run(self):
        with patch(
            "production_test_framework.workload.command_workload.run_cancellable_command",
        ) as m:
            m.return_value = CommandResult(
                returncode=0,
                stdout=NCCL_ALL_REDUCE_OUTPUT,
                stderr="",
            )
            yield m

    def test_is_workload_subclass(self):
        assert issubclass(NcclWorkload, Workload)

    def test_initial_status_is_stopped(self):
        w = NcclWorkload()
        assert w.status == WorkloadStatus.STOPPED
        assert w.get_result().result is None

    def test_default_command_runs_binary_in_container(self):
        w = NcclWorkload()
        cmd = w.build_command()
        assert cmd[:2] == ["docker", "run"]
        assert "--gpus" in cmd and "all" in cmd
        assert "openmosaic/mosaic-nccl-tests:latest" in cmd
        assert "/workspace/bin/all_reduce_perf" in cmd
        assert "mpirun" not in cmd
        # single-node run drives every GPU from one process
        assert cmd[cmd.index("-g") + 1] == "8"

    def test_containerised_run_requires_an_image(self):
        with pytest.raises(ValueError, match="requires image_name"):
            NcclWorkload(use_docker=True, image_name=None)

        w = NcclWorkload(image_name="registry.local/nccl-tests:v2.13")
        assert "registry.local/nccl-tests:v2.13" in w.build_command()

    def test_runs_binary_directly_without_docker(self):
        w = NcclWorkload(use_docker=False, image_name=None)
        cmd = w.build_command()
        assert cmd[0] == "/workspace/bin/all_reduce_perf"
        assert "docker" not in cmd

    def test_gpus_defaults_to_all(self):
        w = NcclWorkload()
        cmd = w.build_command()
        assert cmd[cmd.index("--gpus") + 1] == "all"

    def test_gpu_selection_is_quoted_for_dockers_csv_parser(self):
        # Bare "device=2,3" is split on the comma and rejected by dockerd with
        # "cannot set both Count and DeviceIDs on device request".
        w = NcclWorkload(gpus="device=2,3", gpus_per_host=2)
        assert w.gpus == '"device=2,3"'
        cmd = w.build_command()
        assert cmd[cmd.index("--gpus") + 1] == '"device=2,3"'
        # GPUs are renumbered from 0 in the container, so -g counts them
        assert cmd[cmd.index("-g") + 1] == "2"

    def test_single_gpu_selection_needs_no_quoting(self):
        assert NcclWorkload(gpus="device=2").gpus == "device=2"
        assert NcclWorkload(gpus="2").gpus == "2"

    def test_already_quoted_gpu_selection_is_left_alone(self):
        assert NcclWorkload(gpus='"device=0,1"').gpus == '"device=0,1"'

    def test_network_defaults_to_bridge_so_the_image_finds_eth0(self):
        # The image sets NCCL_SOCKET_IFNAME=eth0; --network host has no eth0 and NCCL's
        # bootstrap aborts with "no socket interface found".
        w = NcclWorkload()
        assert w.docker_network == "bridge"
        cmd = w.build_command()
        assert cmd[cmd.index("--network") + 1] == "bridge"

    def test_network_defaults_to_host_under_mpirun(self):
        w = NcclWorkload(hosts=("gpu01", "gpu02"))
        assert w.docker_network == "host"
        cmd = w.build_command()
        assert cmd[cmd.index("--network") + 1] == "host"

    def test_network_can_be_overridden(self):
        w = NcclWorkload(docker_network="my-net")
        assert w.docker_network == "my-net"
        assert w.build_command()[w.build_command().index("--network") + 1] == "my-net"

    def test_test_selection_and_sizes(self):
        w = NcclWorkload(
            test=NcclTest.ALL_GATHER,
            binary_dir="/usr/local/nccl-tests/build/",
            min_bytes="1M",
            max_bytes="4G",
            step_factor=4,
            iters=50,
            warmup_iters=10,
            check=False,
            test_extra_args=("-z", "1"),
        )
        # Read the flags from the binary onwards: `docker run` takes its own -e (env), which
        # would otherwise shadow the test binary's -e (max bytes).
        full = w.build_command()
        assert "/usr/local/nccl-tests/build/all_gather_perf" in full
        cmd = full[full.index("/usr/local/nccl-tests/build/all_gather_perf") :]
        assert cmd[cmd.index("-b") + 1] == "1M"
        assert cmd[cmd.index("-e") + 1] == "4G"
        assert cmd[cmd.index("-f") + 1] == "4"
        assert cmd[cmd.index("-n") + 1] == "50"
        assert cmd[cmd.index("-w") + 1] == "10"
        assert cmd[cmd.index("-c") + 1] == "0"
        assert cmd[-2:] == ["-z", "1"]

    def test_hosts_switch_the_run_to_mpirun_with_one_gpu_per_rank(self):
        w = NcclWorkload(
            hosts=("gpu01", "gpu02"),
            gpus_per_host=4,
            env={"NCCL_DEBUG": "INFO"},
            mpi_extra_args=("--mca", "btl", "tcp,self"),
        )
        assert w.use_mpi is True
        assert w.num_processes == 8
        cmd = w.build_command()
        # mpirun is the launcher inside the container, so it follows the image name
        assert cmd[cmd.index("openmosaic/mosaic-nccl-tests:latest") + 1] == "mpirun"
        assert cmd[cmd.index("-np") + 1] == "8"
        assert cmd[cmd.index("-H") + 1] == "gpu01:4,gpu02:4"
        assert cmd[cmd.index("-x") + 1] == "NCCL_DEBUG=INFO"
        assert "--mca" in cmd
        assert cmd[cmd.index("-g") + 1] == "1"

    def test_single_host_still_launches_under_mpirun(self):
        w = NcclWorkload(hosts=("gpu01",), gpus_per_host=8, use_docker=False, image_name=None)
        assert w.use_mpi is True
        cmd = w.build_command()
        assert cmd[0] == "mpirun"
        assert cmd[cmd.index("-np") + 1] == "8"
        assert cmd[cmd.index("-H") + 1] == "gpu01:8"
        assert cmd[cmd.index("-g") + 1] == "1"

    def test_no_hosts_means_no_mpirun(self):
        w = NcclWorkload(gpus_per_host=8)
        assert w.use_mpi is False
        assert w.hosts == ()
        assert "mpirun" not in w.build_command()
        # one process drives all 8 GPUs, so the process count is 1 -- not 0, not 8
        assert w.num_processes == 1

    def test_num_processes_override(self):
        w = NcclWorkload(hosts=("gpu01",), gpus_per_host=8, num_processes=2)
        assert w.num_processes == 2
        assert w.build_command()[w.build_command().index("-np") + 1] == "2"

    def test_env_is_passed_to_container_and_mpirun(self):
        w = NcclWorkload(
            hosts=("gpu01",),
            env={"NCCL_IB_DISABLE": "0"},
            use_docker=True,
            image_name="registry.local/nccl-tests:v2.13",
        )
        cmd = w.build_command()
        assert cmd[cmd.index("-e") + 1] == "NCCL_IB_DISABLE=0"
        assert cmd[cmd.index("-x") + 1] == "NCCL_IB_DISABLE=0"

    def test_get_result_after_completion_is_parsed(self, mock_nccl_run):
        w = NcclWorkload()
        w.start()
        w._completion_fut.result(timeout=10.0)
        assert w.status == WorkloadStatus.COMPLETED
        result = w.get_result()
        assert result.status == WorkloadStatus.COMPLETED
        assert result.result.test == "all_reduce_perf"
        assert result.result.avg_bus_bandwidth_gbps == 190.715
        assert result.result.passed is True
        assert result.runtime is not None
        w.shutdown_executor(wait=True)

    def test_parses_table_from_stderr_when_stdout_empty(self, mock_nccl_run):
        mock_nccl_run.return_value = CommandResult(
            returncode=0,
            stdout="",
            stderr=NCCL_ALL_REDUCE_OUTPUT,
        )
        w = NcclWorkload()
        w.start()
        w._completion_fut.result(timeout=10.0)
        assert w.get_result().result.avg_bus_bandwidth_gbps == 190.715
        w.shutdown_executor(wait=True)

    def test_failed_run_sets_error_status(self, mock_nccl_run):
        mock_nccl_run.return_value = CommandResult(
            returncode=1,
            stdout="",
            stderr="NCCL failure: unhandled system error",
        )
        w = NcclWorkload()
        w.start()
        assert w.wait_for_completion(timeout=10.0, poll_interval=0.05) is True
        assert w.status == WorkloadStatus.ERROR
        assert "unhandled system error" in w.get_result().result
        w.shutdown_executor(wait=True)

    def test_stop_cancels_and_clears_result(self, mock_nccl_run):
        def run_until_cancel(cmd, *, timeout, cancel_event, **kwargs):
            for _ in range(500):
                if cancel_event.is_set():
                    return CommandResult(returncode=-1, stdout="", stderr="cancelled")
                time.sleep(0.01)
            return CommandResult(returncode=0, stdout=NCCL_ALL_REDUCE_OUTPUT, stderr="")

        mock_nccl_run.side_effect = run_until_cancel
        w = NcclWorkload()
        w.start()
        time.sleep(0.05)
        w.stop()
        assert mock_nccl_run.call_args[1]["cancel_event"].is_set()
        assert w.status == WorkloadStatus.STOPPED
        assert w.get_result().result is None
        w.shutdown_executor(wait=True)

    def test_second_start_raises_when_already_running(self, mock_nccl_run):
        block = threading.Event()

        def slow_run(*_args, **_kwargs):
            block.wait(timeout=60.0)
            return CommandResult(returncode=0, stdout=NCCL_ALL_REDUCE_OUTPUT, stderr="")

        mock_nccl_run.side_effect = slow_run
        w = NcclWorkload()
        w.start()
        assert w.status == WorkloadStatus.RUNNING
        with pytest.raises(RuntimeError, match="NCCL workload already running"):
            w.start()
        block.set()
        w._completion_fut.result(timeout=10.0)
        w.shutdown_executor(wait=True)


class TestPromptWorkload:
    """Tests for prompt-driven workload against a backend."""

    @pytest.fixture
    def mock_vllm_client_class(self):
        with patch(
            "production_test_framework.workload.prompt_workload.VllmClient",
        ) as m:
            backend = MagicMock()
            backend.wait_for_ready = MagicMock(return_value=True)
            backend.complete = MagicMock(return_value=InferenceResult(success=True, text="model output"))
            m.return_value = backend
            yield m, backend

    def test_default_backend_is_vllm(self, mock_vllm_client_class):
        mock_cls, backend = mock_vllm_client_class
        wl = PromptWorkload("hello world")
        mock_cls.assert_called()
        assert wl.prompt == "hello world"
        backend.wait_for_ready.assert_not_called()
        wl.shutdown_executor(wait=True)

    def test_passes_host_and_port_to_vllm_client(self, mock_vllm_client_class):
        mock_cls, _backend = mock_vllm_client_class
        wl = PromptWorkload(
            "q",
            backend_type=BACKEND_TYPE.VLLM,
            host="vllm.internal",
            port=9090,
        )
        mock_cls.assert_called()
        wl.shutdown_executor(wait=True)

    def test_defaults_to_the_default_model(self, mock_vllm_client_class):
        mock_cls, _backend = mock_vllm_client_class
        wl = PromptWorkload("q")
        assert mock_cls.call_args[0][0].model == DEFAULT_MODEL
        wl.shutdown_executor(wait=True)

    def test_passes_model_to_vllm_client(self, mock_vllm_client_class):
        """`start` submits `complete` with only the prompt, so the model has to travel in the
        config or the request asks for DEFAULT_MODEL and a server serving anything else 404s."""
        mock_cls, _backend = mock_vllm_client_class
        wl = PromptWorkload("q", model="Qwen/Qwen3-32B")
        assert mock_cls.call_args[0][0].model == "Qwen/Qwen3-32B"
        wl.shutdown_executor(wait=True)

    def test_start_waits_for_backend_and_dispatches_completion(self, mock_vllm_client_class):
        mock_cls, backend = mock_vllm_client_class
        wl = PromptWorkload("run this")
        wl.start()

        backend.wait_for_ready.assert_called_once_with(timeout=30)
        backend.complete.assert_called_once()
        fut = wl._completion_fut
        assert fut is not None
        fut.result(timeout=10.0)
        assert wl.status == WorkloadStatus.COMPLETED
        wl.shutdown_executor(wait=True)

    def test_stop_cancels_future(self, mock_vllm_client_class):
        _mock_cls, backend = mock_vllm_client_class
        wl = PromptWorkload("x")
        fake_fut = MagicMock()
        with patch.object(wl, "submit_background", return_value=fake_fut):
            wl.start()
            wl.stop()
        fake_fut.cancel.assert_called_once()
        assert wl.status == WorkloadStatus.STOPPED
        wl.shutdown_executor(wait=True)

    def test_get_result_returns_inference_text_after_completion(self, mock_vllm_client_class):
        mock_cls, backend = mock_vllm_client_class
        backend.complete = MagicMock(return_value=InferenceResult(success=True, text="final text"))
        wl = PromptWorkload("prompt")
        wl.start()
        wl._completion_fut.result(timeout=10.0)
        assert wl.status == WorkloadStatus.COMPLETED
        assert wl.get_result().result.text == "final text"
        assert wl.get_result().status == WorkloadStatus.COMPLETED
        assert wl.get_result().start_time is not None
        assert wl.get_result().end_time is not None
        assert wl.get_result().runtime is not None
        wl.shutdown_executor(wait=True)

    def test_defaults_to_a_single_prompt(self, mock_vllm_client_class):
        _mock_cls, backend = mock_vllm_client_class
        wl = PromptWorkload("prompt")
        wl.start()
        wl._completion_fut.result(timeout=10.0)
        assert backend.complete.call_count == 1
        wl.shutdown_executor(wait=True)

    def test_count_sends_the_prompt_that_many_times(self, mock_vllm_client_class):
        """One completion is not enough traffic for the profiler to export a collective: a new
        aggregation key is held as a primer for several windows before it reports real values,
        so the same prompt has to recur to walk it through."""
        _mock_cls, backend = mock_vllm_client_class
        wl = PromptWorkload("prompt", count=5)
        wl.start()
        wl._completion_fut.result(timeout=10.0)

        assert backend.complete.call_count == 5
        # Repeated verbatim -- identical requests reproduce the same message sizes, so they land
        # on the same keys rather than minting new ones.
        assert {call.args[0] for call in backend.complete.call_args_list} == {"prompt"}
        assert wl.status == WorkloadStatus.COMPLETED
        wl.shutdown_executor(wait=True)

    def test_max_tokens_is_passed_through(self, mock_vllm_client_class):
        _mock_cls, backend = mock_vllm_client_class
        wl = PromptWorkload("prompt", max_tokens=256)
        wl.start()
        wl._completion_fut.result(timeout=10.0)
        assert backend.complete.call_args.kwargs["max_tokens"] == 256
        wl.shutdown_executor(wait=True)

    def test_result_is_the_last_completion(self, mock_vllm_client_class):
        _mock_cls, backend = mock_vllm_client_class
        backend.complete = MagicMock(side_effect=[InferenceResult(success=True, text=f"answer {i}") for i in range(3)])
        wl = PromptWorkload("prompt", count=3)
        wl.start()
        wl._completion_fut.result(timeout=10.0)
        assert wl.get_result().result.text == "answer 2"
        wl.shutdown_executor(wait=True)

    def test_a_failed_prompt_stops_the_run_and_errors(self, mock_vllm_client_class):
        """A 404 or a dropped connection part-way through must not read as a completed run:
        later assertions would be measuring a workload that only partly happened."""
        _mock_cls, backend = mock_vllm_client_class
        backend.complete = MagicMock(
            side_effect=[
                InferenceResult(success=True, text="ok"),
                InferenceResult(success=False, error="HTTP 404: nope"),
                InferenceResult(success=True, text="never reached"),
            ]
        )
        wl = PromptWorkload("prompt", count=3)
        wl.start()
        with pytest.raises(RuntimeError, match="prompt 2/3 failed"):
            wl._completion_fut.result(timeout=10.0)

        assert backend.complete.call_count == 2, "the run must stop at the first failure"
        assert wl.status == WorkloadStatus.ERROR
        wl.shutdown_executor(wait=True)

    def test_count_below_one_is_rejected(self, mock_vllm_client_class):
        with pytest.raises(ValueError, match="count must be at least 1"):
            PromptWorkload("prompt", count=0)

    def test_second_start_raises_when_already_running(self, mock_vllm_client_class):
        _mock_cls, backend = mock_vllm_client_class
        block = threading.Event()

        def blocking_complete(_prompt, **_kwargs):
            block.wait(timeout=60.0)
            return InferenceResult(success=True, text="ok")

        backend.complete = MagicMock(side_effect=blocking_complete)
        wl = PromptWorkload("x")
        wl.start()
        assert wl.status == WorkloadStatus.RUNNING
        with pytest.raises(
            RuntimeError,
            match="Prompt workload already running",
        ):
            wl.start()
        block.set()
        wl._completion_fut.result(timeout=10.0)
        wl.shutdown_executor(wait=True)


class TestContainerCleanup:
    """
    Tests that a containerised run cannot leave its container behind.

    ``docker run --rm`` removes a container when the container exits, which covers a run that
    finishes on its own. It does not cover a stopped or timed-out run: cancellation terminates
    the ``docker run`` client, and because these workloads allocate a TTY the client does not
    proxy that signal to the container. Without an explicit removal the container keeps running,
    ``--rm`` never fires, and for the NCCL workload the GPUs it reserved stay reserved.
    """

    def test_container_removed_after_a_successful_run(self, mock_inferencex_run, no_real_docker_cleanup):
        inferencex_rm = no_real_docker_cleanup
        w = InferencexWorkload()
        w.start()
        w._completion_fut.result(timeout=10.0)

        inferencex_rm.assert_any_call(w.container_name)
        w.shutdown_executor(wait=True)

    def test_container_removed_after_a_stop(self, mock_inferencex_run, no_real_docker_cleanup):
        inferencex_rm = no_real_docker_cleanup
        w = InferencexWorkload()
        w.start()
        w.stop()

        inferencex_rm.assert_any_call(w.container_name)
        w.shutdown_executor(wait=True)

    def test_container_removed_after_a_failed_run(self, mock_inferencex_run, no_real_docker_cleanup):
        # The container must go even when the command itself failed; otherwise a run that
        # errors early leaves the container behind for every later run to trip over.
        inferencex_rm = no_real_docker_cleanup
        mock_inferencex_run.return_value = CommandResult(returncode=1, stdout="", stderr="boom")
        w = InferencexWorkload()
        w.start()
        with pytest.raises(RuntimeError, match="boom"):
            w._completion_fut.result(timeout=10.0)

        inferencex_rm.assert_any_call(w.container_name)
        w.shutdown_executor(wait=True)

    def test_nccl_container_removed_after_a_run(self, no_real_docker_cleanup):
        nccl_rm = no_real_docker_cleanup
        with patch("production_test_framework.workload.command_workload.run_cancellable_command") as m:
            m.return_value = CommandResult(returncode=0, stdout=NCCL_ALL_REDUCE_OUTPUT, stderr="")
            w = NcclWorkload(gpus_per_host=2)
            w.start()
            w._completion_fut.result(timeout=10.0)

        nccl_rm.assert_any_call(w.container_name)
        w.shutdown_executor(wait=True)

    def test_no_container_removal_when_not_using_docker(self, no_real_docker_cleanup):
        # Binaries run directly on the host have no container to remove.
        nccl_rm = no_real_docker_cleanup
        with patch("production_test_framework.workload.command_workload.run_cancellable_command") as m:
            m.return_value = CommandResult(returncode=0, stdout=NCCL_ALL_REDUCE_OUTPUT, stderr="")
            w = NcclWorkload(use_docker=False, image_name=None, gpus_per_host=2)
            w.start()
            w._completion_fut.result(timeout=10.0)

        nccl_rm.assert_not_called()
        w.shutdown_executor(wait=True)

    def test_cleanup_failure_does_not_mask_the_result(self, mock_inferencex_run, no_real_docker_cleanup):
        # A cleanup problem must not turn a good run into an error, nor replace the real reason
        # a bad one failed.
        inferencex_rm = no_real_docker_cleanup
        inferencex_rm.side_effect = RuntimeError("docker unreachable")
        w = InferencexWorkload()
        w.start()
        w._completion_fut.result(timeout=10.0)

        assert w.status == WorkloadStatus.COMPLETED
        assert w.get_result().result.raw_output == BENCHMARK_SERVING_OUTPUT
        w.shutdown_executor(wait=True)


class TestForceRemoveContainer:
    """Tests for the docker removal helper itself."""

    def test_success_reports_removed(self):
        with patch("production_test_framework.docker.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout="c\n", stderr="")
            assert force_remove_container("c") is True
        assert run.call_args[0][0] == ["docker", "rm", "--force", "c"]

    def test_absent_container_is_success(self):
        # The expected case after a clean run, where --rm already removed it.
        with patch("production_test_framework.docker.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                [], 1, stdout="", stderr="Error response from daemon: No such container: c"
            )
            assert force_remove_container("c") is True

    def test_other_failure_reports_false(self):
        with patch("production_test_framework.docker.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, stdout="", stderr="permission denied")
            assert force_remove_container("c") is False

    def test_missing_docker_binary_reports_false(self):
        # Never raises: cleanup runs while a result is being settled.
        with patch("production_test_framework.docker.subprocess.run", side_effect=OSError):
            assert force_remove_container("c") is False

    def test_names_are_unique_per_call(self):
        assert unique_container_name("x") != unique_container_name("x")
        assert unique_container_name("x").startswith("x-")

    def test_argv_can_run_through_sudo(self):
        # Some hosts only allow docker via sudo; the caller decides, rather than this module
        # reading an environment variable of its own.
        assert docker_argv("ps", sudo=True) == ["sudo", "docker", "ps"]
        assert docker_argv("ps") == ["docker", "ps"]

    def test_extra_labels_are_added_alongside_the_framework_one(self):
        args = label_args("c", {"suite": "profiler_otel"})
        assert args[:2] == ["--label", f"{CONTAINER_LABEL}=c"]
        assert args[2:] == ["--label", "suite=profiler_otel"]


class TestContainerLabelSweep:
    """
    Tests for finding and removing containers by label.

    The sweep exists because a container that is still running cannot be cleaned up by
    `docker container prune`, and an orphan's generated name is not usually to hand.
    """

    def test_lists_ids_carrying_the_label(self):
        with patch("production_test_framework.docker.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout="abc123\ndef456\n", stderr="")
            assert list_containers_by_label() == ["abc123", "def456"]
        assert run.call_args[0][0] == [
            "docker",
            "ps",
            "--quiet",
            "--all",
            "--filter",
            f"label={CONTAINER_LABEL}",
        ]

    def test_no_containers_is_not_an_error(self):
        with patch("production_test_framework.docker.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout="\n", stderr="")
            assert list_containers_by_label() == []

    def test_unreachable_docker_is_not_an_error(self):
        # Callers are cleaning up; an unreachable daemon must not fail the session.
        with patch("production_test_framework.docker.subprocess.run", side_effect=OSError):
            assert list_containers_by_label() == []
            assert remove_containers_by_label() == []

    def test_removes_every_listed_id_in_one_call(self):
        with patch("production_test_framework.docker.subprocess.run") as run:
            run.side_effect = [
                subprocess.CompletedProcess([], 0, stdout="abc\ndef\n", stderr=""),
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            ]
            assert remove_containers_by_label() == ["abc", "def"]
        assert run.call_args[0][0] == ["docker", "rm", "--force", "abc", "def"]

    def test_nothing_to_remove_makes_no_removal_call(self):
        with patch("production_test_framework.docker.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            assert remove_containers_by_label() == []
        assert run.call_count == 1, "should not have issued a docker rm with no ids"

    def test_a_custom_label_can_be_swept(self):
        with patch("production_test_framework.docker.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            list_containers_by_label("my.label")
        assert "label=my.label" in run.call_args[0][0]


class TestDockerContainerMixin:
    """
    Tests for the behaviour every containerised workload inherits.

    The point of the base class is that a new containerised workload cannot forget to clean up
    after itself. Forgetting is silent, and a leaked container holding --gpus keeps those GPUs
    from every later run on the host, so these assert on what a subclass gets for free.
    """

    class _Minimal(DockerContainerMixin, CommandWorkload):
        """The least a subclass can define."""

        workload_name = "Minimal"
        container_name_prefix = "minimal"

        def build_command(self) -> list[str]:
            return [*self.docker_run_argv("--network", "host"), "echo", "hi"]

    def test_subclass_gets_a_unique_name_without_asking(self):
        first, second = self._Minimal(image_name="img"), self._Minimal(image_name="img")
        assert first.container_name.startswith("minimal-")
        assert first.container_name != second.container_name

    def test_subclass_gets_name_and_label_in_its_argv(self):
        w = self._Minimal(image_name="img")
        cmd = w.build_command()
        assert cmd[cmd.index("--name") + 1] == w.container_name
        assert cmd[cmd.index("--label") + 1] == f"{CONTAINER_LABEL}={w.container_name}"

    def test_subclass_gets_cleanup_without_defining_it(self, no_real_docker_cleanup):
        # The whole reason this lives on the base rather than in each workload.
        assert "_cleanup_after_run" not in vars(self._Minimal), (
            "the fixture subclass must not define cleanup, or this proves nothing"
        )
        w = self._Minimal(image_name="img")
        with patch("production_test_framework.workload.command_workload.run_cancellable_command") as run:
            run.return_value = CommandResult(returncode=0, stdout="hi", stderr="")
            w.start()
            w._completion_fut.result(timeout=10.0)

        no_real_docker_cleanup.assert_any_call(w.container_name)
        w.shutdown_executor(wait=True)

    def test_flags_precede_the_name_and_image_comes_last(self):
        w = self._Minimal(image_name="img", docker_extra_args=("-v", "/a:/b"))
        cmd = w.docker_run_argv("--gpus", "all")
        assert cmd[:6] == ["docker", "run", "--rm", "-t", "--gpus", "all"]
        assert cmd[-3:] == ["-v", "/a:/b", "img"], "extra args must be last so they can override"

    def test_env_becomes_dash_e_arguments(self):
        w = self._Minimal(image_name="img", env={"A": "1"})
        assert w.env_args() == ["-e", "A=1"]
        # mpirun forwards the same variables with a different flag.
        assert w.env_args("-x") == ["-x", "A=1"]

    def test_uses_docker_defaults_true(self):
        assert self._Minimal(image_name="img").uses_docker is True

    @pytest.mark.parametrize("workload", [InferencexWorkload, NcclWorkload])
    def test_mixin_precedes_the_workload_base(self, workload):
        # Order matters and is easy to get wrong: written the other way round,
        # CommandWorkload's no-op _cleanup_after_run would win and containers would leak
        # again, silently and with every test still passing.
        mro = workload.__mro__
        assert mro.index(DockerContainerMixin) < mro.index(CommandWorkload), (
            f"{workload.__name__} must list DockerContainerMixin first"
        )
        assert workload._cleanup_after_run is DockerContainerMixin._cleanup_after_run

    def test_cooperative_init_passes_the_rest_down_the_mro(self):
        # The mixin consumes only its own arguments; everything else has to reach the
        # workload base, or a timeout set by the caller would be quietly dropped.
        w = self._Minimal(image_name="img", timeout=1234)
        assert w._timeout == 1234

    def test_mixin_does_not_require_a_workload_base_of_its_own(self):
        # It adds a capability rather than being a kind of workload, so it carries no
        # inheritance of its own and can be mixed into anything.
        assert DockerContainerMixin.__mro__[1:] == (object,)
