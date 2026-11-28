# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Unit tests for starting vLLM in docker on a host."""

from unittest.mock import patch

import pytest

from production_test_framework.ssh import CommandResult
from production_test_framework.vllm import server
from production_test_framework.vllm.server import (
    AMD,
    NVIDIA,
    Pool,
    VllmDeployment,
    VllmError,
    aggregated_workers,
    check_docker,
    detect_gpus,
    disaggregated_workers,
    gpu_memory,
    hf_cache,
    needed_gpus,
)

OK = CommandResult(0, "", "")

#: ``amd-smi static --asic --vram --json`` as amd-smi 26 prints it, for two GPUs.
AMD_SMI_STATIC = """{"gpu_data": [
  {"gpu": 0, "asic": {"market_name": "AMD Radeon AI PRO R9700"}, "vram": {"size": {"value": 32624, "unit": "MB"}}},
  {"gpu": 1, "asic": {"market_name": "AMD Radeon AI PRO R9700"}, "vram": {"size": {"value": 32624, "unit": "MB"}}}
]}"""
NO_NVIDIA_DRIVER = CommandResult(9, "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", "")


class FakeHost:
    """A host that records the commands run on it and answers from *replies*, by first word match."""

    def __init__(self, local=True, tools=("nvidia-smi",), replies=None):
        self.local = local
        self.hostname = "localhost" if local else "gpu-node"
        self.address = self.hostname
        self.home = "/home/alice"
        self.tools = tools
        self.replies = replies or {}
        self.commands: list[list[str]] = []
        self.written: dict[str, str] = {}

    def __str__(self):
        return self.address

    def _reply(self, command):
        self.commands.append(command)
        line = " ".join(command)
        return next((reply for prefix, reply in self.replies.items() if line.startswith(prefix)), OK)

    def run(self, command, timeout=60, stdin_data=None):
        return self._reply(command)

    def shell(self, script, *args, timeout=60, stdin_data=None):
        return self._reply(list(args))

    def which(self, tool):
        return f"/usr/bin/{tool}" if tool in self.tools else None

    def getenv(self, name):
        return None

    def write_file(self, path, text):
        self.written[path] = text
        return OK

    def docker_runs(self):
        return [command for command in self.commands if command[:2] == ["docker", "run"]]


class TestWorkers:
    def test_aggregated_takes_the_first_gpus(self):
        (worker,) = aggregated_workers(8080, 4, hostname="gpu-node")
        assert (worker.role, worker.gpus, worker.url) == ("server", (0, 1, 2, 3), "http://gpu-node:8080")

    def test_disaggregated_gives_each_worker_its_gpus_and_ports(self):
        workers = disaggregated_workers(Pool(workers=2, tensor_parallel=1), Pool(tensor_parallel=2))
        assert [(w.role, w.port, w.gpus, w.side_channel_port) for w in workers] == [
            ("prefill", 8100, (0,), 5600),
            ("prefill", 8101, (1,), 5601),
            ("decode", 8200, (2, 3), 5602),
        ]
        assert needed_gpus(workers) == [0, 1, 2, 3]
        assert workers[0].local_url == "http://localhost:8100"


class TestPrerequisites:
    def test_no_driver(self):
        with pytest.raises(VllmError, match="No GPU driver found on this host"):
            gpu_memory(FakeHost(tools=()))

    def test_amd_gpus_from_amd_smi(self):
        host = FakeHost(tools=("amd-smi",), replies={"amd-smi": CommandResult(0, AMD_SMI_STATIC, "")})
        found = detect_gpus(host)
        assert found.vendor is AMD and found.name == "AMD Radeon AI PRO R9700"
        assert found.memory == {0: 32624 * 1024**2, 1: 32624 * 1024**2}

    def test_older_amd_smi_prints_a_bare_list_after_a_warning(self):
        entries = AMD_SMI_STATIC.removeprefix('{"gpu_data": ').removesuffix("}")
        host = FakeHost(tools=("amd-smi",), replies={"amd-smi": CommandResult(0, "WARNING: x\n" + entries, "")})
        assert detect_gpus(host).memory == {0: 32624 * 1024**2, 1: 32624 * 1024**2}

    def test_nvidia_smi_without_nvidia_gpus_falls_back_to_amd(self):
        host = FakeHost(
            tools=("nvidia-smi", "amd-smi"),
            replies={"nvidia-smi": NO_NVIDIA_DRIVER, "amd-smi": CommandResult(0, AMD_SMI_STATIC, "")},
        )
        assert detect_gpus(host).vendor is AMD

    def test_nvidia_is_used_when_it_lists_gpus(self):
        host = FakeHost(
            tools=("nvidia-smi", "amd-smi"),
            replies={
                "nvidia-smi": CommandResult(0, "0, NVIDIA RTX PRO 2000, 16384", ""),
                "amd-smi": CommandResult(0, AMD_SMI_STATIC, ""),
            },
        )
        assert detect_gpus(host).vendor is NVIDIA

    def test_neither_vendor_listing_gpus_says_what_to_check_for_both(self):
        host = FakeHost(
            tools=("nvidia-smi", "amd-smi"),
            replies={"nvidia-smi": NO_NVIDIA_DRIVER, "amd-smi": CommandResult(0, '{"gpu_data": []}', "")},
        )
        with pytest.raises(
            VllmError, match="NVIDIA driver.*amd-smi works but lists no GPUs.*NVIDIA Container.*/dev/kfd"
        ):
            detect_gpus(host)

    def test_nvidia_smi_failing_says_what_to_check(self):
        host = FakeHost(replies={"nvidia-smi": CommandResult(9, "", "NVML: Driver Not Loaded")})
        with pytest.raises(VllmError, match="Driver Not Loaded.*NVIDIA Container Toolkit"):
            gpu_memory(host)

    def test_memory_by_index(self):
        host = FakeHost(
            replies={"nvidia-smi": CommandResult(0, "0, NVIDIA RTX PRO 2000, 16384\n1, NVIDIA RTX PRO 2000, 16384", "")}
        )
        assert gpu_memory(host) == {0: 16 * 1024**3, 1: 16 * 1024**3}

    def test_docker_without_the_group(self):
        host = FakeHost(local=False, replies={"docker info": CommandResult(1, "", "permission denied")})
        with pytest.raises(VllmError, match="on gpu-node .permission denied.*docker group"):
            check_docker(host)

    def test_hf_cache(self):
        assert hf_cache(FakeHost(), "/data/hf") == "/data/hf"
        assert hf_cache(FakeHost(local=False)) == "/home/alice/.cache/huggingface"


def _deployment(host, workers, log_dir, **kwargs):
    return VllmDeployment(host, "acme/model", workers, f"http://{host.hostname}:8080", log_dir=log_dir, **kwargs)


class TestDeployment:
    @pytest.fixture(autouse=True)
    def serving(self):
        with patch.object(server, "serves", return_value=True) as serves:
            yield serves

    def test_aggregated_runs_one_server(self, tmp_path):
        host = FakeHost()
        deployment = _deployment(host, aggregated_workers(8080, 2), log_dir=tmp_path, args={"max_model_len": 65536})
        deployment.start()
        (run,) = host.docker_runs()
        assert run[run.index("--gpus") + 1] == '"device=0,1"'
        assert run[-1] == "vllm serve acme/model --port 8080 --tensor-parallel-size 2 --max-model-len 65536"
        assert run[run.index("--entrypoint") + 1] == "/bin/bash"
        assert run[-3:-1] == ["openmosaic/mosaic-vllm:latest", "-c"]
        assert deployment.worker_containers == deployment.containers

    def test_amd_gets_the_rocm_devices_and_image(self, tmp_path):
        host = FakeHost()
        _deployment(host, aggregated_workers(8080, 2), log_dir=tmp_path, vendor=AMD).start()
        (run,) = host.docker_runs()
        assert "--gpus" not in run
        assert run[run.index("--device") + 1] == "/dev/kfd" and "HIP_VISIBLE_DEVICES=0,1" in run
        assert run[run.index("--pid") + 1] == "host"
        assert AMD.image in run

    def test_an_image_given_replaces_the_vendors(self, tmp_path):
        host = FakeHost()
        _deployment(host, aggregated_workers(8080, 1), log_dir=tmp_path, vendor=AMD, image="rocm/vllm:rdna").start()
        (run,) = host.docker_runs()
        assert "rocm/vllm:rdna" in run and AMD.image not in run

    def test_env_given_is_set_in_the_container(self, tmp_path):
        host = FakeHost()
        env = {"HSA_ENABLE_IPC_MODE_LEGACY": 0}
        _deployment(host, aggregated_workers(8080, 1), log_dir=tmp_path, vendor=AMD, env=env).start()
        (run,) = host.docker_runs()
        assert run[run.index("HSA_ENABLE_IPC_MODE_LEGACY=0") - 1] == "--env"

    def test_disaggregated_starts_the_router_after_the_workers(self, tmp_path):
        host = FakeHost(local=False)
        workers = disaggregated_workers(Pool(), Pool())
        deployment = _deployment(host, workers, log_dir=tmp_path, name="agentx")
        deployment.start()
        prefill, decode, router = host.docker_runs()
        assert "--kv-transfer-config" in prefill[-1] and "VLLM_NIXL_SIDE_CHANNEL_PORT=5601" in decode
        command = router[router.index(server.ROUTER_IMAGE) + 1 :]
        assert command[:2] == ["vllm-router", "--vllm-pd-disaggregation"]
        assert command[2:6] == ["--prefill", "http://localhost:8100", "--decode", "http://localhost:8200"]
        assert command[command.index("--port") + 1] == "8080"
        assert command[command.index("--policy") + 1] == "round_robin"
        assert "--disable-retries" in command and "--disable-circuit-breaker" in command
        assert len(deployment.worker_containers) == 2 and deployment.containers[-1].startswith("agentx-router-")

    def test_hf_token_goes_on_stdin(self, tmp_path):
        host = FakeHost()
        with patch.object(host, "shell", wraps=host.shell) as shell:
            _deployment(host, aggregated_workers(8080, 1), log_dir=tmp_path, hf_token="hf_secret").start()
        assert shell.call_args.kwargs["stdin_data"] == "hf_secret\n"
        assert not any("hf_secret" in part for command in host.commands for part in command)

    def test_a_failed_docker_run_is_still_removed(self, tmp_path):
        failed = CommandResult(125, "", "Unable to find image 'x'\ndocker: nvidia-container-cli: initialization error")
        host = FakeHost(replies={"docker run": failed})
        with (
            patch.object(server, "force_remove_container") as remove,
            pytest.raises(VllmError, match="nvidia-container-cli.*\n.*NVIDIA Container Toolkit"),
        ):
            with _deployment(host, aggregated_workers(8080, 1), tmp_path):
                pass
        assert remove.call_args.args[0].startswith("vllm-") and remove.call_args.kwargs["host"] is host

    def test_an_unknown_kv_transfer(self, tmp_path):
        workers = disaggregated_workers(Pool(), Pool())
        with pytest.raises(VllmError, match="'lmcache' is not one of nixl"):
            _deployment(FakeHost(), workers, log_dir=tmp_path, kv_transfer="lmcache").start()

    def test_a_server_that_exits_fails_with_its_log(self, tmp_path, serving):
        serving.return_value = False
        host = FakeHost(
            replies={"docker inspect": CommandResult(0, "false", ""), "docker logs": CommandResult(0, "CUDA OOM", "")}
        )
        with pytest.raises(VllmError, match=r"exited; its log is .*\.log:\nCUDA OOM"):
            _deployment(host, aggregated_workers(8080, 1), log_dir=tmp_path, startup_timeout=0.1).start()
