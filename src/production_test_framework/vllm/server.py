# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
Start vLLM in docker on a machine for a test, and remove it after.

A deployment is one server (aggregated), or prefill and decode workers on their own GPUs and
ports behind :mod:`~.disagg_proxy` on the endpoint's port (disaggregated), which hands each
request's KV cache from prefill to decode over NIXL. It all runs on a
:class:`~production_test_framework.host.Host`: this machine, or another one over ssh specified
by --agentx-target.
"""

import json
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from ..docker import docker_argv, force_remove_container, label_args, unique_container_name
from ..helper import poll_until
from ..host import Host
from ..workload.inferencex_workload import benchmark_option_argv
from .model_fit import disk_shortfall, gpu_shortfall, model_footprint, rope_override_problem

__all__ = [
    "GPU_VENDORS",
    "KV_CONNECTORS",
    "VLLM_IMAGE",
    "GpuVendor",
    "Pool",
    "VllmDeployment",
    "VllmError",
    "VllmWorker",
    "aggregated_workers",
    "check_docker",
    "check_image_fits",
    "check_model_fits",
    "disaggregated_workers",
    "gpu_memory",
    "hf_cache",
    "needed_gpus",
    "serves",
]

_log = logging.getLogger(__name__)

VLLM_IMAGE = "openmosaic/mosaic-vllm:latest"

#: vLLM's own default, for args that drop gpu_memory_utilization.
VLLM_MEMORY_UTILIZATION = 0.9

PROBE_TIMEOUT = 10
STARTUP_TIMEOUT = 1200.0
POLL_INTERVAL = 5.0

REMOVE_TIMEOUT = 120.0

WORKER_BASE_PORTS = {"prefill": 8100, "decode": 8200}
NIXL_SIDE_CHANNEL_BASE_PORT = 5600

KV_CONNECTORS = {"nixl": "NixlConnector"}

PROXY_SCRIPT = Path(__file__).with_name("disagg_proxy.py")
REMOTE_PROXY_SCRIPT = ".cache/production-test-framework/disagg_proxy.py"

NVIDIA_HINT = (
    "Check on the target that: nvidia-smi lists the GPUs (the NVIDIA driver is installed and loaded); "
    "docker is not the snap package (snap list docker), which cannot see the host's driver libraries; "
    "the NVIDIA Container Toolkit is installed and set up for docker (sudo nvidia-ctk runtime configure "
    "--runtime=docker && sudo systemctl restart docker); and docker run --rm --gpus all ubuntu nvidia-smi "
    "lists the GPUs."
)

PULL_PROGRESS = re.compile(
    r"(Run 'docker run --help'|Unable to find image|[\w.-]+: Pulling from|[0-9a-f]{12}: (Pull complete|Already exists"
    r"|Waiting|Download complete|Verifying Checksum|Pulling fs layer|Extracting|Downloading)|Digest: |Status: )"
)

IMAGE_DISK_PER_DOWNLOAD_BYTE = 3.5

WITH_HF_TOKEN = 'IFS= read -r HF_TOKEN; export HF_TOKEN; exec "$@"'


class VllmError(Exception):
    """vLLM cannot be started on the host. The message says why and what to do about it."""


@dataclass(frozen=True)
class GpuVendor:
    """A GPU vendor, found by the command-line tool its driver installs."""

    name: str
    tool: str
    supported: bool


GPU_VENDORS = (GpuVendor("NVIDIA", "nvidia-smi", supported=True), GpuVendor("AMD", "amd-smi", supported=False))


@dataclass(frozen=True)
class VllmWorker:
    """One vLLM server: the whole deployment when aggregated, else one prefill or decode worker."""

    role: str
    port: int | None
    gpus: tuple[int, ...]
    side_channel_port: int | None = None
    hostname: str = "localhost"

    @property
    def url(self) -> str:
        return f"http://{self.hostname}:{self.port}"

    @property
    def local_url(self) -> str:
        """The worker's URL from its own host, as the proxy beside it reaches it."""
        return f"http://localhost:{self.port}"


@dataclass(frozen=True)
class Pool:
    """A disaggregated deployment's prefill or decode workers."""

    workers: int = 1
    tensor_parallel: int = 1


def _where(host: Host) -> str:
    return "this host" if host.local else str(host)


def serves(url: str) -> bool:
    """Whether an OpenAI-compatible server answers at *url*'s ``/v1/models``."""
    try:
        return requests.get(f"{url}/v1/models", timeout=PROBE_TIMEOUT).ok
    except requests.RequestException:
        return False


def gpu_memory(host: Host) -> dict[int, int]:
    """
    Each GPU's memory in bytes, by index, on *host*. Raises :class:`VllmError` when no supported
    GPU driver works there, since docker's GPU runtime cannot give vLLM the GPUs either.
    """
    where = _where(host)
    found = [vendor for vendor in GPU_VENDORS if host.which(vendor.tool)]
    supported = " and ".join(vendor.name for vendor in GPU_VENDORS if vendor.supported)
    if not found:
        tools = " nor ".join(f"{vendor.tool} ({vendor.name})" for vendor in GPU_VENDORS)
        raise VllmError(
            f"No GPU driver found on {where}: neither {tools} is on PATH. Install the GPU driver; vLLM is "
            f"started on {supported} GPUs."
        )
    if not any(vendor.supported for vendor in found):
        raise VllmError(
            f"{where} has {found[0].name} GPUs ({found[0].tool} is installed), but vLLM is started only on "
            f"{supported} GPUs for now."
        )
    listed = host.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"], timeout=30)
    if not listed.success:
        problem = listed.stderr or listed.stdout or f"exit status {listed.returncode}"
        raise VllmError(
            f"nvidia-smi does not work on {where} ({problem}), so docker cannot give vLLM the GPUs. {NVIDIA_HINT}"
        )
    memory = {}
    for line in listed.stdout.splitlines():
        index, _, mib = line.partition(",")
        if index.strip().isdigit() and mib.strip().isdigit():
            memory[int(index)] = int(mib) * 1024**2
    if not memory:
        raise VllmError(f"nvidia-smi works but lists no GPUs on {where}. {NVIDIA_HINT}")
    return memory


def aggregated_workers(port: int | None, tensor_parallel: int, hostname: str = "localhost") -> list[VllmWorker]:
    """One server on *port* and the first *tensor_parallel* GPUs."""
    return [VllmWorker("server", port, tuple(range(tensor_parallel)), hostname=hostname)]


def disaggregated_workers(prefill: Pool, decode: Pool, hostname: str = "localhost") -> list[VllmWorker]:
    """
    Each prefill then decode worker on its own port and the next ``tensor_parallel`` GPUs, with
    its own NIXL side channel.
    """
    workers: list[VllmWorker] = []
    next_gpu = 0
    for role, pool in (("prefill", prefill), ("decode", decode)):
        for index in range(pool.workers):
            gpus = tuple(range(next_gpu, next_gpu + pool.tensor_parallel))
            side_channel = NIXL_SIDE_CHANNEL_BASE_PORT + len(workers)
            workers.append(VllmWorker(role, WORKER_BASE_PORTS[role] + index, gpus, side_channel, hostname))
            next_gpu += pool.tensor_parallel
    return workers


def needed_gpus(workers: list[VllmWorker]) -> list[int]:
    return sorted({gpu for worker in workers for gpu in worker.gpus})


def hf_cache(host: Host, local_hf_home: str | None = None) -> str:
    """
    The Hugging Face cache vLLM's container mounts: *local_hf_home* or ``~/.cache/huggingface``
    on this machine; on a remote host, its own ``HF_HOME`` or ``~/.cache/huggingface``.
    """
    if host.local:
        return str(Path(local_hf_home or "~/.cache/huggingface").expanduser())
    return host.getenv("HF_HOME") or f"{host.home}/.cache/huggingface"


def check_docker(host: Host) -> None:
    """Raise :class:`VllmError` when docker does not work for this user on *host*."""
    info = host.run(docker_argv("info", "--format", "{{.ServerVersion}}"))
    if not info.success:
        problem = (info.stderr or info.stdout or f"exit status {info.returncode}").splitlines()[-1]
        raise VllmError(
            f"docker does not work for this user on {_where(host)} ({problem}). Install docker there, and add "
            "the user to the docker group (sudo usermod -aG docker <user>, then log in again)."
        )


def _image_download_bytes(host: Host, image: str) -> int | None:
    """The compressed size of *image*'s layers for *host*, from its registry; None if unknown."""
    inspected = host.run(docker_argv("manifest", "inspect", "--verbose", image), timeout=120)
    try:
        manifests = json.loads(inspected.stdout) if inspected.success else None
    except ValueError:
        return None
    if isinstance(manifests, list):
        arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(host.machine, host.machine)
        manifests = next(
            (m for m in manifests if m["Descriptor"].get("platform", {}).get("architecture") == arch), None
        )
    if not manifests:
        return None
    layers = (manifests.get("OCIManifest") or manifests.get("SchemaV2Manifest") or {}).get("layers") or []
    return sum(layer.get("size", 0) for layer in layers)


def _image_store(host: Host) -> str:
    """Where docker keeps pulled images on *host*: containerd's root under its snapshotter, else docker's."""
    info = host.run(docker_argv("info", "--format", "{{.DockerRootDir}}|{{json .DriverStatus}}"))
    root, _, status = info.stdout.partition("|")
    containerd = "/var/lib/containerd"
    return containerd if "io.containerd.snapshotter" in status and host.is_dir(containerd) else root or "/"


def check_image_fits(host: Host, image: str = VLLM_IMAGE) -> None:
    """Raise :class:`VllmError` before a pull of *image* that *host*'s disk has no room for."""
    if host.run(docker_argv("image", "inspect", image)).success:
        return
    download = _image_download_bytes(host, image)
    if not download:
        return
    store = _image_store(host)
    free = host.free_bytes(store)
    if free is None:
        return
    needed = download * IMAGE_DISK_PER_DOWNLOAD_BYTE
    if free < needed:
        raise VllmError(
            f"{image} is not pulled yet and needs about {needed / 1e9:.0f} GB under {store} on {_where(host)} "
            f"({download / 1e9:.1f} GB to download, unpacked beside it), but only {free / 1e9:.0f} GB is "
            "free. Free up space there and run again."
        )


def check_model_fits(
    host: Host,
    model: str,
    workers: list[VllmWorker],
    args: dict[str, Any],
    cache: str,
    memory: dict[int, int],
    *,
    configured_args: dict[str, Any],
    min_context: int,
    source: str,
    hf_token: str | None = None,
) -> str:
    """
    Raise :class:`VllmError` when *model* cannot run on *host* with *args*: the RoPE override in
    *configured_args* (the flags as written in *source*) does not suit it or leaves it short of
    *min_context*, its download does not fit *cache*'s disk, or a worker's GPUs cannot hold its
    weights and the KV cache of one ``max_model_len`` request. Returns a line saying what fits;
    a model whose size cannot be found out is not checked.
    """
    footprint = model_footprint(model, cache, token=hf_token, host=host)
    if footprint is None:
        return f"Could not size {model} from Hugging Face or {cache}; vLLM will report if it does not fit"
    if problem := rope_override_problem(
        footprint, configured_args, args.get("max_model_len"), min_context=min_context, profile=source
    ):
        raise VllmError(problem)
    free = host.free_bytes(cache)
    if free is not None and (problem := disk_shortfall(footprint, cache, free)):
        raise VllmError(problem)
    max_model_len = args.get("max_model_len") or (footprint.config.get("text_config") or footprint.config).get(
        "max_position_embeddings"
    )
    for worker in workers if max_model_len else []:
        problem = gpu_shortfall(
            footprint,
            gpu_bytes=min(memory[gpu] for gpu in worker.gpus),
            tensor_parallel=len(worker.gpus),
            memory_utilization=float(args.get("gpu_memory_utilization") or VLLM_MEMORY_UTILIZATION),
            max_model_len=int(max_model_len),
            worker=f"GPU {', '.join(map(str, worker.gpus))} ({worker.role})",
        )
        if problem:
            raise VllmError(problem)
    fits = f"fits {_where(host)}" if max_model_len else "fits the disk (GPU memory not checked: no max_model_len)"
    return (
        f"{model}: {footprint.weight_bytes / 1e9:.1f} GB of weights, "
        f"{footprint.missing_bytes / 1e9:.1f} GB still to download; {fits}"
    )


class VllmDeployment:
    """
    vLLM containers on *host* serving *model* at *url*: *workers* from :func:`aggregated_workers`
    or :func:`disaggregated_workers`, the latter behind the proxy on *url*'s port. *args* are the
    ``vllm serve`` flags beyond model, port and tensor parallelism, as ``{flag_name: value}``; a
    mapping goes as JSON. Containers are named after *name* and their logs saved to *log_dir*.

    :meth:`start` starts them and waits until *url* answers; :meth:`stop` removes them. Used as
    a context manager it does both, removing what it started if the start fails.
    """

    def __init__(
        self,
        host: Host,
        model: str,
        workers: list[VllmWorker],
        url: str,
        *,
        log_dir: Path,
        args: dict[str, Any] | None = None,
        kv_transfer: str = "nixl",
        image: str = VLLM_IMAGE,
        cache: str | None = None,
        hf_token: str | None = None,
        name: str = "vllm",
        startup_timeout: float = STARTUP_TIMEOUT,
    ) -> None:
        self.host = host
        self.model = model
        self.workers = workers
        self.url = url
        self.log_dir = log_dir
        self.args = args or {}
        self.kv_transfer = kv_transfer
        self.image = image
        self.cache = cache or hf_cache(host)
        self.hf_token = hf_token
        self.name = name
        self.startup_timeout = startup_timeout
        self.worker_containers: list[str] = []
        self.containers: list[str] = []

    @property
    def disaggregated(self) -> bool:
        return any(worker.role != "server" for worker in self.workers)

    def __enter__(self) -> VllmDeployment:
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        """Start the containers and wait until each answers. Raises :class:`VllmError` if one cannot."""
        port = urlparse(self.url).port
        if port is None:
            raise VllmError(f"no server answering at {self.url}/v1/models, and it has no port to start one on")
        connector = KV_CONNECTORS.get(self.kv_transfer)
        if self.disaggregated and connector is None:
            raise VllmError(f"kv_transfer {self.kv_transfer!r} is not one of {', '.join(KV_CONNECTORS)}")
        shared = [
            "--ipc", "host",
            "--network", "host",
            "--volume", f"{self.cache}:/root/.cache/huggingface",
            *(["--env", "HF_TOKEN"] if self.hf_token else []),
        ]  # fmt: skip
        for worker in self.workers:
            argv = [
                "vllm", "serve", self.model,
                "--port", str(worker.port),
                "--tensor-parallel-size", str(len(worker.gpus)),
                *benchmark_option_argv(
                    {key: json.dumps(value) if isinstance(value, dict) else value for key, value in self.args.items()}
                ),
            ]  # fmt: skip
            flags = ["--gpus", f'"device={",".join(map(str, worker.gpus))}"', *shared]
            if self.disaggregated:
                argv += ["--kv-transfer-config", json.dumps({"kv_connector": connector, "kv_role": "kv_both"})]
                flags += ["--env", f"VLLM_NIXL_SIDE_CHANNEL_PORT={worker.side_channel_port}"]
            name = unique_container_name(self.name if worker.role == "server" else f"{self.name}-{worker.role}")
            self.worker_containers.append(name)
            self._run_container(name, flags, shlex.join(argv))
        for name, worker in zip(self.worker_containers, self.workers, strict=True):
            self._wait_until_serving(name, worker.url)

        if self.disaggregated:
            proxy = unique_container_name(f"{self.name}-proxy")
            command = shlex.join([
                "python3", "/disagg_proxy.py",
                "--port", str(port),
                "--prefill", *(worker.local_url for worker in self.workers if worker.role == "prefill"),
                "--decode", *(worker.local_url for worker in self.workers if worker.role == "decode"),
            ])  # fmt: skip
            script = self._proxy_script()
            self._run_container(proxy, ["--network", "host", "--volume", f"{script}:/disagg_proxy.py:ro"], command)
            self._wait_until_serving(proxy, self.url)

    def worker_logs(self) -> list[str]:
        """Each vLLM server's log so far, in the order of *workers*."""
        return [self._log(name) for name in self.worker_containers]

    def save_logs(self) -> list[Path]:
        """Write each container's log to ``<log_dir>/<container>.log``."""
        return [self._save_log(name) for name in self.containers]

    def stop(self) -> None:
        """Remove every container this deployment started."""
        for name in self.containers:
            force_remove_container(name, host=self.host, timeout=REMOVE_TIMEOUT)
        self.containers.clear()
        self.worker_containers.clear()

    def _run_container(self, name: str, flags: list[str], command: str) -> None:
        run = docker_argv("run", "--detach", *flags, "--name", name, *label_args(name), self.image, command)
        _log.info("Starting %s as %s on %s: %s", self.image, name, _where(self.host), command)
        self.containers.append(name)
        if self.hf_token:
            started = self.host.shell(
                WITH_HF_TOKEN, *run, timeout=int(self.startup_timeout), stdin_data=self.hf_token + "\n"
            )
        else:
            started = self.host.run(run, timeout=int(self.startup_timeout))
        if not started.success:
            output = (started.stderr or started.stdout).splitlines()
            error = "\n".join(line for line in output if line.strip() and not PULL_PROGRESS.match(line))
            hint = f"\n{NVIDIA_HINT}" if "nvidia-container" in error else ""
            raise VllmError(f"docker could not start {name}: {error}{hint}")

    def _running(self, name: str) -> bool:
        return self.host.run(docker_argv("inspect", "--format", "{{.State.Running}}", name)).stdout == "true"

    def _log(self, name: str) -> str:
        logs = self.host.run(docker_argv("logs", name), timeout=300)
        return "\n".join(part for part in (logs.stdout, logs.stderr) if part)

    def _save_log(self, name: str) -> Path:
        path = self.log_dir / f"{name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._log(name), encoding="utf-8")
        return path

    def _answers_on_host(self, url: str) -> bool:
        """Whether *url*'s port answers on a remote host itself, when this machine cannot reach it."""
        if self.host.local:
            return False
        probe = f"http://localhost:{urlparse(url).port}/v1/models"
        return self.host.run(["curl", "--silent", "--fail", "--max-time", "10", "--output", "/dev/null", probe]).success

    def _wait_until_serving(self, name: str, url: str) -> None:
        poll_until(lambda: serves(url) or not self._running(name), timeout=self.startup_timeout, interval=POLL_INTERVAL)
        if serves(url):
            return
        state = "exited" if not self._running(name) else f"did not answer in {self.startup_timeout:g}s"
        if self._answers_on_host(url):
            state = (
                f"answers on {self.host} but not from this machine at {url}: open its port in the host's "
                "firewall, or run the test on that host"
            )
        log = self._save_log(name)
        tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-20:])
        raise VllmError(f"{name} {state}; its log is {log}:\n{tail}")

    def _proxy_script(self) -> str:
        """The proxy's path on the host: the packaged copy here, else one written to its home directory."""
        if self.host.local:
            return str(PROXY_SCRIPT)
        path = f"{self.host.home}/{REMOTE_PROXY_SCRIPT}"
        if not (written := self.host.write_file(path, PROXY_SCRIPT.read_text(encoding="utf-8"))).success:
            raise VllmError(f"could not copy {PROXY_SCRIPT.name} to {self.host}:{path}: {written.stderr}")
        return path
