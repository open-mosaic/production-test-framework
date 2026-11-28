# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
Start vLLM in docker on a machine for a test, and remove it after.

A deployment is one server (aggregated), or prefill and decode workers on their own GPUs and
ports behind vLLM Router (https://github.com/vllm-project/router) on the endpoint's port
(disaggregated), which hands each request's KV cache from prefill to decode over NIXL. It all runs on a
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
    "AMD",
    "GPU_VENDORS",
    "KV_CONNECTORS",
    "NVIDIA",
    "ROUTER_IMAGE",
    "VLLM_IMAGE",
    "Gpu",
    "GpuVendor",
    "HostGpus",
    "Pool",
    "VllmDeployment",
    "VllmError",
    "VllmWorker",
    "aggregated_workers",
    "check_docker",
    "check_image_fits",
    "check_model_fits",
    "detect_gpus",
    "disaggregated_workers",
    "gpu_memory",
    "hf_cache",
    "needed_gpus",
    "serves",
]

_log = logging.getLogger(__name__)

VLLM_IMAGE = "openmosaic/mosaic-vllm:latest"
#: vLLM's ROCm build, for AMD GPUs. Not rocm/vllm:latest (vLLM 0.32): its nixl-rocm 1.4.0 moves
#: disaggregated KV transfers through host memory, not GPU to GPU; 0.34's nixl-rocm 1.5.0 does not.
ROCM_VLLM_IMAGE = "vllm/vllm-openai-rocm:v0.34.0"

#: vLLM's own default, for args that drop gpu_memory_utilization.
VLLM_MEMORY_UTILIZATION = 0.9

PROBE_TIMEOUT = 10
STARTUP_TIMEOUT = 1200.0
POLL_INTERVAL = 5.0

REMOVE_TIMEOUT = 120.0

WORKER_BASE_PORTS = {"prefill": 8100, "decode": 8200}
NIXL_SIDE_CHANNEL_BASE_PORT = 5600

KV_CONNECTORS = {"nixl": "NixlConnector"}

ROUTER_IMAGE = "vllm/vllm-router:v0.1.17"
#: Workers in each pool taken in turn, and no retries or circuit breaking, so a run measures the
#: workers rather than the routing and every failed request counts.
ROUTER_POLICY = "round_robin"
ROUTER_REQUEST_TIMEOUT = 7200

NVIDIA_HINT = (
    "Check on the target that: nvidia-smi lists the GPUs (the NVIDIA driver is installed and loaded); "
    "docker is not the snap package (snap list docker), which cannot see the host's driver libraries; "
    "the NVIDIA Container Toolkit is installed and set up for docker (sudo nvidia-ctk runtime configure "
    "--runtime=docker && sudo systemctl restart docker); and docker run --rm --gpus all ubuntu nvidia-smi "
    "lists the GPUs."
)

AMD_HINT = (
    "Check on the target that: amd-smi lists the GPUs (the amdgpu driver is loaded); /dev/kfd and /dev/dri "
    "exist and this user is in the render and video groups; and docker run --rm --device /dev/kfd --device "
    "/dev/dri rocm/rocm-terminal rocminfo lists the GPUs."
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
    """
    A GPU vendor: the command-line tool its driver installs, the vLLM image built for its GPUs,
    and what to check when the GPUs cannot be used.
    """

    name: str
    tool: str
    image: str
    hint: str


NVIDIA = GpuVendor("NVIDIA", "nvidia-smi", VLLM_IMAGE, NVIDIA_HINT)
AMD = GpuVendor("AMD", "amd-smi", ROCM_VLLM_IMAGE, AMD_HINT)

#: In the order they are tried: a host can have a vendor's tool without its GPUs, e.g. nvidia-smi
#: installed with no NVIDIA driver loaded, so the first whose tool lists GPUs is the host's.
GPU_VENDORS = (NVIDIA, AMD)


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    memory_bytes: int


@dataclass(frozen=True)
class HostGpus:
    """The GPUs on a host, and the vendor whose tool listed them."""

    vendor: GpuVendor
    gpus: tuple[Gpu, ...]

    @property
    def memory(self) -> dict[int, int]:
        """Each GPU's memory in bytes, by index."""
        return {gpu.index: gpu.memory_bytes for gpu in self.gpus}

    @property
    def name(self) -> str:
        """The first GPU's name, as its vendor's tool prints it."""
        return self.gpus[0].name


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
        """The worker's URL from its own host, as the router beside it reaches it."""
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


def _nvidia_gpus(host: Host) -> list[Gpu]:
    listed = host.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], timeout=30
    )
    if not listed.success:
        raise VllmError(listed.stderr or listed.stdout or f"exit status {listed.returncode}")
    gpus = []
    for line in listed.stdout.splitlines():
        index, name, mib = (part.strip() for part in (line.split(",") + ["", ""])[:3])
        if index.isdigit() and mib.isdigit():
            gpus.append(Gpu(int(index), name, int(mib) * 1024**2))
    return gpus


def parse_amd_smi_json(text: str) -> list[dict[str, Any]]:
    """
    amd-smi's ``--json`` output, one entry per GPU: amd-smi 26 wraps them in ``gpu_data``, older
    releases print the list itself, and either may print warnings first. Raises ValueError.
    """
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        raise ValueError("no JSON in its output")
    data = json.loads(text[min(starts) :])
    entries = data.get("gpu_data") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("no GPU list in its JSON")
    return [entry for entry in entries if isinstance(entry, dict)]


def amd_smi_json(host: Host, *args: str) -> list[dict[str, Any]]:
    """``amd-smi <args> --json`` on *host*, one entry per GPU. Raises :class:`VllmError` when it fails."""
    listed = host.run(["amd-smi", *args, "--json"], timeout=30)
    if not listed.success:
        raise VllmError(listed.stderr or listed.stdout or f"exit status {listed.returncode}")
    try:
        return parse_amd_smi_json(listed.stdout)
    except ValueError as exc:
        raise VllmError(f"amd-smi {' '.join(args)}: {exc}") from exc


def amd_smi_number(value: Any) -> float | None:
    """A number as amd-smi prints one: ``{"value": 300, "unit": "W"}``, ``"300 W"``, ``300``; None for N/A."""
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, int | float):
        return float(value)
    match = re.match(r"\s*([\d.]+)", str(value or ""))
    return float(match.group(1)) if match else None


def _amd_gpus(host: Host) -> list[Gpu]:
    gpus = []
    for entry in amd_smi_json(host, "static", "--asic", "--vram"):
        mib = amd_smi_number((entry.get("vram") or {}).get("size"))
        if isinstance(entry.get("gpu"), int) and mib:
            name = (entry.get("asic") or {}).get("market_name") or "AMD GPU"
            gpus.append(Gpu(entry["gpu"], str(name), int(mib) * 1024**2))
    return gpus


_LISTERS = {NVIDIA.name: _nvidia_gpus, AMD.name: _amd_gpus}


def detect_gpus(host: Host) -> HostGpus:
    """
    The GPUs on *host*, from the first vendor in :data:`GPU_VENDORS` whose tool is installed and
    lists any. Raises :class:`VllmError` when none does, since docker's GPU runtime cannot give
    vLLM the GPUs either.
    """
    where = _where(host)
    installed = [vendor for vendor in GPU_VENDORS if host.which(vendor.tool)]
    if not installed:
        tools = " nor ".join(f"{vendor.tool} ({vendor.name})" for vendor in GPU_VENDORS)
        raise VllmError(f"No GPU driver found on {where}: neither {tools} is on PATH. Install the GPU driver.")
    problems = []
    for vendor in installed:
        try:
            gpus = _LISTERS[vendor.name](host)
        except VllmError as exc:
            problems.append(f"{vendor.tool} does not work on {where} ({exc}).")
            continue
        if gpus:
            return HostGpus(vendor, tuple(sorted(gpus, key=lambda gpu: gpu.index)))
        problems.append(f"{vendor.tool} works but lists no GPUs on {where}.")
    hints = " ".join(vendor.hint for vendor in installed)
    raise VllmError(f"{' '.join(problems)} So docker cannot give vLLM the GPUs. {hints}")


def gpu_memory(host: Host) -> dict[int, int]:
    """Each GPU's memory in bytes, by index, on *host*, as :func:`detect_gpus` finds them."""
    return detect_gpus(host).memory


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


def gpu_flags(vendor: GpuVendor, gpus: tuple[int, ...]) -> list[str]:
    """The ``docker run`` flags that give a container *gpus*: the NVIDIA runtime's, or ROCm's devices."""
    devices = ",".join(map(str, gpus))
    if vendor is AMD:
        return [
            "--device", "/dev/kfd",
            "--device", "/dev/dri",
            "--group-add", "video",
            "--security-opt", "seccomp=unconfined",
            # ROCm's IPC (HSA_ENABLE_IPC_MODE_LEGACY=0) fails, with hipIpcGetMemHandle: invalid argument,
            # in a second container whose processes have the same pids as the first's: share the host's.
            "--pid", "host",
            "--env", f"HIP_VISIBLE_DEVICES={devices}",
        ]  # fmt: skip
    return ["--gpus", f'"device={devices}"']


class VllmDeployment:
    """
    vLLM containers on *host* serving *model* at *url*: *workers* from :func:`aggregated_workers`
    or :func:`disaggregated_workers`, the latter behind vLLM Router on *url*'s port. *args* are the
    ``vllm serve`` flags beyond model, port and tensor parallelism, as ``{flag_name: value}``; a
    mapping goes as JSON. *env* is set in each vLLM container, over the image's own. Containers
    are named after *name* and their logs saved to *log_dir*.

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
        vendor: GpuVendor = NVIDIA,
        image: str | None = None,
        env: dict[str, Any] | None = None,
        router_image: str = ROUTER_IMAGE,
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
        self.vendor = vendor
        self.image = image or vendor.image
        self.env = env or {}
        self.router_image = router_image
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
            *(flag for key, value in self.env.items() for flag in ("--env", f"{key}={value}")),
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
            flags = [*gpu_flags(self.vendor, worker.gpus), *shared]
            if self.disaggregated:
                argv += ["--kv-transfer-config", json.dumps({"kv_connector": connector, "kv_role": "kv_both"})]
                flags += ["--env", f"VLLM_NIXL_SIDE_CHANNEL_PORT={worker.side_channel_port}"]
            name = unique_container_name(self.name if worker.role == "server" else f"{self.name}-{worker.role}")
            self.worker_containers.append(name)
            # openmosaic/mosaic-vllm's own entrypoint, made explicit for images without it, such as ROCm's.
            flags += ["--entrypoint", "/bin/bash"]
            self._run_container(name, flags, self.image, ["-c", shlex.join(argv)])
        for name, worker in zip(self.worker_containers, self.workers, strict=True):
            self._wait_until_serving(name, worker.url)

        if self.disaggregated:
            check_image_fits(self.host, self.router_image)
            router = unique_container_name(f"{self.name}-router")
            command = [
                "vllm-router", "--vllm-pd-disaggregation",
                *(arg for w in self.workers if w.role == "prefill" for arg in ("--prefill", w.local_url)),
                *(arg for w in self.workers if w.role == "decode" for arg in ("--decode", w.local_url)),
                "--host", "0.0.0.0",
                "--port", str(port),
                "--policy", ROUTER_POLICY,
                "--request-timeout-secs", str(ROUTER_REQUEST_TIMEOUT),
                "--disable-retries",
                "--disable-circuit-breaker",
            ]  # fmt: skip
            self._run_container(router, ["--network", "host"], self.router_image, command)
            self._wait_until_serving(router, self.url)

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

    def _run_container(self, name: str, flags: list[str], image: str, command: list[str]) -> None:
        run = docker_argv("run", "--detach", *flags, "--name", name, *label_args(name), image, *command)
        _log.info("Starting %s as %s on %s: %s", image, name, _where(self.host), " ".join(command))
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
            known = "nvidia-container" in error or "/dev/kfd" in error or "/dev/dri" in error
            hint = f"\n{self.vendor.hint}" if known else ""
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
