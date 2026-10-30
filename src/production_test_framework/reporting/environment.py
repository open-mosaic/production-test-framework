# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.
"""
Description of the machine and the checkout a run happened on, for the report's header.

Everything here is deployment-agnostic: it describes the runner, the code under test, the
GPUs of a host reached over SSH and the workload profile a run targeted.
"""

import os
import platform
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..helper import is_localhost, run_command

__all__ = [
    "benchmark_option_rows",
    "ci_rows",
    "command_output",
    "display_path",
    "endpoint_address",
    "git_rows",
    "gpu_rows",
    "parse_driver_versions",
    "pool_summary",
    "profile_host",
    "profile_rows",
    "run_rows",
    "runner_rows",
    "selection_rows",
    "ssh_target",
]


def display_path(path: Path) -> str:
    """
    A path safe to put in a shared report, with the private prefix dropped: relative to the
    working directory when both are in the same checkout, else repo-relative.
    """
    resolved = Path(path).resolve()
    cwd = Path.cwd().resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists():
            if cwd.is_relative_to(parent) and resolved.is_relative_to(cwd):
                return str(resolved.relative_to(cwd))
            return str(Path(parent.name) / resolved.relative_to(parent))
    return f".../{resolved.parent.name}/{resolved.name}"


def command_output(argv: list[str], timeout: float = 5.0) -> tuple[str | None, str]:
    """
    ``(stdout, reason)`` for *argv* -- stdout is None on any failure, and *reason* says why.

    Never raises: the rows describe the environment, and a probe that cannot run must not be
    what fails the run. ``run_command`` already covers a timeout and a missing binary; any other
    ``OSError`` (a permission error, say) is caught here.
    """
    try:
        result = run_command(argv, timeout=timeout)
    except OSError as exc:
        return None, str(exc)

    if not result.success:
        detail = (result.stderr or result.stdout).splitlines()
        return None, detail[0] if detail else f"exit status {result.returncode}"
    return (result.stdout, "") if result.stdout else (None, "command produced no output")


# =============================================================================
# The runner, the checkout and the CI job
# =============================================================================


def runner_rows() -> list[list[str]]:
    """The machine the tests ran on, and the interpreter they ran under."""
    return [
        ["test runner", f"{platform.node()}  ({platform.system()} {platform.release()})"],
        ["python", platform.python_version()],
    ]


def git_rows(path: Path | None = None, label: str = "code under test") -> list[list[str]]:
    """
    Which commit of *path*'s checkout the run used, and whether it was modified.
    """
    root = Path(path) if path is not None else Path.cwd()
    describe, reason = command_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"])
    if describe is None:
        return [[label, f"not collected -- {reason}"]]

    branch, _ = command_output(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"])
    status, _ = command_output(["git", "-C", str(root), "status", "--porcelain"], timeout=15.0)
    state = "" if status is None else f", {len(status.splitlines())} file(s) modified"
    detail = f"{describe}" + (f" on {branch}" if branch and branch != "HEAD" else "") + state
    return [[label, detail]]


_CI_VARIABLES = (
    ("CI pipeline", "CI_PIPELINE_URL"),
    ("CI job", "CI_JOB_URL"),
    ("CI commit branch", "CI_COMMIT_REF_NAME"),
    ("CI runner", "CI_RUNNER_DESCRIPTION"),
)


def ci_rows() -> list[list[str]]:
    """Links back to the CI pipeline that produced this report, when there was one."""
    return [[label, os.environ[name]] for label, name in _CI_VARIABLES if os.environ.get(name)]


def selection_rows(config) -> list[list[str]]:
    """
    The marker expression the run selected on.

    A nightly report is read by someone asking "did the API tests pass?", and the answer only
    means something alongside what the run took the API tests to be on the night.
    """
    markers = config.getoption("markexpr", "")
    keywords = config.getoption("keyword", "")
    selection = " and ".join(part for part in (markers, keywords) if part)
    return [["selection", f"-m {selection}" if selection else "all collected tests"]]


def run_rows(config, checkout: Path | str | None = None) -> list[list[str]]:
    """
    What every suite's report says about the run itself: the selection, the commit of
    *checkout* (the pytest rootdir when not given), the runner and the CI job.
    """
    root = Path(checkout) if checkout else Path(config.rootpath)
    return selection_rows(config) + git_rows(root) + runner_rows() + ci_rows()


# =============================================================================
# GPU, CUDA and kernel-module versions, collected over SSH
# =============================================================================


def parse_driver_versions(version_text: str) -> dict[str, str]:
    """
    Pull the version fields out of ``nvidia-smi --version`` or a plain ``nvidia-smi`` header.
    """
    raw: dict[str, str] = {}
    for key, pattern in (
        ("nvidia-smi", r"NVIDIA-SMI version\s*:\s*(\S+)"),
        ("NVML", r"NVML version\s*:\s*(\S+)"),
        ("KMD", r"KMD version\s*:\s*(\S+)"),
        ("cuda_umd", r"CUDA UMD version\s*:\s*(\S+)"),
        ("driver_field", r"DRIVER version\s*:\s*(\S+)"),
        ("cuda_field", r"^\s*CUDA version\s*:\s*(\S+)"),
        ("driver_header", r"Driver Version:\s*(\S+)"),
        ("cuda_header", r"CUDA Version:\s*(\S+)"),
    ):
        if match := re.search(pattern, version_text, re.IGNORECASE | re.MULTILINE):
            value = match.group(1)

            if not value.lower().startswith("deprecated"):
                raw[key] = value

    found = {key: raw[key] for key in ("nvidia-smi", "NVML", "KMD") if key in raw}

    if driver := raw.get("driver_field") or raw.get("driver_header") or raw.get("KMD"):
        found["driver"] = driver
    if cuda := raw.get("cuda_umd") or raw.get("cuda_field") or raw.get("cuda_header"):
        found["CUDA"] = cuda
    return found


_GPU_INFO_SCRIPT = (
    "nvidia-smi --version 2>/dev/null || nvidia-smi 2>/dev/null; "
    "echo '===KMD==='; cat /proc/driver/nvidia/version 2>/dev/null; "
    "echo '===GPUS==='; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null"
)


def ssh_target(host: str) -> str:
    """
    ``user@host`` for the probe, or a bare host so ``~/.ssh/config`` decides.
    """
    user = os.getenv("GPU_INFO_SSH_USER") or os.getenv("SSH_USER")
    return f"{user}@{host}" if user else host


def _remote_output(host: str, script: str, timeout: float = 25.0) -> tuple[str | None, str]:
    """
    Run *script* on *host* over SSH, returning ``(stdout, reason)``.
    """
    return command_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "StrictHostKeyChecking=accept-new",
            ssh_target(host),
            script,
        ],
        timeout=timeout,
    )


def gpu_rows(host: str | None) -> list[list[str]]:
    """
    GPU, CUDA and kernel-module versions from *host*.
    """
    host = os.getenv("GPU_INFO_HOST") or host
    if not host:
        return [["GPU driver", "not collected -- no head node host to query (set GPU_INFO_HOST)"]]

    local = is_localhost(host)
    label = f"this host {platform.node()}" if local else f"head node {host}"
    if local:
        output, reason = command_output(["sh", "-c", _GPU_INFO_SCRIPT], timeout=25.0)
    else:
        output, reason = _remote_output(host, _GPU_INFO_SCRIPT)
    if output is None:
        detail = reason.rstrip(". ")
        if local:
            return [["GPU driver", f"not collected -- {detail}"]]
        auth_failure = any(word in reason.lower() for word in ("denied", "authentication"))
        hint = (
            " Set GPU_INFO_SSH_USER if the cluster login differs."
            if auth_failure and not os.getenv("GPU_INFO_SSH_USER")
            else ""
        )
        return [["GPU driver", f"not collected -- ssh {ssh_target(host)}: {detail}.{hint}"]]

    version_text, _, rest = output.partition("===KMD===")
    kmd_text, _, inventory = rest.partition("===GPUS===")

    versions = parse_driver_versions(version_text)

    if "KMD" not in versions and (match := re.search(r"NVRM version:.*?(\d+\.\d+(?:\.\d+)?)", kmd_text)):
        versions["KMD"] = match.group(1)

    rows: list[list[str]] = []
    for name in ("driver", "CUDA", "KMD", "nvidia-smi", "NVML"):
        if name in versions:
            rows.append([f"{name} version ({label})", versions[name]])
    if "open kernel module" in kmd_text.lower():
        rows.append([f"KMD variant ({label})", "open kernel modules"])

    gpu_lines = [line for line in inventory.splitlines() if "," in line]
    if gpu_lines:
        names = sorted({line.split(",", 2)[1].strip() for line in gpu_lines})
        rows.append([f"GPUs ({label})", f"{len(gpu_lines)}x {', '.join(names)}"])

    if not rows:
        rows.append(["GPU driver", f"not collected -- {label} answered but reported no NVIDIA driver"])
    return rows


# =============================================================================
# The workload profile
# =============================================================================
#
# A profile is the YAML a suite targets a deployment with.
# The rows here read it as a plain mapping, so a suite's own loader -- strict or lenient --
# only has to hand over the parsed document. A field the profile leaves out omits its row.


def pool_summary(pool: Mapping[str, Any]) -> str:
    """One line describing a prefill or decode pool, omitting what the profile left unset."""
    parts = []
    if (nodes := pool.get("nodes")) is not None:
        parts.append(f"{nodes} node(s)")
    if (workers := pool.get("workers")) is not None:
        parts.append(f"{workers} worker(s)")
    if (tp := pool.get("tensor_parallel")) is not None:
        parts.append(f"TP={tp}")
    if (ep := pool.get("expert_parallel")) is not None:
        parts.append(f"EP={ep}")
    parts.append("spans nodes" if pool.get("spans_nodes") else "in-node")
    return ", ".join(parts)


def endpoint_address(profile: Mapping[str, Any]) -> str:
    """``host:port``, or the frontend URL a disaggregated deployment sits behind."""
    endpoint = profile.get("endpoint") or {}
    if base_url := endpoint.get("base_url"):
        return str(base_url)
    host, port = endpoint.get("host"), endpoint.get("port")
    return f"{host}:{port}" if host and port else str(host or "")


def profile_host(profile: Mapping[str, Any]) -> str | None:
    """The host to collect GPU versions from: the endpoint's host, or its base URL's."""
    endpoint = profile.get("endpoint") or {}
    if base_url := endpoint.get("base_url"):
        return urlparse(str(base_url)).hostname or None
    return endpoint.get("host") or None


def profile_rows(
    profile: Mapping[str, Any],
    prometheus_url: str,
    grafana_url: str,
    *,
    endpoint: str | None = None,
    deployment: str | None = None,
) -> list[list[str]]:
    """
    What the run was pointed at, as ``(property, value)`` rows.

    Ordered hardware first, then how it is serving, then where to reach it, so every suite's
    report reads the same way. *endpoint* replaces the address the profile names -- for a
    suite that lets the environment override it -- and *deployment*, when given, says how the
    deployment was stood up.
    """
    hardware = profile.get("hardware") or {}
    serving = profile.get("serving") or {}
    coverage = profile.get("coverage") or {}
    timeouts = profile.get("timeouts") or {}

    rows: list[list[str]] = [["profile", f"{profile['name']}  ({display_path(Path(profile['path']))})"]]
    if description := profile.get("description"):
        rows.append(["description", str(description)])

    machines, per_machine = hardware.get("machines"), hardware.get("gpus_per_machine")
    if machines is not None:
        rows.append(["machines", str(machines)])
    if per_machine is not None:
        rows.append(["GPUs per machine", str(per_machine)])
    if machines is not None and per_machine is not None:
        rows.append(["GPUs total", str(machines * per_machine)])
    if sku := hardware.get("sku"):
        rows.append(["GPU SKU", str(sku)])

    if mode := serving.get("mode"):
        rows.append(["serving mode", str(mode)])
    if model := serving.get("model"):
        rows.append(["model", str(model)])
    if (tp := serving.get("tensor_parallel")) is not None:
        rows.append(["tensor parallel", str(tp)])
    if (ep := serving.get("expert_parallel")) is not None:
        rows.append(["expert parallel", str(ep)])

    # Only a disaggregated deployment has these, which is what makes them worth a row: they
    # say how the prefill and decode halves were sized.
    if prefill := serving.get("prefill"):
        rows.append(["prefill pool", pool_summary(prefill)])
    if decode := serving.get("decode"):
        rows.append(["decode pool", pool_summary(decode)])
    if kv := serving.get("kv_transfer"):
        rows.append(["KV transfer", str(kv)])

    if coverage:
        rows.append(
            [
                "coverage expected",
                (
                    f"{coverage.get('hosts')} host(s), {coverage.get('gpus')} GPU(s), "
                    f"{coverage.get('communicators')} comm(s)"
                ),
            ]
        )
    if expected := profile.get("expected_metrics"):
        rows.append(["expected metrics", str(expected)])

    if endpoint := endpoint or endpoint_address(profile):
        rows.append(["endpoint", endpoint])
    if deployment:
        rows.append(["deployment", deployment])
    rows += [["Prometheus", prometheus_url], ["Grafana", grafana_url]]

    if timeouts:
        rows.append(
            [
                "timeouts",
                (
                    f"workload {timeouts.get('workload')}s, "
                    f"metrics {timeouts.get('metrics_available')}s, "
                    f"quiesce {timeouts.get('quiesce')}s"
                ),
            ]
        )
    return rows


# =============================================================================
# Benchmark options
# =============================================================================

#: Where the benchmark writes its own result file inside its container.
_OMITTED_BENCHMARK_OPTIONS = frozenset({"result_dir", "result_filename"})

_BENCHMARK_OPTION_ORDER = (
    "num_prompts",
    "max_concurrency",
    "request_rate",
    "random_input_len",
    "random_output_len",
    "num_warmups",
    "dataset_name",
    "model",
    "backend",
    "ignore_eos",
)


def benchmark_option_rows(options: Mapping[str, Any]) -> list[list[str]]:
    """
    Benchmark options as (option, value, flag) rows, load-shaping options first.

    The flag column is not decoration: these keys are converted to ``benchmark_serving``
    arguments generically, so showing the conversion is what lets a reader reproduce the run
    by hand.
    """

    def rank(item: tuple[int, str]) -> tuple[int, int]:
        position, key = item
        listed = _BENCHMARK_OPTION_ORDER.index(key) if key in _BENCHMARK_OPTION_ORDER else len(_BENCHMARK_OPTION_ORDER)
        return listed, position

    rows: list[list[str]] = []
    for _, key in sorted(enumerate(options), key=rank):
        if key in _OMITTED_BENCHMARK_OPTIONS:
            continue
        value = options[key]
        flag = "--" + key.replace("_", "-")
        if value is None:
            rows.append([key, "unset", "(omitted)"])
        elif isinstance(value, bool):
            rows.append([key, "true" if value else "false", flag if value else "(omitted)"])
        else:
            rows.append([key, str(value), f"{flag} {value}"])
    return rows
