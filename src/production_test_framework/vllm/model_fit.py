# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
Whether a Hugging Face model fits a machine for vLLM: its download in the disk's free space,
and its weights plus the KV cache of one full-length request in each GPU's memory.

Both are estimates made before anything is downloaded or started, so a run that cannot work
fails at once with the reason, rather than partway through a download or after loading the
weights. The sizes come from the Hugging Face API, or from the cache when the model is already
there and the API cannot be reached. The cache can be on another machine, read over ssh.
"""

import fnmatch
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml

from ..host import Host

__all__ = [
    "HF_URL",
    "ModelFootprint",
    "disk_shortfall",
    "gpu_shortfall",
    "model_footprint",
    "rope_override_problem",
]

#: Where a RoPE override sits in vLLM's --hf-overrides: rope_parameters, or the older name.
ROPE_KEYS = ("rope_parameters", "rope_scaling")

HF_URL = "https://huggingface.co"
HF_TIMEOUT = 30.0

GIB = 1024**3

#: What vLLM downloads from a repo: the first weight format it has, in this order (matched
#: anywhere in the repo, so gpt-oss-120b's original/ copy of its weights comes too), and the
#: small files beside them.
WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt")
OTHER_PATTERNS = ("*.json", "*.model", "*.txt", "*.jinja", "*.tiktoken")

#: Memory vLLM's own allocations take on each GPU besides weights and KV cache: activations
#: and buffers, with --enforce-eager.
GPU_OVERHEAD_BYTES = int(0.5 * GIB)

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


@dataclass(frozen=True)
class ModelFootprint:
    """What a model takes: on disk, in GPU memory, and in KV cache per token."""

    repo: str
    download_bytes: int
    cached_bytes: int
    weight_bytes: int
    config: dict[str, Any]

    @property
    def missing_bytes(self) -> int:
        return max(self.download_bytes - self.cached_bytes, 0)

    def kv_bytes_per_token(self, tensor_parallel: int) -> int:
        """KV cache per token on each GPU: keys and values for every layer's share of KV heads."""
        config = self.config.get("text_config") or self.config
        heads = config["num_attention_heads"]
        kv_heads = config.get("num_key_value_heads") or heads
        head_dim = config.get("head_dim") or config["hidden_size"] // heads
        dtype = str(config.get("torch_dtype") or config.get("dtype") or "bfloat16")
        # A KV head is never split: past one per GPU, they are replicated.
        per_gpu = math.ceil(kv_heads / tensor_parallel)
        return 2 * config["num_hidden_layers"] * per_gpu * head_dim * _DTYPE_BYTES.get(dtype, 2)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _vllm_files(sizes: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """``(downloaded, loaded)``: the files vLLM downloads, and of those the weights it loads."""
    for pattern in WEIGHT_PATTERNS:
        weights = {name: size for name, size in sizes.items() if fnmatch.fnmatch(name, pattern)}
        if weights:
            break
    others = {name: size for name, size in sizes.items() if _matches(name, OTHER_PATTERNS)}
    loaded = {name: size for name, size in weights.items() if "/" not in name}
    return {**others, **weights}, loaded


def _snapshot(cache_dir: Path, repo: str) -> Path | None:
    """The most recent local snapshot of *repo* in the Hugging Face cache, if any."""
    snapshots = cache_dir / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    found = sorted(snapshots.glob("*"), key=lambda path: path.stat().st_mtime) if snapshots.is_dir() else []
    return found[-1] if found else None


def _cached_sizes(snapshot: Path | None) -> dict[str, int]:
    """Each complete file in *snapshot*: a snapshot's files link to blobs, missing until downloaded."""
    if snapshot is None:
        return {}
    return {str(path.relative_to(snapshot)): path.stat().st_size for path in snapshot.rglob("*") if path.is_file()}


#: Prints the newest snapshot directory under $1, then each complete file in it as "path<TAB>size".
_REMOTE_SNAPSHOT_SCRIPT = (
    'snapshot=$(ls -1td "$1"/*/ 2>/dev/null | head -n 1); [ -n "$snapshot" ] || exit 0; '
    'echo "$snapshot"; cd "$snapshot" && find -L . -type f -printf \'%P\\t%s\\n\''
)


def _remote_cache(host: Host, cache_dir: str, repo: str) -> tuple[dict[str, int], Callable[[], dict[str, Any]]]:
    """The cached sizes of *repo* in *cache_dir* on *host*, and how to read its cached config."""
    result = host.shell(_REMOTE_SNAPSHOT_SCRIPT, f"{cache_dir}/hub/models--{repo.replace('/', '--')}/snapshots")
    snapshot, *files = result.stdout.splitlines() if result.success and result.stdout else [""]
    cached = {}
    for line in files:
        name, _, size = line.rpartition("\t")
        if size.isdigit():
            cached[name] = int(size)
    return cached, lambda: json.loads(host.run(["cat", f"{snapshot.rstrip('/')}/config.json"]).stdout)


def _local_cache(cache_dir: Path, repo: str) -> tuple[dict[str, int], Callable[[], dict[str, Any]]]:
    snapshot = _snapshot(cache_dir, repo)
    return _cached_sizes(snapshot), lambda: json.loads((snapshot / "config.json").read_text(encoding="utf-8"))


def model_footprint(
    repo: str,
    cache_dir: Path | str,
    token: str | None = None,
    timeout: float = HF_TIMEOUT,
    host: Host | None = None,
) -> ModelFootprint | None:
    """
    *repo*'s footprint, from the Hugging Face API, else from the copy in *cache_dir* (the
    ``HF_HOME`` directory, on *host* when given) when it has its config. None when neither can say.
    """
    if host is None or host.local:
        cached, cached_config = _local_cache(Path(cache_dir), repo)
    else:
        cached, cached_config = _remote_cache(host, str(cache_dir), repo)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        info = requests.get(f"{HF_URL}/api/models/{repo}?blobs=true", headers=headers, timeout=timeout)
        info.raise_for_status()
        sizes = {sibling["rfilename"]: int(sibling.get("size") or 0) for sibling in info.json()["siblings"]}
        config_response = requests.get(f"{HF_URL}/{repo}/resolve/main/config.json", headers=headers, timeout=timeout)
        config_response.raise_for_status()
        config = config_response.json()
    except requests.RequestException, ValueError, KeyError:
        if "config.json" not in cached:
            return None
        sizes = cached
        config = cached_config()

    downloaded, loaded = _vllm_files(sizes)
    return ModelFootprint(
        repo=repo,
        download_bytes=sum(downloaded.values()),
        cached_bytes=sum(size for name, size in downloaded.items() if cached.get(name) == size),
        weight_bytes=sum(loaded.values()),
        config=config,
    )


def _gb(value: float) -> str:
    return f"{value / 1e9:,.1f} GB"


def _gib(value: float) -> str:
    return f"{value / GIB:,.1f} GiB"


def disk_shortfall(footprint: ModelFootprint, cache_dir: Path | str, free_bytes: int) -> str | None:
    """Why the model's download cannot fit in *free_bytes* at *cache_dir*, or None if it can."""
    if footprint.missing_bytes <= free_bytes:
        return None
    return (
        f"{footprint.repo} needs {_gb(footprint.missing_bytes)} more in the Hugging Face cache {cache_dir} "
        f"({_gb(footprint.download_bytes)} of files, {_gb(footprint.cached_bytes)} already there), but that "
        f"disk has {_gb(free_bytes)} free. Free up space there, set HF_HOME to a disk with room, or use a "
        "smaller model."
    )


def gpu_shortfall(
    footprint: ModelFootprint,
    *,
    gpu_bytes: int,
    tensor_parallel: int,
    memory_utilization: float,
    max_model_len: int,
    worker: str = "the server",
) -> str | None:
    """
    Why the model cannot run on GPUs of *gpu_bytes* each at *tensor_parallel*, or None if it can:
    its weights do not fit, or what they leave cannot hold the KV cache of one *max_model_len*
    request, which vLLM needs before it will start.
    """
    usable = gpu_bytes * memory_utilization
    weights = footprint.weight_bytes / tensor_parallel
    kv_room = usable - weights - GPU_OVERHEAD_BYTES
    shape = (
        f"{footprint.repo} on {worker}: each of its {tensor_parallel} GPU(s) has {_gib(gpu_bytes)}, "
        f"{memory_utilization:.0%} of it usable ({_gib(usable)})"
    )
    if weights + GPU_OVERHEAD_BYTES > usable:
        return (
            f"{shape}, but would hold {_gib(weights)} of weights ({_gb(footprint.weight_bytes)} split "
            f"{tensor_parallel} ways). The model does not fit: use more GPUs (tensor_parallel), larger "
            "ones, or a smaller model."
        )
    kv_needed = max_model_len * footprint.kv_bytes_per_token(tensor_parallel)
    if kv_needed > kv_room:
        return (
            f"{shape}. Its {_gib(weights)} of weights leave {_gib(max(kv_room, 0))} for the KV cache, but one "
            f"{max_model_len:,}-token request needs {_gib(kv_needed)}. Lower max_model_len in vllm_args, use "
            "more GPUs (tensor_parallel), or a smaller model."
        )
    return None


def _yaml(mapping: dict[str, Any], indent: int = 0) -> list[str]:
    """*mapping* as YAML lines indented by *indent* spaces, innermost mappings on one line."""

    def inline(value: Any) -> str:
        return yaml.safe_dump(value, default_flow_style=True, sort_keys=False, width=1000).removesuffix("...\n").strip()

    lines = []
    for key, value in mapping.items():
        if isinstance(value, dict) and any(isinstance(inner, dict) for inner in value.values()):
            lines += [f"{' ' * indent}{key}:", *_yaml(value, indent + 2)]
        else:
            lines.append(f"{' ' * indent}{key}: {inline(value)}")
    return lines


def _section(vllm_args: dict[str, Any]) -> str:
    """A profile's vllm_args section, as it would be written there."""
    return "\n".join(["vllm_args:", *_yaml(vllm_args, 2)])


def _commented_out(vllm_args: dict[str, Any], rope_key: str) -> str:
    """The profile's vllm_args with its RoPE override, and any key left empty by it, commented out."""
    hf_overrides = vllm_args["hf_overrides"]
    hf_left = {key: value for key, value in hf_overrides.items() if key != rope_key}
    args_left = len(vllm_args) > 1 or hf_left
    lines = [("" if args_left else "# ") + "vllm_args:"]
    for key, value in vllm_args.items():
        if key != "hf_overrides":
            lines += _yaml({key: value}, 2)
            continue
        lines.append(("  " if hf_left else "# ") + ("hf_overrides:" if hf_left else "  hf_overrides:"))
        for sub_key, sub_value in hf_overrides.items():
            sub_lines = _yaml({sub_key: sub_value}, 4)
            lines += [f"# {line}" if sub_key == rope_key else line for line in sub_lines]
    return "\n".join(lines)


def rope_override_problem(
    footprint: ModelFootprint,
    profile_vllm_args: dict[str, Any],
    max_model_len: int | None,
    *,
    min_context: int = 0,
    profile: str = "the profile",
) -> str | None:
    """
    Why the RoPE override in *profile_vllm_args* (the profile's own ``vllm_args``) does not suit
    the model, with the lines *profile* needs instead; None when it suits.

    An override is wrong for a model whose own context already reaches *max_model_len*: vLLM
    puts it in place of the model's own RoPE settings, which can break the model outright.
    Without one, a model whose context falls short of *max_model_len* will not start; the fix
    is an override, or a max_model_len it reaches when that still holds *min_context* tokens.
    """
    config = footprint.config.get("text_config") or footprint.config
    native = config.get("max_position_embeddings")
    if not native or not max_model_len:
        return None
    hf_overrides = profile_vllm_args.get("hf_overrides") or {}
    rope_key = next((key for key in ROPE_KEYS if hf_overrides.get(key)), None)
    own = config.get("rope_parameters") or config.get("rope_scaling")
    own_text = f" (its own RoPE: {', '.join(f'{k} {v}' for k, v in own.items())})" if own else ""

    if rope_key and native >= max_model_len:
        return (
            f"vllm_args.hf_overrides.{rope_key} in {profile} is not for {footprint.repo}, which handles "
            f"{native:,} tokens on its own{own_text}, already more than max_model_len {max_model_len:,}. vLLM "
            "would put the override in place of the model's own settings, and the model can fail to load "
            f"(gpt-oss models fail with KeyError: 'beta_fast'). Comment out (or delete) these lines in {profile}:\n\n"
            f"{_commented_out(profile_vllm_args, rope_key)}"
        )
    if not rope_key and native < max_model_len:
        stretched = {
            **profile_vllm_args,
            "hf_overrides": {
                **hf_overrides,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": round(max_model_len / native, 2),
                    "original_max_position_embeddings": native,
                    "rope_theta": config.get("rope_theta", 10000),
                },
            },
        }
        reachable = native >= min_context
        options = [
            f"{'either ' if reachable else ''}a RoPE override that stretches it to {max_model_len:,}, using the "
            f"model card's recommended values where it gives them:\n\n{_section(stretched)}"
        ]
        if reachable:
            options.append(
                f"or a max_model_len it reaches:\n\n{_section({**profile_vllm_args, 'max_model_len': native})}"
            )
        return (
            f"{footprint.repo} handles {native:,} tokens on its own{own_text}, less than max_model_len "
            f"{max_model_len:,}, so vLLM will not start it. Set in {profile} " + "\n\n".join(options)
        )
    return None
