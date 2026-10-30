# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Unit tests for the model fit checks."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from production_test_framework.agentx import model_fit
from production_test_framework.agentx.model_fit import (
    GIB,
    disk_shortfall,
    gpu_shortfall,
    model_footprint,
    rope_override_problem,
)

#: Shaped like Qwen3-8B: 36 layers, 8 KV heads of 128, bfloat16.
CONFIG = {
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 4096,
    "torch_dtype": "bfloat16",
}

#: A repo with a second copy of its weights in original/, as gpt-oss-120b has, and a README
#: vLLM does not download.
SIBLINGS = [
    {"rfilename": "model-00001.safetensors", "size": 6 * 10**9},
    {"rfilename": "model-00002.safetensors", "size": 4 * 10**9},
    {"rfilename": "original/model.safetensors", "size": 10 * 10**9},
    {"rfilename": "config.json", "size": 1000},
    {"rfilename": "README.md", "size": 5000},
]


def _hf(url, headers, timeout):
    response = MagicMock()
    response.json.return_value = CONFIG if url.endswith("config.json") else {"siblings": SIBLINGS}
    return response


@pytest.fixture
def hf():
    with patch.object(model_fit.requests, "get", side_effect=_hf) as get:
        yield get


def _cache(tmp_path, files):
    """A Hugging Face cache holding *files* ({name: content}) in one snapshot of acme/model."""
    snapshot = tmp_path / "hub" / "models--acme--model" / "snapshots" / "abc"
    for name, content in files.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


class TestModelFootprint:
    def test_counts_what_vllm_downloads_and_loads(self, hf, tmp_path):
        footprint = model_footprint("acme/model", tmp_path)

        assert footprint.download_bytes == 20 * 10**9 + 1000
        assert footprint.weight_bytes == 10 * 10**9
        assert footprint.missing_bytes == footprint.download_bytes

    def test_files_already_cached_are_not_counted_again(self, hf, tmp_path, monkeypatch):
        monkeypatch.setitem(SIBLINGS[3], "size", len("{}"))
        cache = _cache(tmp_path, {"config.json": "{}"})
        footprint = model_footprint("acme/model", cache)
        assert footprint.cached_bytes == 2

    def test_sends_the_token_for_gated_models(self, hf, tmp_path):
        model_footprint("acme/model", tmp_path, token="secret")
        assert hf.call_args.kwargs["headers"] == {"Authorization": "Bearer secret"}

    def test_falls_back_to_the_cache_when_the_api_is_unreachable(self, tmp_path):
        import json

        cache = _cache(tmp_path, {"config.json": json.dumps(CONFIG), "model.safetensors": "x" * 100})
        with patch.object(model_fit.requests, "get", side_effect=requests.ConnectionError("offline")):
            footprint = model_footprint("acme/model", cache)
        assert footprint.weight_bytes == 100
        assert footprint.missing_bytes == 0

    def test_unknown_without_the_api_or_a_cached_config(self, tmp_path):
        with patch.object(model_fit.requests, "get", side_effect=requests.ConnectionError("offline")):
            assert model_footprint("acme/model", tmp_path) is None

    @pytest.mark.parametrize(("tensor_parallel", "kv_heads_per_gpu"), [(1, 8), (2, 4), (16, 1)])
    def test_kv_heads_split_over_gpus_but_never_below_one(self, hf, tmp_path, tensor_parallel, kv_heads_per_gpu):
        footprint = model_footprint("acme/model", tmp_path)
        assert footprint.kv_bytes_per_token(tensor_parallel) == 2 * 36 * kv_heads_per_gpu * 128 * 2


class TestShortfalls:
    @pytest.fixture
    def footprint(self, hf, tmp_path):
        return model_footprint("acme/model", tmp_path)

    def test_disk_without_room_for_the_download(self, footprint, tmp_path):
        message = disk_shortfall(footprint, tmp_path, free_bytes=5 * 10**9)
        assert "needs 20.0 GB more" in message and "5.0 GB free" in message
        assert disk_shortfall(footprint, tmp_path, free_bytes=25 * 10**9) is None

    def test_weights_that_do_not_fit(self, footprint):
        message = gpu_shortfall(
            footprint, gpu_bytes=8 * GIB, tensor_parallel=1, memory_utilization=0.9, max_model_len=4096
        )
        assert "does not fit" in message

    def test_weights_that_fit_but_leave_too_little_kv_cache(self, footprint):
        message = gpu_shortfall(
            footprint, gpu_bytes=16 * GIB, tensor_parallel=2, memory_utilization=0.9, max_model_len=262144
        )
        assert "for the KV cache, but one 262,144-token request needs" in message

    def test_room_for_weights_and_one_full_request(self, footprint):
        assert (
            gpu_shortfall(footprint, gpu_bytes=16 * GIB, tensor_parallel=2, memory_utilization=0.9, max_model_len=65536)
            is None
        )


YARN = {"rope_parameters": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768}}


def _footprint(**config):
    return model_fit.ModelFootprint("acme/model", 0, 0, 0, {**CONFIG, **config})


class TestRopeOverride:
    def test_an_override_a_long_context_model_does_not_need_is_commented_out(self):
        message = rope_override_problem(
            _footprint(max_position_embeddings=131072, rope_scaling={"rope_type": "yarn", "factor": 32.0}),
            {"hf_overrides": YARN},
            65536,
            profile="agentx/muscae.yaml",
        )
        assert "handles 131,072 tokens on its own" in message
        assert "Comment out (or delete) these lines in agentx/muscae.yaml" in message
        assert message.endswith(
            "# vllm_args:\n#   hf_overrides:\n#     rope_parameters: "
            "{rope_type: yarn, factor: 2.0, original_max_position_embeddings: 32768}"
        )

    def test_other_flags_stay_when_the_override_is_commented_out(self):
        message = rope_override_problem(
            _footprint(max_position_embeddings=131072), {"max_num_seqs": 16, "hf_overrides": YARN}, 65536
        )
        assert message.endswith(
            "vllm_args:\n  max_num_seqs: 16\n#   hf_overrides:\n#     rope_parameters: "
            "{rope_type: yarn, factor: 2.0, original_max_position_embeddings: 32768}"
        )

    def test_a_short_context_model_without_an_override_is_told_the_lines_to_add(self):
        message = rope_override_problem(
            _footprint(max_position_embeddings=40960, rope_theta=1000000), {}, 65536, min_context=44048
        )
        assert "handles 40,960 tokens on its own, less than max_model_len 65,536" in message
        assert (
            "vllm_args:\n  hf_overrides:\n    rope_parameters: {rope_type: yarn, factor: 1.6, "
            "original_max_position_embeddings: 40960, rope_theta: 1000000}"
        ) in message
        assert "max_model_len it reaches" not in message, "40,960 is under the 44,048 AgentX needs"

    def test_a_lower_max_model_len_is_offered_when_it_still_holds_enough(self):
        message = rope_override_problem(_footprint(max_position_embeddings=40960), {}, 65536, min_context=32768)
        assert message.endswith("or a max_model_len it reaches:\n\nvllm_args:\n  max_model_len: 40960")

    @pytest.mark.parametrize(
        ("native", "vllm_args"),
        [(40960, {"hf_overrides": YARN}), (131072, {}), (131072, {"max_num_seqs": 16})],
    )
    def test_a_profile_that_suits_the_model_passes(self, native, vllm_args):
        assert rope_override_problem(_footprint(max_position_embeddings=native), vllm_args, 65536) is None
