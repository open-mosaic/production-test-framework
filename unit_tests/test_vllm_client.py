# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for the vLLM inference client."""

from unittest.mock import MagicMock, patch

import requests

from production_test_framework.vllm.client import (
    DEFAULT_MODEL,
    InferenceResult,
    VllmClient,
    VllmConfig,
)


class TestVllmConfig:
    """Tests for VllmConfig."""

    def test_base_url_default(self):
        config = VllmConfig()
        assert config.base_url == "http://localhost:8080"

    def test_base_url_custom(self):
        config = VllmConfig(host="192.168.1.1", port=9000)
        assert config.base_url == "http://192.168.1.1:9000"

    def test_from_env(self):
        config = VllmConfig.from_env(host="vllm.example.com", port=8081)
        assert config.host == "vllm.example.com"
        assert config.port == 8081

    def test_model_defaults(self):
        assert VllmConfig().model == DEFAULT_MODEL

    def test_model_custom(self):
        assert VllmConfig(model="Qwen/Qwen3-32B").model == "Qwen/Qwen3-32B"

    def test_from_env_carries_model(self):
        config = VllmConfig.from_env(host="vllm", port=8081, model="Qwen/Qwen3-32B")
        assert config.model == "Qwen/Qwen3-32B"


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_success_result(self):
        r = InferenceResult(success=True, text="hello", model="Qwen")
        assert r.success is True
        assert r.text == "hello"
        assert r.error is None

    def test_error_result(self):
        r = InferenceResult(success=False, error="timeout")
        assert r.success is False
        assert r.error == "timeout"


class TestVllmClient:
    """Tests for VllmClient."""

    def test_default_config(self):
        client = VllmClient()
        assert client.base_url == "http://localhost:8080"

    def test_custom_config(self):
        config = VllmConfig(host="vllm", port=9090)
        client = VllmClient(config=config)
        assert client.base_url == "http://vllm:9090"

    @patch("production_test_framework.vllm.client.requests.get")
    def test_health_check_ok(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        client = VllmClient()
        assert client.health_check() is True
        mock_get.assert_called_once_with(
            "http://localhost:8080/health",
            timeout=client.config.health_timeout,
        )

    @patch("production_test_framework.vllm.client.requests.get")
    def test_health_check_fail_status(self, mock_get):
        mock_get.return_value = MagicMock(status_code=503)
        client = VllmClient()
        assert client.health_check() is False

    @patch("production_test_framework.vllm.client.requests.get")
    def test_health_check_request_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()
        client = VllmClient()
        assert client.health_check() is False

    @patch("production_test_framework.vllm.client.requests.post")
    def test_complete_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "choices": [{"text": " world", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ),
        )
        client = VllmClient()
        result = client.complete("Hello", max_tokens=10)

        assert result.success is True
        assert result.text == " world"
        assert result.response_time >= 0
        mock_post.assert_called_once()
        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["prompt"] == "Hello"
        assert call_payload["max_tokens"] == 10
        assert call_payload["model"] == DEFAULT_MODEL
        assert result.model == DEFAULT_MODEL

    @patch("production_test_framework.vllm.client.requests.post")
    def test_complete_uses_configured_model(self, mock_post):
        """A server serving something other than DEFAULT_MODEL 404s unless the config reaches
        the request. This is the case that pinned every caller to Qwen3-8B."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"choices": [{"text": " ok", "finish_reason": "stop"}], "usage": {}}),
        )
        client = VllmClient(VllmConfig(model="Qwen/Qwen3-32B"))
        result = client.complete("Hello")

        assert mock_post.call_args[1]["json"]["model"] == "Qwen/Qwen3-32B"
        assert result.model == "Qwen/Qwen3-32B"

    @patch("production_test_framework.vllm.client.requests.post")
    def test_complete_argument_overrides_configured_model(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"choices": [{"text": " ok", "finish_reason": "stop"}], "usage": {}}),
        )
        client = VllmClient(VllmConfig(model="Qwen/Qwen3-32B"))
        result = client.complete("Hello", model="Qwen/Qwen3-8B")

        assert mock_post.call_args[1]["json"]["model"] == "Qwen/Qwen3-8B"
        assert result.model == "Qwen/Qwen3-8B"

    @patch("production_test_framework.vllm.client.requests.post")
    def test_complete_http_error(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500, text="Internal Server Error")
        client = VllmClient()
        result = client.complete("Hello")

        assert result.success is False
        assert "500" in result.error
        assert result.response_time >= 0

    @patch("production_test_framework.vllm.client.requests.post")
    def test_complete_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout()
        client = VllmClient()
        result = client.complete("Hello")

        assert result.success is False
        assert result.error == "Request timeout"

    @patch("production_test_framework.vllm.client.requests.post")
    def test_complete_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("network error")
        client = VllmClient()
        result = client.complete("Hello")

        assert result.success is False
        assert "network error" in result.error

    @patch.object(VllmClient, "health_check")
    def test_wait_for_ready_success(self, mock_health):
        mock_health.return_value = True
        client = VllmClient()
        assert client.wait_for_ready(timeout=30) is True
        assert mock_health.called

    @patch.object(VllmClient, "health_check")
    def test_wait_for_ready_timeout(self, mock_health):
        mock_health.return_value = False
        client = VllmClient()
        assert client.wait_for_ready(timeout=0.5, poll_interval=0.2) is False
