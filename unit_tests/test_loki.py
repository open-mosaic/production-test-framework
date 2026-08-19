# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for loki module."""

from unittest.mock import MagicMock, patch

import pytest

from production_test_framework.loki import (
    DEFAULT_LOOKBACK_S,
    DEFAULT_QUERY_LIMIT,
    get_loki_base_url,
    logql_lines,
    loki_query_range,
    query_loki,
    response_lines,
)


def range_response(streams, status=200, body_status="success", text=""):
    """A stand-in for the Response a query_range call returns."""
    response = MagicMock()
    response.status_code = status
    response.text = text
    response.json.return_value = {"status": body_status, "data": {"result": streams}}
    return response


class TestGetLokiBaseUrl:
    """Tests for get_loki_base_url."""

    def test_returns_correct_url(self):
        assert get_loki_base_url(3100) == "http://localhost:3100"


class TestQueryLoki:
    """Tests for query_loki."""

    @patch("production_test_framework.loki.requests.get")
    def test_builds_url_and_calls_get(self, mock_get):
        mock_resp = MagicMock()
        mock_get.return_value = mock_resp

        params = {"query": '{namespace="xpt"}'}
        result = query_loki(3100, "/loki/api/v1/query_range", params=params)

        assert result is mock_resp
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://localhost:3100/loki/api/v1/query_range"
        assert call_args[1]["params"] == params


class TestLokiQueryRange:
    """Tests for loki_query_range."""

    @patch("production_test_framework.loki.query_loki")
    def test_defaults(self, mock_query):
        loki_query_range(3100, '{job="x"}')

        params = mock_query.call_args[1]["params"]
        assert mock_query.call_args[0][1] == "/loki/api/v1/query_range"
        assert params["query"] == '{job="x"}'
        assert params["limit"] == str(DEFAULT_QUERY_LIMIT)
        assert params["direction"] == "backward"

    @patch("production_test_framework.loki.query_loki")
    def test_direction_and_limit_are_overridable(self, mock_query):
        loki_query_range(3100, '{job="x"}', direction="forward", limit=5000)

        params = mock_query.call_args[1]["params"]
        assert params["direction"] == "forward"
        assert params["limit"] == "5000"

    @patch("production_test_framework.loki.query_loki")
    def test_window_spans_since_s_in_nanoseconds(self, mock_query):
        loki_query_range(3100, '{job="x"}', since_s=60)

        params = mock_query.call_args[1]["params"]
        span_ns = int(params["end"]) - int(params["start"])
        assert span_ns == pytest.approx(60 * 1e9, rel=1e-6)

    @patch("production_test_framework.loki.query_loki")
    def test_default_window_is_the_default_lookback(self, mock_query):
        loki_query_range(3100, '{job="x"}')

        params = mock_query.call_args[1]["params"]
        span_ns = int(params["end"]) - int(params["start"])
        assert span_ns == pytest.approx(DEFAULT_LOOKBACK_S * 1e9, rel=1e-6)


class TestResponseLines:
    """Tests for response_lines."""

    def test_orders_oldest_first_across_streams(self):
        response = range_response(
            [
                {"values": [["300", "third"], ["100", "first"]]},
                {"values": [["200", "second"]]},
            ]
        )
        assert response_lines(response) == ["first", "second", "third"]

    def test_empty_result_gives_no_lines(self):
        assert response_lines(range_response([])) == []


class TestLogqlLines:
    """Tests for logql_lines."""

    @patch("production_test_framework.loki.query_loki")
    def test_returns_lines_oldest_first(self, mock_query):
        mock_query.return_value = range_response([{"values": [["200", "second"], ["100", "first"]]}])
        assert logql_lines(3100, '{job="x"}') == ["first", "second"]

    @patch("production_test_framework.loki.query_loki")
    def test_passes_direction_through(self, mock_query):
        mock_query.return_value = range_response([])
        logql_lines(3100, '{job="x"}', direction="forward")
        assert mock_query.call_args[1]["params"]["direction"] == "forward"

    @patch("production_test_framework.loki.query_loki")
    def test_fails_on_http_error(self, mock_query):
        mock_query.return_value = range_response([], status=400, text="parse error")

        with pytest.raises(AssertionError, match="HTTP 400"):
            logql_lines(3100, "{")

    @patch("production_test_framework.loki.query_loki")
    def test_fails_when_the_body_reports_an_error(self, mock_query):
        mock_query.return_value = range_response([], body_status="error")

        with pytest.raises(AssertionError, match="returned an error"):
            logql_lines(3100, '{job="x"}')
