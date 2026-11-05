# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Unit tests for wait_for_grafana_ready."""

from unittest.mock import MagicMock, patch

import requests

from production_test_framework.grafana import DEFAULT_TIMEOUT_S, wait_for_grafana_ready


@patch("production_test_framework.helper.requests.get")
def test_ready_checks_org_with_auth(mock_get):
    mock_get.return_value = MagicMock(status_code=200)

    assert wait_for_grafana_ready(3000, auth=("admin", "pw"), timeout=0, interval=0) is True
    mock_get.assert_called_once_with(
        "http://localhost:3000/api/org", timeout=DEFAULT_TIMEOUT_S, auth=("admin", "pw"), headers=None
    )


@patch("production_test_framework.helper.requests.get")
def test_ready_checks_health_without_auth(mock_get):
    mock_get.return_value = MagicMock(status_code=200)

    assert wait_for_grafana_ready(3000, timeout=0, interval=0) is True
    assert mock_get.call_args.args[0] == "http://localhost:3000/api/health"


@patch("production_test_framework.helper.requests.get")
def test_not_ready_on_error_status(mock_get):
    mock_get.return_value = MagicMock(status_code=401)
    assert wait_for_grafana_ready(3000, auth=("admin", "pw"), timeout=0, interval=0) is False


@patch("production_test_framework.helper.requests.get")
def test_not_ready_on_connection_error(mock_get):
    mock_get.side_effect = requests.ConnectionError("refused")
    assert wait_for_grafana_ready(3000, timeout=0, interval=0) is False
