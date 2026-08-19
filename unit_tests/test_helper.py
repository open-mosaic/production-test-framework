# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Unit tests for helper module."""

import threading
import time
from unittest.mock import MagicMock, patch

from production_test_framework.helper import (
    check_tcp_connectivity,
    get_mimir_base_url,
    is_localhost,
    ping,
    poll_until,
    query_mimir,
    run_cancellable_command,
    run_command,
    wait_for_ping,
    wait_for_tcp_connectivity,
)
from production_test_framework.ssh import CommandResult


class TestRunCommand:
    """Tests for run_command."""

    @patch("production_test_framework.helper.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=" out \n", stderr="")
        r = run_command(["/bin/echo", "hi"], timeout=5)
        assert r.success
        assert r.stdout == "out"
        mock_run.assert_called_once()

    @patch("production_test_framework.helper.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess as sp

        mock_run.side_effect = sp.TimeoutExpired(cmd=["x"], timeout=1)
        r = run_command(["sleep", "9"], timeout=1)
        assert r.returncode == -1
        assert "timed out" in r.stderr

    @patch("production_test_framework.helper.subprocess.run")
    def test_executable_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError(2, "No such file")
        r = run_command(["nonexistent-binary-xyz"], timeout=5)
        assert r.returncode == -1
        assert "nonexistent-binary-xyz" in r.stderr


class TestRunCancellableCommand:
    """Tests for run_cancellable_command."""

    def test_success_real_process(self):
        cancel = threading.Event()
        r = run_cancellable_command(
            ["/bin/sh", "-c", "echo ok"],
            timeout=10.0,
            cancel_event=cancel,
            poll_interval=0.2,
        )
        assert r.success
        assert r.stdout == "ok"

    def test_nonzero_exit(self):
        cancel = threading.Event()
        r = run_cancellable_command(
            ["/bin/sh", "-c", "exit 7"],
            timeout=10.0,
            cancel_event=cancel,
            poll_interval=0.2,
        )
        assert r.returncode == 7
        assert not r.success

    def test_cancel_terminates_child(self):
        cancel = threading.Event()
        out = {}

        def run():
            out["r"] = run_cancellable_command(
                ["sleep", "30"],
                timeout=60.0,
                cancel_event=cancel,
                poll_interval=0.1,
            )

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.25)
        cancel.set()
        t.join(timeout=10.0)
        assert "r" in out
        assert out["r"].returncode == -1
        assert "cancelled" in out["r"].stderr.lower()

    def test_wall_clock_timeout(self):
        cancel = threading.Event()
        r = run_cancellable_command(
            ["sleep", "60"],
            timeout=0.4,
            cancel_event=cancel,
            poll_interval=0.1,
        )
        assert r.returncode == -1
        assert "timed out" in r.stderr.lower()


class TestIsLocalhost:
    """Tests for is_localhost."""

    def test_localhost_variants(self):
        assert is_localhost("localhost") is True
        assert is_localhost("LOCALHOST") is True
        assert is_localhost("127.0.0.1") is True
        assert is_localhost("::1") is True
        assert is_localhost("0:0:0:0:0:0:0:1") is True

    def test_not_localhost(self):
        assert is_localhost("") is False
        assert is_localhost("test-host.example.com") is False
        assert is_localhost("10.0.0.1") is False


class TestCheckTcpConnectivity:
    """Tests for check_tcp_connectivity."""

    def test_connectivity_success(self):
        with patch("production_test_framework.helper.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            mock_sock.connect_ex.return_value = 0

            result = check_tcp_connectivity("localhost", 8080)

            assert result is True
            mock_sock.connect_ex.assert_called_once_with(("localhost", 8080))
            mock_sock.close.assert_called_once()

    def test_connectivity_failure(self):
        with patch("production_test_framework.helper.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            mock_sock.connect_ex.return_value = 1

            result = check_tcp_connectivity("example.com", 443)

            assert result is False

    def test_connectivity_socket_error(self):
        with patch("production_test_framework.helper.socket.socket") as mock_socket_cls:
            mock_socket_cls.side_effect = OSError("connection refused")

            result = check_tcp_connectivity("badhost", 9999)

            assert result is False

    def test_uses_custom_timeout(self):
        with patch("production_test_framework.helper.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            mock_sock.connect_ex.return_value = 0

            check_tcp_connectivity("localhost", 80, timeout=10.0)

            mock_sock.settimeout.assert_called_once_with(10.0)


class TestPollUntil:
    """Tests for poll_until."""

    def test_returns_true_without_sleeping_when_already_true(self):
        calls = []

        def predicate() -> bool:
            calls.append(1)
            return True

        with patch("production_test_framework.helper.time.sleep") as mock_sleep:
            assert poll_until(predicate, timeout=5) is True

        assert len(calls) == 1, "predicate must be evaluated once, before any sleep"
        mock_sleep.assert_not_called()

    @patch("production_test_framework.helper.time.sleep")
    def test_retries_until_predicate_passes(self, mock_sleep):
        results = iter([False, False, True])

        assert poll_until(lambda: next(results), timeout=10, interval=0.1) is True
        assert mock_sleep.call_count == 2

    @patch("production_test_framework.helper.time.sleep")
    @patch("production_test_framework.helper.time.monotonic")
    def test_returns_false_on_timeout(self, mock_monotonic, mock_sleep):
        # deadline computed at t=0 (=> 10); first attempt at t=0, then t=100 exits.
        mock_monotonic.side_effect = [0, 0, 100]

        assert poll_until(lambda: False, timeout=10, interval=1) is False


class TestWaitForTcpConnectivity:
    """Tests for wait_for_tcp_connectivity."""

    @patch("production_test_framework.helper.check_tcp_connectivity")
    def test_returns_true_when_immediate_success(self, mock_check):
        mock_check.return_value = True

        result = wait_for_tcp_connectivity("localhost", 8080, timeout=5)

        assert result is True
        assert mock_check.call_count >= 1

    @patch("production_test_framework.helper.check_tcp_connectivity")
    @patch("production_test_framework.helper.time.sleep")
    def test_returns_false_on_timeout(self, mock_sleep, mock_check):
        mock_check.return_value = False

        result = wait_for_tcp_connectivity("localhost", 8080, timeout=1)

        assert result is False


class TestPing:
    """Tests for ping."""

    @patch("production_test_framework.helper.run_command")
    def test_returns_true_on_reply(self, mock_run):
        mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")

        assert ping("10.0.0.1") is True
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "ping"
        assert cmd[-1] == "10.0.0.1"

    @patch("production_test_framework.helper.run_command")
    def test_returns_false_on_no_reply(self, mock_run):
        mock_run.return_value = CommandResult(returncode=1, stdout="", stderr="")

        assert ping("10.0.0.1") is False

    @patch("production_test_framework.helper.run_command")
    def test_passes_interface(self, mock_run):
        mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")

        assert ping("10.0.0.1", interface="eth0") is True
        cmd = mock_run.call_args.args[0]
        assert "-I" in cmd
        assert cmd[cmd.index("-I") + 1] == "eth0"
        assert cmd[-1] == "10.0.0.1"

    @patch("production_test_framework.helper.run_command")
    def test_omits_interface_flag_when_none(self, mock_run):
        mock_run.return_value = CommandResult(returncode=0, stdout="", stderr="")

        assert ping("10.0.0.1") is True
        cmd = mock_run.call_args.args[0]
        assert "-I" not in cmd


class TestWaitForPing:
    """Tests for wait_for_ping."""

    @patch("production_test_framework.helper.ping")
    def test_returns_true_when_immediate_success(self, mock_ping):
        mock_ping.return_value = True

        assert wait_for_ping("10.0.0.1", timeout=5) is True

    @patch("production_test_framework.helper.ping")
    @patch("production_test_framework.helper.time.sleep")
    def test_returns_false_on_timeout(self, mock_sleep, mock_ping):
        mock_ping.return_value = False

        assert wait_for_ping("10.0.0.1", timeout=1, interval=0.1) is False


class TestGetMimirBaseUrl:
    """Tests for get_mimir_base_url."""

    def test_returns_correct_url(self):
        assert get_mimir_base_url(9009) == "http://localhost:9009/prometheus"
        assert get_mimir_base_url(8080) == "http://localhost:8080/prometheus"


class TestQueryMimir:
    """Tests for query_mimir."""

    @patch("production_test_framework.helper.requests.get")
    def test_builds_url_and_calls_get(self, mock_get):
        mock_resp = MagicMock()
        mock_get.return_value = mock_resp

        result = query_mimir(9009, "/api/v1/query", params={"query": "up"})

        assert result is mock_resp
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://localhost:9009/prometheus/api/v1/query"
        assert call_args[1]["params"] == {"query": "up"}
        assert call_args[1]["timeout"] == 10

    @patch("production_test_framework.helper.requests.get")
    def test_default_params_and_timeout(self, mock_get):
        mock_get.return_value = MagicMock()

        query_mimir(9009, "/api/v1/query")

        mock_get.assert_called_once_with(
            "http://localhost:9009/prometheus/api/v1/query",
            params=None,
            timeout=10,
        )
