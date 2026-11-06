# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""
Helper functions for testing.

Provides reusable utilities for testing.
"""

import logging
import os
import signal
import socket
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Mapping

import requests

from production_test_framework.ssh import CommandResult

logger = logging.getLogger(__name__)


def is_localhost(host: str) -> bool:
    """True when host refers to this machine (e.g. loopback), for local vs remote behavior."""
    if not host or not str(host).strip():
        return False
    h = str(host).strip().lower()
    return h in (
        "localhost",
        "127.0.0.1",
        "::1",
        "0:0:0:0:0:0:0:1",
    )


def run_command(
    cmd: list[str],
    timeout: int = 60,
    stdin_data: str | None = None,
) -> CommandResult:
    """
    Run a command locally via subprocess.

    Args:
        cmd: Executable path or name as first element, then arguments (no shell).
        timeout: Seconds before the process is killed.
        stdin_data: Optional string written to stdin.

    Returns:
        CommandResult.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=stdin_data,
        )
        return CommandResult(
            returncode=proc.returncode,
            stdout=(proc.stdout or "").strip(),
            stderr=(proc.stderr or "").strip(),
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
        )
    except FileNotFoundError:
        return CommandResult(
            returncode=-1,
            stdout="",
            stderr=f"{cmd[0]} not found in PATH",
        )
    except Exception as e:
        return CommandResult(returncode=-1, stdout="", stderr=str(e))


def run_cancellable_command(
    cmd: list[str],
    *,
    timeout: float,
    cancel_event: threading.Event,
    poll_interval: float = 0.5,
    stdin_data: str | None = None,
    text: bool = True,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """
    Run a command via Popen; return when the process exits, *timeout* is reached,
    or *cancel_event* is set (child is terminated).

    Args:
        cmd: argv list (no shell).
        timeout: Maximum wall-clock seconds for the run.
        cancel_event: When set from another thread, the child process is terminated.
        poll_interval: How often to wait/check between cancel and completion checks.
        stdin_data: Optional string written to stdin.
        text: Decode stdout/stderr as text.
        env: Environment for the child; the caller's when None.

    Returns:
        CommandResult; returncode -1 for timeout, cancel, or internal errors.
    """
    deadline = time.monotonic() + timeout
    pending_input: str | None = stdin_data

    def _signal_group(proc: subprocess.Popen, sig: signal.Signals) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError, PermissionError:
            pass

    def _finish_terminated(proc: subprocess.Popen, message: str) -> CommandResult:
        # The command's children inherit its pipes, so the whole group goes, or communicate() never returns.
        _signal_group(proc, signal.SIGTERM)
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)
            try:
                out, err = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                return CommandResult(returncode=-1, stdout="", stderr=f"{message}; its output pipes never closed")
        except Exception as e:
            return CommandResult(
                returncode=-1,
                stdout="",
                stderr=f"{message}: {e}",
            )
        stdout = (out or "").strip()
        stderr = (err or "").strip()
        if stderr and message:
            stderr = f"{message}\n{stderr}".strip()
        elif message:
            stderr = message
        return CommandResult(returncode=-1, stdout=stdout, stderr=stderr)

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            env=env,
            start_new_session=True,
        )
    except FileNotFoundError:
        return CommandResult(
            returncode=-1,
            stdout="",
            stderr=f"{cmd[0]} not found in PATH",
        )
    except Exception as e:
        return CommandResult(returncode=-1, stdout="", stderr=str(e))

    while True:
        if cancel_event.is_set():
            return _finish_terminated(proc, "Command cancelled")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _finish_terminated(proc, f"Command timed out after {timeout}s")

        try:
            out, err = proc.communicate(
                input=pending_input,
                timeout=min(poll_interval, max(remaining, 0.01)),
            )
            pending_input = None
            return CommandResult(
                returncode=proc.returncode if proc.returncode is not None else -1,
                stdout=(out or "").strip(),
                stderr=(err or "").strip(),
            )
        except subprocess.TimeoutExpired:
            pending_input = None
            continue
        except Exception as e:
            return CommandResult(returncode=-1, stdout="", stderr=str(e))


def poll_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 30.0,
    interval: float = 1.0,
) -> bool:
    """
    Poll predicate until it returns True or timeout elapses.

    The generic form of the wait_for_* helpers below, for waiting on state a test
    cannot observe synchronously (a controller reconciling, a switch applying a
    change, telemetry reaching a backend).

    predicate is evaluated once immediately, so a condition that is already true
    costs nothing.

    Args:
        predicate: Condition to poll; must be safe to call repeatedly.
        timeout: Total seconds to keep retrying.
        interval: Seconds between attempts.

    Returns:
        The last value predicate returned, so callers can assert on it directly.
    """
    deadline = time.monotonic() + timeout
    result = predicate()
    while not result and time.monotonic() < deadline:
        time.sleep(interval)
        result = predicate()
    return result


def fitted_rise(samples: list[float], *, settle: int = 0) -> tuple[float, float, float]:
    """
    Measure how far a sampled indicator climbed, by the straight line fitted through it.

    This is to determine if a memory drift is in fact a trend and not a range to
    determine whether the drift is a valid test failure.
    """
    kept = samples[settle:] if len(samples) - settle >= 3 else samples
    if len(kept) < 2:
        return 0.0, 0.0, 0.0
    slope, intercept = statistics.linear_regression(range(len(kept)), kept)
    start, end = intercept, intercept + slope * (len(kept) - 1)
    return start, end, end / start - 1.0 if start > 0 else 0.0


def check_tcp_connectivity(host: str, port: int, timeout: float = 5.0) -> bool:
    """
    Test TCP connectivity to a host:port.

    Args:
        host: Target hostname or IP
        port: Target port
        timeout: Connection timeout in seconds

    Returns:
        True if connection successful, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except OSError:
        return False


def wait_for_tcp_connectivity(host: str, port: int, timeout: float = 30) -> bool:
    """Wait for TCP connectivity to a host:port."""
    poll_interval = 0.5
    for i in range(int(timeout / poll_interval)):
        if i % 10 == 0:
            logger.info("Waiting for TCP connectivity to %s:%s... %.0fs", host, port, i * poll_interval)
        if check_tcp_connectivity(host, port):
            return True
        time.sleep(poll_interval)
    return False


def wait_for_http_ok(
    url: str,
    *,
    timeout: float = 30.0,
    interval: float = 1.0,
    request_timeout: float = 5.0,
    auth: tuple[str, str] | None = None,
    headers: dict | None = None,
) -> tuple[bool, str]:
    """
    Wait for GET url to answer with a 2xx status.

    Returns:
        (True, "") once it does, else (False, the last status or connection error).
    """
    last_error = ""

    def ok() -> bool:
        nonlocal last_error
        try:
            resp = requests.get(url, timeout=request_timeout, auth=auth, headers=headers)
        except requests.RequestException as e:
            last_error = str(e)
            return False
        last_error = f"HTTP {resp.status_code}"
        return 200 <= resp.status_code < 300

    if poll_until(ok, timeout=timeout, interval=interval):
        return True, ""
    return False, last_error


def ping(host: str, count: int = 1, timeout: float = 2.0, interface: str | None = None) -> bool:
    """
    Return True if the host answers an ICMP echo request.

    Uses the system `ping` (Linux flags). Runs locally via subprocess, so the
    host must be reachable from wherever this runs.

    Args:
        host: Target hostname or IP.
        count: Number of echo requests to send.
        timeout: Per-reply wait in seconds.
        interface: Source interface (or address) to ping from, passed as
            `ping -I`. If None, the OS chooses the outgoing interface.

    Returns:
        True if at least one reply was received.
    """
    per_reply = max(1, int(timeout))
    cmd = ["ping", "-c", str(count), "-W", str(per_reply)]
    if interface:
        cmd += ["-I", interface]
    cmd.append(host)
    return run_command(cmd, timeout=count * per_reply + 5).success


def wait_for_ping(
    host: str,
    timeout: float = 60.0,
    interval: float = 5.0,
    interface: str | None = None,
) -> bool:
    """Poll ping(host) until it succeeds or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ping(host, interface=interface):
            return True
        time.sleep(interval)
    return False


def get_mimir_base_url(mimir_port: int) -> str:
    """Get the base URL for Mimir Prometheus API."""
    return f"http://localhost:{mimir_port}/prometheus"


def query_mimir(mimir_port: int, endpoint: str, params: dict = None, timeout: int = 10) -> requests.Response:
    """
    Query Mimir Prometheus API.

    Args:
        mimir_port: Local port where Mimir is accessible
        endpoint: API endpoint (e.g., "/api/v1/query")
        params: Optional query parameters
        timeout: Request timeout in seconds

    Returns:
        requests.Response object
    """
    base_url = get_mimir_base_url(mimir_port)
    url = f"{base_url}{endpoint}"
    return requests.get(url, params=params, timeout=timeout)


def power_cycle_via_ssh(host: str, username: str, sudo_password: str | None = None) -> CommandResult:
    """Power-cycle `host` by running `ipmitool chassis power cycle` over `ssh -A`.

    Uses the local ssh (agent, ~/.ssh/config, known_hosts); sudo reads the password from stdin.
    """
    cmd = [
        "ssh",
        "-A",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{username}@{host}",
        "sudo -S -p '' ipmitool chassis power cycle",
    ]
    stdin = f"{sudo_password}\n" if sudo_password is not None else None
    return run_command(cmd, timeout=120, stdin_data=stdin)
