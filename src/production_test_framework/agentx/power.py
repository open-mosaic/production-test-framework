# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.
"""
Power readings on this host: per-GPU power from ``nvidia-smi``, which needs no privileges, and
whole-server wall power from the BMC through ``ipmitool``, which needs root.
"""

import re
import shlex
import statistics
import threading
from dataclasses import dataclass, field

from ..helper import run_command

__all__ = [
    "GPU_POWER_COMMAND",
    "NEEDS_SUDO",
    "SERVER_POWER_COMMAND",
    "PowerReadingUnavailable",
    "PowerSampler",
    "PowerSummary",
    "SudoRequired",
    "gpu_power_watts",
    "server_power_watts",
]

GPU_POWER_COMMAND = ("nvidia-smi", "--query-gpu=power.draw,power.limit", "--format=csv,noheader,nounits")
SERVER_POWER_COMMAND = ("sudo", "ipmitool", "dcmi", "power", "reading")

DEFAULT_SAMPLE_INTERVAL = 5.0

#: Shown in place of a reading that needs root when sudo cannot run without a password.
NEEDS_SUDO = "need sudo privileges"


class PowerReadingUnavailable(RuntimeError):
    """A reading could not be taken; :attr:`command` is what to run by hand instead."""

    def __init__(self, command: str, reason: str):
        super().__init__(f"{command}: {reason}")
        self.command = command
        self.reason = reason


class SudoRequired(PowerReadingUnavailable):
    """The reading needs root, and sudo would not run without a password."""


def gpu_power_watts() -> list[tuple[float, float]]:
    """``(draw, limit)`` watts for each NVIDIA GPU on this host."""
    command = shlex.join(GPU_POWER_COMMAND)
    result = run_command(list(GPU_POWER_COMMAND), timeout=15)
    if not result.success:
        raise PowerReadingUnavailable(command, result.stderr or f"exit status {result.returncode}")
    readings = []
    for line in result.stdout.splitlines():
        try:
            draw, limit = (float(value) for value in line.split(","))
        except ValueError:
            continue
        readings.append((draw, limit))
    if not readings:
        raise PowerReadingUnavailable(command, "reported no GPU power")
    return readings


def server_power_watts() -> float:
    """The BMC's instantaneous wall power for the whole server, through non-interactive sudo."""
    command = shlex.join(SERVER_POWER_COMMAND)
    sudo, *rest = SERVER_POWER_COMMAND
    result = run_command([sudo, "-n", *rest], timeout=30)
    if not result.success:
        reason = (result.stderr or f"exit status {result.returncode}").splitlines()[0]
        raise (SudoRequired if reason.startswith("sudo:") else PowerReadingUnavailable)(command, reason)
    match = re.search(r"Instantaneous power reading:\s*([\d.]+)\s*Watts", result.stdout)
    if match is None:
        raise PowerReadingUnavailable(command, "no instantaneous power reading in its output")
    return float(match.group(1))


@dataclass
class PowerSummary:
    """What a :class:`PowerSampler` saw; per-GPU figures are averaged across the GPUs."""

    samples: int = 0
    gpu_count: int = 0
    gpu_limit_w: float | None = None
    gpu_avg_w: float | None = None
    gpu_max_w: float | None = None
    server_avg_w: float | None = None
    server_max_w: float | None = None
    server_unavailable: str | None = None
    errors: list[str] = field(default_factory=list)


class PowerSampler:
    """
    Sample GPU power, and server power when *server* is set, every *interval* seconds.
    *server_unavailable* says why server power is not sampled, e.g. :data:`NEEDS_SUDO`.
    """

    def __init__(
        self, *, server: bool = False, server_unavailable: str | None = None, interval: float = DEFAULT_SAMPLE_INTERVAL
    ):
        self._server = server
        self._server_unavailable = server_unavailable
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu: list[float] = []
        self._limits: list[float] = []
        self._gpu_count = 0
        self._wall: list[float] = []
        self._samples = 0
        self._errors: dict[str, str] = {}

    def _sample(self) -> None:
        self._samples += 1
        try:
            readings = gpu_power_watts()
            self._gpu_count = len(readings)
            self._gpu.append(statistics.mean(draw for draw, _ in readings))
            self._limits = [limit for _, limit in readings]
        except PowerReadingUnavailable as exc:
            self._errors.setdefault("gpu", str(exc))
        if self._server:
            try:
                self._wall.append(server_power_watts())
            except PowerReadingUnavailable as exc:
                self._errors.setdefault("server", str(exc))

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="power-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> PowerSummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
        return PowerSummary(
            samples=self._samples,
            gpu_count=self._gpu_count,
            gpu_limit_w=statistics.mean(self._limits) if self._limits else None,
            gpu_avg_w=statistics.mean(self._gpu) if self._gpu else None,
            gpu_max_w=max(self._gpu) if self._gpu else None,
            server_avg_w=statistics.mean(self._wall) if self._wall else None,
            server_max_w=max(self._wall) if self._wall else None,
            server_unavailable=self._server_unavailable,
            errors=list(self._errors.values()),
        )
