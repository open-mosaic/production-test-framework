# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Unit tests for the GPU and server power readings."""

from unittest.mock import patch

import pytest

from production_test_framework.agentx.power import (
    NEEDS_SUDO,
    PowerReadingUnavailable,
    PowerSampler,
    SudoRequired,
    gpu_power_watts,
    server_power_watts,
)
from production_test_framework.reporting.agentx import power_summary_rows
from production_test_framework.ssh import CommandResult

DCMI_OUTPUT = """
    Instantaneous power reading:                   225 Watts
    Minimum during sampling period:                225 Watts
    Maximum during sampling period:                225 Watts
    Average power reading over sample period:      225 Watts
    Power reading state is:                   activated
"""


def _patched(*results):
    return patch("production_test_framework.agentx.power.run_command", side_effect=list(results))


class TestGpuPower:
    def test_reads_draw_and_limit_per_gpu(self):
        with _patched(CommandResult(0, "5.03, 70.00\n61.2, 70.00", "")):
            assert gpu_power_watts() == [(5.03, 70.0), (61.2, 70.0)]

    def test_missing_nvidia_smi_names_the_command(self):
        with _patched(CommandResult(-1, "", "nvidia-smi not found in PATH")):
            with pytest.raises(PowerReadingUnavailable) as exc:
                gpu_power_watts()
        assert exc.value.command.startswith("nvidia-smi")


class TestServerPower:
    def test_parses_the_instantaneous_reading(self):
        with _patched(CommandResult(0, DCMI_OUTPUT, "")) as run:
            assert server_power_watts() == 225.0
        assert run.call_args.args[0][:2] == ["sudo", "-n"]

    def test_sudo_needing_a_password_is_unavailable(self):
        with _patched(CommandResult(1, "", "sudo: a password is required")):
            with pytest.raises(SudoRequired) as exc:
                server_power_watts()
        assert exc.value.command == "sudo ipmitool dcmi power reading"
        assert "password is required" in exc.value.reason

    def test_a_bmc_failure_is_not_a_sudo_problem(self):
        with _patched(CommandResult(1, "", "Could not open device at /dev/ipmi0")):
            with pytest.raises(PowerReadingUnavailable) as exc:
                server_power_watts()
        assert not isinstance(exc.value, SudoRequired)


class TestPowerSampler:
    def test_summarises_gpu_and_server_samples(self):
        sampler = PowerSampler(server=True)
        with _patched(
            CommandResult(0, "10, 70\n20, 70", ""),
            CommandResult(0, DCMI_OUTPUT, ""),
            CommandResult(0, "30, 70\n40, 70", ""),
            CommandResult(
                0,
                DCMI_OUTPUT.replace(
                    "Instantaneous power reading:                   225", "Instantaneous power reading: 275"
                ),
                "",
            ),
        ):
            sampler._sample()
            sampler._sample()
        summary = sampler.stop()

        assert (summary.samples, summary.gpu_count, summary.gpu_limit_w) == (2, 2, 70)
        assert (summary.gpu_avg_w, summary.gpu_max_w) == (25, 35)
        assert (summary.server_avg_w, summary.server_max_w) == (250, 275)
        assert summary.errors == []

    def test_errors_are_recorded_once(self):
        sampler = PowerSampler(server=True)
        failure = CommandResult(1, "", "sudo: a password is required")
        with _patched(CommandResult(0, "10, 70", ""), failure, CommandResult(0, "10, 70", ""), failure):
            sampler._sample()
            sampler._sample()
        summary = sampler.stop()
        assert summary.server_avg_w is None
        assert len(summary.errors) == 1

    def test_rows_say_what_was_not_measured(self):
        rows = {row[0]: row[1] for row in power_summary_rows(PowerSampler().stop())}
        assert rows["server wall power mean"] == "not measured"

    def test_rows_say_why_server_power_was_not_measured(self):
        rows = {row[0]: row[1] for row in power_summary_rows(PowerSampler(server_unavailable=NEEDS_SUDO).stop())}
        assert rows["server wall power mean"] == rows["server wall power max"] == "need sudo privileges"
