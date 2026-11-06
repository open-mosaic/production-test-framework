# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""Unit tests for running commands on this machine or over ssh."""

import subprocess
from unittest.mock import patch

import pytest

from production_test_framework.docker import force_remove_container
from production_test_framework.host import SSH_OPTIONS, Host
from production_test_framework.ssh import CommandResult


def _patched(*results):
    return patch("production_test_framework.host.run_command", side_effect=list(results))


class TestAddress:
    @pytest.mark.parametrize("address", ["localhost", "127.0.0.1", "::1", "", "  "])
    def test_loopback_and_empty_are_this_machine(self, address):
        assert Host(address).local

    def test_the_user_is_not_part_of_the_hostname(self):
        host = Host("alice@gpu-node")
        assert (host.hostname, host.local, str(host)) == ("gpu-node", False, "alice@gpu-node")


class TestArgv:
    def test_runs_as_is_locally(self):
        assert Host().argv(["nvidia-smi", "-L"]) == ["nvidia-smi", "-L"]

    def test_wraps_in_batch_mode_ssh_remotely(self):
        argv = Host("gpu-node").argv(["docker", "run", "--gpus", '"device=0,1"', "image"])
        assert argv[: 1 + len(SSH_OPTIONS)] == ["ssh", *SSH_OPTIONS]
        assert "BatchMode=yes" in argv
        assert argv[-2:] == ["gpu-node", "docker run --gpus '\"device=0,1\"' image"]

    def test_shell_passes_arguments_positionally(self):
        with _patched(CommandResult(0, "", "")) as run:
            Host().shell('echo "$1"', "a b")
        assert run.call_args.args[0] == ["sh", "-c", 'echo "$1"', "sh", "a b"]


class TestAccess:
    def test_this_machine_needs_no_ssh(self):
        with _patched() as run:
            assert Host().access_problem() is None
        run.assert_not_called()

    def test_passwordless_ssh_passes(self):
        with _patched(CommandResult(0, "", "")):
            assert Host("gpu-node").access_problem() is None

    def test_a_password_prompt_says_to_copy_the_key(self):
        with _patched(CommandResult(255, "", "alice@gpu-node: Permission denied (publickey,password).")):
            problem = Host("alice@gpu-node").access_problem()
        assert "`ssh -o BatchMode=yes alice@gpu-node true` failed (alice@gpu-node: Permission denied" in problem
        assert problem.endswith("Run `ssh-copy-id alice@gpu-node`, then run again.")


class TestRemoteQueries:
    def test_free_bytes_from_df_in_kib(self):
        with _patched(CommandResult(0, "2048", "")):
            assert Host("gpu-node").free_bytes("/data/hf") == 2 * 1024 * 1024

    def test_free_bytes_unknown_when_df_fails(self):
        with _patched(CommandResult(1, "", "df: not found")):
            assert Host("gpu-node").free_bytes("/data/hf") is None

    def test_free_bytes_locally_from_the_nearest_existing_parent(self, tmp_path):
        assert Host().free_bytes(str(tmp_path / "not" / "yet")) > 0

    def test_unset_variable_is_none(self):
        with _patched(CommandResult(1, "", "")):
            assert Host("gpu-node").getenv("HF_HOME") is None

    def test_write_file_sends_the_text_on_stdin(self):
        with _patched(CommandResult(0, "", "")) as run:
            Host("gpu-node").write_file("/home/alice/proxy.py", "print(1)\n")
        assert run.call_args.kwargs["stdin_data"] == "print(1)\n"
        assert run.call_args.args[0][-1].endswith("/home/alice/proxy.py")


class TestDockerOnAHost:
    def test_force_remove_runs_on_the_host(self):
        with patch("production_test_framework.docker.subprocess.run") as run:
            run.return_value.returncode = 0
            assert force_remove_container("agentx-vllm-1-0", host=Host("gpu-node"))
        argv = run.call_args.args[0]
        assert argv[0] == "ssh" and argv[-1] == "docker rm --force agentx-vllm-1-0"


class TestForceRemove:
    @staticmethod
    def _completed(returncode, stderr=""):
        return subprocess.CompletedProcess([], returncode, "", stderr)

    def test_retries_a_container_still_dying_after_its_kill(self):
        slow = self._completed(
            1, "could not kill container: tried to kill container, but did not receive an exit event"
        )
        with (
            patch("production_test_framework.docker.subprocess.run", side_effect=[slow, self._completed(0)]) as run,
            patch("production_test_framework.docker.time.sleep"),
        ):
            assert force_remove_container("agentx-vllm-1-0")
        assert run.call_count == 2

    def test_a_missing_container_is_already_removed(self):
        with patch(
            "production_test_framework.docker.subprocess.run",
            return_value=self._completed(1, "Error: No such container: x"),
        ):
            assert force_remove_container("x")

    def test_gives_up_on_a_dying_container_at_the_timeout(self):
        dying = self._completed(1, "removal of container x is already in progress")
        with patch("production_test_framework.docker.subprocess.run", return_value=dying) as run:
            assert not force_remove_container("x", timeout=0.5)
        run.assert_called_once()

    def test_other_failures_are_not_retried(self):
        failed = self._completed(1, "Cannot connect to the Docker daemon")
        with patch("production_test_framework.docker.subprocess.run", return_value=failed) as run:
            assert not force_remove_container("x")
        run.assert_called_once()
