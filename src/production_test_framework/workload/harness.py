# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
What the benchmarks that run from a venv of their own share: AgentX and MLPerf each pin their
upstream checkouts and build a venv for them on first use (:class:`VenvHarness`), then run one
command from it per point and parse what it wrote (:class:`HarnessWorkload`).
"""

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from production_test_framework.ssh import CommandResult
from production_test_framework.workload.command_workload import CommandWorkload

__all__ = ["INSTALL_TIMEOUT", "HarnessInstallError", "HarnessWorkload", "VenvHarness", "tail", "venv_environ"]

INSTALL_TIMEOUT = 1800.0


class HarnessInstallError(RuntimeError):
    """The harness could not be installed; the message names the step and its output."""


def tail(text: str | None, lines: int = 25) -> str:
    """The last *lines* lines of *text*, stripped."""
    return "\n".join((text or "").strip().splitlines()[-lines:])


def venv_environ(**env: str) -> dict[str, str]:
    """This process's environment plus *env*, without the test's own venv, for a harness venv's python."""
    environ = {**os.environ, **env}
    environ.pop("VIRTUAL_ENV", None)
    return environ


class VenvHarness:
    """
    Base for a frozen dataclass with a ``root`` field: the venv under it, built once and marked
    with what it was built from, so a change to the pins rebuilds it.

    Subclasses name the benchmark (``name``), the error to raise (``install_error``), and what
    they pin (:meth:`_wanted`).
    """

    root: Path
    name: ClassVar[str] = "harness"
    install_error: ClassVar[type[HarnessInstallError]] = HarnessInstallError

    @property
    def venv(self) -> Path:
        return self.root / "venv"

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    @property
    def marker(self) -> Path:
        return self.root / "installed.json"

    @property
    def entry_point(self) -> Path:
        """The file a finished install leaves; its absence means the venv must be rebuilt."""
        return self.python

    def _wanted(self) -> dict[str, Any]:
        raise NotImplementedError

    def is_installed(self) -> bool:
        """True when the venv exists and was built from what this harness pins."""
        try:
            recorded = json.loads(self.marker.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return False
        return recorded == self._wanted() and self.entry_point.is_file()

    def _start_install(self) -> str:
        """Clear the marker and return uv's path, ready for :meth:`_checkout` and :meth:`_build_venv`."""
        uv = shutil.which("uv")
        if uv is None:
            raise self.install_error(f"uv is not on PATH; it is needed to build the {self.name} venv")
        self.root.mkdir(parents=True, exist_ok=True)
        self.marker.unlink(missing_ok=True)
        return uv

    def _build_venv(self, uv: str, python_version: str, *installs: tuple[str, list[str]]) -> None:
        """Recreate the venv at *python_version*, then run each ``(step, uv pip install args)``."""
        shutil.rmtree(self.venv, ignore_errors=True)
        self._run_step("create the venv", [uv, "venv", "--python", python_version, str(self.venv)])
        env = {**os.environ, "UV_HTTP_TIMEOUT": "120"}
        for step, args in installs:
            self._run_step(step, [uv, "pip", "install", "--python", str(self.python), *args], env)

    def _finish_install(self) -> None:
        self.marker.write_text(json.dumps(self._wanted(), indent=2) + "\n", encoding="utf-8")

    def _run_step(self, step: str, argv: list[str], env: Mapping[str, str] | None = None) -> None:
        """Run one install step, raising :attr:`install_error` with its output on failure."""
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=INSTALL_TIMEOUT, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise self.install_error(f"{self.name} install could not {step}: {exc}") from exc
        if done.returncode != 0:
            raise self.install_error(
                f"{self.name} install could not {step} (exit status {done.returncode}):\n"
                f"{tail(done.stderr or done.stdout)}"
            )

    def _checkout(self, repo: str, directory: Path, ref: str, *, sparse: str | None = None) -> None:
        """Clone *repo* blobless into *directory* if needed, limited to *sparse*, then detach at *ref*."""
        git = ["git", "-C", str(directory)]
        if not (directory / ".git").is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            clone = [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--quiet",
                *(["--sparse"] if sparse else []),
            ]
            self._run_step(f"clone {repo}", [*clone, repo, str(directory)])
            if sparse:
                self._run_step(f"limit {directory.name} to {sparse}", [*git, "sparse-checkout", "set", sparse])

        known = subprocess.run([*git, "cat-file", "-e", f"{ref}^{{commit}}"], capture_output=True)
        if known.returncode != 0:
            self._run_step(f"fetch {ref} into {directory.name}", [*git, "fetch", "--quiet", "origin", ref])
        self._run_step(f"check out {ref} in {directory.name}", [*git, "checkout", "--quiet", "--detach", ref])


class HarnessWorkload(CommandWorkload):
    """
    One run of a :class:`VenvHarness`'s command, writing under *run_dir*: the command line to
    ``benchmark_command.txt`` and its output to :attr:`log_path`, whether it succeeds or not.
    """

    #: The file in the run directory the command's output is saved to.
    log_name = "command.log"

    def __init__(self, *, harness: VenvHarness, run_dir: Path, timeout: float = 7200.0):
        super().__init__(timeout=timeout)
        self._harness = harness
        self._run_dir = Path(run_dir)

    @property
    def harness(self) -> VenvHarness:
        return self._harness

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def log_path(self) -> Path:
        return self._run_dir / self.log_name

    def _record_command(self, argv: list[str]) -> list[str]:
        """Create the run directory and write *argv* to its ``benchmark_command.txt``; returns *argv*."""
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "benchmark_command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")
        return argv

    def _save_log(self, result: CommandResult) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        stderr = f"\n--- stderr ---\n{result.stderr}" if result.stderr else ""
        self.log_path.write_text((result.stdout or "") + stderr, encoding="utf-8")

    def _failure_detail(self, result: CommandResult) -> str:
        self._save_log(result)
        return super()._failure_detail(result)

    def _venv_run(
        self, argv: list[str], timeout: float, cwd: Path | None = None, **env: str
    ) -> subprocess.CompletedProcess:
        """Run *argv* with the harness venv's environment, in *cwd* or the run directory."""
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=venv_environ(**env), cwd=cwd or self._run_dir
        )
