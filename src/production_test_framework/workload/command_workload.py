# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""Shared base for workloads that run a single cancellable subprocess in a background thread."""

import logging
import threading
import time
from abc import abstractmethod
from concurrent.futures import CancelledError, Future
from typing import Any

from production_test_framework.helper import run_cancellable_command
from production_test_framework.ssh import CommandResult
from production_test_framework.workload.workload import Workload, WorkloadResult, WorkloadStatus


class WorkloadCancelled(Exception):
    """The command run was terminated via ``CommandWorkload.stop()``."""


class CommandWorkload(Workload):
    """
    Base class for workloads whose work is "run one command, capture its output".
    """

    workload_name = "Command"

    def __init__(
        self,
        *,
        timeout: float = 600.0,
        poll_interval: float = 0.5,
        max_workers: int = 1,
    ):
        super().__init__(max_workers=max_workers)
        self.logger = logging.getLogger(type(self).__module__)
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._cancel_event = threading.Event()
        self._completion_fut: Future | None = None
        self._result: Any = self._empty_result()
        self._command_result: CommandResult | None = None

    @abstractmethod
    def build_command(self) -> list[str]:
        """Return the argv (no shell) this workload runs."""

    def command_env(self) -> dict[str, str] | None:
        """Environment for the command; None runs it with this process's."""
        return None

    def parse_output(self, result: CommandResult) -> Any:
        """Convert a successful command result into the workload result. Override to structure it."""
        return result.stdout or "(no stdout)"

    def _empty_result(self) -> Any:
        """Value used for ``result`` before a run and after a stop/cancel."""
        return ""

    def _cleanup_after_run(self) -> None:
        """
        Release anything the command left behind. Override for runs with side effects.
        """

    @property
    def command_result(self) -> CommandResult | None:
        """Raw result of the last command run, for debugging failed runs."""
        return self._command_result

    def start(self):
        """Start the workload"""

        if self.status == WorkloadStatus.RUNNING:
            raise RuntimeError(f"{self.workload_name} workload already running")

        self.logger.info("Starting %s workload", self.workload_name.lower())
        self._cancel_event.clear()
        self._start_time = time.time()
        self._workload_status = WorkloadStatus.RUNNING
        self._result = self._empty_result()
        self._command_result = None

        self._completion_fut = self.submit_background(self._run_command_sync)
        self._completion_fut.add_done_callback(self._on_command_done)

    def _run_command_sync(self) -> Any:
        cmd = self.build_command()
        self.logger.info("Running: %s", " ".join(cmd))
        result = run_cancellable_command(
            cmd,
            timeout=self._timeout,
            cancel_event=self._cancel_event,
            poll_interval=self._poll_interval,
            env=self.command_env(),
        )
        self._command_result = result
        self._safe_cleanup()
        if not result.success:
            if self._cancel_event.is_set():
                raise WorkloadCancelled()
            raise RuntimeError(self._failure_detail(result))
        return self.parse_output(result)

    def _failure_detail(self, result) -> str:
        """
        Describe a failed run: its exit status, then stderr and stdout, each labelled.

        Both streams are included since either may hold the cause, and empty ones are left out.
        """
        sections = [f"{self.workload_name.lower()} command failed (exit status {result.returncode})"]
        for name, stream in (("stderr", result.stderr), ("stdout", result.stdout)):
            if stream and stream.strip():
                sections.append(f"--- {name} ---\n{stream.strip()}")
        return "\n".join(sections)

    def _safe_cleanup(self) -> None:
        """Run :meth:`_cleanup_after_run`, logging rather than propagating any failure.

        A cleanup error must not replace the real reason a run failed, nor turn a successful
        run into an error.
        """
        try:
            self._cleanup_after_run()
        except Exception:
            self.logger.exception("%s workload cleanup failed", self.workload_name)

    def _on_command_done(self, fut: Future) -> None:
        try:
            result = fut.result()
        except WorkloadCancelled:
            self.logger.info("%s workload stopped", self.workload_name)
            self._workload_status = WorkloadStatus.STOPPED
            self._result = self._empty_result()
            return
        except CancelledError:
            self.logger.info("%s workload cancelled", self.workload_name)
            self._workload_status = WorkloadStatus.STOPPED
            return
        except Exception as e:
            self.logger.exception("%s workload failed", self.workload_name)
            self._workload_status = WorkloadStatus.ERROR
            self._result = str(e)
            return
        finally:
            self._end_time = time.time()

        self._result = result
        self._workload_status = WorkloadStatus.COMPLETED

    def stop(self):
        """Stop the workload, terminating the running command."""
        self.logger.info("Stopping %s workload", self.workload_name.lower())
        self._cancel_event.set()
        if self._completion_fut is not None:
            self._completion_fut.cancel()
        self._workload_status = WorkloadStatus.STOPPED
        self._completion_fut = None
        self._safe_cleanup()

    def get_result(self) -> WorkloadResult:
        return WorkloadResult(
            start_time=self._start_time, end_time=self._end_time, result=self._result, status=self.status
        )
