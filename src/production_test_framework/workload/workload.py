# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

from production_test_framework.utils.polling import wait_for


class WorkloadStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class WorkloadResult:
    start_time: float
    end_time: float
    result: Any
    status: WorkloadStatus

    @property
    def runtime(self) -> float:
        return self.end_time - self.start_time


class Workload(ABC):
    def __init__(self, max_workers: int = 1):
        self.logger = logging.getLogger(__name__)
        self._workload_status = WorkloadStatus.STOPPED
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="workload",
        )

        # timing metrics for workloads
        self._start_time = 0.0
        self._end_time = 0.0

    @property
    def status(self) -> WorkloadStatus:
        return self._workload_status

    def submit_background(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        """Run *fn* in this workload's thread pool."""
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown_executor(
        self,
        wait: bool = False,
        cancel_futures: bool = True,
    ) -> None:
        """Release thread pool resources (e.g. when discarding the workload)."""
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    @abstractmethod
    def start(self):
        """Start the workload"""

    @abstractmethod
    def stop(self):
        """Stop the workload"""

    @abstractmethod
    def get_result(self) -> WorkloadResult:
        """Get the result of the workload"""

    def stop_if_running(self) -> None:
        """Stop the workload if it is still running, as a test's teardown does."""
        if self.status == WorkloadStatus.RUNNING:
            self.stop()

    def wait_for_completion(self, timeout: float = 300, poll_interval: float = 2.0):
        """Wait for the workload to complete"""
        return wait_for(
            lambda: self.status != WorkloadStatus.RUNNING,
            timeout,
            poll_interval,
        )
