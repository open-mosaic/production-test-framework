# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

import logging
import threading
import time
from concurrent.futures import CancelledError, Future
from enum import Enum

from production_test_framework.vllm import DEFAULT_MODEL, InferenceResult, VllmClient, VllmConfig
from production_test_framework.workload.workload import Workload, WorkloadResult, WorkloadStatus


class BACKEND_TYPE(Enum):
    VLLM = "vllm"


class PromptWorkload(Workload):
    def __init__(
        self,
        prompt: str,
        backend_type: BACKEND_TYPE = BACKEND_TYPE.VLLM,
        host: str = "localhost",
        port: int = 8080,
        model: str = DEFAULT_MODEL,
        count: int = 1,
        max_tokens: int = 100,
    ):
        """
        Send a prompt to a backend *count* times in sequence and hold the last result.

        `count` is the number of prompts sent.The prompt is deliberately repeated verbatim rather
        than varied: identical requests reproduce the same message sizes, so they land on the same
        keys instead of minting new ones.

        `model` goes into the backend's configuration rather than into the request, because
        `start` submits the run as a deferred callable -- there is no call site left at which to
        name a model. A server serving anything other than DEFAULT_MODEL answers 404 unless this
        is set.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.prompt = prompt
        self.backend = None
        self._completion_fut: Future | None = None
        self.prompt_state = WorkloadStatus.STOPPED
        self.prompt_result = None
        self._count = count
        self._max_tokens = max_tokens
        self._stop_event = threading.Event()

        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}")

        match backend_type:
            case BACKEND_TYPE.VLLM:
                self.backend = VllmClient(VllmConfig(host=host, port=port, model=model))

    @property
    def status(self) -> WorkloadStatus:
        return self.prompt_state

    def _send_prompts(self) -> InferenceResult:
        """
        Send the prompt `count` times, returning the last result.
        """
        last: InferenceResult | None = None
        for attempt in range(1, self._count + 1):
            if self._stop_event.is_set():
                self.logger.info("Prompt workload stopping after %d/%d prompts", attempt - 1, self._count)
                break

            result = self.backend.complete(self.prompt, max_tokens=self._max_tokens)
            if not result.success:
                raise RuntimeError(f"prompt {attempt}/{self._count} failed: {result.error}")

            last = result
            self.logger.debug("Prompt %d/%d completed", attempt, self._count)

        if last is None:
            raise RuntimeError("prompt workload stopped before any prompt completed")
        return last

    def start(self):
        """Start the prompt workload"""

        # We currently only support one prompt workload at a time.
        if self.status == WorkloadStatus.RUNNING:
            raise RuntimeError("Prompt workload already running")

        self.logger.info("Starting prompt workload (%d prompt(s)): %s", self._count, self.prompt)
        self.logger.info("waiting for backend to be ready...")
        self.backend.wait_for_ready(timeout=30)

        self.logger.info("sending prompt(s) to backend...")
        self._start_time = time.time()
        self.prompt_state = WorkloadStatus.RUNNING
        self.prompt_result = None
        self._stop_event.clear()
        self._completion_fut = self.submit_background(self._send_prompts)
        self._completion_fut.add_done_callback(self._on_completion_done)

    def stop(self):
        """Stop the prompt workload"""
        self.logger.info("Stopping prompt workload...")
        # Signals the loop as well as cancelling: cancel() only takes effect while the task is
        # still queued, and a multi-prompt run is in the executor for most of its life.
        self._stop_event.set()
        self._completion_fut.cancel()
        self.prompt_state = WorkloadStatus.STOPPED
        self.prompt_result = None
        self._completion_fut = None
        self.logger.info("Prompt workload stopped")

    def get_result(self) -> str:
        """Return inference text after the prompt task has completed."""
        return WorkloadResult(
            start_time=self._start_time, end_time=self._end_time, result=self.prompt_result, status=self.status
        )

    def _on_completion_done(self, fut: Future) -> None:
        # end_time is stamped on every path, but the result and COMPLETED state belong to the
        # success path alone -- setting them in a `finally` also marked failed and cancelled runs
        # completed, and read a `result` that was never bound.
        self._end_time = time.time()
        try:
            result = fut.result()
        except CancelledError:
            self.logger.info("Prompt completion cancelled")
            return
        except Exception:
            self.logger.exception("Prompt workload failed")
            self.prompt_state = WorkloadStatus.ERROR
            return

        self.prompt_result = result
        self.prompt_state = WorkloadStatus.COMPLETED
        self.logger.info("Prompt workload completed: %s", self.prompt_result)
