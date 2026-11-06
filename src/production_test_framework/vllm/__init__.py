# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
vLLM for tests: talking to a server, and starting one.

* :mod:`~.client` -- an inference client for a running server.
* :mod:`~.server` -- start vLLM in docker on a host, aggregated or disaggregated, and remove it.
* :mod:`~.model_fit` -- whether a Hugging Face model fits a host's disk and GPUs.
* ``disagg_proxy.py`` -- the endpoint of a disaggregated deployment, run in the vLLM image.
"""

from .client import DEFAULT_MODEL, InferenceResult, VllmClient, VllmConfig

__all__ = ["DEFAULT_MODEL", "InferenceResult", "VllmClient", "VllmConfig"]
