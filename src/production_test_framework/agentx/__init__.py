# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
Helpers for SemiAnalysis's AgentX benchmark. The workload itself is
:mod:`production_test_framework.workload.agentx_workload` and its report
:mod:`production_test_framework.reporting.agentx`, beside the other workloads and reports.

* :mod:`~.gpu_economics` -- per-GPU power and cost inputs, from InferenceX, and the metrics
  they derive.
* :mod:`~.power` -- GPU and server wall power sampled during a run.
* :mod:`~.model_fit` -- whether a Hugging Face model fits a machine's disk and GPUs for vLLM.
"""
