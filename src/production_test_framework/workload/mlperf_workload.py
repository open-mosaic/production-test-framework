# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
MLPerf Inference llama3.1-8b, run with MLCommons' LoadGen against an already-running
OpenAI-compatible server, and scored the way MLCommons scores it: LoadGen's own result and
validity, and in accuracy mode the reference evaluation.py's ROUGE scores.

The reference implementation runs vLLM in the benchmark's own process, so the requests go
through mlperf_sut_openai.py instead, a SUT beside this module that sends the reference's prompts and
generation settings to the server. LoadGen and the reference checkout are pinned and run from a
venv of their own (:class:`MlperfHarness`), as AgentX's harness is.
"""

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import requests

from production_test_framework.ssh import CommandResult
from production_test_framework.workload.harness import HarnessInstallError, HarnessWorkload, VenvHarness, tail
from production_test_framework.workload.inferencex_workload import benchmark_option_argv

__all__ = [
    "BENCHMARK",
    "DEFAULT_MLPERF_HOME",
    "INFERENCE_REF",
    "LOADGEN_VERSION",
    "MIN_CONTEXT_TOKENS",
    "REFERENCE_DIR",
    "REFERENCE_MODEL",
    "MlperfHarness",
    "MlperfInstallError",
    "MlperfResult",
    "MlperfWorkload",
    "accuracy_problems",
    "parse_detail_log",
    "parse_rouge",
]

INFERENCE_REPO = "https://github.com/mlcommons/inference.git"
INFERENCE_REF = "3fbc329939999c13d0a7b5e67fb2092287e06047"
#: The LoadGen release at INFERENCE_REF (loadgen/VERSION.txt); its mlperf.conf is built in.
LOADGEN_VERSION = "6.0.17"
REFERENCE_DIR = "language/llama3.1-8b"

DEFAULT_MLPERF_HOME = Path("~/.cache/mlperf").expanduser()

PYTHON_VERSION = "3.11"
SUT_REQUIREMENTS = (f"mlcommons-loadgen=={LOADGEN_VERSION}", "aiohttp", "numpy", "pandas")
#: What the reference evaluation.py imports, at its requirements.txt pins where it has them.
ACCURACY_REQUIREMENTS = (
    "transformers==4.46.2",
    "datasets",
    "evaluate==0.4.0",
    "nltk==3.8.1",
    "rouge-score==0.1.2",
    "absl-py==1.4.0",
    "sentencepiece==0.2.0",
)
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

#: The benchmark's name in mlperf.conf, and its reference model.
BENCHMARK = "llama3_1-8b"
REFERENCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

DATASET_BASE_URL = "https://inference.mlcommons-storage.org"
DATASET_FILE = "cnn_eval.json"
DATASET_MD5 = "c41c613c17794d117899fdcdeeadbe3c"
DATASET_URL = f"{DATASET_BASE_URL}/llama3.1_8b/datasets/{DATASET_FILE}"
TOTAL_SAMPLE_COUNT = 13368

#: The longest prompt in cnn_eval.json (2540 tokens) plus the 128 tokens generated.
MIN_CONTEXT_TOKENS = 2668

#: MLCommons' accuracy targets for llama3.1-8b (tools/submission/submission_checker/constants.py):
#: each ROUGE at least 99% of the reference's, and the generated length within 10% of it.
ROUGE_TARGETS = {"rouge1": 38.7792, "rouge2": 15.9075, "rougeL": 24.4957, "rougeLsum": 35.793}
ROUGE_FRACTION = 0.99
REFERENCE_GEN_LEN = 8167644
GEN_LEN_TOLERANCE = 0.10

#: LoadGen's checks of a performance run, in its detail log; a run meeting all of them is VALID.
VALIDITY_CHECKS = (
    "result_min_duration_met",
    "result_min_queries_met",
    "result_perf_constraints_met",
    "early_stopping_met",
)

SUT_SCRIPT = Path(__file__).with_name("mlperf_sut_openai.py")
DETAIL_LOG = "mlperf_log_detail.txt"
SUMMARY_LOG = "mlperf_log_summary.txt"
ACCURACY_LOG = "mlperf_log_accuracy.json"
SUT_STATS = "sut_stats.json"

EVALUATION_TIMEOUT = 3600.0
DOWNLOAD_CHUNK = 1 << 20


class MlperfInstallError(HarnessInstallError):
    """The harness or dataset could not be installed; the message names the step and its output."""


@dataclass(frozen=True)
class MlperfHarness(VenvHarness):
    """The pinned mlcommons/inference checkout under *root*, its dataset, and the venv LoadGen runs in."""

    root: Path = DEFAULT_MLPERF_HOME
    inference_ref: str = INFERENCE_REF

    name: ClassVar[str] = "MLPerf"
    install_error: ClassVar[type[HarnessInstallError]] = MlperfInstallError

    @property
    def inference_dir(self) -> Path:
        return self.root / "inference"

    @property
    def reference_dir(self) -> Path:
        return self.inference_dir / REFERENCE_DIR

    @property
    def dataset_path(self) -> Path:
        return self.root / "datasets" / DATASET_FILE

    @property
    def hf_home(self) -> Path:
        """The Hugging Face cache evaluation.py downloads the tokenizer and ROUGE metric into."""
        return self.root / "huggingface"

    def _wanted(self) -> dict[str, Any]:
        return {
            "inference": self.inference_ref,
            "python": PYTHON_VERSION,
            "requirements": [*SUT_REQUIREMENTS, *ACCURACY_REQUIREMENTS],
        }

    def install(self) -> bool:
        """
        Check out the reference and build the venv; returns False when already installed.

        Raises :class:`MlperfInstallError` naming the step that failed.
        """
        if self.is_installed():
            return False
        uv = self._start_install()
        self._checkout(INFERENCE_REPO, self.inference_dir, self.inference_ref, sparse=REFERENCE_DIR)
        self._build_venv(
            uv,
            PYTHON_VERSION,
            ("install torch (CPU, for the accuracy evaluation)", ["torch", "--index-url", TORCH_INDEX]),
            ("install LoadGen and the evaluation's requirements", [*SUT_REQUIREMENTS, *ACCURACY_REQUIREMENTS]),
        )
        self._finish_install()
        return True

    def download_dataset(self) -> bool:
        """
        Download cnn_eval.json from MLCommons' storage and check its md5, as their r2-downloader
        does; returns False when it is already there.
        """
        path = self.dataset_path
        if path.is_file():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        digest = hashlib.md5()
        try:
            with requests.get(DATASET_URL, stream=True, timeout=60) as response:
                response.raise_for_status()
                with partial.open("wb") as out:
                    for chunk in response.iter_content(DOWNLOAD_CHUNK):
                        out.write(chunk)
                        digest.update(chunk)
        except (requests.RequestException, OSError) as exc:
            partial.unlink(missing_ok=True)
            raise MlperfInstallError(f"could not download {DATASET_URL}: {exc}") from exc
        if digest.hexdigest() != DATASET_MD5:
            partial.unlink(missing_ok=True)
            raise MlperfInstallError(f"{DATASET_URL} has md5 {digest.hexdigest()}, not {DATASET_MD5}; run again")
        partial.rename(path)
        return True


def parse_detail_log(path: Path) -> dict[str, Any]:
    """LoadGen's ``:::MLLOG`` entries in *path*, as ``{key: value}``, the last value of each key."""
    entries: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        _, marker, payload = line.partition(":::MLLOG ")
        if not marker:
            continue
        try:
            entry = json.loads(payload)
        except ValueError:
            continue
        entries[entry["key"]] = entry.get("value")
    return entries


def parse_rouge(output: str) -> dict[str, float]:
    """The scores evaluation.py prints last, e.g. ``{'rouge1': '38.79', ..., 'gen_len': np.int64(8167644)}``."""
    pattern = r"'(rouge1|rouge2|rougeL|rougeLsum|gen_len|gen_num)': (?:np\.\w+\()?'?([\d.]+)"
    return {key: float(value) for key, value in re.findall(pattern, output)}


def accuracy_problems(scores: Mapping[str, float]) -> list[str]:
    """Where *scores* miss MLCommons' llama3.1-8b accuracy targets; empty when they meet them all."""
    problems = []
    for key, reference in ROUGE_TARGETS.items():
        target = reference * ROUGE_FRACTION
        if (score := scores.get(key)) is None or score < target:
            problems.append(f"{key} {score} < {target:.4f} (99% of {reference})")
    low, high = REFERENCE_GEN_LEN * (1 - GEN_LEN_TOLERANCE), REFERENCE_GEN_LEN * (1 + GEN_LEN_TOLERANCE)
    if not low <= (gen_len := scores.get("gen_len") or 0) <= high:
        problems.append(f"gen_len {gen_len:.0f} outside {low:.0f}..{high:.0f} (reference {REFERENCE_GEN_LEN} +/-10%)")
    return problems


@dataclass
class MlperfResult:
    """
    One MLPerf run: LoadGen's detail log entries (:attr:`detail`) and summary, the SUT's request
    counts, and in accuracy mode the ROUGE scores, whose failure to compute is recorded here.
    """

    run_dir: Path
    mode: str
    detail: dict[str, Any]
    summary: str
    sut_stats: dict[str, Any]
    accuracy: dict[str, float] = field(default_factory=dict)
    accuracy_error: str = ""

    @property
    def requests(self) -> int:
        return int(self.sut_stats.get("requests") or 0)

    @property
    def errors(self) -> int:
        return int(self.sut_stats.get("errors") or 0)

    @property
    def error_rate(self) -> float | None:
        return self.errors / self.requests if self.requests else None

    @property
    def validity(self) -> str | None:
        """LoadGen's VALID or INVALID, in performance mode only."""
        return self.detail.get("result_validity")

    @property
    def invalid_reason(self) -> str:
        """Why LoadGen found the run INVALID: its own reason, else the checks it did not meet."""
        if reason := (self.detail.get("result_invalid_reason") or "").strip():
            return reason
        failed = [check for check in VALIDITY_CHECKS if self.detail.get(check) is False]
        return "not met: " + ", ".join(failed) if failed else "see its summary"


class MlperfWorkload(HarnessWorkload):
    """
    One MLPerf llama3.1-8b run, one scenario and mode against one server.

    *sut_options* become sut_openai.py flags by the same rules as AgentX's aiperf_options.
    *loadgen_settings* go into the run's user.conf as ``llama3_1-8b.<scenario>.<key> = <value>``,
    over the mlperf.conf LoadGen has built in.
    """

    workload_name = "MLPerf"
    log_name = "mlperf.log"

    def __init__(
        self,
        *,
        harness: MlperfHarness,
        api_server: str,
        model: str,
        scenario: str,
        mode: str,
        run_dir: Path,
        sut_options: Mapping[str, Any] | None = None,
        loadgen_settings: Mapping[str, Any] | None = None,
        timeout: float = 7200.0,
    ):
        super().__init__(harness=harness, run_dir=run_dir, timeout=timeout)
        self._api_server = api_server
        self._model = model
        self._scenario = scenario
        self._mode = mode
        self._sut_options = dict(sut_options or {})
        self._loadgen_settings = dict(loadgen_settings or {})

    @property
    def harness(self) -> MlperfHarness:
        return self._harness

    @property
    def scenario(self) -> str:
        return self._scenario

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def sut_options(self) -> dict[str, Any]:
        return dict(self._sut_options)

    @property
    def user_conf_lines(self) -> list[str]:
        return [f"{BENCHMARK}.{self._scenario}.{key} = {value}" for key, value in self._loadgen_settings.items()]

    def build_command(self) -> list[str]:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        user_conf = self._run_dir / "user.conf"
        user_conf.write_text("".join(f"{line}\n" for line in self.user_conf_lines), encoding="utf-8")
        return self._record_command([
            str(self._harness.python), str(SUT_SCRIPT),
            "--scenario", self._scenario,
            "--mode", self._mode,
            "--api-server", self._api_server,
            "--model", self._model,
            "--dataset-path", str(self._harness.dataset_path),
            "--lg-model-name", BENCHMARK,
            "--total-sample-count", str(TOTAL_SAMPLE_COUNT),
            "--user-conf", str(user_conf),
            "--output-log-dir", str(self._run_dir),
            *benchmark_option_argv(self._sut_options),
        ])  # fmt: skip

    def parse_output(self, result: CommandResult) -> MlperfResult:
        self._save_log(result)
        try:
            detail = parse_detail_log(self._run_dir / DETAIL_LOG)
            sut_stats = json.loads((self._run_dir / SUT_STATS).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"LoadGen exited cleanly but left no readable logs in {self._run_dir}: {exc}") from exc
        summary_path = self._run_dir / SUMMARY_LOG
        outcome = MlperfResult(
            run_dir=self._run_dir,
            mode=self._mode,
            detail=detail,
            summary=summary_path.read_text(encoding="utf-8") if summary_path.is_file() else "",
            sut_stats=sut_stats,
        )
        if self._mode == "accuracy":
            outcome.accuracy, outcome.accuracy_error = self._evaluate()
        return outcome

    def _evaluate(self) -> tuple[dict[str, float], str]:
        """The reference evaluation.py's ROUGE scores of the accuracy log, or ``({}, reason)``."""
        argv = [
            str(self._harness.python), "evaluation.py",
            "--mlperf-accuracy-file", str(self._run_dir / ACCURACY_LOG),
            "--dataset-file", str(self._harness.dataset_path),
            "--dtype", "int32",
            "--model-name", self._model,
            "--total-sample-count", str(TOTAL_SAMPLE_COUNT),
        ]  # fmt: skip
        try:
            # A Hugging Face cache of its own: vLLM's containers create the usual one as root.
            done = self._venv_run(
                argv, EVALUATION_TIMEOUT, cwd=self._harness.reference_dir, HF_HOME=str(self._harness.hf_home)
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {}, f"evaluation.py did not run: {exc}"
        (self._run_dir / "evaluation.log").write_text(done.stdout + done.stderr, encoding="utf-8")
        scores = parse_rouge(done.stdout)
        if done.returncode != 0 or not scores:
            return scores, f"evaluation.py failed (exit status {done.returncode}):\n{tail(done.stderr or done.stdout)}"
        return scores, ""
