# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
SemiAnalysis AgentX workload: replays agentic-coding traces with the agentx-harness ``aiperf``
client against an already-running OpenAI-compatible server, then scores the run with
InferenceX's AgentX aggregation so the numbers match https://inferencex.semianalysis.com/agentx.

The harness needs Python < 3.14, so both projects are checked out at pinned commits and run
from a venv of their own (:class:`AgentxHarness`).
"""

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from production_test_framework.ssh import CommandResult
from production_test_framework.workload.command_workload import CommandWorkload
from production_test_framework.workload.inferencex_workload import benchmark_option_argv

__all__ = [
    "DEFAULT_AGENTX_HOME",
    "HARNESS_REF",
    "INFERENCEX_REF",
    "AgentxHarness",
    "AgentxInstallError",
    "AgentxResult",
    "AgentxWorkload",
]

HARNESS_REPO = "https://github.com/SemiAnalysisAI/agentx-harness.git"
# agentx-v1.0.6, the release InferenceX's standalone instructions pin.
HARNESS_REF = "89b21867872a5bbc4b0676bf5005c404da5e9f94"

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
INFERENCEX_REF = "b8866018ef92c97dbe3a3ae67066bea99c6d973f"
INFERENCEX_PACKAGE_ROOT = "inferencex-e2e"

DEFAULT_AGENTX_HOME = Path("~/.cache/agentx").expanduser()

PYTHON_VERSION = "3.11"
# Same extra requirement as InferenceX's agentic client: the trace corpora need datasets>=4.7.
EXTRA_REQUIREMENTS = ("datasets>=4.7.0",)

AGGREGATE_FILE = "profile_export_aiperf.json"
RESULT_FILENAME = "agentx_result"
SERVER_LOG_FILENAME = "server.log"

# aiperf draws on a 1600x800 canvas, so its text and markers shrink to nothing at report width.
# Half the canvas at twice the scale keeps the resolution and doubles their relative size. Its
# legends sit inside the plot over the data, so they move above it, at the top right. Its
# time-slice plots label every slice ("0-1", "1-2", ...), which a long run piles into an
# unreadable band; those keep about MAX_TICKS ticks, each at a slice's start, in seconds.
PLOT_COMMAND = """
import math
import sys
import plotly.graph_objects as go
import aiperf.plot.exporters.png.base as png
png.DEFAULT_PLOT_WIDTH, png.DEFAULT_PLOT_HEIGHT, png.DEFAULT_PLOT_DPI = 900, 450, 267
MAX_TICKS = 15
write_image = go.Figure.write_image
def thin_slice_ticks(fig):
    ticks = list(fig.layout.xaxis.tickvals or ())
    if fig.layout.xaxis.tickmode != "array" or len(ticks) <= MAX_TICKS:
        return
    width = ticks[1] - ticks[0]
    starts = [tick - width / 2 for tick in ticks[:: math.ceil(len(ticks) / MAX_TICKS)]]
    fig.update_xaxes(tickvals=starts, ticktext=[f"{start:g}" for start in starts], tickangle=0)
def write_with_legend_above(fig, *args, **kwargs):
    top = fig.layout.margin.t if fig.layout.margin.t is not None else 80
    fig.update_layout(
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.02, "yanchor": "bottom"},
        margin={"t": top + 30},
    )
    thin_slice_ticks(fig)
    return write_image(fig, *args, **kwargs)
go.Figure.write_image = write_with_legend_above
from aiperf.cli import app
sys.argv = ["aiperf", "plot", *sys.argv[1:]]
app()
"""

INSTALL_TIMEOUT = 1800.0
SCORING_TIMEOUT = 900.0
PLOT_TIMEOUT = 900.0


class AgentxInstallError(RuntimeError):
    """The harness could not be installed; the message names the step and its output."""


def _tail(text: str, lines: int = 25) -> str:
    return "\n".join((text or "").strip().splitlines()[-lines:])


@dataclass(frozen=True)
class AgentxHarness:
    """The pinned harness and InferenceX checkouts under *root*, and the venv they run in."""

    root: Path = DEFAULT_AGENTX_HOME
    harness_ref: str = HARNESS_REF
    inferencex_ref: str = INFERENCEX_REF

    @property
    def harness_dir(self) -> Path:
        return self.root / "agentx-harness"

    @property
    def inferencex_dir(self) -> Path:
        return self.root / "InferenceX"

    @property
    def venv(self) -> Path:
        return self.root / "venv"

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    @property
    def aiperf(self) -> Path:
        return self.venv / "bin" / "aiperf"

    @property
    def marker(self) -> Path:
        return self.root / "installed.json"

    def _wanted(self) -> dict[str, str]:
        return {"harness": self.harness_ref, "inferencex": self.inferencex_ref, "python": PYTHON_VERSION}

    def is_installed(self) -> bool:
        """True when the venv exists and was built from the refs this harness pins."""
        try:
            recorded = json.loads(self.marker.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return False
        return recorded == self._wanted() and self.aiperf.is_file()

    def install(self) -> bool:
        """
        Check out both projects and build the venv; returns False when already installed.

        Raises :class:`AgentxInstallError` naming the step that failed.
        """
        if self.is_installed():
            return False

        uv = shutil.which("uv")
        if uv is None:
            raise AgentxInstallError("uv is not on PATH; it is needed to build the AgentX venv")

        self.root.mkdir(parents=True, exist_ok=True)
        self.marker.unlink(missing_ok=True)

        self._checkout(HARNESS_REPO, self.harness_dir, self.harness_ref)
        self._checkout(
            INFERENCEX_REPO, self.inferencex_dir, self.inferencex_ref, sparse=f"{INFERENCEX_PACKAGE_ROOT}/infx"
        )

        shutil.rmtree(self.venv, ignore_errors=True)
        _run_step("create the venv", [uv, "venv", "--python", PYTHON_VERSION, str(self.venv)])
        _run_step(
            "install the harness",
            [uv, "pip", "install", "--python", str(self.python), "-e", str(self.harness_dir), *EXTRA_REQUIREMENTS],
            env={**os.environ, "UV_HTTP_TIMEOUT": "120"},
        )
        if not self.aiperf.is_file():
            raise AgentxInstallError(f"the harness installed, but {self.aiperf} is missing")

        self.marker.write_text(json.dumps(self._wanted(), indent=2) + "\n", encoding="utf-8")
        return True

    def _checkout(self, repo: str, directory: Path, ref: str, *, sparse: str | None = None) -> None:
        """Clone *repo* blobless into *directory* if needed, then detach at *ref*."""
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
            _run_step(f"clone {repo}", [*clone, repo, str(directory)])
            if sparse:
                _run_step(f"limit {directory.name} to {sparse}", [*git, "sparse-checkout", "set", sparse])

        known = subprocess.run([*git, "cat-file", "-e", f"{ref}^{{commit}}"], capture_output=True)
        if known.returncode != 0:
            _run_step(f"fetch {ref} into {directory.name}", [*git, "fetch", "--quiet", "origin", ref])
        _run_step(f"check out {ref} in {directory.name}", [*git, "checkout", "--quiet", "--detach", ref])


def _run_step(step: str, argv: list[str], env: Mapping[str, str] | None = None) -> None:
    """Run one install step, raising :class:`AgentxInstallError` with its output on failure."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=INSTALL_TIMEOUT, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AgentxInstallError(f"AgentX install could not {step}: {exc}") from exc
    if done.returncode != 0:
        raise AgentxInstallError(
            f"AgentX install could not {step} (exit status {done.returncode}):\n{_tail(done.stderr or done.stdout)}"
        )


def find_artifacts(artifact_dir: Path) -> Path:
    """The directory holding aiperf's exports: *artifact_dir* itself, or one run below it."""
    if (artifact_dir / AGGREGATE_FILE).is_file():
        return artifact_dir
    if artifact_dir.is_dir():
        for child in sorted(artifact_dir.iterdir()):
            if (child / AGGREGATE_FILE).is_file():
                return child
    return artifact_dir


def _find_key(data: Any, key: str) -> Any:
    """The first value stored under *key* anywhere in a nested JSON document, or None."""
    if isinstance(data, dict):
        if key in data:
            return data[key]
        values = list(data.values())
    elif isinstance(data, list):
        values = data
    else:
        return None
    for value in values:
        if (found := _find_key(value, key)) is not None:
            return found
    return None


@dataclass
class AgentxResult:
    """
    One AgentX run: aiperf's own summary (:attr:`aggregate`) and InferenceX's scored aggregate
    (:attr:`scored`). Scoring and plotting are best effort, so their failures are recorded here.
    """

    run_dir: Path
    artifact_dir: Path
    aggregate: dict[str, Any]
    scored: dict[str, Any] | None = None
    scoring_error: str = ""
    plots: list[Path] = field(default_factory=list)
    plot_error: str = ""
    exit_error: str = ""

    @property
    def metrics(self) -> dict[str, dict[str, Any]]:
        """Every metric aiperf filled in, as ``{name: {"unit": ..., "avg": ..., "p50": ...}}``."""
        return {
            name: value
            for name, value in self.aggregate.items()
            if isinstance(value, dict) and "unit" in value and any(key != "unit" for key in value)
        }

    def _avg(self, name: str) -> float | None:
        value = (self.aggregate.get(name) or {}).get("avg")
        return None if value is None else float(value)

    @property
    def successful_requests(self) -> int:
        return int(self._avg("request_count") or 0)

    @property
    def error_requests(self) -> int:
        return int(self._avg("error_request_count") or 0)

    @property
    def completed_requests(self) -> int:
        completed = self._avg("completed_request_count")
        return int(completed) if completed is not None else self.successful_requests + self.error_requests

    @property
    def error_rate(self) -> float | None:
        """Failed over completed requests, as InferenceX's validation computes it."""
        completed = self.completed_requests
        return self.error_requests / completed if completed else None

    @property
    def exit_summary(self) -> str:
        """The first ERROR line aiperf logged in :attr:`exit_error`, else its first line."""
        if match := re.search(r"\bERROR\s+(.+?)(?:\s+\(\w+\.py:\d+\))?$", self.exit_error, re.MULTILINE):
            return match.group(1).strip()
        return self.exit_error.strip().partition("\n")[0]

    @property
    def submission_valid(self) -> bool | None:
        """False when the scenario's invariants were overridden, e.g. a run shorter than its minimum."""
        return _find_key(self.aggregate, "submission_valid")

    @property
    def aiperf_version(self) -> str | None:
        return self.aggregate.get("aiperf_version")

    @property
    def dataset(self) -> dict[str, Any]:
        return (self.aggregate.get("metadata") or {}).get("dataset") or {}


class AgentxWorkload(CommandWorkload):
    """
    One AgentX point, one concurrency against one server, run as ``aiperf profile``.

    *aiperf_options* become flags by the same rules as ``InferencexWorkload``'s
    ``benchmark_options``. *scoring_env* describes the deployment to InferenceX's scoring
    (``TP``, ``FRAMEWORK``, ``MODEL`` ...). *server_log* returns the inference servers' logs,
    one per server (every prefill and decode worker when disaggregated), from which the
    scoring reads the GPU KV cache capacity; without it that is left unset.
    """

    workload_name = "AgentX"

    def __init__(
        self,
        *,
        harness: AgentxHarness,
        aiperf_options: Mapping[str, Any],
        run_dir: Path,
        scoring_env: Mapping[str, str] | None = None,
        timeout: float = 7200.0,
        plots: bool = True,
        browser_path: str | None = None,
        server_log: Callable[[], Sequence[str]] | None = None,
    ):
        super().__init__(timeout=timeout)
        self._harness = harness
        self._run_dir = Path(run_dir)
        # The scoring reads aiperf's exports from here, so a caller cannot move them.
        self._options = {**aiperf_options, "output_artifact_dir": str(self.artifact_dir)}
        self._scoring_env = dict(scoring_env or {})
        self._plots = plots
        self._browser_path = browser_path
        self._server_log = server_log

    @property
    def aiperf_options(self) -> dict[str, Any]:
        return dict(self._options)

    @property
    def scoring_env(self) -> dict[str, str]:
        return dict(self._scoring_env)

    @property
    def harness(self) -> AgentxHarness:
        return self._harness

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def artifact_dir(self) -> Path:
        return self._run_dir / "aiperf_artifacts"

    @property
    def log_path(self) -> Path:
        return self._run_dir / "aiperf.log"

    def build_command(self) -> list[str]:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        argv = [str(self._harness.aiperf), "profile", *benchmark_option_argv(self._options)]
        (self._run_dir / "benchmark_command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")
        return argv

    def _save_log(self, result: CommandResult) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        stderr = f"\n--- stderr ---\n{result.stderr}" if result.stderr else ""
        self.log_path.write_text((result.stdout or "") + stderr, encoding="utf-8")

    def _failure_detail(self, result: CommandResult) -> str:
        self._save_log(result)
        return super()._failure_detail(result)

    def _run_command_sync(self) -> AgentxResult:
        # aiperf exits non-zero when too many requests failed, after writing a complete result.
        try:
            return super()._run_command_sync()
        except RuntimeError as exc:
            if self.command_result is None or not (find_artifacts(self.artifact_dir) / AGGREGATE_FILE).is_file():
                raise
            outcome = self.parse_output(self.command_result)
            outcome.exit_error = str(exc)
            return outcome

    def parse_output(self, result: CommandResult) -> AgentxResult:
        self._save_log(result)
        artifacts = find_artifacts(self.artifact_dir)
        aggregate_path = artifacts / AGGREGATE_FILE
        try:
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"aiperf exited cleanly but left no readable {aggregate_path}: {exc}") from exc

        outcome = AgentxResult(run_dir=self._run_dir, artifact_dir=artifacts, aggregate=aggregate)
        outcome.scored, outcome.scoring_error = self._score()
        if self._plots:
            outcome.plots, outcome.plot_error = self._plot(artifacts)
        return outcome

    def _venv_run(self, argv: list[str], timeout: float, **env: str) -> subprocess.CompletedProcess:
        environ = {**os.environ, **env}
        environ.pop("VIRTUAL_ENV", None)
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=environ, cwd=self._run_dir)

    def _score(self) -> tuple[dict[str, Any] | None, str]:
        """InferenceX's AgentX aggregate, or ``(None, reason)`` when it could not be built."""
        # The scoring sums the KV cache capacity each log reports, from RESULT_DIR/server.log or,
        # for several servers, one RESULT_DIR/watchtower-*.out each. One file per server matters:
        # within a file it counts each engine once, and every server's engine is "EngineCore".
        logs = [log for log in (self._server_log() if self._server_log else []) if log]
        names = [SERVER_LOG_FILENAME] if len(logs) == 1 else [f"watchtower-{i}.out" for i in range(len(logs))]
        for name, log in zip(names, logs, strict=True):
            (self._run_dir / name).write_text(log, encoding="utf-8")
        env = {
            "KV_OFFLOADING": "none",
            "IS_MULTINODE": "false",
            **self._scoring_env,
            "CONC": str(self._options.get("concurrency", 0)),
            "RESULT_DIR": str(self._run_dir),
            "RESULT_FILENAME": RESULT_FILENAME,
            "AGENTIC_OUTPUT_DIR": str(self._run_dir),
            "PYTHONPATH": str(self._harness.inferencex_dir / INFERENCEX_PACKAGE_ROOT),
            "PYTHONSAFEPATH": "1",
        }
        argv = [str(self._harness.python), "-m", "infx.results.agentic.process_agentic_result"]
        try:
            done = self._venv_run(argv, SCORING_TIMEOUT, **env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"InferenceX scoring did not run: {exc}"
        (self._run_dir / "scoring.log").write_text(done.stdout + done.stderr, encoding="utf-8")
        if done.returncode != 0:
            return (
                None,
                f"InferenceX scoring failed (exit status {done.returncode}):\n{_tail(done.stderr or done.stdout)}",
            )
        try:
            return json.loads((self._run_dir / f"{RESULT_FILENAME}.json").read_text(encoding="utf-8")), ""
        except (OSError, ValueError) as exc:
            return None, f"InferenceX scoring wrote no readable result: {exc}"

    def _plot(self, artifacts: Path) -> tuple[list[Path], str]:
        """aiperf's PNG plots; kaleido renders them with Chrome, *browser_path* or ``$BROWSER_PATH``."""
        output = self._run_dir / "plots"
        argv = [str(self._harness.python), "-c", PLOT_COMMAND, str(artifacts), "--output", str(output)]
        try:
            done = self._venv_run(
                argv, PLOT_TIMEOUT, **({"BROWSER_PATH": self._browser_path} if self._browser_path else {})
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [], f"aiperf plot did not run: {exc}"
        plots = sorted(output.glob("*.png")) if output.is_dir() else []
        if done.returncode != 0 or not plots:
            detail = _tail(done.stdout + done.stderr, 8)
            return plots, f"aiperf plot produced {len(plots)} plot(s) (exit status {done.returncode}):\n{detail}"
        return plots, ""
