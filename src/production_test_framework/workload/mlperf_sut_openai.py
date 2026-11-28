# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
MLPerf LoadGen system under test (SUT) for an OpenAI-compatible server: each sample LoadGen
issues is sent, as the dataset's token ids, to the server's /v1/completions, and the generated
token ids go back to LoadGen.

Run in the MLPerf harness venv (mlcommons-loadgen, aiohttp, pandas), not the test's own, by
``MlperfWorkload``. The generation settings are those of the llama3.1-8b reference SUT
(language/llama3.1-8b/SUT_VLLM.py): greedy, at most 128 tokens. Offline sends every sample at
once, at most --max-concurrency in flight; Server and SingleStream stream each response and
report its first token to LoadGen, for TTFT and TPOT.
"""

import argparse
import array
import asyncio
import json
import threading
from pathlib import Path

import aiohttp
import mlperf_loadgen as lg
import numpy as np
import pandas as pd

SCENARIOS = {
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "SingleStream": lg.TestScenario.SingleStream,
}

GENERATION = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 42, "max_tokens": 128, "min_tokens": 1}

#: Kept in sut_stats.json, for the test to report why requests failed.
MAX_RECORDED_ERRORS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="Offline")
    parser.add_argument("--mode", choices=["performance", "accuracy"], default="performance")
    parser.add_argument("--api-server", required=True, help="base URL, e.g. http://localhost:8080")
    parser.add_argument("--model", required=True, help="the model name the server serves")
    parser.add_argument("--dataset-path", required=True, help="the benchmark's JSON with a tok_input column")
    parser.add_argument("--lg-model-name", default="llama3_1-8b", help="the benchmark's name in mlperf.conf")
    parser.add_argument("--total-sample-count", type=int, default=13368)
    parser.add_argument("--user-conf", required=True)
    parser.add_argument("--output-log-dir", required=True)
    parser.add_argument("--max-concurrency", type=int, default=256, help="requests in flight at most")
    parser.add_argument("--request-timeout", type=float, default=3600.0, help="seconds per request")
    return parser.parse_args()


def output_token_ids(choice: dict) -> list[int]:
    """
    The token ids in one completion choice: vLLM's ``token_ids`` where it returns them, else the
    ``token_id:N`` strings ``return_tokens_as_token_ids`` puts in its logprobs.
    """
    if choice.get("token_ids") is not None:
        return list(choice["token_ids"])
    tokens = (choice.get("logprobs") or {}).get("tokens") or []
    return [int(token.rpartition("token_id:")[2]) for token in tokens]


class OpenAiSut:
    """Sends LoadGen's samples to *url* from an event loop of its own, *max_concurrency* at a time."""

    def __init__(self, args: argparse.Namespace, prompts: list[list[int]]):
        self.url = args.api_server.rstrip("/") + "/v1/completions"
        self.model = args.model
        self.prompts = prompts
        self.stream = args.scenario != "Offline"
        self.timeout = aiohttp.ClientTimeout(total=args.request_timeout)
        self.max_concurrency = args.max_concurrency
        self.stats = {"requests": 0, "errors": 0, "output_tokens": 0, "error_messages": []}
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="sut-loop", daemon=True)
        self.session: aiohttp.ClientSession | None = None
        self.slots: asyncio.Semaphore | None = None

    def start(self) -> None:
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._open(), self.loop).result()

    async def _open(self) -> None:
        self.slots = asyncio.Semaphore(self.max_concurrency)
        # Idle connections close after 4 s, before vLLM's 5 s keep-alive (VLLM_HTTP_TIMEOUT_KEEP_ALIVE)
        # does: aiohttp's 15 s default reuses ones the server has just closed, and the request fails.
        connector = aiohttp.TCPConnector(limit=self.max_concurrency, keepalive_timeout=4)
        self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self.session.close(), self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()

    def issue_queries(self, samples) -> None:
        for sample in samples:
            asyncio.run_coroutine_threadsafe(self._query(sample.id, sample.index), self.loop)

    def flush_queries(self) -> None:
        pass

    def _payload(self, index: int) -> dict:
        return {
            "model": self.model,
            "prompt": self.prompts[index],
            **GENERATION,
            "stream": self.stream,
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }

    async def _query(self, sample_id: int, index: int) -> None:
        tokens: list[int] = []
        first_reported = False
        async with self.slots:
            self.stats["requests"] += 1
            try:
                async with self.session.post(self.url, json=self._payload(index)) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:300]}")
                    if not self.stream:
                        tokens = output_token_ids((await response.json())["choices"][0])
                    else:
                        async for line in response.content:
                            text = line.decode().strip()
                            if not text.startswith("data:") or text == "data: [DONE]":
                                continue
                            choices = json.loads(text[5:]).get("choices") or []
                            tokens += output_token_ids(choices[0]) if choices else []
                            if tokens and not first_reported:
                                respond(lg.FirstTokenComplete, sample_id, tokens[:1])
                                first_reported = True
                if not tokens:
                    raise RuntimeError("the server returned no token ids")
            except Exception as exc:  # noqa: BLE001 -- every failure is counted, and LoadGen still needs an answer
                self.stats["errors"] += 1
                if len(self.stats["error_messages"]) < MAX_RECORDED_ERRORS:
                    self.stats["error_messages"].append(f"sample {index}: {type(exc).__name__}: {exc}")
        if self.stream and not first_reported:
            respond(lg.FirstTokenComplete, sample_id, tokens[:1])
        self.stats["output_tokens"] += len(tokens)
        respond(lg.QuerySamplesComplete, sample_id, tokens, n_tokens=len(tokens))


def respond(callback, sample_id: int, tokens: list[int], n_tokens: int | None = None) -> None:
    """Hand *tokens* to LoadGen through *callback*, as int32 like the reference SUT."""
    data = array.array("B", np.array(tokens, np.int32).tobytes())
    address, size = data.buffer_info()
    args = (sample_id, address, size) + (() if n_tokens is None else (n_tokens,))
    callback([lg.QuerySampleResponse(*args)])


def main() -> None:
    args = parse_args()
    prompts = pd.read_json(args.dataset_path)["tok_input"].tolist()
    total = min(len(prompts), args.total_sample_count)

    settings = lg.TestSettings()
    settings.scenario = SCENARIOS[args.scenario]
    settings.FromConfig(args.user_conf, args.lg_model_name, args.scenario)
    settings.mode = lg.TestMode.AccuracyOnly if args.mode == "accuracy" else lg.TestMode.PerformanceOnly

    output = Path(args.output_log_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_settings = lg.LogSettings()
    log_settings.log_output.outdir = str(output)
    log_settings.log_output.copy_summary_to_stdout = True
    log_settings.enable_trace = False

    sut = OpenAiSut(args, prompts)
    sut.start()
    qsl = lg.ConstructQSL(total, total, lambda _: None, lambda _: None)
    lg_sut = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    try:
        lg.StartTestWithLogSettings(lg_sut, qsl, settings, log_settings)
    finally:
        sut.stop()
        lg.DestroySUT(lg_sut)
        lg.DestroyQSL(qsl)
        (output / "sut_stats.json").write_text(json.dumps(sut.stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
