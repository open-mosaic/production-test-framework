# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.

"""
The OpenAI endpoint of a disaggregated vLLM deployment, for vLLM's NixlConnector.

Each completion goes to a prefill worker first, capped at one token and asked to keep its KV
cache for a remote decode; the prefill answers with where those blocks are, and the request
then goes to a decode worker with that answer, which reads the blocks over NIXL and generates.
Workers in each pool are taken in turn. ``/v1/models`` comes from a decode worker.

Runs in the vLLM image, which has FastAPI, httpx and uvicorn::

    python3 disagg_proxy.py --port 8080 --prefill http://localhost:8100 --decode http://localhost:8200
"""

import argparse
import itertools

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

#: Asks the prefill worker to hold the prompt's KV blocks for a decode worker to read.
REMOTE_DECODE = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}

app = FastAPI()

client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=None, max_keepalive_connections=None))
prefill_workers: itertools.cycle
decode_workers: itertools.cycle
first_decode: str


def _upstream_response(upstream: httpx.Response) -> Response:
    return Response(upstream.content, upstream.status_code, media_type=upstream.headers.get("content-type"))


@app.get("/v1/models")
async def models() -> Response:
    return _upstream_response(await client.get(f"{first_decode}/v1/models"))


@app.get("/health")
async def health() -> Response:
    return JSONResponse({"status": "ok"})


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def completion(request: Request) -> Response:
    body = await request.json()
    path = request.url.path

    prefill_body = {**body, "kv_transfer_params": REMOTE_DECODE, "stream": False, "max_tokens": 1}
    if "max_completion_tokens" in body:
        prefill_body["max_completion_tokens"] = 1
    for key in ("stream_options", "min_tokens"):
        prefill_body.pop(key, None)
    prefilled = await client.post(f"{next(prefill_workers)}{path}", json=prefill_body)
    if prefilled.status_code != 200:
        return _upstream_response(prefilled)

    decode_body = {**body, "kv_transfer_params": prefilled.json().get("kv_transfer_params")}
    decode_url = f"{next(decode_workers)}{path}"
    if not body.get("stream"):
        return _upstream_response(await client.post(decode_url, json=decode_body))
    upstream = await client.send(client.build_request("POST", decode_url, json=decode_body), stream=True)
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        background=BackgroundTask(upstream.aclose),
    )


def main() -> None:
    global prefill_workers, decode_workers, first_decode
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefill", nargs="+", required=True, help="prefill worker base URLs")
    parser.add_argument("--decode", nargs="+", required=True, help="decode worker base URLs")
    args = parser.parse_args()
    prefill_workers, decode_workers = itertools.cycle(args.prefill), itertools.cycle(args.decode)
    first_decode = args.decode[0]
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
