# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""
Loki HTTP API helpers.
"""

import time
from typing import Literal

import requests

# How far back a range query looks when the caller does not say otherwise.
DEFAULT_LOOKBACK_S = 300

DEFAULT_QUERY_LIMIT = 1000


def get_loki_base_url(loki_port: int) -> str:
    """Get the base URL for the Loki HTTP API."""
    return f"http://localhost:{loki_port}"


def query_loki(loki_port: int, endpoint: str, params: dict = None, timeout: int = 10) -> requests.Response:
    """
    Query the Loki HTTP API.
    """
    base_url = get_loki_base_url(loki_port)
    url = f"{base_url}{endpoint}"
    return requests.get(url, params=params, timeout=timeout)


def loki_query_range(
    loki_port: int,
    query: str,
    *,
    since_s: float = DEFAULT_LOOKBACK_S,
    limit: int = DEFAULT_QUERY_LIMIT,
    direction: Literal["backward", "forward"] = "backward",
    timeout: int = 30,
) -> requests.Response:
    """
    Run a LogQL range query over the last since_s seconds.
    """
    now = time.time()
    return query_loki(
        loki_port,
        "/loki/api/v1/query_range",
        params={
            "query": query,
            "start": f"{int((now - since_s) * 1e9)}",
            "end": f"{int(now * 1e9)}",
            "limit": str(limit),
            "direction": direction,
        },
        timeout=timeout,
    )


def response_lines(response: requests.Response) -> list[str]:
    """Return the log lines of a query_range response, oldest first."""
    entries: list[tuple[int, str]] = []
    for stream in response.json().get("data", {}).get("result", []):
        for timestamp, line in stream.get("values", []):
            entries.append((int(timestamp), line))
    return [line for _, line in sorted(entries)]


def logql_lines(
    loki_port: int,
    query: str,
    *,
    since_s: float = DEFAULT_LOOKBACK_S,
    limit: int = DEFAULT_QUERY_LIMIT,
    direction: Literal["backward", "forward"] = "backward",
    timeout: int = 30,
) -> list[str]:
    """
    Run a range query and return its log lines, oldest first.
    """
    response = loki_query_range(loki_port, query, since_s=since_s, limit=limit, direction=direction, timeout=timeout)
    assert response.status_code == 200, (
        f"Loki query {query!r} failed: HTTP {response.status_code} {response.text[:200]}"
    )

    body = response.json()
    assert body.get("status") == "success", f"Loki query {query!r} returned an error: {body}"

    return response_lines(response)
