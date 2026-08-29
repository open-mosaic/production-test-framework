# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.

"""
Grafana HTTP API and browser UI helpers.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import requests

from production_test_framework.helper import poll_until

if TYPE_CHECKING:  # pragma: no cover - import only for annotations
    from playwright.sync_api import Page

# =============================================================================
# Constants
# =============================================================================

DEFAULT_GRAFANA_PORT = 3000

# Grafana answers requests it serves itself quickly. Anything that leaves the pod
# for a datasource - a health check, a proxied query - gets the longer timeout.
DEFAULT_TIMEOUT_S = 15
QUERY_TIMEOUT_S = 60

# Grafana serves /api/health while it is still starting, so readiness is a login
# that works rather than a health check that answers.
READY_TIMEOUT_S = 180.0
READY_INTERVAL_S = 2.0

# /api/search pages its results. A deployment has dashboards in the tens, so one
# page large enough to hold all of them keeps the callers simple.
SEARCH_LIMIT = 5000

# Key that query_errors() reports a whole-request failure under, when the request
# failed before Grafana got as far as running the individual queries.
REQUEST_ERROR_KEY = "_request"

# What Grafana stores for a template variable with its "All" option selected.
ALL_VALUE = "$__all"

# Grafana computes $__interval from the panel width and the time range, which a
# test has neither of. These stand in, so a query is reproducible run to run.
DEFAULT_INTERVAL = "1m"
DEFAULT_INTERVAL_MS = 60_000
DEFAULT_MAX_DATA_POINTS = 100

# Default window a panel query is run over.
DEFAULT_TIME_FROM = "now-1h"
DEFAULT_TIME_TO = "now"

# Polling a panel's query back to life after the datasource behind it was
# restarted: a replaced pod replays a write-ahead log before it serves reads.
DATA_RECOVERY_TIMEOUT_S = 300.0
DATA_RECOVERY_INTERVAL_S = 5.0

# Loki panels cap the lines they return; an uncapped query on a busy stream is
# slow enough to time the request out.
DEFAULT_MAX_LINES = 100

# =============================================================================
# Browser selectors
# =============================================================================

LOGIN_PATH = "/login"
LOGOUT_PATH = "/logout"

LOGIN_USER_INPUT = "input[name='user']"
LOGIN_PASSWORD_INPUT = "input[name='password']"
LOGIN_SUBMIT = "[data-testid='data-testid Login button'], button[type='submit']"
LOGIN_ALERT = "[data-testid='data-testid Alert error'], [role='alert']"

# Shown only when the account still has a default password; skipped, not filled.
SKIP_PASSWORD_CHANGE = "[data-testid='data-testid Skip change password button']"

# Present on every signed-in page and on none of the signed-out ones.
SIGNED_IN_MARKER = "[data-testid='data-testid Nav toolbar'], [data-testid='data-testid Toggle menu']"

# The testid carries the panel title: "data-testid Panel header <title>".
PANEL_HEADER = "[data-testid^='data-testid Panel header']"
PANEL_HEADER_TESTID_PREFIX = "data-testid Panel header "

DASHBOARD_SCROLL_CONTAINER = "[data-testid='data-testid Dashboard canvas scroll container'], .scrollbar-view, main"

DASHBOARD_VIEWPORT = {"width": 1920, "height": 2400}

# Playwright waits, in milliseconds. Grafana's frontend bundle is large and a
# dashboard's first paint waits on its datasources.
PAGE_LOAD_TIMEOUT_MS = 60_000
PANEL_LOAD_TIMEOUT_MS = 90_000
SHORT_WAIT_MS = 5_000

# One scroll step when forcing lazily mounted panels to render
SCROLL_STEP_PX = 700
SCROLL_MAX_STEPS = 40
SCROLL_SETTLE_MS = 250


class GrafanaError(RuntimeError):
    """Grafana answered, but not with what was asked for."""


# =============================================================================
# HTTP API
# =============================================================================


def grafana_base_url(port: int) -> str:
    """Base URL of a Grafana reached through a local port-forward."""
    return f"http://localhost:{port}"


def grafana_request(
    port: int,
    path: str,
    *,
    method: str = "GET",
    auth: tuple[str, str] | None = None,
    params: dict | None = None,
    payload: dict | None = None,
    headers: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> requests.Response:
    """
    Send one request to Grafana and return the response unexamined.
    """
    return requests.request(
        method,
        f"{grafana_base_url(port)}{path}",
        auth=auth,
        params=params,
        json=payload,
        headers=headers,
        timeout=timeout,
    )


def grafana_json(
    port: int,
    path: str,
    *,
    auth: tuple[str, str] | None = None,
    params: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> Any:
    """
    GET a Grafana endpoint that is expected to succeed, and return its JSON.
    """
    response = grafana_request(port, path, auth=auth, params=params, timeout=timeout)
    if response.status_code != 200:
        raise GrafanaError(f"GET {path} returned HTTP {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError as error:
        raise GrafanaError(f"GET {path} did not return JSON: {response.text[:200]}") from error


def wait_for_grafana_ready(
    port: int,
    *,
    auth: tuple[str, str] | None = None,
    timeout: float = READY_TIMEOUT_S,
    interval: float = READY_INTERVAL_S,
) -> bool:
    """
    Poll until Grafana serves an authenticated API call.
    """
    path = "/api/org" if auth else "/api/health"

    def answered() -> bool:
        try:
            return grafana_request(port, path, auth=auth).status_code == 200
        except requests.exceptions.RequestException:
            return False

    return poll_until(answered, timeout=timeout, interval=interval)


# =============================================================================
# Datasources
# =============================================================================


def datasources(port: int, *, auth: tuple[str, str]) -> list[dict]:
    """Every datasource Grafana has, provisioned or added by hand."""
    return grafana_json(port, "/api/datasources", auth=auth)


def datasource_uids(port: int, *, auth: tuple[str, str]) -> set[str]:
    """The UIDs panels can refer to, for checking a dashboard's are all present."""
    return {entry.get("uid", "") for entry in datasources(port, auth=auth)}


def datasource_proxy(
    port: int,
    uid: str,
    path: str,
    *,
    auth: tuple[str, str],
    params: dict | None = None,
    timeout: int = QUERY_TIMEOUT_S,
) -> requests.Response:
    """
    Call a datasource's own API through Grafana's proxy.
    """
    return grafana_request(
        port,
        f"/api/datasources/proxy/uid/{uid}{path}",
        auth=auth,
        params=params,
        timeout=timeout,
    )


def prometheus_series(
    port: int,
    uid: str,
    match: str,
    *,
    auth: tuple[str, str],
    timeout: int = QUERY_TIMEOUT_S,
) -> list[dict[str, str]]:
    """
    Label sets of the series a selector matches.
    """
    response = datasource_proxy(port, uid, "/api/v1/series", auth=auth, params={"match[]": match}, timeout=timeout)
    if response.status_code != 200:
        return []
    return list(response.json().get("data") or [])


# =============================================================================
# Dashboards
# =============================================================================


def search_dashboards(
    port: int,
    *,
    auth: tuple[str, str],
    query: str | None = None,
    tag: str | None = None,
    limit: int = SEARCH_LIMIT,
) -> list[dict]:
    """Dashboards Grafana knows about, as its own search returns them."""
    params: dict[str, Any] = {"type": "dash-db", "limit": limit}
    if query:
        params["query"] = query
    if tag:
        params["tag"] = tag
    return grafana_json(port, "/api/search", auth=auth, params=params)


def find_dashboard(port: int, title: str, *, auth: tuple[str, str]) -> dict | None:
    """
    The search entry for a dashboard with exactly this title, or None.
    """
    wanted = title.strip().casefold()
    for entry in search_dashboards(port, auth=auth, query=title):
        if entry.get("title", "").strip().casefold() == wanted:
            return entry
    return None


def wait_for_dashboards(
    port: int,
    titles: Iterable[str],
    *,
    auth: tuple[str, str],
    timeout: float = READY_TIMEOUT_S,
    interval: float = READY_INTERVAL_S,
) -> bool:
    """
    Poll until Grafana has every named dashboard. False if some never arrived.
    """
    wanted = list(titles)

    def provisioned() -> bool:
        try:
            found = {entry.get("title", "") for entry in search_dashboards(port, auth=auth)}
        except requests.exceptions.RequestException, GrafanaError:
            return False
        return all(title in found for title in wanted)

    return poll_until(provisioned, timeout=timeout, interval=interval)


def dashboard_by_uid(port: int, uid: str, *, auth: tuple[str, str]) -> dict:
    """The stored dashboard, wrapper and all, as Grafana serves it."""
    return grafana_json(port, f"/api/dashboards/uid/{uid}", auth=auth)


def dashboard_spec(payload: dict) -> dict:
    """
    The dashboard itself, from whichever envelope Grafana wrapped it in.
    """
    for key in ("dashboard", "spec"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return payload


@dataclass(frozen=True)
class Panel:
    """One panel of a dashboard, flattened out of whichever schema held it."""

    title: str
    type: str
    datasource: dict
    targets: tuple[dict, ...]
    # Title of the row the panel sits under, empty when it sits above them all.
    row: str
    # True when the row is collapsed, so the browser will not render the panel
    # until someone expands it.
    collapsed: bool


def _classic_panel(entry: dict, *, row: str, collapsed: bool) -> Panel:
    """Build a Panel from a classic-schema panel object."""
    return Panel(
        title=str(entry.get("title") or ""),
        type=str(entry.get("type") or ""),
        datasource=entry.get("datasource") or {},
        targets=tuple(entry.get("targets") or []),
        row=row,
        collapsed=collapsed,
    )


def _schema_v2_panel(element: dict) -> Panel:
    """
    Build a Panel from a dashboard.grafana.app/v2 element.
    """
    spec = element.get("spec") or {}
    targets: list[dict] = []
    for query in (spec.get("data") or {}).get("spec", {}).get("queries") or []:
        query_spec = query.get("spec") or {}
        inner = query_spec.get("query") or {}
        target = dict(inner.get("spec") or {})
        target["refId"] = query_spec.get("refId") or "A"
        target["datasource"] = {"type": inner.get("group") or ""}
        if not query_spec.get("hidden"):
            targets.append(target)

    return Panel(
        title=str(spec.get("title") or ""),
        type=str((spec.get("vizConfig") or {}).get("group") or (spec.get("vizConfig") or {}).get("kind") or ""),
        datasource={},
        targets=tuple(targets),
        row="",
        collapsed=False,
    )


def dashboard_panels(spec: dict) -> list[Panel]:
    """
    Every panel of a dashboard, rows flattened away.
    """
    elements = spec.get("elements")
    if isinstance(elements, dict):
        return [
            _schema_v2_panel(element)
            for element in elements.values()
            if isinstance(element, dict) and element.get("kind") == "Panel"
        ]

    panels: list[Panel] = []
    row = ""
    collapsed = False

    for entry in spec.get("panels") or []:
        if entry.get("type") == "row":
            row = str(entry.get("title") or "")
            collapsed = bool(entry.get("collapsed"))
            # A collapsed row carries its children inline; an expanded one leaves
            # them as siblings that follow it.
            for child in entry.get("panels") or []:
                panels.append(_classic_panel(child, row=row, collapsed=True))
            continue
        panels.append(_classic_panel(entry, row=row, collapsed=collapsed))

    return panels


def panel_title_matches(expected: str, rendered: str) -> bool:
    """
    True when a rendered panel title is the expected one.
    """
    expected = expected.strip()
    rendered = rendered.strip()
    if "$" not in expected:
        return rendered == expected
    return rendered.startswith(expected.split("$", 1)[0].strip())


def find_panel(panels: list[Panel], expected: str) -> Panel | None:
    """The first panel whose title matches, or None when the dashboard has none."""
    return next((panel for panel in panels if panel_title_matches(expected, panel.title)), None)


def missing_panel_titles(expected: Iterable[str], titles: Iterable[str]) -> list[str]:
    """Expected panel titles that nothing in titles matches."""
    found = list(titles)
    return [want for want in expected if not any(panel_title_matches(want, title) for title in found)]


# =============================================================================
# Template variables
# =============================================================================


def dashboard_variables(spec: dict) -> dict[str, str]:
    """
    Each template variable's current value, ready to substitute into a query.
    """
    values: dict[str, str] = {}
    for entry in spec.get("templating", {}).get("list") or []:
        name = entry.get("name")
        if not name or entry.get("type") == "adhoc":
            continue
        values[str(name)] = _variable_value(entry)
    return values


def _variable_value(entry: dict) -> str:
    """The single string a variable interpolates to."""
    value = (entry.get("current") or {}).get("value")
    if isinstance(value, list):
        value = value[0] if value else ""
    if value in (None, ""):
        query = entry.get("query")
        value = query if isinstance(query, str) else ""
    if value == ALL_VALUE:
        # "All" with no explicit all-value is a match-everything regex.
        value = entry.get("allValue") or ".*"
    return str(value)


# $name, ${name} and ${name:modifier}, the three forms Grafana accepts.
_VARIABLE_PATTERN = re.compile(r"\$(?:\{(?P<braced>[A-Za-z0-9_]+)(?::(?P<modifier>[^}]+))?\}|(?P<bare>[A-Za-z0-9_]+))")

# Built-ins Grafana derives from the time range rather than from a variable.
_INTERVAL_BUILTINS = frozenset({"__interval", "__rate_interval", "__auto", "__auto_interval"})


def interpolate(text: str, variables: dict[str, str], *, interval: str = DEFAULT_INTERVAL) -> str:
    """
    Substitute Grafana's variables into a query the way the frontend would.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("bare") or ""
        if name in _INTERVAL_BUILTINS:
            return interval
        if name == "__interval_ms":
            return str(DEFAULT_INTERVAL_MS)
        if name not in variables:
            return match.group(0)

        value = variables[name]
        if (match.group("modifier") or "") == "regex" and value != ".*":
            return re.escape(value)
        return value

    return _VARIABLE_PATTERN.sub(replace, text)


def interpolate_deep(value: Any, variables: dict[str, str], *, interval: str = DEFAULT_INTERVAL) -> Any:
    """
    Interpolate every string inside a nested structure.
    """
    if isinstance(value, str):
        return interpolate(value, variables, interval=interval)
    if isinstance(value, dict):
        return {key: interpolate_deep(item, variables, interval=interval) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate_deep(item, variables, interval=interval) for item in value]
    return value


# =============================================================================
# Running a panel's queries
# =============================================================================


def panel_queries(
    panel: Panel,
    *,
    variables: dict[str, str] | None = None,
    datasource_uids_by_type: dict[str, str] | None = None,
    interval: str = DEFAULT_INTERVAL,
    interval_ms: int = DEFAULT_INTERVAL_MS,
    max_data_points: int = DEFAULT_MAX_DATA_POINTS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[dict]:
    """
    A panel's own targets, resolved into a body /api/ds/query will accept.
    """
    variables = variables or {}
    uids_by_type = datasource_uids_by_type or {}
    queries: list[dict] = []

    for target in panel.targets:
        if target.get("hide"):
            continue

        query = interpolate_deep(dict(target), variables, interval=interval)
        datasource = query.get("datasource") or panel.datasource or {}
        datasource = interpolate_deep(dict(datasource), variables, interval=interval)

        if not datasource.get("uid"):
            uid = uids_by_type.get(str(datasource.get("type") or ""))
            if uid:
                datasource = {**datasource, "uid": uid}

        query["datasource"] = datasource
        query.setdefault("refId", "A")
        query.setdefault("intervalMs", interval_ms)
        query.setdefault("maxDataPoints", max_data_points)

        if datasource.get("type") == "loki":
            query.setdefault("queryType", "range")
            query.setdefault("maxLines", max_lines)

        queries.append(query)

    return queries


def query_datasource(
    port: int,
    queries: list[dict],
    *,
    auth: tuple[str, str],
    time_from: str = DEFAULT_TIME_FROM,
    time_to: str = DEFAULT_TIME_TO,
    timeout: int = QUERY_TIMEOUT_S,
) -> requests.Response:
    """
    Run queries through the same endpoint the dashboard frontend uses (Grafana).
    """
    payload = {"from": time_from, "to": time_to, "queries": queries}
    return grafana_request(port, "/api/ds/query", method="POST", auth=auth, payload=payload, timeout=timeout)


def query_results(response: requests.Response) -> dict[str, dict]:
    """The per-refId results of a /api/ds/query response, empty if it has none."""
    if response.status_code >= 400:
        return {}
    try:
        return response.json().get("results") or {}
    except ValueError:
        return {}


def query_errors(response: requests.Response) -> dict[str, str]:
    """
    {refId: message} for every query that failed, empty when they all succeeded.

    A request that failed outright is reported once under REQUEST_ERROR_KEY, so
    a caller can treat "the request was rejected" and "one query was rejected"
    the same way.
    """
    if response.status_code >= 400:
        message = response.text[:300]
        try:
            body = response.json()
            message = body.get("message") or body.get("error") or message
        except ValueError:
            pass
        return {REQUEST_ERROR_KEY: f"HTTP {response.status_code}: {message}"}

    errors: dict[str, str] = {}
    for ref_id, result in query_results(response).items():
        message = result.get("error") or ""
        if not message:
            # Newer Grafana reports a list of structured errors instead.
            details = result.get("errors") or []
            message = "; ".join(str(item.get("message") or item) for item in details)
        if not message and result.get("status", 200) >= 400:
            message = f"status {result['status']}"
        if message:
            errors[ref_id] = str(message)
    return errors


def query_row_counts(response: requests.Response) -> dict[str, int]:
    """
    {refId: rows returned}, the count that separates "loaded" from "loaded data".

    A frame with no rows is what Grafana draws as "No data": the query ran, the
    datasource answered, and nothing matched.
    """
    counts: dict[str, int] = {}
    for ref_id, result in query_results(response).items():
        rows = 0
        for frame in result.get("frames") or []:
            values = (frame.get("data") or {}).get("values") or []
            rows += max((len(column) for column in values), default=0)
        counts[ref_id] = rows
    return counts


def query_row_total(response: requests.Response) -> int:
    """Rows across every query of one request."""
    return sum(query_row_counts(response).values())


def run_panel_queries(
    port: int,
    panel: Panel,
    *,
    auth: tuple[str, str],
    variables: dict[str, str] | None = None,
    datasource_uids_by_type: dict[str, str] | None = None,
    time_from: str = DEFAULT_TIME_FROM,
    time_to: str = DEFAULT_TIME_TO,
    timeout: int = QUERY_TIMEOUT_S,
) -> requests.Response | None:
    """
    Resolve and run one panel's queries; None when the panel has none to run.
    """
    queries = panel_queries(
        panel,
        variables=variables,
        datasource_uids_by_type=datasource_uids_by_type,
    )
    if not queries:
        return None
    return query_datasource(port, queries, auth=auth, time_from=time_from, time_to=time_to, timeout=timeout)


def datasource_has_data(
    port: int,
    uid: str,
    expr: str,
    *,
    auth: tuple[str, str],
    kind: str = "prometheus",
    query_body: dict | None = None,
    time_from: str = DEFAULT_TIME_FROM,
    time_to: str = DEFAULT_TIME_TO,
    timeout: int = QUERY_TIMEOUT_S,
) -> bool:
    """
    True when one query returns rows, for gating a test on a signal existing.
    """
    if query_body is not None:
        query: dict[str, Any] = {**query_body, "refId": "probe", "datasource": {"uid": uid, "type": kind}}
    else:
        query = {
            "refId": "probe",
            "datasource": {"uid": uid, "type": kind},
            "expr": expr,
            "intervalMs": DEFAULT_INTERVAL_MS,
            "maxDataPoints": DEFAULT_MAX_DATA_POINTS,
        }
        if kind == "loki":
            query["queryType"] = "range"
            query["maxLines"] = 1
        else:
            query["instant"] = True

    try:
        response = query_datasource(port, [query], auth=auth, time_from=time_from, time_to=time_to, timeout=timeout)
    except requests.exceptions.RequestException:
        return False

    return not query_errors(response) and query_row_total(response) > 0


def dashboard_panel_results(
    port: int,
    spec: dict,
    *,
    auth: tuple[str, str],
    variables: dict[str, str] | None = None,
    datasource_uids_by_type: dict[str, str] | None = None,
    time_from: str = DEFAULT_TIME_FROM,
    time_to: str = DEFAULT_TIME_TO,
    timeout: int = QUERY_TIMEOUT_S,
) -> tuple[dict[str, str], dict[str, int]]:
    """
    Run every panel of a dashboard, returning (errors, rows) keyed by panel title.
    """
    errors: dict[str, str] = {}
    rows: dict[str, int] = {}
    # A dashboard may give two panels the same title - a gauge and the timeseries
    # beside it - and one of them failing must not be hidden by the other
    # succeeding, so repeats are numbered rather than overwritten.
    seen: dict[str, int] = {}

    for panel in dashboard_panels(spec):
        response = run_panel_queries(
            port,
            panel,
            auth=auth,
            variables=variables,
            datasource_uids_by_type=datasource_uids_by_type,
            time_from=time_from,
            time_to=time_to,
            timeout=timeout,
        )
        if response is None:
            continue

        seen[panel.title] = seen.get(panel.title, 0) + 1
        label = panel.title if seen[panel.title] == 1 else f"{panel.title} ({seen[panel.title]})"

        panel_errors = query_errors(response)
        if panel_errors:
            errors[label] = "; ".join(f"{ref}: {message}" for ref, message in sorted(panel_errors.items()))
        rows[label] = query_row_total(response)

    return errors, rows


def wait_for_panel_data(
    port: int,
    panel: Panel,
    *,
    auth: tuple[str, str],
    variables: dict[str, str] | None = None,
    datasource_uids_by_type: dict[str, str] | None = None,
    time_from: str = DEFAULT_TIME_FROM,
    time_to: str = DEFAULT_TIME_TO,
    timeout: float = DATA_RECOVERY_TIMEOUT_S,
    interval: float = DATA_RECOVERY_INTERVAL_S,
) -> bool:
    """
    Poll until a panel's own query answers with rows again.
    """

    def loaded() -> bool:
        try:
            response = run_panel_queries(
                port,
                panel,
                auth=auth,
                variables=variables,
                datasource_uids_by_type=datasource_uids_by_type,
                time_from=time_from,
                time_to=time_to,
            )
        except requests.exceptions.RequestException:
            return False
        if response is None:
            return False
        return not query_errors(response) and query_row_total(response) > 0

    return poll_until(loaded, timeout=timeout, interval=interval)


# =============================================================================
# Browser UI
# =============================================================================


def dashboard_url(
    base_url: str,
    uid: str,
    *,
    slug: str = "",
    time_from: str | None = None,
    time_to: str | None = None,
    variables: dict[str, str] | None = None,
    refresh: str | None = None,
    kiosk: bool = False,
) -> str:
    """
    URL of a dashboard, with its time range and variables pinned.
    """
    url = f"{base_url}/d/{uid}"
    if slug:
        url = f"{url}/{slug}"

    params: dict[str, str] = {}
    if time_from:
        params["from"] = time_from
    if time_to:
        params["to"] = time_to
    if refresh:
        params["refresh"] = refresh
    for name, value in (variables or {}).items():
        params[f"var-{name}"] = value
    if kiosk:
        params["kiosk"] = ""

    return f"{url}?{urlencode(params)}" if params else url


def _timeout_error() -> type[BaseException]:
    """Playwright's timeout exception, imported only where the UI half is used."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    return PlaywrightTimeoutError


def open_login(page: Page, base_url: str, *, timeout: int = PAGE_LOAD_TIMEOUT_MS) -> None:
    """Load the login page and wait for its form."""
    page.goto(f"{base_url}{LOGIN_PATH}", wait_until="domcontentloaded", timeout=timeout)
    page.locator(LOGIN_USER_INPUT).wait_for(state="visible", timeout=timeout)


def submit_login(page: Page, username: str, password: str) -> None:
    """
    Fill the login form and submit it, without waiting for the outcome.
    """
    page.locator(LOGIN_USER_INPUT).fill(username)
    page.locator(LOGIN_PASSWORD_INPUT).fill(password)
    page.locator(LOGIN_SUBMIT).first.click()


def login(
    page: Page,
    base_url: str,
    username: str,
    password: str,
    *,
    timeout: int = PAGE_LOAD_TIMEOUT_MS,
) -> None:
    """
    Sign in and wait until a signed-in page is on screen.
    """
    if LOGIN_PATH not in page.url:
        open_login(page, base_url, timeout=timeout)
    submit_login(page, username, password)

    page.wait_for_url(lambda url: LOGIN_PATH not in url, timeout=timeout)

    skip = page.locator(SKIP_PASSWORD_CHANGE).first
    try:
        skip.click(timeout=SHORT_WAIT_MS)
    except _timeout_error():
        pass

    page.locator(SIGNED_IN_MARKER).first.wait_for(state="visible", timeout=timeout)


def logout(page: Page, base_url: str, *, timeout: int = PAGE_LOAD_TIMEOUT_MS) -> None:
    """
    Sign out by visiting the logout route.
    """
    page.goto(f"{base_url}{LOGOUT_PATH}", wait_until="domcontentloaded", timeout=timeout)
    page.locator(LOGIN_USER_INPUT).wait_for(state="visible", timeout=timeout)


def wait_for_signed_in(page: Page, *, timeout: int = PAGE_LOAD_TIMEOUT_MS) -> bool:
    """
    Wait for a signed-in Grafana to finish rendering. False if it never does.
    """
    try:
        page.locator(SIGNED_IN_MARKER).first.wait_for(state="visible", timeout=timeout)
    except _timeout_error():
        return False
    return True


def is_signed_in(page: Page) -> bool:
    """
    True when the page shows signed-in Grafana right now, without waiting.
    """
    return page.locator(SIGNED_IN_MARKER).first.is_visible()


def login_alert_text(page: Page, *, timeout: int = SHORT_WAIT_MS) -> str | None:
    """The login page's error message, or None when it is showing none."""
    alert = page.locator(LOGIN_ALERT).first
    try:
        alert.wait_for(state="visible", timeout=timeout)
    except _timeout_error():
        return None
    return alert.inner_text().strip()


def open_dashboard(
    page: Page,
    base_url: str,
    uid: str,
    *,
    timeout: int = PANEL_LOAD_TIMEOUT_MS,
    **url_args: Any,
) -> str:
    """
    Open a dashboard and wait for its first panel to mount. Returns the URL used.
    """
    url = dashboard_url(base_url, uid, **url_args)
    page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    page.locator(PANEL_HEADER).first.wait_for(state="visible", timeout=timeout)
    return url


# Reads every panel's state in one pass, so a dashboard costs one round trip
# rather than one per panel.
_PANEL_STATE_SCRIPT = """
([headerSelector, titlePrefix]) => {
  const headers = Array.from(document.querySelectorAll(headerSelector));
  return headers.map((header) => {
    const testid = header.getAttribute('data-testid') || '';
    const root =
      header.closest('[data-viz-panel-key]') ||
      header.closest('section') ||
      header.closest('.panel-container') ||
      header.parentElement ||
      header;
    const errorEl = root.querySelector(
      "[data-testid='data-testid Panel status error'], [aria-label='Panel status error']"
    );
    const text = root.innerText || '';
    return {
      title: testid.startsWith(titlePrefix) ? testid.slice(titlePrefix.length) : testid,
      error: errorEl
        ? (errorEl.getAttribute('title') || errorEl.getAttribute('aria-label') || errorEl.innerText || 'error').trim()
        : '',
      no_data: /(^|\\n)\\s*No data\\s*(\\n|$)/.test(text),
      loading: !!root.querySelector("[data-testid='data-testid Panel loading bar']"),
    };
  });
}
"""


@dataclass(frozen=True)
class PanelState:
    """What one panel put on screen, as the browser sees it."""

    title: str
    # Grafana's own error text for the panel, empty when it is not in error.
    error: str
    # The panel rendered, and its query matched nothing.
    no_data: bool
    # The panel is still waiting on its query.
    loading: bool

    @property
    def failed(self) -> bool:
        """True when Grafana is showing this panel as broken."""
        return bool(self.error)


def panel_states(page: Page) -> list[PanelState]:
    """Every panel currently mounted on the open dashboard."""
    raw = page.evaluate(_PANEL_STATE_SCRIPT, [PANEL_HEADER, PANEL_HEADER_TESTID_PREFIX])
    return [
        PanelState(
            title=str(entry.get("title") or ""),
            error=str(entry.get("error") or ""),
            no_data=bool(entry.get("no_data")),
            loading=bool(entry.get("loading")),
        )
        for entry in raw
    ]


def rendered_panel_titles(page: Page) -> list[str]:
    """Titles of the panels on screen, with their variables already resolved."""
    return [state.title for state in panel_states(page)]


def wait_for_panels(page: Page, *, timeout: int = PANEL_LOAD_TIMEOUT_MS, interval: float = 1.0) -> bool:
    """
    Poll until no panel is still loading. False if some never settled.
    """
    deadline_s = timeout / 1000

    def settled() -> bool:
        states = panel_states(page)
        return bool(states) and not any(state.loading for state in states)

    return poll_until(settled, timeout=deadline_s, interval=interval)


def load_all_panels(
    page: Page,
    *,
    step_px: int = SCROLL_STEP_PX,
    max_steps: int = SCROLL_MAX_STEPS,
    settle_ms: int = SCROLL_SETTLE_MS,
) -> None:
    """
    Scroll a dashboard end to end so every panel mounts, then return to the top.
    """
    script = """
    ([selector, step]) => {
      const el = document.querySelector(selector);
      if (el && el.scrollHeight > el.clientHeight) {
        el.scrollTop += step;
        return el.scrollTop + el.clientHeight < el.scrollHeight - 1;
      }
      window.scrollBy(0, step);
      return window.scrollY + window.innerHeight < document.body.scrollHeight - 1;
    }
    """
    for _ in range(max_steps):
        page.wait_for_timeout(settle_ms)
        if not page.evaluate(script, [DASHBOARD_SCROLL_CONTAINER, step_px]):
            break

    page.wait_for_timeout(settle_ms)
    page.evaluate(
        """
        (selector) => {
          const el = document.querySelector(selector);
          if (el) { el.scrollTop = 0; }
          window.scrollTo(0, 0);
        }
        """,
        DASHBOARD_SCROLL_CONTAINER,
    )
