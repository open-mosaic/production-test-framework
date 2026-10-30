# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.
"""
Combined result report for the mosaic test suites.

The report has five parts, in this order:

1. a header -- logo, title, when the run started, and a tally of outcomes;
2. the environment the run targeted, as tables the session records up front;
3. test results grouped into categories, one section per category;
4. every failure in one table, with a stack-trace snippet for the first few;
5. the detail tables individual tests emitted while running.

Parts 1, 3 and 4 come from pytest itself via :class:`~.report_plugin.ReportPlugin`. Parts 2 and 5 come from
whatever the suite chooses to record, so a suite that records nothing still gets a report.
"""

import datetime as dt
import html
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .assets import Logo
from .formatting import (
    STATUS_CLASS,
    ReportFormat,
    Table,
    render_table,
    render_table_html,
    render_table_markdown,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_FAILURE_SNIPPETS",
    "DEFAULT_TITLE",
    "FAILED_OUTCOMES",
    "Reporter",
    "UNCATEGORIZED",
]

#: What the report is called when nothing overrides it.
DEFAULT_TITLE = "Test Report"

#: Section heading for tests carrying none of the configured category markers.
UNCATEGORIZED = "uncategorized"

FAILED_OUTCOMES = frozenset({"failed", "error"})

#: How many stack-trace snippets the failure section shows.
DEFAULT_MAX_FAILURE_SNIPPETS = 5

#: Characters kept from each traceback, taken from the end
MAX_SNIPPET_CHARS = 1600


def failure_snippet(text: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    """
    The tail of *text*, which for a traceback is the assertion and its location.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return "... (truncated)\n" + text[-max_chars:].lstrip()


def summary_line(text: str) -> str:
    """
    The one line of a traceback that names the error, for the failure table's last column.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    for line in lines:
        if line == "E" or line.startswith("E "):
            return line[1:].strip()
    for line in reversed(lines):
        if "Error" in line or "Exception" in line or line.startswith("assert"):
            return line
    return lines[-1]


def _anchor(section: int, title: str) -> str:
    """The id of a table's heading: unique per section, so two tests' tables never collide."""
    return f"s{section}-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


@dataclass
class _Outcome:
    """One row of the summary half: what pytest reported for a test."""

    nodeid: str
    outcome: str
    duration: float
    category: str = UNCATEGORIZED
    failure_text: str = ""

    @property
    def failed(self) -> bool:
        return self.outcome in FAILED_OUTCOMES

    @property
    def name(self) -> str:
        """The test's own name, without the file path that the category already implies."""
        return self.nodeid.partition("::")[2] or self.nodeid


@dataclass
class _Note:
    """A line of prose in a detail section."""

    text: str


@dataclass
class _Figure:
    """An inline SVG figure. HTML only -- Markdown has nowhere to put it."""

    markup: str
    title: str = ""
    data_tables: tuple[str, ...] = ()


@dataclass
class _Output:
    """
    A block of captured program output, shown verbatim.
    """

    text: str
    title: str = ""


_CSS = """
:root { color-scheme: light dark;
  --fg:#1c1c1e; --bg:#fbfbfd; --muted:#6b6b70; --rule:#d8d8dd; --panel:#fff;
  --logo:#203f5b;
  --ok:#1a7f47; --ok-bg:#e7f6ec; --warn:#8a5a00; --warn-bg:#fdf3e0; --bad:#b3261e; --bad-bg:#fdeceb;
  --c0:#86b6ef; --c1:#5598e7; --c2:#2a78d6; --c3:#1c5cab; --c4:#104281; }
@media (prefers-color-scheme: dark) { :root {
  --fg:#e8e8ea; --bg:#16161a; --muted:#9a9aa2; --rule:#33333a; --panel:#1e1e24;
  --logo:#e6ebf0;
  --ok:#5cd68f; --ok-bg:#122a1c; --warn:#e0b25c; --warn-bg:#2b2213; --bad:#ff8a80; --bad-bg:#2e1614;
  --c0:#9ec5f4; --c1:#6da7ec; --c2:#3987e5; --c3:#256abf; --c4:#184f95; } }
body { margin:0; padding:2rem 1.5rem; background:var(--bg); color:var(--fg);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:1150px; margin:0 auto; }
header.masthead { display:flex; align-items:center; gap:1.1rem; flex-wrap:wrap;
  padding-bottom:1rem; border-bottom:2px solid var(--rule); margin-bottom:1.25rem; }
header.masthead .logo { color:var(--logo); flex:0 0 auto; display:flex; align-items:center; }
header.masthead .logo svg, header.masthead .logo img { height:44px; width:auto; display:block; }
header.masthead .titles { flex:1 1 16rem; min-width:0; }
h1 { font-size:1.45rem; margin:0 0 .25rem; }
h2 { font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.35rem; border-bottom:2px solid var(--rule); }
h3 { font-size:1rem; margin:1.75rem 0 .5rem; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  overflow-wrap:anywhere; }
h3.category { font-family:inherit; font-size:1rem; margin:1.5rem 0 .5rem; }
h3.category .count { color:var(--muted); font-weight:400; font-size:.85rem; margin-left:.5rem; }
h4 { font-size:.8rem; font-weight:600; color:var(--muted); margin:1.25rem 0 .4rem;
  text-transform:uppercase; letter-spacing:.05em; }
.meta { color:var(--muted); margin:0 0 .5rem; }
.tally { margin:0 0 1.5rem; display:flex; gap:.5rem; flex-wrap:wrap; }
.tally span { padding:.15rem .55rem; border-radius:999px; font-weight:600; font-size:.82rem;
  border:1px solid var(--rule); }
.tally .ok { color:var(--ok); background:var(--ok-bg); }
.tally .bad { color:var(--bad); background:var(--bad-bg); }
.tally .warn { color:var(--warn); background:var(--warn-bg); }
.note { margin:.5rem 0; }
.empty { color:var(--muted); font-style:italic; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; background:var(--panel); }
th, td { padding:.4rem .6rem; border-bottom:1px solid var(--rule); text-align:left; white-space:nowrap; }
th { font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
td.ok { color:var(--ok); background:var(--ok-bg); font-weight:600; }
td.warn { color:var(--warn); background:var(--warn-bg); font-weight:600; }
td.bad { color:var(--bad); background:var(--bad-bg); font-weight:600; }
tbody tr:last-child td { border-bottom:none; }
pre.failure, pre.output { color:var(--fg); padding:.75rem 1rem; overflow-x:auto;
  font-size:12px; line-height:1.45; margin:.5rem 0 0; }
pre.failure { background:var(--bad-bg); border-left:3px solid var(--bad); }
pre.output { background:var(--panel); border-left:3px solid var(--rule); }
a { color:var(--c3); }
div.figure { border-top:2px solid var(--rule); margin-top:2.5rem; padding-top:.15rem; }
div.figure > h4 { margin-top:.9rem; color:var(--fg); font-size:.85rem; }
a.top { color:var(--muted); font-size:.8rem; text-decoration:none; }
.charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr)); gap:1.1rem 1.5rem;
  margin:.6rem 0 .2rem; }
.chart { margin:0; min-width:0; }
.chart svg { width:100%; height:auto; overflow:visible; display:block; }
.chart figcaption { color:var(--muted); font-size:.78rem; margin-top:.3rem; }
.chart figcaption b { color:var(--fg); font-weight:600; }
.chart figcaption.above { margin:0 0 .4rem; }
/* Values and labels wear text tokens; the bar alone carries the encoding. */
.chart .cl { fill:var(--muted); font-size:10px; text-anchor:end; dominant-baseline:middle; }
.chart .cv { fill:var(--fg); font-size:10px; dominant-baseline:middle; font-variant-numeric:tabular-nums; }
.chart .ca { stroke:var(--rule); stroke-width:1; }
.chart .cb0 { fill:var(--c0); } .chart .cb1 { fill:var(--c1); } .chart .cb2 { fill:var(--c2); }
.chart .cb3 { fill:var(--c3); } .chart .cb4 { fill:var(--c4); }
@media (max-width:640px) {
  body { padding:1.25rem 1rem; }
  header.masthead .logo svg, header.masthead .logo img { height:34px; }
}
"""


class Reporter:
    """
    Collect run outcomes and result tables, then render both into one file.
    """

    def __init__(
        self,
        path: Path | None = None,
        fmt: ReportFormat | str = ReportFormat.HTML,
        title: str = DEFAULT_TITLE,
        *,
        logo: Logo | None = None,
        category_order: list[str] | None = None,
        max_failure_snippets: int = DEFAULT_MAX_FAILURE_SNIPPETS,
    ):
        self._path = path
        self._format = ReportFormat(fmt)
        self._title = title
        self._logo = logo if logo is not None else Logo("")
        self._category_order = list(category_order or [])
        self._max_failure_snippets = max_failure_snippets
        self._started = dt.datetime.now().astimezone()
        self._sections: list[tuple[str, list[Table | _Note | _Output | _Figure]]] = []
        self._outcomes: list[_Outcome] = []
        self._header_tables: list[Table] = []

    @property
    def writes_file(self) -> bool:
        return self._path is not None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def format(self) -> ReportFormat:
        return self._format

    @property
    def title(self) -> str:
        return self._title

    # -- header tables -------------------------------------------------------------------

    def add_header_table(
        self,
        rows: list[list[str]],
        *,
        title: str,
        headers: list[str] | None = None,
        left: set[int] | None = None,
    ) -> None:
        """
        Add a table above the run summary, describing the run as a whole.
        """
        if any(table.title == title for table in self._header_tables):
            return
        table = Table(
            headers=headers or ["property", "value"],
            rows=[[str(cell) for cell in row] for row in rows],
            title=title,
            left=left if left is not None else {1},
        )
        self._header_tables.append(table)
        if not self.writes_file:
            logger.info(title)
            logger.info(render_table(table))
            return
        self.flush()

    def set_environment(self, rows: list[list[str]], title: str = "Environment") -> None:
        """What this run was pointed at -- profile, hardware shape, endpoints, driver."""
        self.add_header_table(rows, title=title)

    # -- detail half ---------------------------------------------------------------------

    def start_test(self, nodeid: str) -> None:
        """Open a detail section for *nodeid*, reusing it if it already exists."""
        if self.writes_file:
            self._section(nodeid)

    def note(self, text: str) -> None:
        """A line of prose -- a heading for the table that follows, or a warning."""
        if not self.writes_file:
            logger.info(text.strip())
            return
        self._append(_Note(text.strip()))

    def output(self, text: str, *, title: str = "", max_chars: int = MAX_SNIPPET_CHARS) -> None:
        """
        Add a block of captured output -- a command's stderr, a tool's summary.
        """
        text = failure_snippet(text, max_chars)
        if not text:
            return
        if not self.writes_file:
            if title:
                logger.info(title)
            logger.info(text)
            return
        self._append(_Output(text, title))

    def figure(self, markup: str, *, title: str = "", data_tables: tuple[str, ...] = ()) -> None:
        """
        Add an inline SVG figure, built by :mod:`.charts`.

        *data_tables* names the tables in this section holding the figure's numbers; each one
        present is linked from the figure.
        """
        if not markup or not self.writes_file:
            return
        self._append(_Figure(markup, title, tuple(data_tables)))

    def table(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        title: str = "",
        left: set[int] | None = None,
        status_column: int | None = None,
        status_classes: Mapping[str, str] | None = None,
    ) -> None:
        """
        Add one table. Cells must already be strings -- use the ``format_*`` helpers.

        *status_column* is coloured by its text: pytest's outcomes are known already, and
        *status_classes* maps any other status this table uses to ``ok``, ``warn`` or ``bad``.
        """
        built = Table(
            headers=headers,
            rows=[[str(cell) for cell in row] for row in rows],
            title=title,
            left=left or set(),
            status_column=status_column,
            status_classes=status_classes or {},
        )
        if not self.writes_file:
            if title:
                logger.info(title)
            logger.info(render_table(built))
            return
        self._append(built)

    # -- summary half -----------------------------------------

    def record_outcome(
        self,
        nodeid: str,
        outcome: str,
        duration: float,
        failure_text: str = "",
        category: str = UNCATEGORIZED,
    ) -> None:
        """
        Record one test's result for the summary table.
        """
        self._outcomes.append(
            _Outcome(
                nodeid=nodeid,
                outcome=outcome,
                duration=duration,
                category=category,
                failure_text=failure_text,
            )
        )
        self.flush()

    # -- rendering -----------------------------------------------------------------------

    def _section(self, nodeid: str) -> list[Table | _Note | _Output | _Figure]:
        """The item list for *nodeid*, created on first use."""
        for name, items in self._sections:
            if name == nodeid:
                return items
        items: list[Table | _Note | _Output | _Figure] = []
        self._sections.append((nodeid, items))
        return items

    def _append(self, item: Table | _Note | _Output | _Figure) -> None:
        if not self._sections:
            self._sections.append(("", []))
        self._sections[-1][1].append(item)
        self.flush()

    def flush(self) -> None:
        """Write the report file. A no-op when there is no report file."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        body = self._render_markdown() if self._format is ReportFormat.MD else self._render_html()
        self._path.write_text(body, encoding="utf-8")

    def _categories(self) -> list[tuple[str, list[_Outcome]]]:
        """
        Outcomes grouped into category sections, ordered as the run configured them.
        """
        grouped: dict[str, list[_Outcome]] = {}
        for outcome in self._outcomes:
            grouped.setdefault(outcome.category, []).append(outcome)

        def rank(name: str) -> tuple[int, str]:
            if name in self._category_order:
                return self._category_order.index(name), ""
            return (len(self._category_order) + (1 if name == UNCATEGORIZED else 0), name)

        return [(name, grouped[name]) for name in sorted(grouped, key=rank)]

    def _category_table(self, outcomes: list[_Outcome]) -> Table:
        """One category's tests: what each is called, how it ended, how long it took."""
        return Table(
            headers=["test", "outcome", "duration"],
            rows=[[o.name, o.outcome, f"{o.duration:.2f}s"] for o in outcomes],
            status_column=1,
        )

    def _category_counts(self, outcomes: list[_Outcome]) -> str:
        """Pass Fail counts for a category heading."""
        counts: dict[str, int] = {}
        for outcome in outcomes:
            counts[outcome.outcome] = counts.get(outcome.outcome, 0) + 1
        return ", ".join(f"{counts[name]} {name}" for name in sorted(counts))

    def _failures(self) -> list[_Outcome]:
        return [o for o in self._outcomes if o.failed]

    def _failure_table(self, failures: list[_Outcome]) -> Table:
        """Every failed test in one table, whatever category it ran under."""
        return Table(
            headers=["test", "category", "outcome", "error"],
            rows=[
                [o.name, o.category, o.outcome, summary_line(o.failure_text) or "(no detail captured)"]
                for o in failures
            ],
            left={1, 2, 3},
            status_column=2,
        )

    def _tally(self) -> list[tuple[str, str]]:
        """(label, outcome-or-empty) pairs for the counts line, in a stable order."""
        counts: dict[str, int] = {}
        for outcome in self._outcomes:
            counts[outcome.outcome] = counts.get(outcome.outcome, 0) + 1
        total = sum(outcome.duration for outcome in self._outcomes)
        pairs = [(f"{counts[name]} {name}", name) for name in sorted(counts)]
        pairs.append((f"{total:.1f}s total", ""))
        return pairs

    def _render_html(self) -> str:
        detail: list[str] = []
        for index, (nodeid, items) in enumerate(self._sections):
            if not items:
                continue
            if nodeid:
                detail.append(f"<h3>{html.escape(nodeid)}</h3>")
            tables = {item.title for item in items if isinstance(item, Table) and item.title}
            for item in items:
                match item:
                    case _Note(text):
                        detail.append(f"<p class='note'>{html.escape(text)}</p>")
                    case _Output(text, title):
                        heading = f"<h4>{html.escape(title)}</h4>" if title else ""
                        detail.append(f"{heading}<pre class='output'>{html.escape(text)}</pre>")
                    case _Figure(markup, title, data_tables):
                        heading = f"<h4>{html.escape(title)}</h4>" if title else ""
                        links = ", ".join(
                            f"<a href='#{_anchor(index, name)}'>{html.escape(name)}</a>"
                            for name in data_tables
                            if name in tables
                        )
                        data = f"<p class='note'>Data: {links}</p>" if links else ""
                        detail.append(f"<div class='figure'>{heading}{data}{markup}</div>")
                    case Table() as table:
                        heading = (
                            f"<h4 id='{_anchor(index, table.title)}'>{html.escape(table.title)}</h4>"
                            if table.title
                            else ""
                        )
                        detail.append(f"{heading}<div class='scroll'>{render_table_html(table)}</div>")

        chips = "".join(
            f"<span class='{STATUS_CLASS.get(outcome, '')}'>{html.escape(label)}</span>"
            for label, outcome in self._tally()
        )
        details_section = "<h2>Details</h2>" + "".join(detail) if detail else ""
        environment = "".join(
            f"<h2>{html.escape(table.title)}</h2><div class='scroll'>{render_table_html(table)}</div>"
            for table in self._header_tables
        )

        categories = self._categories()
        if categories:
            results = "".join(
                f"<h3 class='category'>{html.escape(name)}"
                f"<span class='count'>{html.escape(self._category_counts(outcomes))}</span></h3>"
                f"<div class='scroll'>{render_table_html(self._category_table(outcomes))}</div>"
                for name, outcomes in categories
            )
        else:
            results = "<p class='empty'>(no tests ran)</p>"

        failures = self._failures()
        failures_section = ""
        if failures:
            snippets = "".join(
                f"<h3>{html.escape(o.nodeid)}</h3>"
                f"<pre class='failure'>{html.escape(failure_snippet(o.failure_text))}</pre>"
                for o in failures[: self._max_failure_snippets]
                if o.failure_text
            )
            hidden = len(failures) - self._max_failure_snippets
            more = (
                f"<p class='note'>{hidden} further failure(s) are listed above; their output is in the job log.</p>"
                if hidden > 0
                else ""
            )
            failures_section = (
                f"<h2>Failures ({len(failures)})</h2>"
                f"<div class='scroll'>{render_table_html(self._failure_table(failures))}</div>"
                f"{snippets}{more}"
            )

        logo = f"<div class='logo'>{self._logo.markup}</div>" if not self._logo.is_empty else ""
        return (
            "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(self._title)}</title><style>{_CSS}</style></head><body><main>"
            f"<header class='masthead'>{logo}<div class='titles'>"
            f"<h1>{html.escape(self._title)}</h1>"
            f"<p class='meta'>{self._started:%Y-%m-%d %H:%M:%S %Z}</p>"
            "</div></header>"
            f"<p class='tally'>{chips}</p>"
            f"{environment}"
            "<h2>Test results</h2>"
            f"{results}"
            f"{failures_section}"
            f"{details_section}"
            "</main></body></html>\n"
        )

    def _render_markdown(self) -> str:
        out: list[str] = [
            f"# {self._title}",
            "",
            f"{self._started:%Y-%m-%d %H:%M:%S %Z} — " + ", ".join(label for label, _ in self._tally()),
            "",
        ]
        for table in self._header_tables:
            out += [f"## {table.title}", "", render_table_markdown(table), ""]

        out += ["## Test results", ""]
        categories = self._categories()
        if not categories:
            out += ["_(no tests ran)_", ""]
        for name, outcomes in categories:
            out += [
                f"### {name} — {self._category_counts(outcomes)}",
                "",
                render_table_markdown(self._category_table(outcomes)),
                "",
            ]

        failures = self._failures()
        if failures:
            out += [
                f"## Failures ({len(failures)})",
                "",
                render_table_markdown(self._failure_table(failures)),
                "",
            ]
            for outcome in failures[: self._max_failure_snippets]:
                if not outcome.failure_text:
                    continue
                out += [f"### {outcome.nodeid}", "", "```text", failure_snippet(outcome.failure_text), "```", ""]
            hidden = len(failures) - self._max_failure_snippets
            if hidden > 0:
                out += [f"_{hidden} further failure(s) are listed above; their output is in the job log._", ""]

        detail: list[str] = []
        for nodeid, items in self._sections:
            if not items:
                continue
            if nodeid:
                detail += [f"### {nodeid}", ""]
            for item in items:
                match item:
                    case _Note(text):
                        detail += [text, ""]
                    case _Output(text, title):
                        if title:
                            detail += [f"#### {title}", ""]
                        detail += ["```text", text, "```", ""]
                    case _Figure():
                        continue
                    case Table() as table:
                        if table.title:
                            detail += [f"#### {table.title}", ""]
                        detail += [render_table_markdown(table), ""]

        if detail:
            out += ["## Details", "", *detail]
        return "\n".join(out).rstrip() + "\n"
