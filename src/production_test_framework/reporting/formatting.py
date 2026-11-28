# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.
"""
Sink-agnostic building blocks for a report: number formatting, tables, status vocabulary.
"""

import html
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "COMPACT_DECIMALS",
    "SIGNIFICANT_DECIMALS",
    "STATUS_CLASS",
    "ReportFormat",
    "Table",
    "format_cell",
    "format_delta",
    "format_duration",
    "format_for_unit",
    "format_number",
    "format_value",
    "md_escape",
    "render_table",
    "render_table_html",
    "render_table_markdown",
]


class ReportFormat(StrEnum):
    """
    Output format for the report file.
    """

    HTML = "html"
    MD = "md"

    @classmethod
    def for_path(cls, path: Path) -> ReportFormat:
        """Infer a format from *path*'s suffix, defaulting to HTML for anything unfamiliar."""
        return cls.MD if path.suffix.lower() in (".md", ".markdown") else cls.HTML


#: How a status cell is coloured: ``ok``, ``warn`` or ``bad``, keyed by the cell's text. Only
#: pytest's own outcomes are known here; a table adds its suite's statuses with
#: ``status_classes``.
STATUS_CLASS = {
    "passed": "ok",
    "failed": "bad",
    "error": "bad",
    "skipped": "warn",
    "xfailed": "warn",
    "xpassed": "warn",
}


SIGNIFICANT_DECIMALS = 4

COMPACT_DECIMALS = 2


def _decimal_places(value: float, significant: int | None = None) -> int:
    """
    How many decimals to show for *value*: as many as it has, capped at *significant* ones.
    """
    if significant is None:
        significant = SIGNIFICANT_DECIMALS

    text = repr(float(value))
    if "e" in text or "E" in text:
        exponent = math.floor(math.log10(abs(value)))
        return max(0, -exponent - 1) + significant

    fraction = text.partition(".")[2].rstrip("0")
    if len(fraction) <= COMPACT_DECIMALS:
        return len(fraction)
    leading_zeros = len(fraction) - len(fraction.lstrip("0"))
    return min(len(fraction), leading_zeros + significant)


def _format_decimal(value: float, significant: int | None = None) -> str:
    """
    Grouped decimal notation, never scientific.
    """
    if not math.isfinite(value):
        return str(value)
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.{_decimal_places(value, significant)}f}"


def format_value(value: float | None, significant: int | None = None) -> str:
    """Render a metric total, distinguishing an absent series from a zero one."""
    return "absent" if value is None else _format_decimal(value, significant)


def format_number(value: float | None, spec: str | None = None, significant: int | None = None) -> str:
    """
    Render a benchmark number, or "-" when the run did not report it.
    """
    if value is None:
        return "-"
    return f"{value:{spec}}" if spec else _format_decimal(value, significant)


def _unit_decimals(unit: str) -> int | None:
    """Decimal places shown for *unit*: 1 for a token rate, 2 for a dollar amount."""
    if unit.startswith("$"):
        return 2
    if "tok/s" in unit or "tokens/sec" in unit:
        return 1
    return None


def format_for_unit(value: float, unit: str = "", significant: int | None = None) -> str:
    """*value* rounded, halves up, to its unit's decimal places, or as :func:`format_number` shows it."""
    places = _unit_decimals(unit)
    if places is None:
        return format_number(value, significant=significant)
    rounded = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{rounded:,.{places}f}"


def format_cell(value: Any, unit: str = "") -> str:
    """A table cell for any value: "-" for None, true/false, a number for its *unit*, a list joined."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return format_for_unit(value, unit)
    if isinstance(value, list):
        return ", ".join(format_cell(item) for item in value) or "-"
    return str(value)


def format_duration(seconds: float | None) -> str:
    """
    Converts seconds as ``1 hr 12 mins 3 secs`` format, or "-" when the run did not report it.
    """
    if seconds is None:
        return "-"
    if not math.isfinite(seconds):
        return str(seconds)
    if abs(seconds) < 60:
        return f"{_format_decimal(seconds)} {'sec' if seconds == 1 else 'secs'}"

    whole = round(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    for amount, unit in ((hours, "hr"), (minutes, "min"), (secs, "sec")):
        if amount:
            parts.append(f"{amount} {unit}" + ("" if amount == 1 else "s"))
    return " ".join(parts)


def format_delta(baseline: float | None, current: float | None) -> str:
    """
    Percentage change from *baseline* to *current*.
    """
    if baseline is None or current is None:
        return "-"
    if baseline == 0:
        return "new" if current > 0 else "+0.00%"
    return f"{(current - baseline) / baseline * 100:+.2f}%"


@dataclass
class Table:
    """
    Table formatting data class
    """

    headers: list[str]
    rows: list[list[str]]
    title: str = ""
    left: set[int] = field(default_factory=set)
    status_column: int | None = None
    status_classes: Mapping[str, str] = field(default_factory=dict)


def render_table(table: Table, indent: str = "  ") -> str:
    """Render *table* as fixed-width text, returned as one string."""
    if not table.rows:
        return f"{indent}(no rows)"

    left_aligned = {0} | table.left
    widths = [max(len(row[i]) for row in (table.headers, *table.rows)) for i in range(len(table.headers))]

    def line(cells: list[str]) -> str:
        rendered = (
            cell.ljust(widths[i]) if i in left_aligned else cell.rjust(widths[i]) for i, cell in enumerate(cells)
        )
        return (indent + "  ".join(rendered)).rstrip()

    separator = ["-" * width for width in widths]
    return "\n".join([line(table.headers), line(separator), *(line(row) for row in table.rows)])


def render_table_html(table: Table) -> str:
    """Render *table* as an HTML table, escaping every cell."""
    if not table.rows:
        return "<p class='empty'>(no rows)</p>"

    left_aligned = {0} | table.left

    def cell(tag: str, index: int, text: str) -> str:
        classes = [] if index in left_aligned else ["num"]
        if tag == "td" and index == table.status_column:
            classes.append(table.status_classes.get(text) or STATUS_CLASS.get(text, ""))
        present = [c for c in classes if c]
        attr = f" class='{' '.join(present)}'" if present else ""
        return f"<{tag}{attr}>{html.escape(text)}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(table.headers))
    body = "".join(
        "<tr>" + "".join(cell("td", i, value) for i, value in enumerate(row)) + "</tr>" for row in table.rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def md_escape(text: str) -> str:
    """Escape the one character that would break a Markdown table cell."""
    return text.replace("|", r"\|")


def render_table_markdown(table: Table) -> str:
    """
    Render *table* as a Markdown table with every column left-aligned and padded.
    """
    if not table.rows:
        return "_(no rows)_"

    cells = [[md_escape(c) for c in table.headers], *[[md_escape(c) for c in row] for row in table.rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(table.headers))]

    def line(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"

    rule = [":" + "-" * (width - 1) if width > 1 else ":" for width in widths]
    return "\n".join([line(cells[0]), "| " + " | ".join(rule) + " |", *(line(row) for row in cells[1:])])
