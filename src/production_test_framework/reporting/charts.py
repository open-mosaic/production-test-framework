# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2025 Delos Data, Inc.
"""
Inline SVG figures for the report.
"""

import html
from collections.abc import Callable

from .formatting import format_number

__all__ = ["latency_percentile_figure", "scatter_figure"]

LATENCY_FAMILIES = (
    ("ttft", "TTFT", "time to first token"),
    ("tpot", "TPOT", "time per output token"),
    ("itl", "ITL", "inter-token latency"),
    ("e2el", "E2EL", "end-to-end latency"),
)

PERCENTILES = (
    ("mean", "mean"),
    ("median", "p50"),
    ("p90", "p90"),
    ("p99", "p99"),
    ("p99_9", "p99.9"),
)

MIN_BARS = 2

_WIDTH = 360
_ROW = 20
_BAR = 13
_LABEL_GUTTER = 46
_VALUE_GUTTER = 74
_RADIUS = 4


def _bar_path(x: float, y: float, width: float, height: float, radius: float = _RADIUS) -> str:
    """
    A bar with its data-end rounded and its baseline end square.
    """
    r = min(radius, width, height / 2)
    if r <= 0:
        return f"M{x},{y} h{width} v{height} h{-width} Z"
    right = x + width
    return (
        f"M{x},{y} H{right - r} A{r},{r} 0 0 1 {right},{y + r} "
        f"V{y + height - r} A{r},{r} 0 0 1 {right - r},{y + height} H{x} Z"
    )


def _family_values(latency: dict[str, float], suffix: str) -> list[tuple[str, float]]:
    """``[(label, value)]`` for one family, in percentile order, skipping what the run omitted."""
    found = []
    for key, label in PERCENTILES:
        value = latency.get(f"{key}_{suffix}")
        if value is not None:
            found.append((label, float(value)))
    return found


def _panel(name: str, description: str, values: list[tuple[str, float]]) -> str:
    """One small multiple: a horizontal bar per percentile, on this family's own scale."""
    height = _ROW * len(values)
    largest = max(value for _, value in values) or 1.0
    span = _WIDTH - _LABEL_GUTTER - _VALUE_GUTTER

    last = len(values) - 1
    steps = [round(index * 4 / last) if last else 0 for index in range(len(values))]

    marks = []
    for index, ((label, value), step) in enumerate(zip(values, steps, strict=True)):
        y = index * _ROW + (_ROW - _BAR) / 2
        width = max(value / largest * span, 1.0)
        reading = f"{format_number(value)} ms"
        marks.append(
            f"<g><title>{html.escape(f'{name} {label}: {reading}')}</title>"
            f"<text class='cl' x='{_LABEL_GUTTER - 8}' y='{y + _BAR / 2}'>{html.escape(label)}</text>"
            f"<path class='cb{step}' d='{_bar_path(_LABEL_GUTTER, y, width, _BAR)}'/>"
            f"<text class='cv' x='{_LABEL_GUTTER + width + 6}' y='{y + _BAR / 2}'>{html.escape(reading)}</text>"
            "</g>"
        )

    return (
        "<figure class='chart'>"
        f"<figcaption class='above'><b>{html.escape(name)}</b> — {html.escape(description)} (ms)</figcaption>"
        f"<svg viewBox='0 0 {_WIDTH} {height}' role='img' "
        f"aria-label='{html.escape(f'{name} latency percentiles, milliseconds')}'>"
        f"<line class='ca' x1='{_LABEL_GUTTER}' y1='0' x2='{_LABEL_GUTTER}' y2='{height}'/>"
        f"{''.join(marks)}</svg>"
        "</figure>"
    )


def latency_percentile_figure(latency: dict[str, float]) -> str:
    """
    The latency distribution as small multiples, or "" when the run reported too little.

    One panel per family, each on **its own scale** -- TTFT and TPOT differ by two orders of
    magnitude, and a shared axis would flatten whichever is smaller into nothing. Every bar is
    labelled with its value so the panels are never compared by length alone.
    """
    panels = [
        _panel(name, description, values)
        for suffix, name, description in LATENCY_FAMILIES
        if len(values := _family_values(latency, suffix)) >= MIN_BARS
    ]
    if not panels:
        return ""
    return f"<div class='charts'>{''.join(panels)}</div>"


_SCATTER_W = 360
_SCATTER_H = 200
_SCATTER_PAD = (78, 12, 14, 30)  # left, right, top, bottom


def _two_places(value: float, unit: str) -> str:
    return format_number(value, significant=2)


#: A y-axis title longer than this puts its "(unit)" on a second line.
_Y_TITLE_ONE_LINE = 18


def _y_axis_title(text: str, middle: float, axis_x: float) -> str:
    """
    The rotated y-axis title, centred on the axis and just left of it, its unit on a second line
    when it is long. The axis's numbers sit only at its ends, so the title never meets them.
    """
    name, separator, unit = text.rpartition(" (")
    lines = [name, f"({unit}"] if separator and len(text) > _Y_TITLE_ONE_LINE else [text]
    # Rotated -90 degrees, each later line sits to the right of the one before, nearer the plot.
    spans = "".join(
        f"<tspan x='0' dy='{'0' if index == 0 else '1.15em'}'>{html.escape(line)}</tspan>"
        for index, line in enumerate(lines)
    )
    # Each line's baseline is its right edge once rotated; the last one sits 6px off the axis.
    x = axis_x - 6 - 11.5 * (len(lines) - 1)
    return (
        f"<text class='cv' x='0' y='0' text-anchor='middle' transform='translate({x} {middle}) rotate(-90)'>"
        f"{spans}</text>"
    )


def _scatter_panel(
    title: str,
    x_label: str,
    y_label: str,
    points: list[tuple[float, float, str, bool]],
    x_title: str = "",
    y_title: str = "",
    *,
    formatter: Callable[[float, str], str] = _two_places,
) -> str:
    left, right, top, bottom = _SCATTER_PAD
    width, height = _SCATTER_W - left - right, _SCATTER_H - top - bottom
    xs = [x for x, _, _, _ in points]
    ys = [y for _, y, _, _ in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(0.0, min(ys)), max(ys)
    x_span = (x_hi - x_lo) or abs(x_hi) or 1.0
    y_span = (y_hi - y_lo) or abs(y_hi) or 1.0
    x_lo, x_hi = x_lo - x_span * 0.1, x_hi + x_span * 0.1
    if min(xs) >= 0:
        x_lo = max(0.0, x_lo)
    y_hi += y_span * 0.1

    def place(x: float, y: float) -> tuple[float, float]:
        return left + (x - x_lo) / (x_hi - x_lo) * width, top + height - (y - y_lo) / (y_hi - y_lo) * height

    marks = []
    for x, y, label, current in sorted(points, key=lambda point: point[3]):
        px, py = place(x, y)
        reading = f"{label}: {formatter(x, x_label)} {x_label}, {formatter(y, y_label)} {y_label}"
        marks.append(
            f"<g><title>{html.escape(reading)}</title>"
            f"<circle class='{'cb4' if current else 'cb1'}' cx='{px:.1f}' cy='{py:.1f}' r='{5 if current else 3.5}'/>"
            "</g>"
        )
    tick = {
        name: html.escape(formatter(value, unit))
        for name, value, unit in (
            ("x_lo", x_lo, x_label),
            ("x_hi", x_hi, x_label),
            ("y_lo", y_lo, y_label),
            ("y_hi", y_hi, y_label),
        )
    }
    base, middle = top + height, left + width / 2
    axes = (
        f"<line class='ca' x1='{left}' y1='{base}' x2='{left + width}' y2='{base}'/>"
        f"<line class='ca' x1='{left}' y1='{top}' x2='{left}' y2='{base}'/>"
        f"<text class='cl' x='{left - 4}' y='{top}'>{tick['y_hi']}</text>"
        f"<text class='cl' x='{left - 4}' y='{base}'>{tick['y_lo']}</text>"
        f"<text class='cv' x='{left}' y='{base + 12}'>{tick['x_lo']}</text>"
        f"<text class='cl' x='{left + width}' y='{base + 12}'>{tick['x_hi']}</text>"
        f"<text class='cv' x='{middle}' y='{base + 24}' text-anchor='middle'>{html.escape(x_title or x_label)}</text>"
        f"{_y_axis_title(y_title or y_label, top + height / 2, left)}"
    )
    return (
        "<figure class='chart'>"
        f"<figcaption class='above'><b>{html.escape(title)}</b> ({html.escape(y_label)})</figcaption>"
        f"<svg viewBox='0 0 {_SCATTER_W} {_SCATTER_H}' role='img' aria-label='{html.escape(title)}'>"
        f"{axes}{''.join(marks)}</svg>"
        "</figure>"
    )


def _scatter_legend(entries: list[tuple[str, bool]]) -> str:
    """One row naming each kind of point the panels draw."""
    marks, x = [], 6.0
    for label, current in entries:
        marks.append(
            f"<circle class='{'cb4' if current else 'cb1'}' cx='{x}' cy='9' r='{5 if current else 3.5}'/>"
            f"<text class='cv' x='{x + 10}' y='9'>{html.escape(label)}</text>"
        )
        x += 24 + 6.5 * len(label)
    # Drawn 1.4x so its text matches the panels' scaled-up labels.
    size = f"viewBox='0 0 {x:.0f} 18' style='width:{x * 1.4:.0f}px'"
    return f"<div class='chart'><svg {size} role='img' aria-label='legend'>{''.join(marks)}</svg></div>"


def scatter_figure(
    panels: list[tuple],
    legend: tuple[str, str] = ("this run", "other runs"),
    formatter: Callable[[float, str], str] = _two_places,
) -> str:
    """
    Small-multiple scatter plots, one per ``(title, x_unit, y_unit, points[, x_title, y_title])``
    panel, under a legend. The axis titles default to the units.

    Each point is ``(x, y, label, current)``; current points are drawn larger and darker and named
    by ``legend[0]``, the rest by ``legend[1]``. Panels with no point are left out, and "" is
    returned when none remain. *formatter* renders a value given its axis's unit; each point's
    label shows when it is hovered.
    """
    drawn = [_scatter_panel(*panel, formatter=formatter) for panel in panels if panel[3]]
    if not drawn:
        return ""
    kinds = {current for panel in panels for *_, current in panel[3]}
    entries = [(name, current) for name, current in ((legend[0], True), (legend[1], False)) if current in kinds]
    return f"{_scatter_legend(entries)}<div class='charts'>{''.join(drawn)}</div>"
