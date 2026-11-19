"""What the results view shows: tables of a steady state and charts of a run in time.

**Pure** (no aquakin, no trame). The worker reads a study's results
(``study_results``: values by stream and channel, with units); this module
turns them into rows of text and SVG charts the page displays as they are.
"""

from __future__ import annotations

import math
from html import escape

__all__ = ["NO_VALUE", "channels_of", "fmt", "line_chart", "steady_table"]

#: What a missing (or not finite) value reads as: an en dash.
NO_VALUE = "\u2013"
#: The most hover targets one chart carries; a longer run is sampled evenly.
MAX_HOVER = 240
#: The most points one line is drawn with; a longer run is thinned evenly.
MAX_POINTS = 2000


def fmt(value) -> str:
    """A value as the tables and tooltips show it: four significant figures,
    thousands grouped, never an exponent between 0.001 and a billion."""
    if value is None:
        return NO_VALUE
    value = float(value)
    if value == 0:
        return "0"
    size = abs(value)
    if size >= 1e9 or size < 1e-3:
        return f"{value:.3e}"
    if size >= 1000:
        return f"{value:,.0f}"
    decimals = max(0, 3 - math.floor(math.log10(size)))
    return f"{value:.{decimals}f}"


def _unit(units: dict, stream: str, channel: str) -> str:
    return (units.get(stream) or {}).get(channel, "")


def steady_table(values: dict, units: dict) -> dict:
    """``{stream: {channel: value}}`` as a table: a column per stream, a row per
    channel (in the order they first appear), each with its unit.

    Returns
    -------
    dict
        ``{"columns": [stream, ...], "rows": [{"channel", "unit", "values": [text]}]}``.
    """
    streams = list(values)
    channels: list[str] = []
    for stream in streams:
        channels += [c for c in values[stream] if c not in channels]
    rows = []
    for channel in channels:
        unit = next((_unit(units, s, channel) for s in streams if _unit(units, s, channel)), "")
        rows.append(
            {
                "channel": channel,
                "unit": unit,
                "values": [fmt(values[s].get(channel)) for s in streams],
            }
        )
    return {"columns": streams, "rows": rows}


def channels_of(results: dict, stream: str) -> list[str]:
    """The channels a run in time recorded for ``stream``: its measurements
    first, then its flow, species and temperature."""
    out: list[str] = []
    for key in ("measured", "streams"):
        out += [c for c in (results.get(key) or {}).get(stream, {}) if c not in out]
    return out


def _series(results: dict, stream: str, channel: str) -> list:
    for key in ("measured", "streams"):
        values = (results.get(key) or {}).get(stream, {}).get(channel)
        if values is not None:
            return values
    raise KeyError(f"{stream} has no channel {channel!r}")


def _ticks(low: float, high: float, count: int = 4) -> list[float]:
    """Round tick values covering ``[low, high]``."""
    if not high - low > 1e-9 * max(abs(low), abs(high)):  # flat: centre it
        pad = 0.05 * abs(low) or 1.0
        low, high = low - pad, high + pad
    raw = (high - low) / count
    step = 10 ** math.floor(math.log10(raw))  # so step <= raw < 10 step
    step *= next(f for f in (1, 2, 5, 10) if raw <= f * step)
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value < high + step * 0.999:
        ticks.append(round(value, 12))
        value += step
    return ticks


def _tick_text(value: float, step: float) -> str:
    if abs(value) >= 10_000:
        return f"{value / 1000:,.0f}K" if step >= 1000 else f"{value:,.0f}"
    decimals = max(0, -math.floor(math.log10(step))) if step < 1 else 0
    return f"{value:,.{decimals}f}"


def line_chart(results: dict, stream: str, channel: str, *, width=420, height=200) -> str:
    """One channel of a stream over time, as an SVG chart.

    A single 2px line on a hairline grid; the title names the channel and its
    unit. Hovering anywhere over the plot shows a crosshair and the value at the
    nearest saved time (a tooltip of the browser's own).

    Parameters
    ----------
    results : dict
        The worker's ``study_results`` reply for a run in time.
    stream, channel : str
        What to draw.
    width, height : int
        The chart's size in pixels (it scales to its container).

    Returns
    -------
    str
        The ``<svg>`` element.
    """
    t = results["t"]
    y = _series(results, stream, channel)
    unit = _unit(results.get("units") or {}, stream, channel)
    time_unit = results.get("time_unit") or ""
    points = [(a, b) for a, b in zip(t, y) if a is not None and b is not None]
    title = escape(f"{channel}" + (f" ({unit})" if unit else ""))
    left, right, top, bottom = 52, 12, 26, 22
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [
        f'<svg class="ak-chart" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="{title} of {escape(stream)} over time">',
        f'<text class="ak-chart-title" x="{left}" y="16">{title}</text>',
    ]
    if not points:
        parts.append(
            f'<text class="ak-chart-axis" x="{left}" y="{top + plot_h / 2}">no finite values</text>'
        )
        return "".join(parts) + "</svg>"
    ts = [p[0] for p in points]
    ys = [p[1] for p in points]
    t0, t1 = min(ts), max(ts)
    y_ticks = _ticks(min(ys), max(ys))
    x_ticks = _ticks(t0, t1, 5) if t1 > t0 else [t0]
    y0, y1 = y_ticks[0], y_ticks[-1]
    x_lo, x_hi = (t0, t1) if t1 > t0 else (t0 - 0.5, t0 + 0.5)

    def sx(v):
        return left + (v - x_lo) / (x_hi - x_lo) * plot_w

    def sy(v):
        return top + plot_h - (v - y0) / ((y1 - y0) or 1.0) * plot_h

    y_step = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else 1.0
    for v in y_ticks:
        parts.append(
            f'<line class="ak-chart-grid" x1="{left}" x2="{left + plot_w}" '
            f'y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
            f'<text class="ak-chart-axis" x="{left - 6}" y="{sy(v) + 4:.1f}" '
            f'text-anchor="end">{_tick_text(v, y_step)}</text>'
        )
    x_step = x_ticks[1] - x_ticks[0] if len(x_ticks) > 1 else 1.0
    for v in x_ticks:
        if x_lo - 1e-12 <= v <= x_hi + 1e-12:
            parts.append(
                f'<text class="ak-chart-axis" x="{sx(v):.1f}" y="{top + plot_h + 16}" '
                f'text-anchor="middle">{_tick_text(v, x_step)}</text>'
            )
    if time_unit:
        parts.append(
            f'<text class="ak-chart-axis" x="{left + plot_w}" y="16" '
            f'text-anchor="end">time ({escape(time_unit)})</text>'
        )
    stride = max(1, math.ceil(len(points) / MAX_POINTS))
    drawn = points[::stride] if points[-1] in points[::stride] else [*points[::stride], points[-1]]
    path = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in drawn)
    parts.append(f'<polyline class="ak-chart-line" points="{path}"/>')
    # Hover: a band per sampled time, each with its crosshair, dot and tooltip.
    hover = points[:: max(1, math.ceil(len(points) / MAX_HOVER))]
    edges = [left] + [(sx(a[0]) + sx(b[0])) / 2 for a, b in zip(hover, hover[1:])] + [left + plot_w]
    for (a, b), x_from, x_to in zip(hover, edges, edges[1:]):
        tip = escape(f"t = {fmt(a)}{(' ' + time_unit) if time_unit else ''}: {fmt(b)} {unit}")
        parts.append(
            f'<g class="ak-chart-hit"><title>{tip}</title>'
            f'<rect x="{x_from:.1f}" y="{top}" width="{max(x_to - x_from, 0.5):.1f}" '
            f'height="{plot_h}"/>'
            f'<line x1="{sx(a):.1f}" x2="{sx(a):.1f}" y1="{top}" y2="{top + plot_h}"/>'
            f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="4"/></g>'
        )
    return "".join(parts) + "</svg>"
