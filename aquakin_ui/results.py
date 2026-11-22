"""What the results view shows: tables of a steady state and charts of a run in time.

**Pure** (no aquakin, no trame). The worker reads a study's results
(``study_results``: values by stream and channel, with units); this module
turns them into rows of text and SVG charts the page displays as they are.
"""

from __future__ import annotations

import math
from html import escape

__all__ = [
    "NO_VALUE",
    "analysis_view",
    "channels_of",
    "fmt",
    "line_chart",
    "steady_table",
    "xy_chart",
]

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
    unit = _unit(results.get("units") or {}, stream, channel)
    time_unit = results.get("time_unit") or ""
    return xy_chart(
        title=channel + (f" ({unit})" if unit else ""),
        label=f"{channel} of {stream} over time",
        x_name="t",
        x_unit=time_unit,
        y_unit=unit,
        line=list(zip(results["t"], _series(results, stream, channel))),
        width=width,
        height=height,
    )


def xy_chart(
    *,
    title: str,
    label: str,
    x_name: str,
    x_unit: str = "",
    y_unit: str = "",
    line=None,
    dots=None,
    reference: tuple[float, str] | None = None,
    x_axis: str | None = None,
    width=420,
    height=200,
) -> str:
    """An SVG chart of one quantity: a line, data points, or both on one axis.

    Parameters
    ----------
    title : str
        Named above the plot (with the unit, when it has one).
    label : str
        What the chart shows, for a screen reader.
    x_name, x_unit, y_unit : str
        For the hover tooltips (``t = 0.5 d: 2.3 g/m³``) and the x-axis title.
    line, dots : list of (x, y), optional
        A 2px line (the palette's first colour) and points (its second); with
        both, a legend names them. ``None`` values are skipped.
    reference : (value, text), optional
        A horizontal hairline at ``value`` with its text (a threshold).
    x_axis : str, optional
        The x-axis title (default ``time (<x_unit>)`` when there is a unit).
    width, height : int
        The chart's size in pixels (it scales to its container).

    Returns
    -------
    str
        The ``<svg>`` element.
    """
    line = [(a, b) for a, b in (line or []) if a is not None and b is not None]
    dots = [(a, b) for a, b in (dots or []) if a is not None and b is not None]
    title = escape(title)
    left, right, top, bottom = 52, 12, 26, 22
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [
        f'<svg class="ak-chart" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="{escape(label)}">',
        f'<text class="ak-chart-title" x="{left}" y="16">{title}</text>',
    ]
    points = line + dots
    if not points:
        parts.append(
            f'<text class="ak-chart-axis" x="{left}" y="{top + plot_h / 2}">no finite values</text>'
        )
        return "".join(parts) + "</svg>"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points] + ([reference[0]] if reference else [])
    t0, t1 = min(xs), max(xs)
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
    axis = x_axis if x_axis is not None else (f"time ({x_unit})" if x_unit else "")
    legend = line and dots
    if axis and not legend:
        parts.append(
            f'<text class="ak-chart-axis" x="{left + plot_w}" y="16" '
            f'text-anchor="end">{escape(axis)}</text>'
        )
    if legend:
        x = left + plot_w
        parts.append(
            f'<text class="ak-chart-axis" x="{x}" y="16" text-anchor="end">fit</text>'
            f'<line class="ak-chart-line" x1="{x - 38}" x2="{x - 22}" y1="12" y2="12"/>'
            f'<text class="ak-chart-axis" x="{x - 46}" y="16" text-anchor="end">data</text>'
            f'<circle class="ak-chart-dot" cx="{x - 80}" cy="12" r="4"/>'
        )
    if reference:
        value, text = reference
        parts.append(
            f'<line class="ak-chart-ref" x1="{left}" x2="{left + plot_w}" '
            f'y1="{sy(value):.1f}" y2="{sy(value):.1f}"/>'
            f'<text class="ak-chart-axis" x="{left + 4}" y="{sy(value) - 4:.1f}">'
            f"{escape(text)}</text>"
        )
    if line:
        stride = max(1, math.ceil(len(line) / MAX_POINTS))
        drawn = line[::stride] if line[-1] in line[::stride] else [*line[::stride], line[-1]]
        path = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in drawn)
        parts.append(f'<polyline class="ak-chart-line" points="{path}"/>')
    unit_x = f" {x_unit}" if x_unit else ""
    for a, b in dots[:MAX_POINTS]:
        tip = escape(f"data {x_name} = {fmt(a)}{unit_x}: {fmt(b)} {y_unit}".rstrip())
        parts.append(
            f'<circle class="ak-chart-dot" cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="4">'
            f"<title>{tip}</title></circle>"
        )
    # Hover on the line: a band per sampled point, with its crosshair, dot and tooltip.
    hover = line[:: max(1, math.ceil(len(line) / MAX_HOVER))] if line else []
    edges = [left] + [(sx(a[0]) + sx(b[0])) / 2 for a, b in zip(hover, hover[1:])] + [left + plot_w]
    for (a, b), x_from, x_to in zip(hover, edges, edges[1:]):
        tip = escape(f"{x_name} = {fmt(a)}{unit_x}: {fmt(b)} {y_unit}".rstrip())
        parts.append(
            f'<g class="ak-chart-hit"><title>{tip}</title>'
            f'<rect x="{x_from:.1f}" y="{top}" width="{max(x_to - x_from, 0.5):.1f}" '
            f'height="{plot_h}"/>'
            f'<line x1="{sx(a):.1f}" x2="{sx(a):.1f}" y1="{top}" y2="{top + plot_h}"/>'
            f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="4"/></g>'
        )
    return "".join(parts) + "</svg>"


def _table(title: str, columns: list[str], rows: list[list]) -> dict:
    return {"title": title, "columns": columns, "rows": rows}


def _fisher_view(results: dict, summary: dict, lines: list, tables: list, charts: list) -> None:
    """A Fisher study: the parameters' values and standard deviations, the design
    criteria and the information matrix."""
    values = summary.get("values") or {}
    std = summary.get("std")
    tables.append(
        _table(
            "Parameters",
            ["parameter", "value", "standard deviation"],
            [[n, fmt(v), fmt((std or {}).get(n))] for n, v in values.items()],
        )
    )
    if std is None:
        lines.append(
            "The information matrix is singular: these measurements cannot tell the "
            "free parameters apart."
        )
    criteria = summary.get("criteria") or {}
    tables.append(
        _table(
            "Design criteria",
            ["criterion", "value"],
            [[c, fmt(v)] for c, v in criteria.items()],
        )
    )
    matrix = results.get("matrix")
    if matrix:
        tables.append(
            _table(
                "Information matrix (log-parameters where transformed)",
                ["", *matrix["names"]],
                [[n, *(fmt(v) for v in row)] for n, row in zip(matrix["names"], matrix["rows"])],
            )
        )


def _rank_view(results: dict, summary: dict, lines: list, tables: list, charts: list) -> None:
    """A rank study: the sensor sets, best first."""
    table = results.get("table") or {}
    criterion = results.get("criterion", "D")
    n = len(table.get("candidate") or [])
    rows = [
        [
            str(i + 1),
            table["candidate"][i],
            fmt((table.get(criterion) or [None] * n)[i]),
            fmt((table.get("improvement") or [None] * n)[i]),
        ]
        for i in range(n)
    ]
    tables.append(
        _table("Sensor sets, best first", ["rank", "sensor set", criterion, "improvement"], rows)
    )
    lines.append(
        "Improvement is over the sensors already in place (the base); it is empty "
        "when there is no base."
    )


def _calibrate_view(results: dict, summary: dict, lines: list, tables: list, charts: list) -> None:
    """A calibration: the fitted parameters, how the fit ended, any estimated
    measurement error, and a chart of fit and data per channel."""
    fitted = summary.get("fitted") or {}
    std = summary.get("std") or {}
    if fitted:
        tables.append(
            _table(
                "Fitted parameters",
                ["parameter", "value", "standard deviation"],
                [[n, fmt(v), fmt(std.get(n))] for n, v in fitted.items()],
            )
        )
    if summary:
        state = "converged" if summary.get("converged") else "did not converge"
        lines.append(
            f"The fit {state} after {summary.get('iterations')} iterations "
            f"({summary.get('message')})."
        )
    estimate = summary.get("sigma_estimate")
    if estimate:
        tables.append(
            _table(
                "Estimated measurement error",
                ["channel", "standard deviation"],
                [[n, fmt(v)] for n, v in estimate.items()],
            )
        )
    fit = results.get("fitted") or {}
    data = results.get("data") or {}
    for stream, channels in (fit.get("values") or {}).items():
        for channel, values in channels.items():
            observed = (data.get("values") or {}).get(stream, {}).get(channel)
            charts.append(
                {
                    "channel": f"{stream}.{channel}",
                    "svg": xy_chart(
                        title=f"{stream} {channel}",
                        label=f"{channel} of {stream}: data and fit",
                        x_name="t",
                        line=list(zip(fit["t"], values)),
                        dots=list(zip(data.get("t") or [], observed or [])),
                        x_axis="time",
                    ),
                }
            )


def _profile_view(results: dict, summary: dict, lines: list, tables: list, charts: list) -> None:
    """A profile: the best value and interval, a chart of the profile and its table."""
    table = results.get("table") or {}
    parameter = summary.get("parameter")
    low, high = (summary.get("ci") or [None, None])[:2]
    bounds = (
        "below the grid" if low is None else fmt(low),
        "above the grid" if high is None else fmt(high),
    )
    lines.append(
        f"Best of the grid: {parameter} = {fmt(summary.get('mle'))}. 95 % interval: "
        f"{bounds[0]} to {bounds[1]} (where the curve stays under the threshold)."
    )
    if isinstance(parameter, str) and parameter in table:
        charts.append(
            {
                "channel": parameter,
                "svg": xy_chart(
                    title=f"Profile of {parameter}",
                    label=f"likelihood profile of {parameter}",
                    x_name=parameter,
                    line=list(zip(table[parameter], table.get("delta_loss") or [])),
                    reference=(summary.get("delta", 1.92), "threshold"),
                    x_axis=parameter,
                ),
            }
        )
        tables.append(
            _table(
                "Profile",
                [parameter, "loss", "rise from the best"],
                [
                    [fmt(x), fmt(v), fmt(d)]
                    for x, v, d in zip(
                        table[parameter], table.get("loss") or [], table.get("delta_loss") or []
                    )
                ],
            )
        )


#: The kind of study shown by the editor's own tables and charts (a steady state
#: or a run in time), not by :func:`analysis_view`.
SIMULATE = "simulate"

#: Each other kind's view; ``tests/ui/test_protocol.py`` pins that these and
#: :data:`SIMULATE` are exactly aquakin's study kinds.
ANALYSIS_VIEWS = {
    "fisher": _fisher_view,
    "rank": _rank_view,
    "calibrate": _calibrate_view,
    "profile": _profile_view,
}


def analysis_view(results: dict) -> dict:
    """What a Fisher, rank, calibrate or profile study found, ready to show.

    Parameters
    ----------
    results : dict
        The worker's ``study_results`` reply.

    Returns
    -------
    dict
        ``{"lines": [text], "tables": [{"title", "columns", "rows"}],
        "charts": [{"channel", "svg"}]}``; every number as text.

    Raises
    ------
    ValueError
        If the kind has no view here (rather than drawing it as another kind).
    """
    kind = results["kind"]
    view = ANALYSIS_VIEWS.get(kind)
    if view is None:
        raise ValueError(f"no view for a {kind!r} study")
    lines: list[str] = []
    tables: list[dict] = []
    charts: list[dict] = []
    view(results, results.get("summary") or {}, lines, tables, charts)
    return {"lines": lines, "tables": tables, "charts": charts}
