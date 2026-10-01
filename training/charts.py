"""
Dependency-free terminal charts rendered with Unicode braille dots.

Each character cell holds a 2×4 dot grid, so a 60×12 cell chart has 120×48
plot resolution. Charts are Rich renderables and size themselves to whatever
space the layout gives them.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

# braille dot bits indexed by [x % 2][y % 4]
_DOTS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))
_BRAILLE_BASE = 0x2800

# Colour-blind friendly palette in a fixed order so a series keeps its colour
PALETTE = ["#4e9fe6", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#ff9da7"]


@dataclass
class Series:
    label: str
    xs: Sequence[float]
    ys: Sequence[float]
    color: str = PALETTE[0]
    dotted: bool = False  # used for reference/previous runs


@dataclass
class LineChart:
    series: List[Series]
    y_range: Optional[Tuple[float, float]] = None
    x_range: Optional[Tuple[float, float]] = None
    height: int = 10
    y_fmt: Callable[[float], str] = field(default=lambda v: f"{v:.2f}")
    x_fmt: Callable[[float], str] = field(default=lambda v: f"{v:g}")
    empty_message: str = "waiting for data…"
    legend: bool = True

    def _bounds(self) -> Optional[Tuple[float, float, float, float]]:
        xs = [x for s in self.series for x in s.xs]
        ys = [y for s in self.series for y in s.ys if not math.isnan(y)]
        if not xs or not ys:
            return None
        x0, x1 = self.x_range or (min(xs), max(xs))
        y0, y1 = self.y_range or (min(ys), max(ys))
        if x1 == x0:
            x0, x1 = x0 - 1, x1 + 1
        if y1 == y0:
            pad = abs(y0) * 0.1 or 1.0
            y0, y1 = y0 - pad, y1 + pad
        return x0, x1, y0, y1

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width
        height = options.height or self.height
        bounds = self._bounds()
        if bounds is None or width < 16 or height < 3:
            yield Text(self.empty_message, style="dim")
            for _ in range(max(0, height - 1)):
                yield Text("")
            return
        x0, x1, y0, y1 = bounds

        label_w = max(len(self.y_fmt(v)) for v in (y0, y1, (y0 + y1) / 2)) + 1
        plot_w = max(4, width - label_w - 1)
        legend_rows = 1 if self.legend else 0
        plot_h = max(2, height - 1 - legend_rows)
        dots_w, dots_h = plot_w * 2, plot_h * 4

        bits = [[0] * plot_w for _ in range(plot_h)]
        colors: List[List[Optional[str]]] = [[None] * plot_w for _ in range(plot_h)]

        def to_dot(x: float, y: float) -> Tuple[int, int]:
            dx = round((x - x0) / (x1 - x0) * (dots_w - 1))
            dy = round((y1 - y) / (y1 - y0) * (dots_h - 1))
            return min(max(dx, 0), dots_w - 1), min(max(dy, 0), dots_h - 1)

        def plot(dx: int, dy: int, color: str) -> None:
            cx, cy = dx // 2, dy // 4
            bits[cy][cx] |= _DOTS[dx % 2][dy % 4]
            colors[cy][cx] = color

        # reference series first so live data draws on top
        for s in sorted(self.series, key=lambda s: not s.dotted):
            pts = [to_dot(x, y) for x, y in zip(s.xs, s.ys) if not math.isnan(y)]
            if len(pts) == 1:
                plot(*pts[0], s.color)
            step = 0
            for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                # Bresenham between consecutive points
                ddx, ddy = abs(bx - ax), -abs(by - ay)
                sx, sy = (1 if ax < bx else -1), (1 if ay < by else -1)
                err = ddx + ddy
                while True:
                    if not s.dotted or step % 3 == 0:
                        plot(ax, ay, s.color)
                    step += 1
                    if ax == bx and ay == by:
                        break
                    e2 = 2 * err
                    if e2 >= ddy:
                        err += ddy
                        ax += sx
                    if e2 <= ddx:
                        err += ddx
                        ay += sy

        axis = Style(color="grey50")
        tick_rows = {0: y1, plot_h // 2: (y0 + y1) / 2, plot_h - 1: y0}
        for row in range(plot_h):
            label = self.y_fmt(tick_rows[row]).rjust(label_w - 1) + " " if row in tick_rows else " " * label_w
            yield Segment(label, axis)
            yield Segment("┤" if row in tick_rows else "│", axis)
            for col in range(plot_w):
                b = bits[row][col]
                yield Segment(chr(_BRAILLE_BASE + b) if b else " ", Style(color=colors[row][col]) if b else None)
            yield Segment.line()

        # x axis labels: left, middle, right
        xl, xm, xr = self.x_fmt(x0), self.x_fmt((x0 + x1) / 2), self.x_fmt(x1)
        axis_line = [" "] * plot_w
        labels = [(0, xl), (plot_w - len(xr), xr)]
        if xm not in (xl, xr):
            labels.append(((plot_w - len(xm)) // 2, xm))
        for pos, lab in labels:
            for i, ch in enumerate(lab):
                if 0 <= pos + i < plot_w:
                    axis_line[pos + i] = ch
        yield Segment(" " * label_w + "└", axis)
        yield Segment("".join(axis_line), axis)
        yield Segment.line()

        if self.legend:
            legend = Text(" " * (label_w + 1))
            for s in self.series:
                legend.append("┄ " if s.dotted else "━ ", style=s.color)
                legend.append(s.label + "  ", style="dim" if s.dotted else "")
            legend.truncate(width, overflow="ellipsis")
            yield legend


def bar(fraction: float, width: int, color: str = "cyan", empty_style: str = "grey23") -> Text:
    """Single-line horizontal bar with eighth-block precision."""
    fraction = min(max(fraction if fraction == fraction else 0.0, 0.0), 1.0)
    eighths = round(fraction * width * 8)
    full, rem = divmod(eighths, 8)
    text = Text("█" * full, style=color)
    if rem and full < width:
        text.append("▏▎▍▌▋▊▉"[rem - 1], style=color)
        full += 1
    text.append("━" * (width - full), style=empty_style)
    return text


def sparkline(values: Sequence[float], width: int, lo: float = 0.0, hi: float = 100.0, color: str = "cyan") -> Text:
    """Compact block sparkline of the most recent `width` values."""
    ticks = "▁▂▃▄▅▆▇█"
    vals = list(values)[-width:]
    out = Text(" " * (width - len(vals)))
    for v in vals:
        idx = int((min(max(v, lo), hi) - lo) / ((hi - lo) or 1) * (len(ticks) - 1))
        out.append(ticks[idx], style=color)
    return out
