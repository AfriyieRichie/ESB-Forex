"""Render zones over price, as of a given bar.

This is the validation loop: the detector draws what it thinks the levels are,
and the trader marks where it disagrees with their eye. It is the only place in
the project where human judgement is the ground truth, so the picture has to be
honest about what was knowable.

Two rules the rendering obeys:

  - No future bars. The chart ends at `as_of_bar`. Showing what happened next
    would have the trader validating with hindsight, judging whether a zone
    turned out useful rather than whether it was the level they would have
    drawn at the time. Pass `reveal` to deliberately look forward afterwards.
  - Zone bands start at the bar the zone became knowable, and use the bounds
    it had at `as_of_bar` - not its final, widest bounds.

Colour encoding: hue carries support vs resistance (blue vs orange, which
survives colour-vision deficiency where the conventional green/red does not);
texture carries native vs flipped; opacity and border carry tier. Candles stay
neutral - hollow for up, filled for down - because they are context, and
spending colour on them would compete with the zones, which are the subject.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from fxlab.zones.builder import ZoneBook, ZoneView  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Validated blue/orange pair (CVD delta-E 24.7, normal-vision 33.6).
SUPPORT = "#2a78d6"
RESISTANCE = "#eb6834"

CANDLE_WIDTH = 0.62


def _zone_style(view: ZoneView) -> dict:
    colour = SUPPORT if view.kind == "support" else RESISTANCE
    primary = view.tier == "primary"
    return {
        "colour": colour,
        "alpha": 0.20 if primary else 0.09,
        "linewidth": 1.3 if primary else 0.8,
        "linestyle": "solid" if primary else (0, (4, 3)),
        "hatch": "///" if view.origin == "flipped" else None,
    }


def _draw_candles(ax, frame: pl.DataFrame) -> None:
    opens = frame["open"].to_numpy()
    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    closes = frame["close"].to_numpy()

    for x, (o, h, low, c) in enumerate(zip(opens, highs, lows, closes)):
        ax.plot([x, x], [low, h], color=SECONDARY_INK, linewidth=0.7, zorder=3)
        bottom, height = min(o, c), abs(c - o) or 1e-9
        ax.add_patch(
            Rectangle(
                (x - CANDLE_WIDTH / 2, bottom),
                CANDLE_WIDTH,
                height,
                facecolor=SURFACE if c >= o else SECONDARY_INK,
                edgecolor=SECONDARY_INK,
                linewidth=0.7,
                zorder=4,
            )
        )


def render(
    bars: pl.DataFrame,
    books: ZoneBook | list[ZoneBook],
    as_of_bar: int,
    out_path: Path | str,
    *,
    symbol: str = "",
    window: int = 180,
    reveal: int = 0,
    min_prior_touches: int | None = None,
    show_close_line: bool = True,
    levels: dict[str, float] | None = None,
    markers: list[dict] | None = None,
    title: str | None = None,
    hide_axes: bool = False,
) -> Path:
    """Draw `window` bars ending at `as_of_bar`, with zones known at that bar.

    `reveal` optionally draws N bars beyond `as_of_bar`, separated by a rule,
    for checking outcomes after judgement has already been made.

    `levels` draws proposed trade levels (entry / stop / target).

    `hide_axes` and `title` support blind review: with the symbol, the dates and
    the price scale removed, a judgement has to come from the price action
    rather than from remembering what that market did.
    """
    if isinstance(books, ZoneBook):
        books = [books]

    # Bar indices are per-timeframe: D1 bar 400 and H4 bar 400 are different
    # moments. Mixing books here would silently draw zones at the wrong dates.
    # Overlaying D1 zones on an H4 chart needs the session-date bridge first.
    timeframes = {book.timeframe for book in books}
    if len(timeframes) > 1:
        raise ValueError(
            f"books span multiple timeframes {sorted(timeframes)}; bar indices are "
            "not comparable across timeframes"
        )

    start = max(0, as_of_bar - window + 1)
    visible = bars.filter(pl.col("bar").is_between(start, as_of_bar + reveal))
    if visible.is_empty():
        raise ValueError(f"no bars in range {start}..{as_of_bar}")

    history = visible.filter(pl.col("bar") <= as_of_bar)
    n_history = len(history)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    _draw_candles(ax, visible)

    if show_close_line:
        # Zones are built from closes, so the line the detector actually sees
        # is drawn alongside the candles.
        ax.plot(
            range(len(visible)),
            visible["close"].to_numpy(),
            color=INK,
            linewidth=1.0,
            alpha=0.55,
            zorder=5,
        )

    lows = visible["low"].to_numpy()
    highs = visible["high"].to_numpy()
    pad = (highs.max() - lows.min()) * 0.08
    y_lo, y_hi = lows.min() - pad, highs.max() + pad

    right_edge = len(visible) - 1
    drawn = 0
    for book in books:
        for view in book.zones_as_of(as_of_bar, min_prior_touches=min_prior_touches):
            if view.upper < y_lo or view.lower > y_hi:
                continue  # off-screen; not useful for judging this window
            style = _zone_style(view)
            x0 = max(0, view.created_bar - start)
            ax.add_patch(
                Rectangle(
                    (x0 - 0.5, view.lower),
                    right_edge - x0 + 1,
                    max(view.upper - view.lower, 1e-9),
                    facecolor=style["colour"],
                    edgecolor=style["colour"],
                    alpha=style["alpha"],
                    hatch=style["hatch"],
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                    zorder=1,
                )
            )
            ax.annotate(
                f"{view.touch_count}x",
                xy=(right_edge + 1, view.mid),
                fontsize=8,
                color=MUTED,
                va="center",
                annotation_clip=False,
            )
            drawn += 1

    if levels:
        for name, price in levels.items():
            if not (y_lo <= price <= y_hi):
                continue
            style = "solid" if name == "entry" else (0, (5, 3))
            ax.axhline(price, color=INK, linewidth=1.1, linestyle=style, alpha=0.75, zorder=7)
            ax.annotate(
                name,
                xy=(0, price),
                xytext=(2, 3),
                textcoords="offset points",
                fontsize=8,
                color=SECONDARY_INK,
            )

    # Setup annotations: mark specific bars (e.g. the two touches, the trigger
    # candle) so the chart shows WHY the detector called it a setup.
    if markers:
        for i, m in enumerate(markers):
            x = int(m["bar"]) - start
            if not (0 <= x <= right_edge):
                continue
            color = m.get("color", SECONDARY_INK)
            ax.axvline(x, color=color, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.7, zorder=6)
            ax.annotate(
                m.get("label", ""),
                xy=(x, y_hi),
                xytext=(0, -2 - (i % 3) * 13),  # stagger so adjacent labels don't collide
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                fontweight="600",
                color=color,
            )

    if reveal:
        ax.axvline(n_history - 0.5, color=MUTED, linewidth=1.0, linestyle=(0, (2, 3)), zorder=6)

    ax.set_xlim(-1, right_edge + 3)
    ax.set_ylim(y_lo, y_hi)

    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ticks = list(range(0, len(visible), max(1, len(visible) // 9)))
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [visible["ts_open"][t].strftime("%Y-%m-%d") for t in ticks],
            rotation=0,
            fontsize=8,
            color=MUTED,
        )
        ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)

    if title is None:
        as_of_ts = bars.filter(pl.col("bar") == as_of_bar)["ts_open"][0]
        timeframes = "/".join(dict.fromkeys(b.timeframe for b in books))
        title = (
            f"{symbol}  as of {as_of_ts:%Y-%m-%d}  (bar {as_of_bar})   "
            f"{drawn} zones from {timeframes}"
        )
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=12)

    ax.legend(
        handles=[
            Patch(facecolor=SUPPORT, alpha=0.30, label="support"),
            Patch(facecolor=RESISTANCE, alpha=0.30, label="resistance"),
            Patch(facecolor=MUTED, alpha=0.20, hatch="///", label="flipped"),
            Line2D([], [], color=MUTED, linestyle=(0, (4, 3)), label="secondary (H4)"),
            Line2D([], [], color=MUTED, linestyle="solid", label="primary (D1)"),
        ],
        loc="upper left",
        fontsize=8,
        frameon=False,
        labelcolor=SECONDARY_INK,
        ncols=5,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path
