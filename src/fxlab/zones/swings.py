"""Swing detection on the close series.

Zones are drawn off a line chart, so pivots are found on closes rather than on
highs and lows. Closes are markedly less noisy, which is why a small fractal
window is enough.

Every swing carries two prices for two different jobs:
  - `close`, the line-chart value, which is what zone geometry is built from
  - `low` / `high`, the actual wick extremes, which is what stops reference

Conflating them puts stops inside the candle that formed the level, where they
get taken out trivially.

Every swing also carries two bar indices:
  - `pivot_bar`, where the swing is
  - `confirmed_bar`, when it first became knowable (pivot + window)

A fractal cannot be identified until `window` bars have printed after it.
Anything reading `pivot_bar` as if the swing were known at that moment is
reading the future.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

Kind = Literal["low", "high"]

DEFAULT_WINDOW = 2


@dataclass(frozen=True)
class Swing:
    pivot_bar: int
    confirmed_bar: int
    kind: Kind
    close: float
    low: float
    high: float
    ts_open: dt.datetime

    @property
    def stop_reference(self) -> float:
        """The wick extreme a stop would sit beyond."""
        return self.low if self.kind == "low" else self.high


def _fractal_mask(values: np.ndarray, window: int, *, find_low: bool) -> np.ndarray:
    """Mark bars that are the local extreme of [i-window, i+window].

    Comparison is strict against earlier bars and non-strict against later
    ones, so a flat run resolves to its first bar rather than marking every
    bar in the run.
    """
    n = len(values)
    mask = np.ones(n, dtype=bool)
    if n <= 2 * window:
        return np.zeros(n, dtype=bool)

    for k in range(1, window + 1):
        if find_low:
            mask[k:] &= values[k:] < values[:-k]
            mask[:-k] &= values[:-k] <= values[k:]
        else:
            mask[k:] &= values[k:] > values[:-k]
            mask[:-k] &= values[:-k] >= values[k:]

    # Edges lack a full window on one side, so they cannot be confirmed.
    mask[:window] = False
    mask[n - window :] = False
    return mask


def detect_swings(bars: pl.DataFrame, *, window: int = DEFAULT_WINDOW) -> list[Swing]:
    """Find close-based swing highs and lows, in confirmation order."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    close = bars["close"].to_numpy()
    low = bars["low"].to_numpy()
    high = bars["high"].to_numpy()
    ts = bars["ts_open"].to_list()
    bar_index = bars["bar"].to_numpy()

    swings: list[Swing] = []
    for kind, find_low in (("low", True), ("high", False)):
        for i in np.flatnonzero(_fractal_mask(close, window, find_low=find_low)):
            swings.append(
                Swing(
                    pivot_bar=int(bar_index[i]),
                    confirmed_bar=int(bar_index[i]) + window,
                    kind=kind,  # type: ignore[arg-type]
                    close=float(close[i]),
                    low=float(low[i]),
                    high=float(high[i]),
                    ts_open=ts[i],
                )
            )

    # Confirmation order is the order the zone builder must consume them in.
    swings.sort(key=lambda s: (s.confirmed_bar, s.pivot_bar))
    return swings


def to_frame(swings: list[Swing]) -> pl.DataFrame:
    if not swings:
        return pl.DataFrame(
            schema={
                "pivot_bar": pl.Int64,
                "confirmed_bar": pl.Int64,
                "kind": pl.Utf8,
                "close": pl.Float64,
                "low": pl.Float64,
                "high": pl.Float64,
                "ts_open": pl.Datetime("us", "UTC"),
            }
        )
    return pl.DataFrame([vars(s) for s in swings])
