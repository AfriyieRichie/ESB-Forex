"""Indicators. All are point-in-time by construction: the value at bar i uses
only bars <= i.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def true_range(bars: pl.DataFrame) -> np.ndarray:
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    prev_close = np.concatenate(([np.nan], bars["close"].to_numpy()[:-1]))

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    tr[0] = high[0] - low[0]
    return tr


def atr(bars: pl.DataFrame, period: int = 20) -> np.ndarray:
    """Simple moving average of true range.

    A plain SMA rather than Wilder smoothing: this is used to scale zone
    tolerances, where transparency matters more than convention. Values before
    `period` bars are NaN, so callers must handle the warmup explicitly rather
    than silently treating early bars as zero-volatility.
    """
    tr = true_range(bars)
    out = np.full(len(tr), np.nan)
    if len(tr) >= period:
        cumsum = np.cumsum(np.insert(tr, 0, 0.0))
        out[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
    return out
