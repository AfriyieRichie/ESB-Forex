"""Align the basket into shared date-indexed matrices for the portfolio engine.

Pairs keep different holidays, so their D1 series do not line up one-to-one. A
single missing session is forward-filled (a zero-return hold), which the engine
treats as holding through a closed market rather than being forced out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from fxlab import instruments
from fxlab.data import load_bars
from fxlab.data.resample import session_date


@dataclass(frozen=True)
class Basket:
    symbols: list[str]
    dates: np.ndarray  # session dates, length T
    closes: np.ndarray  # T x N, forward-filled
    returns: np.ndarray  # T x N simple returns; row t over [t-1, t]; NaN pre-history
    cost_fraction: np.ndarray  # T x N round-trip cost as fraction of notional


def load_basket(symbols: list[str], year_lo: int, year_hi: int) -> Basket:
    frames = []
    for symbol in symbols:
        d1 = (
            load_bars(symbol, "D1")
            .filter(pl.col("ts_open").dt.year().is_between(year_lo, year_hi))
            .with_columns(session_date(pl.col("ts_open")).alias("date"))
            .select("date", pl.col("close").alias(symbol))
        )
        frames.append(d1)

    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.join(frame, on="date", how="full", coalesce=True)
    wide = wide.sort("date")

    dates = wide["date"].to_numpy()
    closes = np.column_stack([wide[s].to_numpy() for s in symbols]).astype(float)

    # Forward-fill single-market holidays so a hole is a hold, not an exit.
    for j in range(closes.shape[1]):
        col = closes[:, j]
        last = np.nan
        for i in range(len(col)):
            if np.isnan(col[i]):
                col[i] = last
            else:
                last = col[i]

    returns = np.full_like(closes, np.nan)
    returns[1:] = closes[1:] / closes[:-1] - 1.0

    # Round-trip cost as a fraction of notional: (spread + 2*slippage) pips,
    # converted at that day's price. Slippage matches the setup backtest.
    slippage_pips = 0.3
    cost_fraction = np.full_like(closes, np.nan)
    for j, symbol in enumerate(symbols):
        inst = instruments.get(symbol)
        pip_value = (inst.spread_pips + 2 * slippage_pips) * inst.pip
        cost_fraction[:, j] = pip_value / closes[:, j]

    return Basket(
        symbols=list(symbols),
        dates=dates,
        closes=closes,
        returns=returns,
        cost_fraction=cost_fraction,
    )


def weekly_rebalance_mask(dates: np.ndarray, every: int = 5) -> np.ndarray:
    """Rebalance every `every` sessions (weekly at D1)."""
    mask = np.zeros(len(dates), dtype=bool)
    mask[::every] = True
    return mask
