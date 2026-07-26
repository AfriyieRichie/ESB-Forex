"""Carry signal and carry accrual, aligned to the basket's dates.

Each pair is a bet on the policy-rate differential of its two currencies:
  - signal  = sign(rate_base - rate_quote): long the pair if the base currency
    out-yields the quote, short otherwise.
  - accrual = (rate_base - rate_quote)/100/252 per day: the interest you earn
    (long) or pay (short) simply for holding the position. This is most of the
    carry premium and the spot-only engine does not see it, so it is added to
    the returns fed to the engine.

Point-in-time: policy rates are public the day they change, so the rate in
effect on a date (as-of, backward) is knowable then. Monthly granularity is
coarse but the rate is a step function, so forward-fill loses almost nothing.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fxlab.data.rates import load_policy_rates
from fxlab.portfolio.basket import Basket


def _daily_rate_by_currency(currencies: list[str], dates: np.ndarray) -> dict[str, np.ndarray]:
    """Policy rate (%) in effect on each basket date, per currency."""
    rates = load_policy_rates(currencies)
    date_frame = pl.DataFrame({"date": dates}).with_columns(
        pl.col("date").cast(pl.Date)
    ).sort("date")

    out = {}
    for currency in currencies:
        monthly = (
            rates.filter(pl.col("currency") == currency)
            .select(pl.col("month").alias("date"), "rate_pct")
            .sort("date")
        )
        joined = date_frame.join_asof(monthly, on="date", strategy="backward")
        out[currency] = joined["rate_pct"].to_numpy().astype(float)
    return out


def build_carry(basket: Basket) -> tuple[np.ndarray, np.ndarray]:
    """Return (carry_sign, carry_daily), both T x N, aligned to basket.dates.

    carry_sign  in {-1, 0, +1}: the position direction the carry rule wants.
    carry_daily is the daily accrual for a LONG unit of each pair.
    """
    bases = [s[:3] for s in basket.symbols]
    quotes = [s[3:] for s in basket.symbols]
    currencies = sorted(set(bases) | set(quotes))

    rate = _daily_rate_by_currency(currencies, basket.dates)

    n_days = len(basket.dates)
    n_assets = len(basket.symbols)
    carry_daily = np.full((n_days, n_assets), np.nan)
    carry_sign = np.zeros((n_days, n_assets))

    for j, (base, quote) in enumerate(zip(bases, quotes)):
        diff_pct = rate[base] - rate[quote]
        carry_daily[:, j] = diff_pct / 100.0 / 252.0
        carry_sign[:, j] = np.sign(diff_pct)

    return carry_sign, carry_daily


def total_returns(basket: Basket, carry_daily: np.ndarray) -> np.ndarray:
    """Spot return plus carry accrual — what a held position actually earns."""
    total = basket.returns.copy()
    # Accrual for the return over [t-1, t] uses the rate in effect at t-1.
    accrual = np.zeros_like(total)
    accrual[1:] = carry_daily[:-1]
    return total + np.where(np.isnan(total), np.nan, accrual)
