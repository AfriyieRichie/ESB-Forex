"""Portfolio equity simulation for continuously-held, vol-sized positions.

The one rule that matters: a weight decided on day t (from data through t) earns
the return from t to t+1. The simulation is written around that shift so the
lookahead cannot creep in through indexing.

Simplifications, stated so they can be argued with:
  - Positions are assumed re-set to target at each rebalance; intra-week drift
    of weights as prices move is ignored. Small at a weekly cadence.
  - A missing bar (one market's holiday) is treated as a zero-return hold, not
    a forced exit.
  - Costs are a fraction of the notional *change* at each rebalance: full spread
    plus slippage, converted from pips at that day's price. Optimistic for the
    2005-2016 era, as elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioResult:
    daily_returns: np.ndarray  # net, aligned to `dates[1:]`
    equity: np.ndarray  # starts at 1.0, aligned to `dates`
    gross_exposure: np.ndarray  # sum |weight| actually held, per day
    turnover: np.ndarray  # sum |dw| at each rebalance


def simulate(
    returns: np.ndarray,
    target_weights: np.ndarray,
    cost_fraction: np.ndarray,
    rebalance_mask: np.ndarray,
    *,
    gross_cap: float = 3.0,
) -> PortfolioResult:
    """Simulate a weighted portfolio net of turnover costs.

    returns[t, i]        simple return of asset i over [t-1, t]; NaN = no data
    target_weights[t, i] desired weight from info through t; NaN = flat
    cost_fraction[t, i]  round-trip cost as a fraction of notional, per unit
                         turnover, at day t
    rebalance_mask[t]    whether weights are re-set to target on day t

    Weights set on day t earn returns[t+1]. Cost for a rebalance is charged on
    day t, reducing that step's return.
    """
    n_days, n_assets = returns.shape
    held = np.zeros(n_assets)

    daily = np.zeros(n_days - 1)
    gross = np.zeros(n_days)
    turnover = np.zeros(n_days)

    for t in range(n_days - 1):
        if rebalance_mask[t]:
            desired = np.where(np.isnan(target_weights[t]), 0.0, target_weights[t])
            gross_sum = np.abs(desired).sum()
            if gross_sum > gross_cap and gross_sum > 0:
                desired = desired * (gross_cap / gross_sum)  # scale book to cap
            change = np.abs(desired - held)
            turnover[t] = change.sum()
            cost = float(np.nansum(change * cost_fraction[t]))
            held = desired
        else:
            cost = 0.0

        gross[t] = np.abs(held).sum()
        step_return = np.nansum(held * returns[t + 1])  # weights_t x return_{t+1}
        daily[t] = step_return - cost

    gross[-1] = np.abs(held).sum()

    equity = np.empty(n_days)
    equity[0] = 1.0
    equity[1:] = np.cumprod(1.0 + daily)
    return PortfolioResult(daily_returns=daily, equity=equity, gross_exposure=gross, turnover=turnover)
