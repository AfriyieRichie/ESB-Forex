"""Portfolio performance statistics.

Trend-following is judged on risk-adjusted return and the shape of the
drawdowns, not on per-trade expectancy. A high return with a 60% drawdown is
untradeable; the ratio metrics say so where a raw return number would not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class PortfolioStats:
    days: int
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    hit_days: float  # fraction of days positive
    total_return: float

    def line(self, label: str = "") -> str:
        return (
            f"{label:<16} Sharpe={self.sharpe:>+5.2f}  ret={self.ann_return:>+6.1%}  "
            f"vol={self.ann_vol:>5.1%}  maxDD={self.max_drawdown:>6.1%}  "
            f"Calmar={self.calmar:>+5.2f}"
        )


def max_drawdown(equity: np.ndarray) -> float:
    """Most negative peak-to-trough fraction (<= 0)."""
    peaks = np.maximum.accumulate(equity)
    return float((equity / peaks - 1.0).min())


def summarize(daily_returns: np.ndarray, equity: np.ndarray, trading_days: int = TRADING_DAYS) -> PortfolioStats:
    n = len(daily_returns)
    if n == 0:
        return PortfolioStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mean = float(daily_returns.mean())
    std = float(daily_returns.std(ddof=1)) if n > 1 else 0.0
    ann_return = mean * trading_days
    ann_vol = std * np.sqrt(trading_days)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    mdd = max_drawdown(equity)

    return PortfolioStats(
        days=n,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=mdd,
        calmar=ann_return / abs(mdd) if mdd < 0 else 0.0,
        hit_days=float((daily_returns > 0).mean()),
        total_return=float(equity[-1] - 1.0),
    )
