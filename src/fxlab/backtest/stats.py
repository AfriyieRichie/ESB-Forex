"""Trade statistics.

Expectancy is always reported with an interval. A bare "+0.08R" is unreadable:
it could be a real edge or a coin landing heads slightly more often over 300
tries, and the difference is the entire question.

The interval is a normal approximation on the mean of R, which **overstates
confidence here** for two reasons: trades across a basket sharing USD on both
sides are correlated, and trades overlapping in time on the same pair are not
independent draws. Treat a marginal result as worse than it looks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fxlab.backtest.engine import Trade


@dataclass(frozen=True)
class Summary:
    setups: int
    filled: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    expectancy: float  # mean R per filled trade
    stderr: float
    total_r: float
    avg_win: float
    avg_loss: float

    @property
    def fill_rate(self) -> float:
        return self.filled / self.setups if self.setups else 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        margin = 1.96 * self.stderr
        return self.expectancy - margin, self.expectancy + margin

    # Below this, an interval is arithmetic rather than evidence. A single
    # trade has zero sample variance, so its CI collapses to a point and would
    # otherwise be flagged as a certainty.
    MIN_N_FOR_INFERENCE = 20

    @property
    def significant(self) -> bool:
        """Whether the 95% interval excludes zero. Necessary, nowhere near sufficient."""
        if self.filled < self.MIN_N_FOR_INFERENCE or self.stderr <= 0:
            return False
        low, high = self.ci95
        return low > 0 or high < 0

    def line(self, label: str = "") -> str:
        if self.filled == 0:
            return f"{label:<22} n=    0  (no trades)"
        low, high = self.ci95
        star = " *" if self.significant else ""
        if self.filled < self.MIN_N_FOR_INFERENCE:
            star = " (n too small)"
        return (
            f"{label:<22} n={self.filled:>5}  win={self.win_rate:>5.1%}  "
            f"E[R]={self.expectancy:>+6.3f}  95%CI=[{low:>+6.3f},{high:>+6.3f}]  "
            f"totalR={self.total_r:>+8.1f}{star}"
        )


def summarize(trades: list[Trade]) -> Summary:
    filled = [t for t in trades if t.filled]
    returns = [t.r_multiple for t in filled]
    n = len(returns)

    if n == 0:
        return Summary(len(trades), 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1) if n > 1 else 0.0
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    return Summary(
        setups=len(trades),
        filled=n,
        wins=len(wins),
        losses=len(losses),
        timeouts=sum(1 for t in filled if t.outcome == "timeout"),
        win_rate=len(wins) / n,
        expectancy=mean,
        stderr=math.sqrt(variance / n) if n > 1 else 0.0,
        total_r=sum(returns),
        avg_win=sum(wins) / len(wins) if wins else 0.0,
        avg_loss=sum(losses) / len(losses) if losses else 0.0,
    )


def slice_by(trades: list[Trade], attribute: str) -> dict[str, Summary]:
    buckets: dict[str, list[Trade]] = {}
    for trade in trades:
        buckets.setdefault(str(getattr(trade, attribute)), []).append(trade)
    return {key: summarize(value) for key, value in sorted(buckets.items())}
