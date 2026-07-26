"""Event-driven backtest for stop-entry, stop-loss, zone-target trades.

Written rather than borrowed, because every library hides its assumptions about
fills, and the assumptions are the entire result. The ones made here are listed
explicitly so they can be argued with.

**Same-bar ambiguity.** When one bar's range contains both the stop and the
target, OHLC cannot say which came first. This assumes the **stop**. That is
pessimistic and it is meant to be: the optimistic reading turns every volatile
bar into a winner and is the single most common way a backtest flatters itself.
`pessimistic_same_bar=False` exists to measure how much of a result depends on
the assumption - if flipping it changes the verdict, the verdict was never real.

**Gaps.** A stop order that gaps through fills at the open, not at its price.
Entries gap to a worse price; exits gap to a worse price. Markets do not fill
you at the level you wanted just because you wanted it.

**Costs.** One full spread per round trip, plus slippage on both stop-entry and
stop-loss. Limit exits at the target are not slipped: if price trades through a
resting limit, it fills. Spreads are modern-tight and therefore optimistic for
2005-2016.

**Risk accounting.** R is measured against the *planned* risk at order
placement, since that is what position size would have been based on. Slippage
and gaps then show up as R degradation rather than being quietly hidden by
re-basing R on the actual fill.

**Scope.** Each setup is simulated independently. This measures whether the
pattern has edge; it is not a portfolio simulation and says nothing about
concurrent exposure or correlated drawdown.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from fxlab.instruments import Instrument
from fxlab.setups.patterns import Setup

Outcome = Literal["target", "stop", "timeout", "no_fill"]


@dataclass(frozen=True)
class BacktestConfig:
    # "stop" is the traded rule: a resting stop order beyond the signal candle,
    # which by design does not always fill. "close" enters at market on the
    # signal candle's close and therefore always fills - not a proposed
    # strategy, but the control needed to ask whether the stop order rejects
    # setups that would have won.
    entry_mode: Literal["stop", "close"] = "stop"
    # How long the entry stop order rests before being cancelled.
    entry_valid_bars: int = 3
    # Time barrier. At H4, 120 bars is about four trading weeks.
    max_holding_bars: int = 120
    slippage_pips: float = 0.3
    pessimistic_same_bar: bool = True
    spread_pips: float | None = None  # overrides the instrument default


@dataclass(frozen=True)
class Trade:
    symbol: str
    pattern: str
    direction: str
    zone_tier: str
    zone_origin: str
    confluent: bool
    planned_rr: float
    signal_bar: int
    entry_bar: int | None
    exit_bar: int | None
    entry_price: float | None
    stop_price: float
    target_price: float
    exit_price: float | None
    outcome: Outcome
    r_multiple: float
    bars_held: int
    ts: dt.datetime

    @property
    def filled(self) -> bool:
        return self.outcome != "no_fill"


@dataclass
class _Bars:
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    pos_of_bar: dict[int, int]

    @classmethod
    def of(cls, bars: pl.DataFrame) -> _Bars:
        return cls(
            opens=bars["open"].to_numpy(),
            highs=bars["high"].to_numpy(),
            lows=bars["low"].to_numpy(),
            closes=bars["close"].to_numpy(),
            pos_of_bar={int(b): i for i, b in enumerate(bars["bar"].to_numpy())},
        )


def _no_fill(setup: Setup) -> Trade:
    return Trade(
        symbol=setup.symbol,
        pattern=setup.pattern,
        direction=setup.direction,
        zone_tier=setup.zone_tier,
        zone_origin=setup.zone_origin,
        confluent=setup.confluent,
        planned_rr=setup.reward_risk,
        signal_bar=setup.signal_bar,
        entry_bar=None,
        exit_bar=None,
        entry_price=None,
        stop_price=setup.stop,
        target_price=setup.target,
        exit_price=None,
        outcome="no_fill",
        r_multiple=0.0,
        bars_held=0,
        ts=setup.ts,
    )


def simulate(
    setup: Setup, bars: _Bars, inst: Instrument, config: BacktestConfig
) -> Trade:
    """Run one setup forward through the bars."""
    sign = 1 if setup.direction == "long" else -1
    slippage = config.slippage_pips * inst.pip
    spread = (
        config.spread_pips * inst.pip if config.spread_pips is not None else inst.spread
    )
    planned_risk = abs(setup.entry - setup.stop)
    if planned_risk <= 0:
        return _no_fill(setup)

    signal_pos = bars.pos_of_bar.get(setup.signal_bar)
    if signal_pos is None:
        return _no_fill(setup)

    # --- entry ---------------------------------------------------------------
    entry_pos = None
    entry_fill = 0.0

    if config.entry_mode == "close":
        # Market on the signal close. Always fills, so exits begin next bar.
        entry_pos = signal_pos
        entry_fill = float(bars.closes[signal_pos]) + sign * slippage
        exit_start = signal_pos + 1
    else:
        last = min(signal_pos + config.entry_valid_bars, len(bars.opens) - 1)
        for pos in range(signal_pos + 1, last + 1):
            triggered = (
                bars.highs[pos] >= setup.entry if sign > 0 else bars.lows[pos] <= setup.entry
            )
            if not triggered:
                continue
            # A gap past the trigger fills at the open, never at the wanted price.
            raw = (
                max(setup.entry, bars.opens[pos])
                if sign > 0
                else min(setup.entry, bars.opens[pos])
            )
            entry_fill = raw + sign * slippage
            entry_pos = pos
            break

        if entry_pos is None:
            return _no_fill(setup)
        exit_start = entry_pos

    if exit_start >= len(bars.closes):
        return _no_fill(setup)

    # --- exit: stop, target, or time ----------------------------------------
    outcome: Outcome = "timeout"
    exit_pos = min(exit_start + config.max_holding_bars, len(bars.closes) - 1)
    exit_price = float(bars.closes[exit_pos])

    for pos in range(exit_start, exit_pos + 1):
        hit_stop = (
            bars.lows[pos] <= setup.stop if sign > 0 else bars.highs[pos] >= setup.stop
        )
        hit_target = (
            bars.highs[pos] >= setup.target
            if sign > 0
            else bars.lows[pos] <= setup.target
        )

        if hit_stop and (config.pessimistic_same_bar or not hit_target):
            gapped = (
                min(setup.stop, bars.opens[pos])
                if sign > 0
                else max(setup.stop, bars.opens[pos])
            )
            exit_price = float(gapped - sign * slippage)
            exit_pos, outcome = pos, "stop"
            break
        if hit_target:
            exit_price = float(setup.target)
            exit_pos, outcome = pos, "target"
            break

    pnl = sign * (exit_price - entry_fill) - spread
    return Trade(
        symbol=setup.symbol,
        pattern=setup.pattern,
        direction=setup.direction,
        zone_tier=setup.zone_tier,
        zone_origin=setup.zone_origin,
        confluent=setup.confluent,
        planned_rr=setup.reward_risk,
        signal_bar=setup.signal_bar,
        entry_bar=int(entry_pos),
        exit_bar=int(exit_pos),
        entry_price=float(entry_fill),
        stop_price=setup.stop,
        target_price=setup.target,
        exit_price=exit_price,
        outcome=outcome,
        r_multiple=float(pnl / planned_risk),
        bars_held=int(exit_pos - entry_pos),
        ts=setup.ts,
    )


def run(
    setups: list[Setup],
    bars: pl.DataFrame,
    inst: Instrument,
    config: BacktestConfig | None = None,
) -> list[Trade]:
    config = config or BacktestConfig()
    prepared = _Bars.of(bars)
    return [simulate(s, prepared, inst, config) for s in setups]


def to_frame(trades: list[Trade]) -> pl.DataFrame:
    if not trades:
        return pl.DataFrame()
    return pl.DataFrame([vars(t) for t in trades])
