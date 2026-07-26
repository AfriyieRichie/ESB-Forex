"""Wammie and moolah detection.

A wammie is a double bottom at an already-established support zone; a moolah is
its exact mirror at resistance. They are one detector with a direction, not two
implementations - mirrored logic written twice drifts apart and gets separately
buggy, and any asymmetry in results would then be indistinguishable from a real
long/short edge.

The rules, as traded:

  1. price touches an established zone twice
  2. the second touch is slightly shallower than the first
  3. at least N bars separate the touches
  4. a reversal candle prints on the second touch
  5. entry is a stop order beyond that candle
  6. the stop sits beyond the *first* touch's wick
  7. the target is the nearest zone paying at least `min_reward_risk`

Three causality traps this code is shaped around:

  - The signal candle cannot be "the last bar of the touch". Which bar was last
    is only knowable once a later bar fails to touch. The signal is instead the
    *first* qualifying bar inside the second touch, which is knowable the
    moment it closes.
  - Establishment is checked as of the *first* touch, not at signal time. The
    two touches themselves generate swings that become forming touches on the
    zone, so checking later would let a setup satisfy "already established"
    using its own touches - reading (a) enforced in name only.
  - Zone geometry is always read at the bar being evaluated, never at the end
    of history.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from fxlab.indicators import atr
from fxlab.zones.builder import Kind, Origin, Tier, Zone, ZoneBook

Direction = Literal["long", "short"]


@dataclass(frozen=True)
class SetupConfig:
    """Free parameters of the setup rules. Part of the overfitting surface."""

    min_bars_between_touches: int = 6
    # Without an upper bound, two dips a year apart read as one double bottom.
    max_bars_between_touches: int = 60
    # "a bit higher" - bounded both ways. Too little is a flat double bottom;
    # too much is an uptrend pullback, a different animal entirely.
    min_second_touch_offset_atr: float = 0.05
    max_second_touch_offset_atr: float = 0.75
    entry_buffer_atr: float = 0.10
    stop_buffer_atr: float = 0.10
    min_reward_risk: float = 1.5
    # A real ceiling, not just a filter: it changes which zone is chosen as
    # target. Uncapped, "nearest zone paying at least 1.5R" reached targets 33x
    # the risk away, which means no zone existed in between and the level was
    # not a target in any meaningful sense. A setup with no zone in the band is
    # now skipped rather than aimed at the horizon.
    max_reward_risk: float | None = 2.0
    min_prior_touches: int = 2
    require_close_outside_zone: bool = True
    atr_period: int = 20

    # --- Filters derived from the blind-review notes (2026-07-24) ------------
    # All default OFF so the baseline reproduces. Each is DATA-SUGGESTED, so a
    # positive result in research means nothing until it survives 2017-2021.
    # See TRIALS.md.

    # "room to the left": the level was established, then abandoned for a
    # stretch, and price is only now returning - as opposed to grinding at the
    # level continuously. Encoded as a minimum gap between the most recent
    # establishing touch and the setup's own approach.
    require_clearance: bool = False
    clearance_bars: int = 20

    # "the body should close above/below the second touch": the reversal candle
    # must close beyond the prior bar's body, not merely be the right colour.
    require_close_beyond_body: bool = False

    # "the double tap should be the reversing point of a trend" / "price is
    # ranging, risky": price must have travelled directionally INTO the zone
    # over the recent lookback (down into support for a long, up into
    # resistance for a short) by at least min_trend_atr, so the setup is a
    # potential reversal rather than a stall inside a range. This reads the
    # move into the zone, not the whole chart - the choice left open in review.
    require_trend_context: bool = False
    trend_lookback: int = 20
    min_trend_atr: float = 1.5


@dataclass(frozen=True)
class _Spec:
    direction: Direction
    pattern: str
    zone_kind: Kind
    sign: int  # +1 long, -1 short; flips every price comparison


LONG = _Spec("long", "wammie", "support", 1)
SHORT = _Spec("short", "moolah", "resistance", -1)
SPECS = (LONG, SHORT)


@dataclass(frozen=True)
class TouchEvent:
    """One visit to a zone: price enters, interacts, and leaves."""

    start_bar: int
    end_bar: int
    extreme_bar: int
    extreme: float  # lowest low (support) or highest high (resistance)


@dataclass(frozen=True)
class Setup:
    symbol: str
    pattern: str
    direction: Direction
    zone_id: str
    zone_tier: Tier
    zone_origin: Origin
    confluent: bool
    first_touch_bar: int
    second_touch_bar: int
    signal_bar: int
    entry: float
    stop: float
    target: float
    target_zone_id: str
    target_tier: Tier
    reward_risk: float
    risk: float
    ts: dt.datetime

    @property
    def detected_bar(self) -> int:
        """When the setup became knowable - the signal candle's close."""
        return self.signal_bar

    @property
    def setup_id(self) -> str:
        """Stable identifier, for joining decisions back to outcomes."""
        return f"{self.symbol}:{self.pattern}:{self.zone_id}:{self.signal_bar}"


def find_touch_events(
    zone: Zone,
    *,
    highs: np.ndarray,
    lows: np.ndarray,
    query_bars: np.ndarray,
    bar_indices: np.ndarray,
    spec: _Spec,
) -> list[TouchEvent]:
    """Maximal runs of consecutive bars interacting with `zone`.

    Runs, not bars: three consecutive bars inside the zone are one touch, and
    a second touch requires price to have left in between.
    """
    events: list[TouchEvent] = []
    start = end = extreme_bar = -1
    extreme = np.nan

    # Only scan the zone's active window. query_bars is non-decreasing (an h4
    # bar index, or the bridge's latest-knowable D1 bar), so a bar can touch
    # this zone only where created_bar <= query_bars[pos] < ended_bar. Outside
    # that, is_active is False and the bar would be skipped anyway - but a zone
    # lives a few hundred bars, not the whole 18k-bar history, so bounding the
    # loop here is the difference between a minute and an hour per basket pass.
    lo = int(np.searchsorted(query_bars, zone.created_bar, side="left"))
    ended = zone.ended_bar
    hi = int(np.searchsorted(query_bars, ended, side="left")) if ended is not None else len(bar_indices)

    for pos in range(lo, hi):
        query = int(query_bars[pos])
        bar = int(bar_indices[pos])
        bounds = zone.bounds_as_of(query) if query >= 0 and zone.is_active(query) else None

        touching = bounds is not None and lows[pos] <= bounds[1] and highs[pos] >= bounds[0]
        if touching:
            candidate = lows[pos] if spec.sign > 0 else highs[pos]
            if start < 0:
                start, extreme_bar, extreme = bar, bar, candidate
            elif spec.sign * (candidate - extreme) < 0:
                extreme_bar, extreme = bar, candidate
            end = bar
        elif start >= 0:
            events.append(TouchEvent(start, end, extreme_bar, float(extreme)))
            start = end = extreme_bar = -1

    if start >= 0:
        events.append(TouchEvent(start, end, extreme_bar, float(extreme)))
    return events


def _is_confluent(
    zone: Zone,
    bounds: tuple[float, float],
    books: list[ZoneBook],
    query_by_book: list[np.ndarray],
    pos: int,
) -> bool:
    """Whether a zone on the other tier overlaps this one.

    A D1 and an H4 level sitting on the same price is stronger than either
    alone. Flagged rather than merged, so it stays available as a filter
    without committing to it being real.
    """
    for book, query_bars in zip(books, query_by_book):
        if book.timeframe == zone.timeframe:
            continue
        query = int(query_bars[pos])
        if query < 0:
            continue
        for view in book.zones_as_of(query, min_prior_touches=1):
            if view.kind == zone.kind and view.lower <= bounds[1] and view.upper >= bounds[0]:
                return True
    return False


def _pick_target(
    books: list[ZoneBook],
    query_by_book: list[np.ndarray],
    pos: int,
    entry: float,
    risk: float,
    spec: _Spec,
    config: SetupConfig,
) -> tuple[float, str, Tier] | None:
    """Nearest zone beyond entry that pays at least `min_reward_risk`.

    The near edge of the zone is used, not its middle or far side: price is
    assumed to stall where the level starts, which is the conservative read.
    """
    candidates = []
    for book, query_bars in zip(books, query_by_book):
        query = int(query_bars[pos])
        if query < 0:
            continue
        for view in book.zones_as_of(query, min_prior_touches=config.min_prior_touches):
            near_edge = view.lower if spec.sign > 0 else view.upper
            distance = spec.sign * (near_edge - entry)
            if distance <= 0:
                continue
            candidates.append((distance, near_edge, view))

    for distance, near_edge, view in sorted(candidates, key=lambda c: c[0]):
        reward_risk = distance / risk
        if reward_risk < config.min_reward_risk:
            continue
        if config.max_reward_risk is not None and reward_risk > config.max_reward_risk:
            continue
        return near_edge, view.zone_id, view.tier
    return None


def detect_setups(
    symbol: str,
    bars: pl.DataFrame,
    books: list[ZoneBook],
    *,
    query_by_book: list[np.ndarray],
    config: SetupConfig | None = None,
    specs: tuple[_Spec, ...] = SPECS,
) -> list[Setup]:
    """Find wammies and moolahs on `bars`.

    `query_by_book[i]` maps each position in `bars` to the bar index at which
    `books[i]` should be queried - identity for a book on this timeframe, the
    timeframe bridge for a book on a higher one.
    """
    config = config or SetupConfig()

    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    opens = bars["open"].to_numpy()
    closes = bars["close"].to_numpy()
    bar_indices = bars["bar"].to_numpy()
    timestamps = bars["ts_open"].to_list()
    atr_values = atr(bars, period=config.atr_period)

    pos_of_bar = {int(b): i for i, b in enumerate(bar_indices)}
    setups: list[Setup] = []

    for spec in specs:
        for book, query_bars in zip(books, query_by_book):
            for zone in book.zones:
                if zone.kind != spec.zone_kind:
                    continue

                events = find_touch_events(
                    zone,
                    highs=highs,
                    lows=lows,
                    query_bars=query_bars,
                    bar_indices=bar_indices,
                    spec=spec,
                )

                for first, second in zip(events, events[1:]):
                    setup = _evaluate(
                        symbol=symbol,
                        spec=spec,
                        zone=zone,
                        first=first,
                        second=second,
                        books=books,
                        query_by_book=query_by_book,
                        query_bars=query_bars,
                        pos_of_bar=pos_of_bar,
                        bar_indices=bar_indices,
                        highs=highs,
                        lows=lows,
                        opens=opens,
                        closes=closes,
                        atr_values=atr_values,
                        timestamps=timestamps,
                        config=config,
                    )
                    if setup is not None:
                        setups.append(setup)

    setups.sort(key=lambda s: (s.signal_bar, s.zone_id))
    return setups


def _evaluate(
    *,
    symbol: str,
    spec: _Spec,
    zone: Zone,
    first: TouchEvent,
    second: TouchEvent,
    books: list[ZoneBook],
    query_by_book: list[np.ndarray],
    query_bars: np.ndarray,
    pos_of_bar: dict[int, int],
    bar_indices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    atr_values: np.ndarray,
    timestamps: list,
    config: SetupConfig,
) -> Setup | None:
    """Test one pair of touches against the rules. Returns None on any failure."""
    first_pos = pos_of_bar.get(first.start_bar)
    if first_pos is None:
        return None

    # Reading (a): the level must already have been proven by earlier price
    # action, judged before this setup's own touches could count toward it.
    established_query = int(query_bars[first_pos])
    if established_query < 0:
        return None
    if zone.touch_count_as_of(established_query) < config.min_prior_touches:
        return None

    # "Room to the left": the most recent establishing touch, knowable before
    # this setup's approach, must be at least clearance_bars in the past. Uses
    # forming touches (which carry confirmation lag), never the zone's mutated
    # end-of-history interaction state, which would read the future.
    if config.require_clearance:
        prior_pivots = [
            t.swing.pivot_bar
            for t in zone.touches
            if t.confirmed_bar <= established_query and t.swing.pivot_bar < first.start_bar
        ]
        if not prior_pivots or first.start_bar - max(prior_pivots) < config.clearance_bars:
            return None

    reference_atr = atr_values[first_pos]
    if np.isnan(reference_atr):
        return None

    if config.require_trend_context:
        lookback_pos = pos_of_bar.get(first.start_bar)
        if lookback_pos is None or lookback_pos < config.trend_lookback:
            return None
        start_close = closes[lookback_pos - config.trend_lookback]
        approach_close = closes[lookback_pos]
        displacement = spec.sign * (start_close - approach_close)  # >0 = moved toward zone
        if displacement < config.min_trend_atr * reference_atr:
            return None

    found = _find_signal(
        first=first,
        second=second,
        zone=zone,
        spec=spec,
        query_bars=query_bars,
        pos_of_bar=pos_of_bar,
        highs=highs,
        lows=lows,
        opens=opens,
        closes=closes,
        reference_atr=float(reference_atr),
        config=config,
    )
    if found is None:
        return None
    signal_pos, second_extreme_bar = found

    signal_atr = atr_values[signal_pos]
    if np.isnan(signal_atr):
        return None

    entry_extreme = highs[signal_pos] if spec.sign > 0 else lows[signal_pos]
    entry = entry_extreme + spec.sign * config.entry_buffer_atr * signal_atr
    stop = first.extreme - spec.sign * config.stop_buffer_atr * signal_atr
    risk = spec.sign * (entry - stop)
    if risk <= 0:
        return None

    target = _pick_target(
        books, query_by_book, signal_pos, entry, risk, spec, config
    )
    if target is None:
        return None
    target_price, target_zone_id, target_tier = target

    signal_bounds = zone.bounds_as_of(int(query_bars[signal_pos]))
    confluent = signal_bounds is not None and _is_confluent(
        zone, signal_bounds, books, query_by_book, signal_pos
    )

    return Setup(
        symbol=symbol,
        pattern=spec.pattern,
        direction=spec.direction,
        zone_id=zone.zone_id,
        zone_tier=zone.tier,
        zone_origin=zone.origin,
        confluent=confluent,
        first_touch_bar=first.extreme_bar,
        second_touch_bar=second_extreme_bar,
        signal_bar=int(bar_indices[signal_pos]),
        entry=float(entry),
        stop=float(stop),
        target=float(target_price),
        target_zone_id=target_zone_id,
        target_tier=target_tier,
        reward_risk=float(spec.sign * (target_price - entry) / risk),
        risk=float(risk),
        ts=timestamps[signal_pos],
    )


def _find_signal(
    *,
    first: TouchEvent,
    second: TouchEvent,
    zone: Zone,
    spec: _Spec,
    query_bars: np.ndarray,
    pos_of_bar: dict[int, int],
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    reference_atr: float,
    config: SetupConfig,
) -> tuple[int, int] | None:
    """Walk the second touch bar by bar, testing every rule causally.

    Returns (signal position, bar of the running extreme) or None.

    Two things make this a loop rather than a set of up-front checks:

    - The signal must be the *first* qualifying candle. Whether a bar was the
      touch's last is only knowable once a later bar fails to touch, so keying
      on the final bar would peek.
    - The depth of the second touch must be measured by the extreme *so far*,
      not the extreme of the whole event. The event's true low can land after
      the signal candle, and judging "the second touch is slightly higher"
      against a low that has not printed yet reads the future - which is
      precisely what a trader standing at the signal bar cannot do.
    """
    running_extreme: float | None = None
    running_extreme_bar = -1

    for bar in range(second.start_bar, second.end_bar + 1):
        pos = pos_of_bar.get(bar)
        if pos is None:
            continue

        candidate = lows[pos] if spec.sign > 0 else highs[pos]
        if running_extreme is None or spec.sign * (candidate - running_extreme) < 0:
            running_extreme, running_extreme_bar = float(candidate), bar

        gap = running_extreme_bar - first.extreme_bar
        if not (config.min_bars_between_touches <= gap <= config.max_bars_between_touches):
            continue

        # Shallower than the first touch, but only slightly.
        offset = spec.sign * (running_extreme - first.extreme)
        if not (
            config.min_second_touch_offset_atr * reference_atr
            <= offset
            <= config.max_second_touch_offset_atr * reference_atr
        ):
            continue

        if spec.sign * (closes[pos] - opens[pos]) <= 0:
            continue

        # Close beyond the prior bar's body: the reversal must reclaim ground,
        # not merely print the right colour. For a long, close above the
        # previous candle's body top; mirror for a short.
        if config.require_close_beyond_body and pos > 0:
            prior_body_edge = (
                max(opens[pos - 1], closes[pos - 1])
                if spec.sign > 0
                else min(opens[pos - 1], closes[pos - 1])
            )
            if spec.sign * (closes[pos] - prior_body_edge) <= 0:
                continue

        if config.require_close_outside_zone:
            query = int(query_bars[pos])
            bounds = zone.bounds_as_of(query) if query >= 0 else None
            if bounds is None:
                continue
            edge = bounds[1] if spec.sign > 0 else bounds[0]
            if spec.sign * (closes[pos] - edge) <= 0:
                continue

        return pos, running_extreme_bar
    return None
