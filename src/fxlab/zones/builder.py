"""Zone construction and lifecycle.

A zone is a band of price that has repeatedly turned the market. It is built by
clustering close-based swings, it widens as touches accumulate, and it dies when
price closes decisively through it.

The invariant that matters most: a zone's geometry at bar i is derived only
from touches confirmed at or before bar i. Bounds are therefore never stored as
fields - they are computed on demand from the visible touch set. Storing them
would guarantee that some caller eventually reads a zone's final, widest bounds
while standing at a bar where the zone was still narrow.

Two distinct notions of "touch" live in this project and must not be conflated:

  - FormingTouch (here) is a confirmed swing that establishes the level. It
    carries fractal confirmation lag, so it is knowable `swing_window` bars
    after its pivot.
  - A trading touch - price returning to an established zone - is knowable at
    the close of the bar it happens on, with no lag, and belongs to the setup
    detector rather than to zone construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from fxlab.indicators import atr
from fxlab.zones.swings import Swing, detect_swings

Kind = Literal["support", "resistance"]
Tier = Literal["primary", "secondary"]
Origin = Literal["native", "flipped"]

TIER_BY_TIMEFRAME: dict[str, Tier] = {"D1": "primary", "H4": "secondary"}


@dataclass(frozen=True)
class ZoneConfig:
    """Every free parameter in zone construction, in one place.

    This is the overfitting surface. Sweeps over these should be looking for
    plateaus - a setting that works only at one exact value is noise.
    """

    swing_window: int = 2
    atr_period: int = 20
    cluster_tolerance_atr: float = 0.4
    min_zone_width_atr: float = 0.2
    max_zone_width_atr: float = 1.0
    break_margin_atr: float = 0.5
    min_prior_touches: int = 2
    flip_enabled: bool = True
    # Whether a flipped zone arrives pre-qualified with its parent's touch
    # count. Inheriting makes every break mint an instantly-established level,
    # which quietly bypasses min_prior_touches for the majority of the book.
    # Off by default: a flipped level must prove itself in its new role.
    flip_inherits_touches: bool = False
    # A level price has not revisited in this many bars stops being a level.
    # Without expiry the book only grows: zones break, flip, and the flipped
    # remnant survives forever, which is how 90%+ of a book ends up being
    # flip debris nobody would draw.
    max_untouched_bars: int | None = 250


@dataclass(frozen=True)
class FormingTouch:
    """A confirmed swing that helps define a zone."""

    swing: Swing

    @property
    def confirmed_bar(self) -> int:
        return self.swing.confirmed_bar

    @property
    def price(self) -> float:
        return self.swing.close


@dataclass(frozen=True)
class ZoneView:
    """Immutable snapshot of a zone as it stood at one bar."""

    zone_id: str
    kind: Kind
    tier: Tier
    timeframe: str
    origin: Origin
    lower: float
    upper: float
    touch_count: int
    created_bar: int
    last_touch_bar: int
    confluent: bool = False

    @property
    def mid(self) -> float:
        return (self.lower + self.upper) / 2

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper


@dataclass
class Zone:
    zone_id: str
    kind: Kind
    tier: Tier
    timeframe: str
    created_bar: int
    min_width: float
    max_width: float
    seed_lower: float
    seed_upper: float
    touches: list[FormingTouch] = field(default_factory=list)
    broken_bar: int | None = None
    expired_bar: int | None = None
    origin: Origin = "native"
    flipped_from: str | None = None
    inherited_touches: int = 0
    last_interaction_bar: int = -1

    def __post_init__(self) -> None:
        if self.last_interaction_bar < 0:
            self.last_interaction_bar = self.created_bar

    @property
    def ended_bar(self) -> int | None:
        """First bar at which the zone stopped being tradeable, either way."""
        ends = [b for b in (self.broken_bar, self.expired_bar) if b is not None]
        return min(ends) if ends else None

    def visible_touches(self, bar: int) -> list[FormingTouch]:
        return [t for t in self.touches if t.confirmed_bar <= bar]

    def bounds_as_of(self, bar: int) -> tuple[float, float] | None:
        """Zone band using only what was knowable at `bar`."""
        if bar < self.created_bar:
            return None

        lower, upper = self.seed_lower, self.seed_upper
        for touch in self.visible_touches(bar):
            lower = min(lower, touch.price)
            upper = max(upper, touch.price)

        if upper - lower < self.min_width:
            mid = (lower + upper) / 2
            lower, upper = mid - self.min_width / 2, mid + self.min_width / 2
        return lower, upper

    def would_admit(self, price: float, bar: int, tolerance: float) -> bool:
        """Whether `price` can join this zone without bloating it past its cap.

        The width cap is what stops zone drift: without it, each touch widens
        the band, the wider band matches more distant swings, and a level
        ratchets outward until it spans an entire trading range.
        """
        bounds = self.bounds_as_of(bar)
        if bounds is None:
            return False
        lower, upper = bounds
        if not (lower - tolerance <= price <= upper + tolerance):
            return False
        return max(upper, price) - min(lower, price) <= self.max_width

    def touch_count_as_of(self, bar: int) -> int:
        return len(self.visible_touches(bar)) + self.inherited_touches

    def is_active(self, bar: int) -> bool:
        if bar < self.created_bar:
            return False
        ended = self.ended_bar
        return ended is None or bar < ended

    def view(self, bar: int) -> ZoneView | None:
        bounds = self.bounds_as_of(bar)
        if bounds is None:
            return None
        visible = self.visible_touches(bar)
        return ZoneView(
            zone_id=self.zone_id,
            kind=self.kind,
            tier=self.tier,
            timeframe=self.timeframe,
            origin=self.origin,
            lower=bounds[0],
            upper=bounds[1],
            touch_count=len(visible) + self.inherited_touches,
            created_bar=self.created_bar,
            last_touch_bar=max((t.swing.pivot_bar for t in visible), default=self.created_bar),
        )


class ZoneBook:
    """All zones for one instrument on one timeframe, queryable at any bar."""

    def __init__(self, symbol: str, timeframe: str, zones: list[Zone], config: ZoneConfig):
        self.symbol = symbol
        self.timeframe = timeframe
        self.zones = zones
        self.config = config

    def __len__(self) -> int:
        return len(self.zones)

    def zones_as_of(
        self,
        bar: int,
        *,
        kind: Kind | None = None,
        min_prior_touches: int | None = None,
    ) -> list[ZoneView]:
        """Active, established zones at `bar`, nearest-first by mid price.

        `min_prior_touches` defaults to the config value. Raising it demands the
        level was already proven before a setup forms on it; lowering it lets
        setups form on levels the market has barely tested.
        """
        threshold = (
            self.config.min_prior_touches if min_prior_touches is None else min_prior_touches
        )
        out = []
        for zone in self.zones:
            if not zone.is_active(bar):
                continue
            if kind is not None and zone.kind != kind:
                continue
            if zone.touch_count_as_of(bar) < threshold:
                continue
            view = zone.view(bar)
            if view is not None:
                out.append(view)
        return sorted(out, key=lambda v: v.mid)


def build_zones(
    bars: pl.DataFrame,
    symbol: str,
    timeframe: str,
    config: ZoneConfig | None = None,
) -> ZoneBook:
    """Walk bars forward, growing and killing zones as the market prints.

    Strictly causal: at bar i only swings confirmed by i have been consumed,
    and breaks are judged against that bar's close.
    """
    config = config or ZoneConfig()
    tier = TIER_BY_TIMEFRAME.get(timeframe)
    if tier is None:
        raise ValueError(f"no tier defined for timeframe {timeframe!r}")

    swings = detect_swings(bars, window=config.swing_window)
    atr_values = atr(bars, period=config.atr_period)
    closes = bars["close"].to_numpy()
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    bar_indices = bars["bar"].to_numpy()

    swings_by_confirmation: dict[int, list[Swing]] = {}
    for swing in swings:
        swings_by_confirmation.setdefault(swing.confirmed_bar, []).append(swing)

    zones: list[Zone] = []
    active: list[Zone] = []
    counter = 0

    for position, bar in enumerate(bar_indices):
        bar = int(bar)
        current_atr = atr_values[position]
        if np.isnan(current_atr):
            continue  # ATR warmup; no scale to measure tolerance against yet

        # 1. Absorb swings that became knowable on this bar.
        for swing in swings_by_confirmation.get(bar, []):
            kind: Kind = "support" if swing.kind == "low" else "resistance"
            tolerance = config.cluster_tolerance_atr * current_atr

            candidates = [
                z
                for z in active
                if z.kind == kind and z.would_admit(swing.close, bar, tolerance)
            ]
            if candidates:
                nearest = min(
                    candidates,
                    key=lambda z: abs(swing.close - sum(z.bounds_as_of(bar)) / 2),  # type: ignore[arg-type]
                )
                nearest.touches.append(FormingTouch(swing))
            else:
                counter += 1
                width = config.min_zone_width_atr * current_atr
                zone = Zone(
                    zone_id=f"{symbol}-{timeframe}-{counter:05d}",
                    kind=kind,
                    tier=tier,
                    timeframe=timeframe,
                    created_bar=bar,
                    min_width=width,
                    max_width=config.max_zone_width_atr * current_atr,
                    seed_lower=swing.close - width / 2,
                    seed_upper=swing.close + width / 2,
                    touches=[FormingTouch(swing)],
                )
                zones.append(zone)
                active.append(zone)

        # 2. Kill zones this bar broke through, and retire ones price has
        #    stopped visiting.
        close = closes[position]
        high, low = highs[position], lows[position]
        margin = config.break_margin_atr * current_atr
        still_active = []
        for zone in active:
            bounds = zone.bounds_as_of(bar)
            if bounds is None:
                still_active.append(zone)
                continue
            lower, upper = bounds

            if low <= upper and high >= lower:
                zone.last_interaction_bar = bar

            broken = (
                close < lower - margin if zone.kind == "support" else close > upper + margin
            )
            if not broken:
                stale = (
                    config.max_untouched_bars is not None
                    and bar - zone.last_interaction_bar > config.max_untouched_bars
                )
                if stale:
                    zone.expired_bar = bar
                else:
                    still_active.append(zone)
                continue

            zone.broken_bar = bar
            if config.flip_enabled:
                counter += 1
                flipped = Zone(
                    zone_id=f"{symbol}-{timeframe}-{counter:05d}",
                    kind="resistance" if zone.kind == "support" else "support",
                    tier=zone.tier,
                    timeframe=timeframe,
                    created_bar=bar,
                    min_width=zone.min_width,
                    max_width=zone.max_width,
                    seed_lower=lower,
                    seed_upper=upper,
                    origin="flipped",
                    flipped_from=zone.zone_id,
                    # The level was proven as support; whether it holds as
                    # resistance is exactly the untested claim. Tagged either
                    # way so the question stays answerable instead of assumed.
                    inherited_touches=(
                        zone.touch_count_as_of(bar) if config.flip_inherits_touches else 0
                    ),
                )
                zones.append(flipped)
                still_active.append(flipped)
        active = still_active

    return ZoneBook(symbol, timeframe, zones, config)
