import datetime as dt

import polars as pl
import pytest

from fxlab.zones.builder import ZoneConfig, build_zones

BASE = 1.1000
RANGE = 0.0010
WARMUP = 25  # ATR period plus headroom


def make_bars(
    closes: list[float],
    *,
    lows: dict[int, float] | None = None,
    highs: dict[int, float] | None = None,
) -> pl.DataFrame:
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    lows = lows or {}
    highs = highs or {}
    return pl.DataFrame(
        {
            "bar": list(range(len(closes))),
            "ts_open": [start + dt.timedelta(days=i) for i in range(len(closes))],
            "open": closes,
            "high": [highs.get(i, c + RANGE) for i, c in enumerate(closes)],
            "low": [lows.get(i, c - RANGE) for i, c in enumerate(closes)],
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


def flat_series(n: int, dips: dict[int, float]) -> list[float]:
    """A flat baseline with dips at chosen bars, so pivots are unambiguous."""
    return [dips.get(i, BASE) for i in range(n)]


def test_a_dip_creates_a_support_zone():
    bars = make_bars(flat_series(WARMUP + 20, {30: 1.0900}))
    book = build_zones(bars, "TEST", "D1")

    supports = [z for z in book.zones if z.kind == "support"]
    assert len(supports) == 1
    zone = supports[0]

    # Knowable only after fractal confirmation, not at the pivot.
    assert zone.created_bar == 30 + ZoneConfig().swing_window
    assert zone.tier == "primary"
    assert zone.origin == "native"


def test_bounds_ignore_touches_that_have_not_happened_yet():
    """The invariant the whole design exists to protect."""
    bars = make_bars(flat_series(WARMUP + 60, {30: 1.0900, 60: 1.0895}))
    book = build_zones(bars, "TEST", "D1")

    zone = next(z for z in book.zones if z.kind == "support")
    assert zone.touch_count_as_of(80) == 2, "both dips should join one zone"

    early = zone.bounds_as_of(40)
    late = zone.bounds_as_of(80)
    assert early is not None and late is not None

    # The second dip is lower, so it may only widen the zone downward later.
    assert early[0] > late[0]
    assert late[0] == pytest.approx(1.0895)


def test_zone_is_invisible_before_it_is_created():
    bars = make_bars(flat_series(WARMUP + 20, {30: 1.0900}))
    book = build_zones(bars, "TEST", "D1")
    zone = next(z for z in book.zones if z.kind == "support")

    assert zone.bounds_as_of(29) is None
    assert not zone.is_active(29)
    assert zone.is_active(zone.created_bar)
    assert book.zones_as_of(29, kind="support", min_prior_touches=1) == []


def test_close_beyond_the_zone_breaks_it():
    closes = flat_series(WARMUP + 60, {30: 1.0900, 50: 1.0850})
    book = build_zones(make_bars(closes), "TEST", "D1")

    zone = next(z for z in book.zones if z.kind == "support" and z.origin == "native")
    assert zone.broken_bar == 50
    assert not zone.is_active(51)


def test_a_wick_through_the_zone_does_not_break_it():
    """Only closes kill a level; wicks through support are ordinary."""
    closes = flat_series(WARMUP + 60, {30: 1.0900})
    bars = make_bars(closes, lows={50: 1.0700})

    book = build_zones(bars, "TEST", "D1")
    zone = next(z for z in book.zones if z.kind == "support")

    assert zone.broken_bar is None
    assert zone.is_active(55)


def test_break_flips_the_zone_and_tags_its_origin():
    closes = flat_series(WARMUP + 60, {30: 1.0900, 50: 1.0850})
    book = build_zones(make_bars(closes), "TEST", "D1")

    broken = next(z for z in book.zones if z.origin == "native" and z.broken_bar == 50)
    flipped = next(z for z in book.zones if z.origin == "flipped")

    assert flipped.kind == "resistance"
    assert flipped.flipped_from == broken.zone_id
    assert flipped.created_bar == 50
    # By default a flipped level starts unproven in its new role, so it cannot
    # satisfy min_prior_touches purely by virtue of having been broken.
    assert flipped.inherited_touches == 0

    inheriting = build_zones(
        make_bars(closes), "TEST", "D1", ZoneConfig(flip_inherits_touches=True)
    )
    flipped_inheriting = next(z for z in inheriting.zones if z.origin == "flipped")
    assert flipped_inheriting.inherited_touches > 0


def test_flips_can_be_disabled():
    closes = flat_series(WARMUP + 60, {30: 1.0900, 50: 1.0850})
    config = ZoneConfig(flip_enabled=False)
    book = build_zones(make_bars(closes), "TEST", "D1", config)

    assert not [z for z in book.zones if z.origin == "flipped"]


def test_min_prior_touches_gates_which_zones_are_visible():
    bars = make_bars(flat_series(WARMUP + 60, {30: 1.0900, 60: 1.0895}))
    book = build_zones(bars, "TEST", "D1")

    # After one dip the level is not yet established under reading (a).
    assert book.zones_as_of(45, kind="support", min_prior_touches=2) == []
    assert len(book.zones_as_of(45, kind="support", min_prior_touches=1)) == 1
    assert len(book.zones_as_of(80, kind="support", min_prior_touches=2)) == 1


def test_secondary_tier_for_h4():
    bars = make_bars(flat_series(WARMUP + 20, {30: 1.0900}))
    book = build_zones(bars, "TEST", "H4")

    assert all(z.tier == "secondary" for z in book.zones)


def test_unknown_timeframe_is_rejected():
    bars = make_bars(flat_series(WARMUP + 20, {30: 1.0900}))
    with pytest.raises(ValueError, match="no tier"):
        build_zones(bars, "TEST", "M15")
