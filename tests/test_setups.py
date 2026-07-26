import datetime as dt

import numpy as np
import polars as pl
import pytest

from fxlab.setups.patterns import (
    LONG,
    SHORT,
    SetupConfig,
    detect_setups,
    find_touch_events,
)
from fxlab.zones.builder import ZoneConfig, Zone, build_zones

MIRROR_AXIS = 2.0


def frame_from_closes(closes: np.ndarray, spread: float = 0.0005) -> pl.DataFrame:
    start = dt.datetime(2005, 1, 3, tzinfo=dt.timezone.utc)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    return pl.DataFrame(
        {
            "bar": list(range(len(closes))),
            "ts_open": [start + dt.timedelta(hours=4 * i) for i in range(len(closes))],
            "ts_close": [start + dt.timedelta(hours=4 * (i + 1)) for i in range(len(closes))],
            "open": opens,
            "high": np.maximum(opens, closes) + spread,
            "low": np.minimum(opens, closes) - spread,
            "close": closes,
            "volume": np.ones(len(closes)),
        }
    )


def random_walk(n: int = 2500, seed: int = 11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 1.1 + np.cumsum(rng.normal(0, 0.0012, n))
    return frame_from_closes(closes)


def mirror(bars: pl.DataFrame) -> pl.DataFrame:
    """Reflect prices about a fixed axis. Highs and lows swap roles."""
    return bars.with_columns(
        (MIRROR_AXIS - pl.col("open")).alias("open"),
        (MIRROR_AXIS - pl.col("low")).alias("high"),
        (MIRROR_AXIS - pl.col("high")).alias("low"),
        (MIRROR_AXIS - pl.col("close")).alias("close"),
    )


def run(bars: pl.DataFrame, config: SetupConfig | None = None):
    book = build_zones(bars, "TEST", "H4", ZoneConfig())
    return detect_setups(
        "TEST",
        bars,
        [book],
        query_by_book=[bars["bar"].to_numpy()],
        config=config or SetupConfig(),
    )


# --- touch events -----------------------------------------------------------


def make_zone(lower: float, upper: float) -> Zone:
    return Zone(
        zone_id="Z",
        kind="support",
        tier="primary",
        timeframe="D1",
        created_bar=0,
        min_width=upper - lower,
        max_width=upper - lower,
        seed_lower=lower,
        seed_upper=upper,
    )


def test_consecutive_bars_in_a_zone_are_one_touch():
    zone = make_zone(1.0990, 1.1010)
    n = 10
    highs = np.full(n, 1.1050)
    lows = np.full(n, 1.1040)
    lows[3:6] = 1.1000  # three consecutive bars dip in
    highs[3:6] = 1.1020

    events = find_touch_events(
        zone,
        highs=highs,
        lows=lows,
        query_bars=np.arange(n),
        bar_indices=np.arange(n),
        spec=LONG,
    )

    assert len(events) == 1
    assert (events[0].start_bar, events[0].end_bar) == (3, 5)


def test_leaving_and_returning_makes_two_touches():
    zone = make_zone(1.0990, 1.1010)
    n = 14
    highs = np.full(n, 1.1050)
    lows = np.full(n, 1.1040)
    for span in (slice(2, 4), slice(8, 10)):
        lows[span] = 1.1000
        highs[span] = 1.1020

    events = find_touch_events(
        zone,
        highs=highs,
        lows=lows,
        query_bars=np.arange(n),
        bar_indices=np.arange(n),
        spec=LONG,
    )

    assert len(events) == 2
    assert events[0].end_bar < events[1].start_bar


def test_touch_extreme_uses_low_for_support_and_high_for_resistance():
    zone = make_zone(1.0990, 1.1010)
    n = 8
    highs = np.full(n, 1.1050)
    lows = np.full(n, 1.1040)
    lows[3:5] = [1.1000, 1.0995]
    highs[3:5] = [1.1020, 1.1015]

    long_events = find_touch_events(
        zone, highs=highs, lows=lows, query_bars=np.arange(n),
        bar_indices=np.arange(n), spec=LONG,
    )
    short_events = find_touch_events(
        zone, highs=highs, lows=lows, query_bars=np.arange(n),
        bar_indices=np.arange(n), spec=SHORT,
    )

    assert long_events[0].extreme == pytest.approx(1.0995)
    assert long_events[0].extreme_bar == 4
    assert short_events[0].extreme == pytest.approx(1.1020)
    assert short_events[0].extreme_bar == 3


def test_zone_that_is_not_active_is_never_touched():
    zone = make_zone(1.0990, 1.1010)
    zone.broken_bar = 2
    n = 10
    highs = np.full(n, 1.1020)
    lows = np.full(n, 1.1000)

    events = find_touch_events(
        zone, highs=highs, lows=lows, query_bars=np.arange(n),
        bar_indices=np.arange(n), spec=LONG,
    )

    assert all(e.end_bar < 2 for e in events)


# --- the detector -----------------------------------------------------------


def test_detector_finds_setups_on_a_random_walk():
    setups = run(random_walk())
    assert setups, "a 2500-bar walk should produce at least a few setups"


def test_wammies_and_moolahs_are_exact_mirrors():
    """The strongest structural check on the shared-direction design.

    Reflecting price turns every double bottom into a double top. If the long
    and short paths have diverged at all, the counts stop matching.
    """
    bars = random_walk()
    normal = run(bars)
    reflected = run(mirror(bars))

    assert sum(s.pattern == "wammie" for s in normal) == sum(
        s.pattern == "moolah" for s in reflected
    )
    assert sum(s.pattern == "moolah" for s in normal) == sum(
        s.pattern == "wammie" for s in reflected
    )


def test_mirrored_setups_have_mirrored_prices():
    bars = random_walk()
    normal = sorted(run(bars), key=lambda s: (s.signal_bar, s.pattern))
    reflected = sorted(run(mirror(bars)), key=lambda s: (s.signal_bar, s.pattern))

    pairs = [
        (a, b)
        for a, b in zip(normal, reflected)
        if a.signal_bar == b.signal_bar and a.pattern != b.pattern
    ]
    assert pairs, "expected mirrored setups to line up on the same bars"
    for a, b in pairs[:20]:
        assert a.entry == pytest.approx(MIRROR_AXIS - b.entry, abs=1e-9)
        assert a.stop == pytest.approx(MIRROR_AXIS - b.stop, abs=1e-9)


def test_every_setup_respects_the_reward_risk_floor():
    config = SetupConfig(min_reward_risk=2.0)
    for setup in run(random_walk(), config):
        assert setup.reward_risk >= 2.0


def test_reward_risk_cap_is_enforced_when_set():
    capped = run(random_walk(), SetupConfig(min_reward_risk=1.5, max_reward_risk=2.0))
    assert capped, "capping R:R should not eliminate every setup"
    for setup in capped:
        assert 1.5 <= setup.reward_risk <= 2.0


def test_touch_spacing_bounds_are_respected():
    config = SetupConfig(min_bars_between_touches=10, max_bars_between_touches=30)
    for setup in run(random_walk(), config):
        gap = setup.second_touch_bar - setup.first_touch_bar
        assert 10 <= gap <= 30


def test_stop_sits_beyond_the_first_touch_and_risk_is_positive():
    for setup in run(random_walk()):
        assert setup.risk > 0
        if setup.direction == "long":
            assert setup.stop < setup.entry < setup.target
        else:
            assert setup.stop > setup.entry > setup.target


def test_signal_never_precedes_the_second_touch():
    """The setup cannot be knowable before the candle that triggers it."""
    for setup in run(random_walk()):
        assert setup.signal_bar >= setup.second_touch_bar
        assert setup.detected_bar == setup.signal_bar
        assert setup.first_touch_bar < setup.second_touch_bar


def test_raising_min_prior_touches_cannot_add_setups():
    """A stricter establishment rule is a filter, never a generator."""
    loose = run(random_walk(), SetupConfig(min_prior_touches=1))
    strict = run(random_walk(), SetupConfig(min_prior_touches=3))
    assert len(strict) <= len(loose)


# --- note-derived filters (data-suggested, off by default) ------------------


def test_new_filters_are_off_by_default():
    config = SetupConfig()
    assert not config.require_clearance
    assert not config.require_close_beyond_body
    assert not config.require_trend_context


@pytest.mark.parametrize(
    "config",
    [
        SetupConfig(require_clearance=True, clearance_bars=15),
        SetupConfig(require_close_beyond_body=True),
        SetupConfig(require_trend_context=True, min_trend_atr=1.0),
    ],
)
def test_each_filter_only_removes_setups(config):
    """Every note-derived filter must be a strict subset of the baseline."""
    bars = random_walk()
    baseline = {s.setup_id for s in run(bars)}
    filtered = {s.setup_id for s in run(bars, config)}
    assert filtered <= baseline


def test_close_beyond_body_is_stricter_than_close_colour_alone():
    bars = random_walk()
    lenient = run(bars, SetupConfig(require_close_beyond_body=False))
    strict = run(bars, SetupConfig(require_close_beyond_body=True))
    assert len(strict) <= len(lenient)


def test_trend_context_demands_directional_approach():
    """A long needs price to have fallen into support; a dead-flat run yields none."""
    flat = frame_from_closes(1.1 + 0.0 * __import__("numpy").arange(2500))
    assert run(flat, SetupConfig(require_trend_context=True, min_trend_atr=1.0)) == []
