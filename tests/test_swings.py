import datetime as dt

import polars as pl
import pytest

from fxlab.zones.swings import detect_swings


def make_bars(closes: list[float], *, spread: float = 0.5) -> pl.DataFrame:
    """Bars whose closes are `closes` and whose wicks extend past them."""
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    return pl.DataFrame(
        {
            "bar": list(range(len(closes))),
            "ts_open": [start + dt.timedelta(hours=4 * i) for i in range(len(closes))],
            "open": closes,
            "high": [c + spread for c in closes],
            "low": [c - spread for c in closes],
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


def test_finds_the_obvious_pivots():
    bars = make_bars([5, 4, 3, 4, 5, 6, 5, 4, 5])
    swings = detect_swings(bars, window=2)

    lows = [s.pivot_bar for s in swings if s.kind == "low"]
    highs = [s.pivot_bar for s in swings if s.kind == "high"]

    assert lows == [2]
    assert highs == [5]


def test_confirmation_lags_the_pivot_by_the_window():
    bars = make_bars([5, 4, 3, 4, 5, 6, 5, 4, 5])

    for window in (1, 2, 3):
        for swing in detect_swings(bars, window=window):
            assert swing.confirmed_bar == swing.pivot_bar + window


def test_edges_cannot_be_swings():
    """A pivot needs a full window on both sides, so the tails are excluded."""
    closes = [3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5]
    bars = make_bars(closes)
    window = 2

    swings = detect_swings(bars, window=window)
    pivots = [s.pivot_bar for s in swings]

    assert all(window <= p < len(closes) - window for p in pivots)
    # Index 2 is the lowest close overall but index 0 is not a candidate.
    assert 2 in pivots


def test_flat_run_resolves_to_a_single_bar():
    """Equal closes must not each register as their own swing."""
    bars = make_bars([5, 4, 3, 3, 3, 4, 5])
    swings = detect_swings(bars, window=2)

    lows = [s.pivot_bar for s in swings if s.kind == "low"]
    assert len(lows) == 1


def test_close_drives_geometry_but_wicks_are_kept():
    bars = make_bars([5, 4, 3, 4, 5, 6, 5, 4, 5], spread=0.25)
    swing = next(s for s in detect_swings(bars, window=2) if s.kind == "low")

    assert swing.close == pytest.approx(3.0)
    assert swing.low == pytest.approx(2.75)
    # Stops reference the wick, never the close that defined the level.
    assert swing.stop_reference == pytest.approx(2.75)


def test_swings_are_returned_in_confirmation_order():
    bars = make_bars([5, 4, 3, 4, 5, 6, 5, 4, 5, 7, 8, 7, 6])
    swings = detect_swings(bars, window=2)

    confirmed = [s.confirmed_bar for s in swings]
    assert confirmed == sorted(confirmed)


def test_series_shorter_than_the_window_yields_nothing():
    assert detect_swings(make_bars([1, 2, 3]), window=2) == []
