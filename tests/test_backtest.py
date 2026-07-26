import datetime as dt

import polars as pl
import pytest

from fxlab import instruments
from fxlab.backtest import BacktestConfig, run
from fxlab.setups.patterns import Setup

EURUSD = instruments.get("EURUSD")
PIP = EURUSD.pip

ENTRY = 1.1000
STOP = 1.0950  # risk = 50 pips
TARGET = 1.1100  # 2.0 R


def make_setup(direction: str = "long", **overrides) -> Setup:
    defaults = dict(
        symbol="EURUSD",
        pattern="wammie" if direction == "long" else "moolah",
        direction=direction,
        zone_id="Z",
        zone_tier="primary",
        zone_origin="native",
        confluent=False,
        first_touch_bar=0,
        second_touch_bar=0,
        signal_bar=0,
        entry=ENTRY,
        stop=STOP,
        target=TARGET,
        target_zone_id="T",
        target_tier="primary",
        reward_risk=2.0,
        risk=abs(ENTRY - STOP),
        ts=dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc),
    )
    return Setup(**{**defaults, **overrides})


def make_bars(rows: list[tuple[float, float, float, float]]) -> pl.DataFrame:
    """rows are (open, high, low, close)."""
    start = dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc)
    return pl.DataFrame(
        {
            "bar": list(range(len(rows))),
            "ts_open": [start + dt.timedelta(hours=4 * i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1.0] * len(rows),
        }
    )


def one(setup, bars, config=None):
    return run([setup], bars, EURUSD, config or BacktestConfig())[0]


def test_clean_target_hit_pays_close_to_planned_r():
    bars = make_bars([
        (1.0990, 1.1000, 1.0980, 1.0995),   # signal
        (1.0990, 1.1010, 1.0985, 1.1005),   # entry triggers
        (1.1005, 1.1150, 1.1000, 1.1120),   # target
    ])
    trade = one(make_setup(), bars)

    assert trade.outcome == "target"
    assert trade.entry_bar == 1
    # Slightly under 2.0: spread and entry slippage are real money.
    assert 1.90 < trade.r_multiple < 2.00


def test_clean_stop_hit_loses_slightly_more_than_one_r():
    bars = make_bars([
        (1.0990, 1.1000, 1.0980, 1.0995),
        (1.0990, 1.1010, 1.0985, 1.1005),
        (1.1000, 1.1010, 1.0900, 1.0920),   # stop
    ])
    trade = one(make_setup(), bars)

    assert trade.outcome == "stop"
    assert -1.15 < trade.r_multiple < -1.00


def test_same_bar_stop_and_target_resolves_pessimistically():
    """Unknowable from OHLC, so the loss is assumed."""
    bars = make_bars([
        (1.0990, 1.1000, 1.0980, 1.0995),
        (1.0990, 1.1010, 1.0985, 1.1005),
        (1.1005, 1.1150, 1.0900, 1.1000),   # spans both barriers
    ])

    pessimistic = one(make_setup(), bars, BacktestConfig(pessimistic_same_bar=True))
    optimistic = one(make_setup(), bars, BacktestConfig(pessimistic_same_bar=False))

    assert pessimistic.outcome == "stop"
    assert optimistic.outcome == "target"
    assert pessimistic.r_multiple < 0 < optimistic.r_multiple


def test_order_expires_unfilled():
    bars = make_bars([
        (1.0990, 1.1000, 1.0980, 1.0995),
        (1.0985, 1.0995, 1.0980, 1.0990),
        (1.0985, 1.0995, 1.0980, 1.0990),
        (1.0985, 1.0995, 1.0980, 1.0990),
        (1.0985, 1.1200, 1.0980, 1.1150),   # too late, order already cancelled
    ])
    trade = one(make_setup(), bars, BacktestConfig(entry_valid_bars=3))

    assert trade.outcome == "no_fill"
    assert not trade.filled
    assert trade.r_multiple == 0.0


def test_gap_through_entry_fills_at_the_open_not_the_wanted_price():
    bars = make_bars([
        (1.0990, 1.1000, 1.0980, 1.0995),
        (1.1050, 1.1060, 1.1045, 1.1055),   # gaps straight past the stop order
        (1.1055, 1.1150, 1.1050, 1.1120),
    ])
    trade = one(make_setup(), bars)

    assert trade.entry_price == pytest.approx(1.1050 + 0.3 * PIP)
    # Paying 50 pips more than planned turns a 2R idea into much less.
    assert trade.r_multiple < 1.0


def test_gap_through_stop_fills_at_the_open():
    bars = make_bars([
        (1.0990, 1.1000, 1.0980, 1.0995),
        (1.0990, 1.1010, 1.0985, 1.1005),
        (1.0900, 1.0910, 1.0850, 1.0870),   # opens below the stop
    ])
    trade = one(make_setup(), bars)

    assert trade.outcome == "stop"
    assert trade.exit_price == pytest.approx(1.0900 - 0.3 * PIP)
    assert trade.r_multiple < -1.5, "a gap should hurt more than a clean stop"


def test_timeout_exits_at_the_close():
    flat = (1.0990, 1.1010, 1.0985, 1.1005)
    bars = make_bars([(1.0990, 1.1000, 1.0980, 1.0995)] + [flat] * 6)
    trade = one(make_setup(), bars, BacktestConfig(max_holding_bars=3))

    assert trade.outcome == "timeout"
    assert trade.bars_held == 3


def test_short_is_the_mirror_of_long():
    long_bars = make_bars([
        (1.0990, 1.1000, 1.0980, 1.0995),
        (1.0990, 1.1010, 1.0985, 1.1005),
        (1.1005, 1.1150, 1.1000, 1.1120),
    ])
    axis = 2.0
    short_bars = long_bars.with_columns(
        (axis - pl.col("open")).alias("open"),
        (axis - pl.col("low")).alias("high"),
        (axis - pl.col("high")).alias("low"),
        (axis - pl.col("close")).alias("close"),
    )

    long_trade = one(make_setup("long"), long_bars)
    short_trade = one(
        make_setup("short", entry=axis - ENTRY, stop=axis - STOP, target=axis - TARGET),
        short_bars,
    )

    assert long_trade.outcome == short_trade.outcome
    assert long_trade.r_multiple == pytest.approx(short_trade.r_multiple, abs=1e-9)


def test_r_is_measured_against_planned_risk_not_the_actual_fill():
    """Slippage must show as degraded R, not be hidden by re-basing R."""
    bars = make_bars([
        (1.0990, 1.1000, 1.0980, 1.0995),
        (1.1020, 1.1030, 1.1015, 1.1025),   # filled 20 pips late
        (1.1025, 1.1150, 1.1020, 1.1120),
    ])
    trade = one(make_setup(), bars)

    assert trade.outcome == "target"
    # Re-basing on the actual fill would still report ~2R; planned risk does not.
    assert trade.r_multiple < 1.7
