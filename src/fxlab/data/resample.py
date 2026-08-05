"""Build H4 and D1 bars from H1.

Both timeframes hang off one anchor: the 17:00 New York close that defines the
FX trading day. That keeps exactly 6 H4 bars inside every D1 bar, so "six
candles between touches" means the same thing whichever tier a setup forms on.

Why New York rather than a fixed UTC offset: the market week runs Sunday 17:00
to Friday 17:00 New York, so an NY anchor yields exactly five sessions a week
with no weekend stubs. A fixed UTC offset only matches for half the year -
UTC+3 was tried first and produced Saturday stub bars every winter week,
inflating 2024 from 262 sessions to 279.

Bucketing runs on a naive "session clock" (NY time shifted so 17:00 becomes
midnight). Bucket boundaries land on 17/21/01/05/09/13 NY, and DST transitions
happen at 02:00 NY, so no boundary is ever ambiguous or nonexistent.

ts_open and ts_close are nominal session boundaries, not the first and last
ticks observed. ts_close is the moment the bar becomes knowable, and is the
only timestamp downstream logic may gate on.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

SESSION_TZ = "America/New_York"
SESSION_CLOSE_HOUR = 17
_TO_MIDNIGHT = dt.timedelta(hours=24 - SESSION_CLOSE_HOUR)

_PERIOD = {"H4": dt.timedelta(hours=4), "D1": dt.timedelta(days=1)}
_EVERY = {"H4": "4h", "D1": "1d"}

SCHEMA = {
    "ts_open": pl.Datetime("us", "UTC"),
    "ts_close": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def session_clock(expr: pl.Expr) -> pl.Expr:
    """UTC instant -> naive session clock (17:00 NY becomes midnight)."""
    return (
        expr.dt.convert_time_zone(SESSION_TZ).dt.replace_time_zone(None) + _TO_MIDNIGHT
    )


def session_date(expr: pl.Expr) -> pl.Expr:
    """UTC instant -> the trading day it belongs to.

    Every H4 bar in a session maps to the same date as its D1 bar, which is
    what makes the two tiers joinable. Grouping on the raw NY date instead
    splits each session 1/5 across two calendar days.
    """
    return session_clock(expr).dt.date()


def _clock_to_utc(expr: pl.Expr) -> pl.Expr:
    """Naive session clock -> real UTC instant."""
    return (
        (expr - _TO_MIDNIGHT)
        .dt.replace_time_zone(SESSION_TZ)
        .dt.convert_time_zone("UTC")
    )


def resample(
    h1: pl.DataFrame, timeframe: str, *, drop_incomplete: bool = True
) -> pl.DataFrame:
    """Aggregate H1 bars up to `timeframe` ("H4" or "D1")."""
    if timeframe not in _PERIOD:
        raise ValueError(f"timeframe must be one of {sorted(_PERIOD)}, got {timeframe!r}")
    if h1.is_empty():
        return pl.DataFrame(schema=SCHEMA)

    grouped = (
        h1.sort("ts_open")
        .with_columns(session_clock(pl.col("ts_open")).alias("_clock"))
        .group_by_dynamic("_clock", every=_EVERY[timeframe], closed="left", label="left")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        )
        .with_columns(
            _clock_to_utc(pl.col("_clock")).alias("ts_open"),
            _clock_to_utc(pl.col("_clock") + _PERIOD[timeframe]).alias("ts_close"),
        )
        .select(*SCHEMA)
        .sort("ts_open")
    )

    if drop_incomplete:
        # The newest bucket is usually still forming. Letting a partial bar
        # count as closed would allow a not-yet-final close to set a swing.
        h1_end = h1.select(pl.col("ts_open").max()).item() + dt.timedelta(hours=1)
        grouped = grouped.filter(pl.col("ts_close") <= h1_end)

    return grouped
