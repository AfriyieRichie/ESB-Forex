"""Sanity-check stored bars: session counts, weekday spread, H4-per-D1 ratio.

Catches the class of bug that silently corrupts everything downstream - stub
weekend sessions, misaligned timeframes, gaps - before any zone logic runs.
"""

from __future__ import annotations

import argparse

import polars as pl

from fxlab.data import load_bars
from fxlab.data.resample import session_date

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="EURUSD")
    args = parser.parse_args()

    d1 = load_bars(args.symbol, "D1")
    h4 = load_bars(args.symbol, "H4")

    print(f"{args.symbol}  D1={len(d1)} bars  H4={len(h4)} bars")

    print("\nD1 sessions per year (expect ~255-262):")
    per_year = (
        d1.group_by(pl.col("ts_close").dt.year().alias("year"))
        .len()
        .sort("year")
    )
    for year, count in per_year.iter_rows():
        flag = "" if 240 <= count <= 266 else "   <-- suspicious"
        print(f"  {year}: {count:>4}{flag}")

    print("\nD1 sessions by weekday of ts_close (expect Mon-Fri only):")
    # ts_close is 17:00 NY, so a session closing Monday 17:00 NY is the Monday
    # session. Saturday or Sunday here means weekend stubs are leaking through.
    by_dow = (
        d1.with_columns(
            pl.col("ts_close")
            .dt.convert_time_zone("America/New_York")
            .dt.weekday()
            .alias("dow")
        )
        .group_by("dow")
        .len()
        .sort("dow")
    )
    for dow, count in by_dow.iter_rows():
        flag = "   <-- WEEKEND STUB" if dow >= 6 else ""
        print(f"  {WEEKDAYS[dow - 1]}: {count:>5}{flag}")

    print("\nH4 bars per D1 session (expect 6, fewer on holidays):")
    h4_per_session = (
        h4.with_columns(session_date(pl.col("ts_open")).alias("session"))
        .group_by("session")
        .len(name="n_bars")
        .group_by("n_bars")
        .len(name="n_sessions")
        .sort("n_bars")
    )
    for n_bars, n_sessions in h4_per_session.iter_rows():
        print(f"  {n_bars} H4 bars: {n_sessions:>5} sessions")

    print("\nprice sanity:")
    stats = d1.select(
        pl.col("low").min().alias("min"),
        pl.col("high").max().alias("max"),
        pl.col("close").last().alias("last"),
    )
    print(f"  low={stats['min'][0]:.5f}  high={stats['max'][0]:.5f}  last={stats['last'][0]:.5f}")

    print("\nfirst 3 and last 3 D1 sessions:")
    for row in d1.head(3).iter_rows(named=True):
        print(f"  {row['ts_open']:%Y-%m-%d %H:%M} -> {row['ts_close']:%Y-%m-%d %H:%M}  c={row['close']:.5f}")
    print("  ...")
    for row in d1.tail(3).iter_rows(named=True):
        print(f"  {row['ts_open']:%Y-%m-%d %H:%M} -> {row['ts_close']:%Y-%m-%d %H:%M}  c={row['close']:.5f}")


if __name__ == "__main__":
    main()
