"""Download H1 history and build H4/D1 bars.

    uv run python scripts/download.py --symbols EURUSD --start 2024-01 --end 2024-12
    uv run python scripts/download.py                     # whole basket, 2005-present

Raw payloads are cached, so re-runs and interrupted downloads are cheap.
"""

from __future__ import annotations

import argparse
import datetime as dt

import polars as pl

from fxlab import instruments
from fxlab.data import Client, download_h1, resample, save_bars
from fxlab.data.store import RAW_DIR


def month_arg(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m").date().replace(day=1)


def main() -> None:
    today = dt.date.today()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None, help="default: whole basket")
    parser.add_argument("--start", type=month_arg, default=dt.date(2005, 1, 1))
    parser.add_argument("--end", type=month_arg, default=today.replace(day=1))
    parser.add_argument("--min-interval", type=float, default=0.4, help="seconds between requests")
    args = parser.parse_args()

    symbols = args.symbols or [i.symbol for i in instruments.BASKET]
    client = Client(RAW_DIR, min_interval=args.min_interval)

    for symbol in symbols:
        inst = instruments.get(symbol)
        print(f"{symbol}: {args.start:%Y-%m} -> {args.end:%Y-%m}", flush=True)
        h1 = download_h1(client, inst, args.start, args.end)
        if h1.is_empty():
            print(f"  {symbol}: no data", flush=True)
            continue

        save_bars(h1, symbol, "H1")
        for timeframe in ("H4", "D1"):
            bars = resample(h1, timeframe)
            save_bars(bars, symbol, timeframe)
            first = bars.select(pl.col("ts_open").min()).item()
            last = bars.select(pl.col("ts_open").max()).item()
            print(
                f"  {symbol} {timeframe}: {len(bars):>7} bars  "
                f"{first:%Y-%m-%d} -> {last:%Y-%m-%d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
