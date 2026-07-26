"""Sweep zone parameters and report how many levels a trader would actually see.

The target is not a number of zones in the book - it is how many are visible at
a given moment. A human marking a D1 chart draws roughly 4-10 levels. A
detector showing 20 is not describing the same thing they are.
"""

from __future__ import annotations

import argparse
import random
import statistics

from fxlab.data import load_bars
from fxlab.indicators import atr
from fxlab.zones.builder import ZoneConfig, build_zones

RESEARCH_END_YEAR = 2016


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="D1")
    parser.add_argument("--windows", type=int, nargs="*", default=[2, 3, 5, 8, 12])
    parser.add_argument(
        "--stale", type=int, nargs="*", default=[0, 500, 250, 120, 60],
        help="max bars a zone survives untouched; 0 means never expire",
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    bars = load_bars(args.symbol, args.timeframe)
    research = bars.filter(bars["ts_open"].dt.year() <= RESEARCH_END_YEAR)
    atr_values = atr(research, period=20)
    median_atr = float(statistics.median(v for v in atr_values if v == v))

    rng = random.Random(args.seed)
    probes = sorted(rng.sample(range(200, len(research)), min(args.samples, len(research) - 200)))

    print(f"{args.symbol} {args.timeframe}: {len(research)} bars, median ATR {median_atr:.5f}")
    print("target: a human marks roughly 4-10 levels on a chart\n")
    print(f"{'window':>7} {'stale':>7} {'zones':>7} {'visible':>8} {'p90':>5} {'flipped%':>9}")

    for window in args.windows:
        for stale in args.stale:
            config = ZoneConfig(
                swing_window=window, max_untouched_bars=stale if stale > 0 else None
            )
            book = build_zones(research, args.symbol, args.timeframe, config)

            counts = []
            for bar in probes:
                counts.append(len(book.zones_as_of(bar)))

            visible_flipped = [
                sum(1 for v in book.zones_as_of(bar) if v.origin == "flipped")
                for bar in probes[::10]
            ]
            visible_total = [len(book.zones_as_of(bar)) for bar in probes[::10]]
            flip_share = sum(visible_flipped) / max(sum(visible_total), 1)

            counts.sort()
            marker = " <--" if 4 <= statistics.median(counts) <= 10 else ""
            print(
                f"{window:>7} {stale if stale else '-':>7} {len(book):>7} "
                f"{statistics.median(counts):>8.1f} "
                f"{counts[int(len(counts) * 0.9)]:>5} "
                f"{flip_share:>8.0%}{marker}"
            )


if __name__ == "__main__":
    main()
