"""Is a filter's improvement broad, or one lucky pair/year?

A filter that lifts expectancy only because of one pair or one year has found a
coincidence, not a rule, and will not survive out-of-sample. This checks the
per-bucket delta before any OOS budget is spent.

    uv run python scripts/concentration.py --filter close_beyond_body
"""

from __future__ import annotations

import argparse
import collections

import polars as pl

from fxlab import instruments
from fxlab.backtest import BacktestConfig, run, summarize
from fxlab.bridge import build_bridge
from fxlab.data import load_bars
from fxlab.setups import SetupConfig, detect_setups
from fxlab.zones.builder import ZoneConfig, build_zones

RESEARCH = (2005, 2016)

CONFIGS = {
    "clearance": SetupConfig(require_clearance=True, clearance_bars=20),
    "close_beyond_body": SetupConfig(require_close_beyond_body=True),
    "trend_context": SetupConfig(require_trend_context=True),
}


def collect(symbols, setup_config):
    zone_config = ZoneConfig(swing_window=3, max_untouched_bars=250)
    trades = []
    for symbol in symbols:
        inst = instruments.get(symbol)
        h4 = load_bars(symbol, "H4").filter(pl.col("ts_open").dt.year().is_between(*RESEARCH))
        d1 = load_bars(symbol, "D1").filter(pl.col("ts_open").dt.year().is_between(*RESEARCH))
        if h4.is_empty() or d1.is_empty():
            continue
        setups = detect_setups(
            symbol,
            h4,
            [build_zones(h4, symbol, "H4", zone_config), build_zones(d1, symbol, "D1", zone_config)],
            query_by_book=[h4["bar"].to_numpy(), build_bridge(h4, d1)],
            config=setup_config,
        )
        trades.extend(run(setups, h4, inst, BacktestConfig()))
    return trades


def bucket_expectancy(trades, key):
    buckets = collections.defaultdict(list)
    for t in trades:
        if t.filled:
            buckets[key(t)].append(t.r_multiple)
    return {k: (len(v), sum(v) / len(v)) for k, v in buckets.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()

    symbols = args.symbols or [i.symbol for i in instruments.BASKET]

    base = collect(symbols, SetupConfig())
    filtered = collect(symbols, CONFIGS[args.filter])

    print("=" * 76)
    print(f"CONCENTRATION: {args.filter} vs baseline, research 2005-2016")
    print("=" * 76)
    print(summarize(base).line("baseline"))
    print(summarize(filtered).line(args.filter))

    for name, key in (("pair", lambda t: t.symbol), ("year", lambda t: t.ts.year)):
        print(f"\n--- by {name}: baseline E[R] -> filtered E[R] ---")
        b = bucket_expectancy(base, key)
        f = bucket_expectancy(filtered, key)
        positive = negative = 0
        for bucket in sorted(set(b) | set(f)):
            bn, be = b.get(bucket, (0, 0.0))
            fn, fe = f.get(bucket, (0, 0.0))
            delta = fe - be if bn and fn else 0.0
            if bn and fn:
                positive += delta > 0
                negative += delta < 0
            mark = "  +" if delta > 0.05 else ("  -" if delta < -0.05 else "   ")
            print(
                f"  {str(bucket):<8} n {bn:>3}->{fn:>3}   "
                f"{be:>+6.3f} -> {fe:>+6.3f}   Δ{delta:>+6.3f}{mark}"
            )
        print(f"  buckets improved: {positive}   worsened: {negative}")

    print("\nbroad improvement -> a real hypothesis worth one OOS look.")
    print("one pair/year carrying it -> a coincidence; do not spend OOS budget.")


if __name__ == "__main__":
    main()
