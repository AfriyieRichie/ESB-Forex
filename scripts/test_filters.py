"""Test the note-derived filters, one at a time, against the baseline.

Each filter came out of the blind-review notes, so each is DATA-SUGGESTED: a
positive result here is a hypothesis, not a finding, and means nothing until it
survives the out-of-sample window. This script runs research only.

A filter earns an out-of-sample look only if it does BOTH:
  - lifts expectancy by a margin worth the trades it discards, and
  - the lift is not concentrated in one pair or one year.

    uv run python scripts/test_filters.py                 # research window
    uv run python scripts/test_filters.py --oos           # 2017-2021, once, only for a survivor
"""

from __future__ import annotations

import argparse

import polars as pl

from fxlab import instruments
from fxlab.backtest import BacktestConfig, run, summarize
from fxlab.bridge import build_bridge
from fxlab.data import load_bars
from fxlab.setups import SetupConfig, detect_setups
from fxlab.zones.builder import ZoneConfig, build_zones

RESEARCH = (2005, 2016)
OOS = (2017, 2021)

FILTERS = {
    "baseline": SetupConfig(),
    "clearance": SetupConfig(require_clearance=True, clearance_bars=20),
    "close_beyond_body": SetupConfig(require_close_beyond_body=True),
    "trend_context": SetupConfig(require_trend_context=True),
    "all_three": SetupConfig(
        require_clearance=True,
        require_close_beyond_body=True,
        require_trend_context=True,
    ),
}


def prepare(symbols, years):
    """Build the per-pair inputs once. Zones and the bridge do not depend on
    SetupConfig, so they are shared across every filter rather than rebuilt -
    the zone build is the expensive step and rebuilding it per filter made this
    six times slower than it needs to be."""
    lo, hi = years
    zone_config = ZoneConfig(swing_window=3, max_untouched_bars=250)
    prepared = []
    for symbol in symbols:
        inst = instruments.get(symbol)
        h4 = load_bars(symbol, "H4").filter(pl.col("ts_open").dt.year().is_between(lo, hi))
        d1 = load_bars(symbol, "D1").filter(pl.col("ts_open").dt.year().is_between(lo, hi))
        if h4.is_empty() or d1.is_empty():
            continue
        books = [
            build_zones(h4, symbol, "H4", zone_config),
            build_zones(d1, symbol, "D1", zone_config),
        ]
        query = [h4["bar"].to_numpy(), build_bridge(h4, d1)]
        prepared.append((symbol, inst, h4, books, query))
        print(f"  prepared {symbol}", flush=True)
    return prepared


def backtest(prepared, setup_config):
    trades = []
    for symbol, inst, h4, books, query in prepared:
        setups = detect_setups(symbol, h4, books, query_by_book=query, config=setup_config)
        trades.extend(run(setups, h4, inst, BacktestConfig()))
    return trades


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--oos", action="store_true", help="run 2017-2021 (use sparingly)")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to named filters")
    args = parser.parse_args()

    symbols = args.symbols or [i.symbol for i in instruments.BASKET]
    years = OOS if args.oos else RESEARCH
    window = "OUT-OF-SAMPLE 2017-2021" if args.oos else "RESEARCH 2005-2016"

    filters = FILTERS if args.only is None else {k: FILTERS[k] for k in args.only}

    print("=" * 92)
    print(f"FILTER COMPARISON  -  {window}  -  {len(symbols)} pairs")
    if args.oos:
        print("!!! out-of-sample: every run here spends budget. Only survivors belong here.")
    print("=" * 92)

    print("preparing zones (once)...", flush=True)
    prepared = prepare(symbols, years)

    base = summarize(backtest(prepared, FILTERS["baseline"]))
    for name, config in filters.items():
        summary = summarize(backtest(prepared, config))
        delta = summary.expectancy - base.expectancy
        tag = "" if name == "baseline" else f"   d(E[R])={delta:+.3f}"
        print(summary.line(name) + tag)

    print("\nreading: a filter is only interesting if it lifts E[R] materially AND keeps")
    print("enough trades to matter. A tiny sample with a pretty number is noise.")


if __name__ == "__main__":
    main()
