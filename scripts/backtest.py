"""Backtest wammies and moolahs over the research window.

    uv run python scripts/backtest.py
    uv run python scripts/backtest.py --symbols EURUSD GBPUSD

Reports expectancy in R with confidence intervals, sliced by the dimensions
worth knowing about, plus a sensitivity check on the same-bar assumption.
"""

from __future__ import annotations

import argparse
import collections

import polars as pl

from fxlab import instruments
from fxlab.backtest import BacktestConfig, run, slice_by, summarize
from fxlab.bridge import build_bridge
from fxlab.data import load_bars
from fxlab.setups import SetupConfig, detect_setups
from fxlab.zones.builder import ZoneConfig, build_zones

RESEARCH_END_YEAR = 2016


def collect(symbols, zone_config, setup_config, backtest_config):
    trades = []
    for symbol in symbols:
        inst = instruments.get(symbol)
        h4 = load_bars(symbol, "H4").filter(pl.col("ts_open").dt.year() <= RESEARCH_END_YEAR)
        d1 = load_bars(symbol, "D1").filter(pl.col("ts_open").dt.year() <= RESEARCH_END_YEAR)
        if h4.is_empty() or d1.is_empty():
            continue

        setups = detect_setups(
            symbol,
            h4,
            [build_zones(h4, symbol, "H4", zone_config), build_zones(d1, symbol, "D1", zone_config)],
            query_by_book=[h4["bar"].to_numpy(), build_bridge(h4, d1)],
            config=setup_config,
        )
        trades.extend(run(setups, h4, inst, backtest_config))
    return trades


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--swing-window", type=int, default=3)
    parser.add_argument("--stale", type=int, default=250)
    parser.add_argument("--entry-valid-bars", type=int, default=3)
    parser.add_argument("--max-holding-bars", type=int, default=120)
    parser.add_argument("--min-rr", type=float, default=SetupConfig().min_reward_risk)
    parser.add_argument(
        "--max-rr", type=float, default=SetupConfig().max_reward_risk,
        help="pass a negative value to remove the cap entirely",
    )
    parser.add_argument("--label", default="", help="recorded in TRIALS.md")
    args = parser.parse_args()

    symbols = args.symbols or [i.symbol for i in instruments.BASKET]
    zone_config = ZoneConfig(swing_window=args.swing_window, max_untouched_bars=args.stale)
    setup_config = SetupConfig(
        min_reward_risk=args.min_rr,
        max_reward_risk=None if args.max_rr is not None and args.max_rr < 0 else args.max_rr,
    )
    print(
        f"trial: {args.label or 'unlabelled'}  "
        f"min_rr={setup_config.min_reward_risk}  max_rr={setup_config.max_reward_risk}  "
        f"swing_window={args.swing_window}  stale={args.stale}"
    )
    backtest_config = BacktestConfig(
        entry_valid_bars=args.entry_valid_bars,
        max_holding_bars=args.max_holding_bars,
    )

    trades = collect(symbols, zone_config, setup_config, backtest_config)
    if not trades:
        print("no trades")
        return

    overall = summarize(trades)
    print("=" * 92)
    print(f"RESEARCH WINDOW 2005-{RESEARCH_END_YEAR}   {len(symbols)} pairs")
    print("=" * 92)
    print(f"setups={overall.setups}  filled={overall.filled} ({overall.fill_rate:.0%})")
    print(overall.line("ALL"))
    print(
        f"{'':22} avg win={overall.avg_win:>+6.3f}R  avg loss={overall.avg_loss:>+6.3f}R  "
        f"timeouts={overall.timeouts}"
    )

    outcomes = collections.Counter(t.outcome for t in trades)
    print("\noutcomes: " + "  ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))

    for attribute in ("pattern", "zone_tier", "zone_origin", "confluent", "direction"):
        print(f"\n--- by {attribute} ---")
        for key, summary in slice_by(trades, attribute).items():
            print(summary.line(key))

    print("\n--- by year ---")
    by_year: dict[int, list] = {}
    for trade in trades:
        by_year.setdefault(trade.ts.year, []).append(trade)
    for year in sorted(by_year):
        print(summarize(by_year[year]).line(str(year)))

    print("\n--- by pair ---")
    for key, summary in slice_by(trades, "symbol").items():
        print(summary.line(key))

    # If flipping the same-bar assumption changes the verdict, there is no
    # verdict - the result was an artifact of an unknowable ordering.
    print("\n--- sensitivity: same-bar stop/target assumption ---")
    optimistic = collect(
        symbols,
        zone_config,
        setup_config,
        BacktestConfig(
            entry_valid_bars=args.entry_valid_bars,
            max_holding_bars=args.max_holding_bars,
            pessimistic_same_bar=False,
        ),
    )
    print(summarize(trades).line("pessimistic"))
    print(summarize(optimistic).line("optimistic"))

    print("\n* = 95% CI excludes zero. Necessary, not sufficient:")
    print("  these pairs correlate and overlapping trades are not independent,")
    print("  so the true interval is wider than the one printed.")


if __name__ == "__main__":
    main()
