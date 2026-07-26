"""Is the stop-entry rule rejecting the setups that would have won?

29% of setups never trigger. Those have no outcome under the traded rule, so
the question cannot be answered by looking at them directly.

The design: run **market-at-close entry on every setup**, then split the
results by whether the stop order would have filled. Because the same entry
rule is applied to both groups, the difference between them isolates the
selection effect rather than the price advantage of entering earlier.

  - rejected group much better  -> the stop entry is anti-selective, a real
                                   structural flaw worth fixing
  - groups similar              -> the stop entry is neutral; it costs trades
                                   but does not systematically pick losers
  - rejected group much worse   -> the stop entry is doing useful filtering

    uv run python scripts/diagnose_entry.py
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

RESEARCH_END_YEAR = 2016


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--max-rr", type=float, default=-1, help="negative removes the cap")
    args = parser.parse_args()

    symbols = args.symbols or [i.symbol for i in instruments.BASKET]
    zone_config = ZoneConfig(swing_window=3, max_untouched_bars=250)
    setup_config = SetupConfig(
        max_reward_risk=None if args.max_rr < 0 else args.max_rr
    )

    filled_group, rejected_group, stop_rule = [], [], []

    for symbol in symbols:
        inst = instruments.get(symbol)
        h4 = load_bars(symbol, "H4").filter(pl.col("ts_open").dt.year() <= RESEARCH_END_YEAR)
        d1 = load_bars(symbol, "D1").filter(pl.col("ts_open").dt.year() <= RESEARCH_END_YEAR)
        if h4.is_empty() or d1.is_empty():
            continue

        setups = detect_setups(
            symbol,
            h4,
            [
                build_zones(h4, symbol, "H4", zone_config),
                build_zones(d1, symbol, "D1", zone_config),
            ],
            query_by_book=[h4["bar"].to_numpy(), build_bridge(h4, d1)],
            config=setup_config,
        )

        by_stop = run(setups, h4, inst, BacktestConfig(entry_mode="stop"))
        by_close = run(setups, h4, inst, BacktestConfig(entry_mode="close"))
        stop_rule.extend(by_stop)

        for stop_trade, close_trade in zip(by_stop, by_close):
            if not close_trade.filled:
                continue
            (filled_group if stop_trade.filled else rejected_group).append(close_trade)

    print("=" * 92)
    print("ENTRY DIAGNOSTIC - market-at-close applied to both groups")
    print("=" * 92)
    print(summarize(stop_rule).line("traded rule (stop)"))
    print()
    print(summarize(filled_group + rejected_group).line("close entry, ALL"))
    print(summarize(filled_group).line("  stop WOULD fill"))
    print(summarize(rejected_group).line("  stop would NOT fill"))

    a, b = summarize(filled_group), summarize(rejected_group)
    if a.filled and b.filled:
        gap = b.expectancy - a.expectancy
        # Difference of two independent means.
        stderr = (a.stderr**2 + b.stderr**2) ** 0.5
        print(
            f"\ndifference (rejected - filled): {gap:+.3f}R  "
            f"95%CI=[{gap - 1.96 * stderr:+.3f},{gap + 1.96 * stderr:+.3f}]"
        )
        if abs(gap) < 1.96 * stderr:
            print("  -> not distinguishable: the stop entry is not selecting badly.")
        elif gap > 0:
            print("  -> rejected setups did better: the stop entry is anti-selective.")
        else:
            print("  -> rejected setups did worse: the stop entry filters usefully.")


if __name__ == "__main__":
    main()
