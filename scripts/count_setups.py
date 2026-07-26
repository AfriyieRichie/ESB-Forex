"""Count wammies and moolahs before building anything that trades them.

This is the cheap milestone that decides whether the strict reading of the
rules is testable at all. If the whole basket yields 60 setups over 11 years,
no backtest built on top of it can say anything, and the honest move is to
relax a rule deliberately rather than discover the problem after building an
engine there is nothing to feed.

    uv run python scripts/count_setups.py
"""

from __future__ import annotations

import argparse
import collections

import polars as pl

from fxlab import instruments
from fxlab.bridge import build_bridge
from fxlab.data import load_bars
from fxlab.setups import SetupConfig, detect_setups
from fxlab.zones.builder import ZoneConfig, build_zones

RESEARCH_END_YEAR = 2016


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--swing-window", type=int, default=3)
    parser.add_argument("--stale", type=int, default=250)
    parser.add_argument("--min-prior-touches", type=int, default=2)
    # Defaults mirror SetupConfig. Defaulting --max-rr to None here would
    # silently override the config's cap and quietly undo it.
    parser.add_argument("--min-rr", type=float, default=SetupConfig().min_reward_risk)
    parser.add_argument("--max-rr", type=float, default=SetupConfig().max_reward_risk)
    args = parser.parse_args()

    symbols = args.symbols or [i.symbol for i in instruments.BASKET]
    zone_config = ZoneConfig(
        swing_window=args.swing_window,
        max_untouched_bars=args.stale if args.stale > 0 else None,
        min_prior_touches=args.min_prior_touches,
    )
    setup_config = SetupConfig(
        min_prior_touches=args.min_prior_touches,
        min_reward_risk=args.min_rr,
        max_reward_risk=args.max_rr,
    )

    all_setups = []
    per_symbol: dict[str, int] = {}

    for symbol in symbols:
        h4 = load_bars(symbol, "H4").filter(pl.col("ts_open").dt.year() <= RESEARCH_END_YEAR)
        d1 = load_bars(symbol, "D1").filter(pl.col("ts_open").dt.year() <= RESEARCH_END_YEAR)
        if h4.is_empty() or d1.is_empty():
            print(f"{symbol}: no research-window bars")
            continue

        h4_book = build_zones(h4, symbol, "H4", zone_config)
        d1_book = build_zones(d1, symbol, "D1", zone_config)
        bridge = build_bridge(h4, d1)

        setups = detect_setups(
            symbol,
            h4,
            [h4_book, d1_book],
            query_by_book=[h4["bar"].to_numpy(), bridge],
            config=setup_config,
        )
        all_setups.extend(setups)
        per_symbol[symbol] = len(setups)
        print(
            f"{symbol}: {len(h4)} H4 bars, "
            f"{len(d1_book)} primary + {len(h4_book)} secondary zones, "
            f"{len(setups)} setups"
        )

    if not all_setups:
        print("\nno setups found")
        return

    print(f"\n{'=' * 58}\nTOTAL: {len(all_setups)} setups across {len(per_symbol)} pairs")

    by_pattern = collections.Counter(s.pattern for s in all_setups)
    by_tier = collections.Counter(s.zone_tier for s in all_setups)
    by_origin = collections.Counter(s.zone_origin for s in all_setups)
    by_year = collections.Counter(s.ts.year for s in all_setups)

    print("\nby pattern:  " + "  ".join(f"{k}={v}" for k, v in sorted(by_pattern.items())))
    print("by tier:     " + "  ".join(f"{k}={v}" for k, v in sorted(by_tier.items())))
    print("by origin:   " + "  ".join(f"{k}={v}" for k, v in sorted(by_origin.items())))

    print("\nper year:")
    for year in sorted(by_year):
        print(f"  {year}: {by_year[year]:>4}  {'#' * (by_year[year] // 4)}")

    rr = sorted(s.reward_risk for s in all_setups)
    print(
        f"\nreward:risk   min={rr[0]:.2f}  p25={rr[len(rr) // 4]:.2f}  "
        f"median={rr[len(rr) // 2]:.2f}  p75={rr[3 * len(rr) // 4]:.2f}  max={rr[-1]:.2f}"
    )

    print("\nsetups per pair per year (effective sample is smaller - these pairs correlate):")
    years = max(1, len(by_year))
    for symbol, count in sorted(per_symbol.items(), key=lambda kv: -kv[1]):
        print(f"  {symbol}: {count / years:>5.1f}")


if __name__ == "__main__":
    main()
