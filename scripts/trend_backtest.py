"""Trend-following (time-series momentum) backtest — the pre-registered Phase 2 spec.

    uv run python scripts/trend_backtest.py                 # research 2005-2016
    uv run python scripts/trend_backtest.py --oos           # 2017-2021, one look only

Spec is fixed in TRIALS.md. No parameter sweeping here: the defaults are the
canonical TSMOM values, and picking a better-performing lookback on this data
would be the overfit the whole project exists to avoid.
"""

from __future__ import annotations

import argparse
import collections

import numpy as np

from fxlab import instruments
from fxlab.portfolio import simulate, summarize, target_weights
from fxlab.portfolio.basket import load_basket, weekly_rebalance_mask
from fxlab.portfolio.signals import GROSS_CAP

RESEARCH = (2005, 2016)
OOS = (2017, 2021)

# Pre-registered acceptance bar (see TRIALS.md).
BAR_SHARPE = 0.40
BAR_MIN_POSITIVE_PAIRS = 6
BAR_MAX_YEAR_SHARE = 0.50


def build_weights(basket) -> np.ndarray:
    n = len(basket.symbols)
    cols = [target_weights(basket.closes[:, j], n_assets=n) for j in range(n)]
    return np.column_stack(cols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--oos", action="store_true")
    args = parser.parse_args()

    symbols = args.symbols or [i.symbol for i in instruments.BASKET]
    years = OOS if args.oos else RESEARCH
    window = "OUT-OF-SAMPLE 2017-2021" if args.oos else "RESEARCH 2005-2016"

    basket = load_basket(symbols, *years)
    weights = build_weights(basket)
    mask = weekly_rebalance_mask(basket.dates)

    result = simulate(basket.returns, weights, basket.cost_fraction, mask, gross_cap=GROSS_CAP)
    stats = summarize(result.daily_returns, result.equity)

    print("=" * 84)
    print(f"TREND-FOLLOWING (TSMOM 252d)  -  {window}  -  {len(symbols)} pairs")
    if args.oos:
        print("!!! out-of-sample: this is the one pre-registered look.")
    print("=" * 84)
    print(stats.line("portfolio"))
    print(
        f"{'':16} avg gross={np.nanmean(result.gross_exposure):.2f}  "
        f"total turnover={result.turnover.sum():.0f}  hit days={stats.hit_days:.1%}"
    )

    # Per-pair standalone (each traded alone, same rules) for the breadth test.
    print("\n--- standalone per pair ---")
    positive = 0
    for j, symbol in enumerate(symbols):
        w = target_weights(basket.closes[:, j], n_assets=1).reshape(-1, 1)
        r = basket.returns[:, j].reshape(-1, 1)
        c = basket.cost_fraction[:, j].reshape(-1, 1)
        single = simulate(r, w, c, mask, gross_cap=GROSS_CAP)
        single_stats = summarize(single.daily_returns, single.equity)
        positive += single_stats.total_return > 0
        print(single_stats.line(symbol) + f"  total={single_stats.total_return:>+7.1%}")

    # P&L concentration by calendar year.
    print("\n--- P&L by year ---")
    years_arr = basket.dates[1:].astype("datetime64[Y]").astype(int) + 1970
    by_year = collections.defaultdict(float)
    for y, r in zip(years_arr, result.daily_returns):
        by_year[int(y)] += r
    total_abs = sum(abs(v) for v in by_year.values()) or 1.0
    max_share = 0.0
    for y in sorted(by_year):
        share = by_year[y] / total_abs
        max_share = max(max_share, abs(share))
        bar = "#" * int(abs(by_year[y]) * 200)
        print(f"  {y}: {by_year[y]:>+7.3f}  {bar}")

    if not args.oos:
        print("\n--- pre-registered acceptance bar ---")
        checks = [
            (f"Sharpe >= {BAR_SHARPE}", stats.sharpe >= BAR_SHARPE, f"{stats.sharpe:+.2f}"),
            (
                f">= {BAR_MIN_POSITIVE_PAIRS}/10 pairs positive",
                positive >= BAR_MIN_POSITIVE_PAIRS,
                f"{positive}/10",
            ),
            (
                f"no year > {BAR_MAX_YEAR_SHARE:.0%} of P&L",
                max_share <= BAR_MAX_YEAR_SHARE,
                f"max {max_share:.0%}",
            ),
        ]
        passed = all(ok for _, ok, _ in checks)
        for name, ok, value in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<28} ({value})")
        print(f"\n  => {'CLEARS the bar - eligible for ONE OOS look' if passed else 'FAILS - stop, do not spend OOS'}")


if __name__ == "__main__":
    main()
