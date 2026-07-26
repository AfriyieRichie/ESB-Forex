"""Carry and trend+carry combination backtest — pre-registered Phase 2b spec.

    uv run python scripts/carry_backtest.py            # research 2005-2016
    uv run python scripts/carry_backtest.py --oos      # 2017-2021, one look, combo only

Carry P&L includes daily interest accrual, not just spot. The combination is
the pre-registered hypothesis; carry standalone is diagnostic. Blend is fixed
equal-weight of two vol-targeted sleeves, never tuned.
"""

from __future__ import annotations

import argparse
import collections

import numpy as np

from fxlab import instruments
from fxlab.portfolio import simulate, summarize, target_weights
from fxlab.portfolio.basket import load_basket, weekly_rebalance_mask
from fxlab.portfolio.carry import build_carry, total_returns
from fxlab.portfolio.signals import GROSS_CAP, scale_by_vol

RESEARCH = (2005, 2016)
OOS = (2017, 2021)

BAR_SHARPE = 0.40
BAR_MIN_POSITIVE_YEARS = 8
BAR_MAX_YEAR_SHARE = 0.50


def nan_to_zero(a):
    return np.where(np.isnan(a), 0.0, a)


def run_strategy(name, returns, weights, basket, mask):
    result = simulate(returns, weights, basket.cost_fraction, mask, gross_cap=GROSS_CAP)
    return name, result, summarize(result.daily_returns, result.equity)


def year_breakdown(dates, daily_returns):
    years = dates[1:].astype("datetime64[Y]").astype(int) + 1970
    by_year = collections.defaultdict(float)
    for y, r in zip(years, daily_returns):
        by_year[int(y)] += r
    return by_year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--oos", action="store_true")
    args = parser.parse_args()

    symbols = args.symbols or [i.symbol for i in instruments.BASKET]
    years = OOS if args.oos else RESEARCH
    window = "OUT-OF-SAMPLE 2017-2021" if args.oos else "RESEARCH 2005-2016"
    n = len(symbols)

    basket = load_basket(symbols, *years)
    mask = weekly_rebalance_mask(basket.dates)

    carry_sign, carry_daily = build_carry(basket)
    returns = total_returns(basket, carry_daily)  # spot + accrual

    trend_w = np.column_stack([target_weights(basket.closes[:, j], n_assets=n) for j in range(n)])
    carry_w = np.column_stack(
        [scale_by_vol(carry_sign[:, j], basket.closes[:, j], n_assets=n) for j in range(n)]
    )
    combo_w = 0.5 * nan_to_zero(trend_w) + 0.5 * nan_to_zero(carry_w)

    print("=" * 84)
    print(f"CARRY & COMBINATION  -  {window}  -  {n} pairs")
    if args.oos:
        print("!!! out-of-sample: the one pre-registered look, combination only.")
    print("=" * 84)

    runs = [
        run_strategy("carry", returns, carry_w, basket, mask),
        run_strategy("trend(+accrual)", returns, trend_w, basket, mask),
        run_strategy("COMBINATION", returns, combo_w, basket, mask),
    ]
    for name, _result, stats in runs:
        print(stats.line(name))

    combo_result = runs[-1][1]
    combo_stats = runs[-1][2]

    by_year = year_breakdown(basket.dates, combo_result.daily_returns)
    total_abs = sum(abs(v) for v in by_year.values()) or 1.0
    positive_years = sum(1 for v in by_year.values() if v > 0)
    max_share = max(abs(v) / total_abs for v in by_year.values())

    print("\n--- COMBINATION P&L by year ---")
    for y in sorted(by_year):
        bar = "#" * int(abs(by_year[y]) * 200)
        print(f"  {y}: {by_year[y]:>+7.3f}  {bar}")

    if not args.oos:
        print("\n--- pre-registered bar (COMBINATION) ---")
        checks = [
            (f"Sharpe >= {BAR_SHARPE}", combo_stats.sharpe >= BAR_SHARPE, f"{combo_stats.sharpe:+.2f}"),
            (
                f">= {BAR_MIN_POSITIVE_YEARS}/12 years positive",
                positive_years >= BAR_MIN_POSITIVE_YEARS,
                f"{positive_years}/{len(by_year)}",
            ),
            (
                f"no year > {BAR_MAX_YEAR_SHARE:.0%} of P&L",
                max_share <= BAR_MAX_YEAR_SHARE,
                f"max {max_share:.0%}",
            ),
        ]
        passed = all(ok for _, ok, _ in checks)
        for label, ok, value in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:<30} ({value})")
        print(
            f"\n  => {'CLEARS the bar - eligible for ONE OOS look' if passed else 'FAILS - stop, do not spend OOS'}"
        )


if __name__ == "__main__":
    main()
