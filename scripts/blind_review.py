"""Build an anonymised pack of setups for blind take/skip judgement.

The open question is no longer whether the mechanical rules have edge - they do
not. It is whether **selectivity** does: of the ~1,378 setups the detector
fires, a trader takes a small fraction, and that filtering is the untested
variable.

A forward journal cannot answer this in reasonable time. At roughly 110 setups
a year across the basket, separating a 0.3R selection effect would need about
570 decisions, or five years. Blind replay compresses that into a few sessions.

Anonymity is the point. Charts carry no symbol, no dates and no price scale, so
judgement has to come from the price action rather than from remembering what a
market did. Each chart stops at the signal bar: no future is visible.

This runs on the **research window**, which costs no out-of-sample budget. The
strategy's aggregate expectancy there is already known to be about zero; the
question is whether a human's subset beats that, and the taken-vs-skipped
comparison lives entirely inside the sample.

Power: at 150 setups splitting roughly evenly, this separates a difference of
about 0.45R between taken and skipped. A selection skill worth trading should
clear that comfortably; a subtler one this cannot see.

    uv run python scripts/blind_review.py --count 150
    # ... fill in decisions.csv ...
    uv run python scripts/review_report.py --pack review/pack-01
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import polars as pl

from fxlab import instruments
from fxlab.backtest import BacktestConfig, run
from fxlab.bridge import build_bridge
from fxlab.data import load_bars
from fxlab.setups import SetupConfig, detect_setups
from fxlab.viz import render
from fxlab.zones.builder import ZoneConfig, build_zones

RESEARCH_END_YEAR = 2016


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--window", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("review/pack-01"))
    parser.add_argument("--max-rr", type=float, default=-1, help="negative removes the cap")
    args = parser.parse_args()

    symbols = args.symbols or [i.symbol for i in instruments.BASKET]
    zone_config = ZoneConfig(swing_window=3, max_untouched_bars=250)
    setup_config = SetupConfig(max_reward_risk=None if args.max_rr < 0 else args.max_rr)

    pool = []
    for symbol in symbols:
        inst = instruments.get(symbol)
        h4 = load_bars(symbol, "H4").filter(pl.col("ts_open").dt.year() <= RESEARCH_END_YEAR)
        d1 = load_bars(symbol, "D1").filter(pl.col("ts_open").dt.year() <= RESEARCH_END_YEAR)
        if h4.is_empty() or d1.is_empty():
            continue

        h4_book = build_zones(h4, symbol, "H4", zone_config)
        d1_book = build_zones(d1, symbol, "D1", zone_config)
        setups = detect_setups(
            symbol,
            h4,
            [h4_book, d1_book],
            query_by_book=[h4["bar"].to_numpy(), build_bridge(h4, d1)],
            config=setup_config,
        )
        trades = run(setups, h4, inst, BacktestConfig())

        for setup, trade in zip(setups, trades):
            if setup.signal_bar < args.window:
                continue  # not enough history behind it to draw a fair chart
            pool.append((setup, trade, h4, h4_book))
        print(f"{symbol}: {len(setups)} setups", flush=True)

    if not pool:
        raise SystemExit("no setups available")

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.count, len(pool)))
    rng.shuffle(sample)

    charts = args.out / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    key_rows, decision_rows = [], []
    for i, (setup, trade, bars, book) in enumerate(sample, 1):
        review_id = f"{i:03d}"
        render(
            bars,
            book,
            setup.signal_bar,
            charts / f"{review_id}.png",
            window=args.window,
            levels={"entry": setup.entry, "stop": setup.stop, "target": setup.target},
            title=(
                f"Setup {review_id}     {setup.direction.upper()}     "
                f"planned {setup.reward_risk:.1f}R"
            ),
            hide_axes=True,
        )
        key_rows.append(
            {
                "review_id": review_id,
                "setup_id": setup.setup_id,
                "symbol": setup.symbol,
                "pattern": setup.pattern,
                "ts": setup.ts.isoformat(),
                "zone_tier": setup.zone_tier,
                "zone_origin": setup.zone_origin,
                "planned_rr": setup.reward_risk,
                "outcome": trade.outcome,
                "r_multiple": trade.r_multiple,
            }
        )
        decision_rows.append(review_id)
        if i % 25 == 0:
            print(f"  rendered {i}/{len(sample)}", flush=True)

    with (args.out / "decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["review_id", "decision", "confidence", "note"])
        for review_id in decision_rows:
            writer.writerow([review_id, "", "", ""])

    key_path = args.out / "ANSWER_KEY_do_not_open.jsonl"
    with key_path.open("w", encoding="utf-8") as handle:
        for row in key_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"\npack written to {args.out}")
    print(f"  {len(sample)} charts in {charts}")
    print("  fill in decisions.csv: decision = take | skip, confidence 1-5, note optional")
    print(f"  {key_path.name} holds the outcomes - opening it invalidates the experiment")


if __name__ == "__main__":
    main()
