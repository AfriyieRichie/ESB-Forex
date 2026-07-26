"""Render a sample of charts with detected zones, for eyeball validation.

    uv run python scripts/render_zones.py --symbol EURUSD --count 50

Dates are drawn at random from the research window so the sample is not
cherry-picked. The seed is fixed, so the same sample can be re-rendered after a
parameter change to see what actually moved.

Charts stop at the as-of bar. Judge whether these are the levels you would have
drawn standing there - not whether they turned out to be useful.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from fxlab.data import load_bars
from fxlab.viz import render
from fxlab.zones.builder import ZoneConfig, build_zones

# Research window only. Later years stay unseen so they remain a real test.
RESEARCH_END_YEAR = 2016


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="D1", choices=["D1", "H4"])
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--window", type=int, default=180, help="bars of history per chart")
    parser.add_argument("--reveal", type=int, default=0, help="bars to show past the as-of bar")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("charts"))
    parser.add_argument("--min-prior-touches", type=int, default=None)
    parser.add_argument("--swing-window", type=int, default=3)
    parser.add_argument("--stale", type=int, default=250, help="0 disables expiry")
    args = parser.parse_args()

    bars = load_bars(args.symbol, args.timeframe)
    research = bars.filter(bars["ts_open"].dt.year() <= RESEARCH_END_YEAR)
    if research.is_empty():
        raise SystemExit(f"no {args.symbol} bars at or before {RESEARCH_END_YEAR}")

    config = ZoneConfig(
        swing_window=args.swing_window,
        max_untouched_bars=args.stale if args.stale > 0 else None,
    )
    book = build_zones(research, args.symbol, args.timeframe, config)
    print(f"{args.symbol} {args.timeframe}: {len(research)} bars, {len(book)} zones built")

    candidates = [
        int(b)
        for b in research["bar"].to_numpy()
        if b >= args.window  # need a full window of history behind the chart
    ]
    rng = random.Random(args.seed)
    picks = sorted(rng.sample(candidates, min(args.count, len(candidates))))

    out_dir = args.out / f"{args.symbol}_{args.timeframe}"
    for i, as_of in enumerate(picks, 1):
        path = render(
            research,
            book,
            as_of,
            out_dir / f"{i:03d}_bar{as_of}.png",
            symbol=f"{args.symbol} {args.timeframe}",
            window=args.window,
            reveal=args.reveal,
            min_prior_touches=args.min_prior_touches,
        )
        if i % 10 == 0 or i == len(picks):
            print(f"  {i}/{len(picks)} rendered -> {path.parent}")


if __name__ == "__main__":
    main()
