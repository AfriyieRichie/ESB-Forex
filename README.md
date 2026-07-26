# fxlab

Research harness for price-action trading: wammies, moolahs and kangaroo tails
at support/resistance zones. Personal use, D1 primary zones with H4 secondary,
H4 entries.

## Verdict (2026-07-25)

**Three canonical FX edges tested under strict pre-registration; none clears a
tradeable bar on this universe. Systematic research is stopped.** Full detail in
`TRIALS.md`.

| Edge | Family | Result |
|---|---|---|
| wammie/moolah | mean-reversion | null; E[R] ≈ 0, CI excludes any edge > +0.06R |
| — discretion overlay | (blind review) | selection +0.001R, p=0.997 — added nothing |
| — best note-derived filter | | overfit; reversed sign out-of-sample |
| TSMOM | trend | weak but real, Sharpe 0.24 — below 0.40 bar |
| policy-rate carry + combo | carry | dead in the zero-rate era, Sharpe 0.09; combo 0.26 |

Every hypothesis was pre-registered and died in research, so **the 2022–2026
holdout was never spent** — the discipline worked exactly as intended.

This claims only what it tested: retail, FX-only, systematic, these families,
2005–2016, after realistic costs. It does not claim the edges are fake (they are
documented over broader universes), that multi-asset trend-following would fail
(untested, a bigger project), or that discretionary trading can't work.

What survives the null: a verified data layer, point-in-time-safe zone/setup
detection, an auditable event-driven backtest **and** a vol-targeted portfolio
engine (with BIS policy-rate carry), the blind-review harness, and a full trials
log. The risk and journaling pieces have standalone value for discretionary
trading. The methodology — pre-registration, plateaus-not-peaks, a sacred
holdout — is the reusable asset.

## Status

| Component | State |
|---|---|
| Data layer (Dukascopy → Parquet) | done, verified |
| Swing detection | done, tested |
| Zone builder + lifecycle | done, tested; charts eyeball-approved |
| Render / validation harness | done |
| Timeframe bridge (H4 → knowable D1) | done, tested |
| Wammie + moolah detector | done, tested |
| Backtest engine | done, tested |
| Blind-review harness | done |
| Out-of-sample look | spent once (F2), failed |
| Holdout 2022–2026 | untouched, pristine |

## Running it

```bash
uv run python scripts/download.py                        # whole basket, 2005-present
uv run python scripts/verify_bars.py --symbol EURUSD     # session integrity checks
uv run python scripts/tune_zones.py --symbol EURUSD      # parameter sweep
uv run python scripts/render_zones.py --count 50         # validation charts
uv run pytest tests -q
```

Raw `.bi5` payloads are cached under `data/raw/`, so re-runs and interrupted
downloads are cheap.

## Data

Dukascopy, no account needed. H1 bars are the only thing downloaded; H4 and D1
are built locally so the session boundary is ours to control.

Two data facts that took empirical work to establish, both easy to get wrong:

- `.bi5` records are 24-byte big-endian `>IIIIIf`, and the field order is
  **open, close, low, high, volume** — not OHLC.
- Bars with `volume == 0` are synthetic market-closed fills where all four
  prices are equal. They are dropped at decode time; left in, they read as
  pivots and fabricate zones out of weekends.

Sessions run **17:00–17:00 New York**, which yields exactly five sessions a
week with no weekend stubs. A fixed UTC offset was tried first and produced a
Saturday stub bar every winter week, inflating 2024 from 262 sessions to 279.
`scripts/verify_bars.py` checks for exactly this class of failure.

## The lookahead discipline

Every bar carries `ts_close`, the moment it becomes knowable. Gate on that,
never on `ts_open`.

Every swing carries `pivot_bar` (where it is) and `confirmed_bar` (when it
became knowable — pivot + window). A fractal cannot be seen until `window` bars
have printed after it. Anything reading `pivot_bar` as if the swing were known
then is reading the future.

Zone bounds are **never stored as fields**. They are computed on demand from
touches confirmed at or before the query bar, because a stored bound guarantees
some caller eventually reads a zone's final, widest geometry while standing at
a bar where it was still narrow.

Two distinct notions of "touch" exist and must not be conflated:

- **Forming touch** — a confirmed swing that establishes a level. Carries
  fractal confirmation lag.
- **Trading touch** — price returning to an established zone. Knowable at that
  bar's close, no lag. Belongs to the setup detector, not to zone construction.

## Parameter registry

This is the overfitting surface. Sweeps should look for **plateaus, not peaks** —
a setting that works only at one exact value is noise.

| Parameter | Default | Role |
|---|---|---|
| `swing_window` | 3 | fractal half-width on closes |
| `atr_period` | 20 | scale for every tolerance below |
| `cluster_tolerance_atr` | 0.4 | swing→zone merge distance |
| `min_zone_width_atr` | 0.2 | floor width for thin zones |
| `max_zone_width_atr` | 1.0 | cap; prevents zone drift |
| `break_margin_atr` | 0.5 | close beyond bounds to kill a zone |
| `max_untouched_bars` | 250 | retire levels price has abandoned |
| `min_prior_touches` | 2 | reading (a): level established *before* a setup |
| `flip_enabled` | True | broken support becomes resistance |
| `flip_inherits_touches` | False | flipped levels must re-prove themselves |

### Why three of these exist

Each was added because the render harness exposed a failure, not by design:

- **`max_zone_width_atr`** — clustering matched swings against a zone's
  *bounds*, and joining widened those bounds, which widened the match region.
  Levels ratcheted outward until one spanned 200+ pips (>2 ATR).
- **`max_untouched_bars`** — with no expiry the book only grows. Zones broke,
  flipped, and the flipped remnant survived forever; 90%+ of every book was
  flip debris, and a typical chart showed 65 levels against a human's 4–10.
- **`flip_inherits_touches`** — flipped zones originally inherited their
  parent's touch count, so every break minted an instantly-"established" level.
  That silently bypassed `min_prior_touches` for ~85% of the book, meaning
  reading (a) was not actually being enforced.

## Setups

Wammies and moolahs are **one detector with a direction**, not two
implementations. Mirrored logic written twice drifts apart and gets separately
buggy, and any asymmetry in results would then be indistinguishable from a real
long/short edge. `tests/test_setups.py` reflects a price series about a fixed
axis and asserts every wammie becomes a moolah at mirrored prices.

| Parameter | Default | Role |
|---|---|---|
| `min_bars_between_touches` | 6 | rule 3 |
| `max_bars_between_touches` | 60 | without it, dips a year apart pair up |
| `min_second_touch_offset_atr` | 0.05 | "a bit higher", lower bound |
| `max_second_touch_offset_atr` | 0.75 | above this it is a trend pullback |
| `entry_buffer_atr` | 0.10 | "a few pips above" |
| `stop_buffer_atr` | 0.10 | "a few pips below" |
| `min_reward_risk` | 1.5 | target is the nearest zone paying this |
| `max_reward_risk` | None | see note below |
| `require_close_outside_zone` | True | signal candle must reject the level |

`max_reward_risk` is unset by default because refusing a 3R trade for paying
too well is unusual. Set it to 2.0 if the stated 1.5–2 band is a real ceiling
rather than a typical range — it changes which zone gets selected as target,
not just which trades are filtered.

### Observed counts (research window 2005–2016, 10 pairs)

**1,378 setups** — 671 wammies, 707 moolahs. Between 73 and 157 per year, with
no dead years. Roughly 9.6–13.3 setups per pair per year.

Composition: 479 primary / 899 secondary; 457 native / 921 flipped.

The headline number overstates the real sample. Seven of the ten pairs contain
USD, so these are nowhere near 1,378 independent observations — treat the
effective count as a few hundred when judging significance.

### The causality trap in the signal candle

Two rules here cannot be checked up front, which is why `_find_signal` is a
bar-by-bar loop rather than a set of predicates:

- **The signal is the first qualifying candle, not the last.** Whether a bar
  was the touch's final bar is only knowable once a later bar fails to touch.
- **Touch depth is the extreme *so far*, not the extreme of the whole touch.**
  A touch event's true low often lands *after* the signal candle. The first
  implementation validated "the second touch is slightly higher" against the
  full-event extreme and then emitted a signal bar that preceded it — reading a
  low that had not printed yet. Caught by
  `test_signal_never_precedes_the_second_touch`, which found a setup signalling
  at bar 108 off an extreme at bar 111.

## Backtest assumptions

Written rather than borrowed: every library hides its assumptions about fills,
and the assumptions are the entire result. These are the ones made, stated so
they can be argued with.

| Assumption | Choice | Why |
|---|---|---|
| Same bar hits stop **and** target | assume **stop** | Unknowable from OHLC. The optimistic read turns every volatile bar into a winner. |
| Stop order gaps past its price | fill at the **open** | Markets do not fill you at your price because you wanted it. |
| Costs | one full spread per round trip + slippage on stop entries and stop losses | Limit exits at target are not slipped: price trading through a resting limit fills it. |
| R denominator | **planned** risk at order placement | Position size was based on it. Re-basing R on the actual fill would hide slippage. |
| Concurrency | each setup simulated independently | Measures whether the pattern has edge. Says nothing about portfolio exposure or correlated drawdown. |

`scripts/backtest.py` prints a sensitivity row flipping the same-bar
assumption. **If that flip changes the verdict, there is no verdict** — the
result was an artifact of an unknowable ordering.

Spreads are set slightly wider than today's tightest quotes but remain
optimistic for 2005–2016, when retail spreads were materially worse. A strategy
that only works at these costs is not a strategy.

Significance is suppressed below 20 trades. A single trade has zero sample
variance, so its interval collapses to a point and would otherwise be printed
as a certainty.

## Result: the mechanical rules have no edge

Two pre-registered trials (see `TRIALS.md`), 2005–2016, 10 pairs:

| Trial | n | win | E[R] | 95% CI |
|---|---|---|---|---|
| R:R band 1.5–2.0 | 287 | 40.4% | +0.053 | [−0.102, +0.207] |
| R:R floor 1.5, uncapped | 983 | 28.5% | −0.050 | [−0.158, +0.057] |

The second is the informative one: at n=983 the interval **rules out any edge
above roughly +0.06R per trade**. Both sit within noise of break-even (40.4% vs
38.5% needed; 28.5% vs 29.9% needed). Flipping the same-bar assumption moves
expectancy by 0.003R, so this is not a modelling artifact.

This tests the *mechanical encoding* of the rules. It does not test the trader's
selectivity, which is the next experiment.

## The discretion experiment

Roughly 1,378 setups fire across the basket; a trader takes a small fraction.
That filtering is untested, and it is where any real edge plausibly lives.

A forward journal cannot answer it in reasonable time — at ~110 setups a year,
separating a 0.3R selection effect needs about 570 decisions, or five years.
So the test is **blind replay**:

```bash
uv run python scripts/blind_review.py --count 150   # build a pack
# fill in review/pack-01/decisions.csv
uv run python scripts/review_report.py --pack review/pack-01
```

Charts carry **no symbol, no dates and no price scale**, and stop at the signal
bar. Judgement has to come from the price action rather than from remembering
what a market did.

It runs on the research window, which costs no out-of-sample budget: the
comparison is taken vs skipped *within the same pool*, so the pool's own
expectancy cancels out. Significance comes from a permutation test over the
take/skip labels — exact, and free of distributional assumptions.

At 150 setups the experiment separates a difference of about 0.45R. A
selection skill worth trading should clear that; a subtler one it cannot see.

## Data splits — do not violate

| Window | Use |
|---|---|
| 2005–2016 | research; iterate freely |
| 2017–2021 | out-of-sample; a handful of looks, total |
| 2022–2026 | holdout; touched **once**, at the end |

Peeking at the holdout and continuing to research destroys the only honest
estimate this project will ever produce.

## Open questions

1. **Is 1.5–2 R:R a band or a floor?** Currently a floor. With no cap the
   observed median is ~2.3R and the tail reaches 13R, which means some targets
   are zones far away with nothing in between. Setting `max_reward_risk=2.0`
   would reject those.
2. **Zone parameters are not yet validated by eye.** `swing_window=3`,
   `max_untouched_bars=250` were chosen to land in the 4–10 visible-zone band,
   not because the levels were confirmed as the right ones.
