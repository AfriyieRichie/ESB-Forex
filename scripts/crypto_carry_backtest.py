"""Liquid crypto funding-carry backtest — the honest 'shoestring test'.

Tests the ONE thing that would have to be true for a small carry sleeve to be
worth running (the red-team's condition): on the liquid names (BTC/ETH), is the
net-of-cost funding-carry return positive AND materially above T-bills, and does
a funding-regime filter (only harvest when funding is meaningfully positive)
help?

Model (delta-neutral long-spot / short-perp): spot and perp price moves cancel,
so the carry return per funding interval is the funding rate received, minus
costs. This is the dominant, honest term; we don't pretend to model the
counterparty tail here (that's a separate risk panel — and the reason any live
sizing must be tiny).

    uv run python scripts/crypto_carry_backtest.py

Data: Binance USDⓈ-M funding history via CCXT, back to ~2019. BTC/ETH are
survivors, so this liquid base is survivorship-clean.
"""

from __future__ import annotations

import time

import numpy as np

COINS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
TBILL_ANN = 0.04           # ~risk-free hurdle (3m T-bill ~3.9-4%)
ROUND_TRIP_COST = 0.0030   # 30 bps fee+slippage per regime switch (per coin)
REGIME_LOOKBACK = 21       # funding intervals (~7 days at 8h) for the regime read
HOLDOUT_START = "2025-01-01"


def fetch_funding(exchange, symbol: str) -> list[tuple[int, float]]:
    """All funding (timestamp_ms, rate) for a symbol, paginated from 2019."""
    out: dict[int, float] = {}
    since = exchange.parse8601("2019-09-01T00:00:00Z")
    while True:
        for _ in range(6):  # ride out DNS blips
            try:
                batch = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  {symbol}: retry ({type(e).__name__})", flush=True)
                time.sleep(5)
        else:
            break
        if not batch:
            break
        for row in batch:
            out[row["timestamp"]] = float(row["fundingRate"])
        nxt = batch[-1]["timestamp"] + 1
        if nxt <= since or len(batch) < 1000:
            since = nxt
            if len(batch) < 1000:
                break
        else:
            since = nxt
        time.sleep(exchange.rateLimit / 1000)
    return sorted(out.items())


def ann_stats(rets: np.ndarray, per_year: float) -> dict:
    if len(rets) == 0:
        return dict(ann=0, vol=0, sharpe=0, maxdd=0)
    mean, sd = rets.mean(), rets.std(ddof=1) if len(rets) > 1 else 0.0
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    maxdd = float((equity / peak - 1).min())
    ann = mean * per_year
    vol = sd * np.sqrt(per_year)
    return dict(ann=ann, vol=vol, sharpe=ann / vol if vol else 0.0, maxdd=maxdd)


def main() -> None:
    import ccxt

    ex = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 20000})
    print("fetching funding history (BTC, ETH)...", flush=True)
    series = {}
    for c in COINS:
        f = fetch_funding(ex, c)
        series[c] = dict(f)
        first = time.strftime("%Y-%m-%d", time.gmtime(f[0][0] / 1000))
        last = time.strftime("%Y-%m-%d", time.gmtime(f[-1][0] / 1000))
        print(f"  {c}: {len(f)} intervals, {first} -> {last}", flush=True)

    # Common timeline (BTC/ETH settle on the same 8h grid).
    ts = sorted(set(series[COINS[0]]) & set(series[COINS[1]]))
    if not ts:
        raise SystemExit("no overlapping funding timestamps")
    per_year = len(ts) / ((ts[-1] - ts[0]) / (365.25 * 864e5))
    holdout_ms = ex.parse8601(HOLDOUT_START + "T00:00:00Z")
    cash_per = TBILL_ANN / per_year

    fund = {c: np.array([series[c][t] for t in ts]) for c in COINS}
    ts_arr = np.array(ts)

    def run(regime_filter: bool):
        """Return per-interval net return array + %-deployed."""
        rets = np.zeros(len(ts))
        deployed_flags = []
        for c in COINS:
            f = fund[c]
            # trailing mean funding known BEFORE interval i (point-in-time)
            trail = np.full(len(f), np.nan)
            for i in range(REGIME_LOOKBACK, len(f)):
                trail[i] = f[i - REGIME_LOOKBACK:i].mean()
            deploy = np.ones(len(f), bool)
            if regime_filter:
                # harvest only when trailing funding clears break-even (cash rate)
                deploy = trail > cash_per
                deploy[:REGIME_LOOKBACK] = False
            # per-interval return for this coin's sleeve (half the book each)
            coin_ret = np.where(deploy, f, cash_per)
            # switching cost when deploy state flips
            switches = np.abs(np.diff(deploy.astype(int), prepend=0))
            coin_ret = coin_ret - switches * ROUND_TRIP_COST
            rets += 0.5 * coin_ret
            deployed_flags.append(deploy)
        pct_deployed = float(np.mean([d.mean() for d in deployed_flags]))
        return rets, pct_deployed

    print(f"\n{'=' * 76}")
    print(f"LIQUID CRYPTO CARRY (BTC/ETH, delta-neutral)  |  {len(ts)} intervals, "
          f"~{per_year:.0f}/yr\n{'=' * 76}")
    print(f"hurdle: T-bill {TBILL_ANN:.1%}   |   switch cost {ROUND_TRIP_COST*1e4:.0f} bps   "
          f"|   regime lookback ~{REGIME_LOOKBACK*8/24:.0f}d")

    hold = ts_arr >= holdout_ms
    for label, rf in [("Layer 0: always-on carry", False), ("Layer 1: + funding-regime filter", True)]:
        rets, dep = run(rf)
        full = ann_stats(rets, per_year)
        ho = ann_stats(rets[hold], per_year)
        print(f"\n--- {label} ---")
        print(f"  full sample : ann {full['ann']:+.1%}  vol {full['vol']:.1%}  Sharpe {full['sharpe']:+.2f}  maxDD {full['maxdd']:.1%}  deployed {dep:.0%}")
        print(f"  2025->now   : ann {ho['ann']:+.1%}  vol {ho['vol']:.1%}  Sharpe {ho['sharpe']:+.2f}  maxDD {ho['maxdd']:.1%}")
        edge = full["ann"] - TBILL_ANN
        print(f"  vs T-bill (full): {edge:+.1%} {'ABOVE' if edge > 0 else 'BELOW'} cash")

    # --- usable monitor: deploy-or-wait right now ---
    deploy_thresh = TBILL_ANN + 0.05  # red-team hurdle: cash + 5% to pay for the tail
    print(f"\n--- CURRENT SIGNAL (deploy tiny only if trailing funding > {deploy_thresh:.0%} ann) ---")
    for c in COINS:
        f = fund[c]
        trail_ann = f[-REGIME_LOOKBACK:].mean() * per_year
        sig = "DEPLOY (tiny)" if trail_ann > deploy_thresh else "WAIT — funding too thin for the tail"
        print(f"  {c}: trailing ~7d funding {trail_ann:+.1%} ann  ->  {sig}")

    print(f"\n{'=' * 76}\nHONEST READ")
    print("=" * 76)
    print("The go/no-go isn't Sharpe — it's whether net carry clears T-bills by enough")
    print("to justify the exchange tail (FTX / ADL / depeg) this backtest CANNOT show.")
    print("Red-team hurdle: need ~T-bill + 5% net to compensate that tail. Judge below.")


if __name__ == "__main__":
    main()
