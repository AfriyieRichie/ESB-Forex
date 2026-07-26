"""Setup scanner — runs the wammie/moolah detector over the basket and caches
the results (plus the bars/zones needed to draw each one).

The detector IS the checklist encoded: an established zone (>=2 prior touches),
two touches with the second shallower, >=6 bars apart, a reversal candle closing
back out of the zone, and a zone-based entry/stop/target at 1.5-2R. Every signal
it returns has already satisfied those rules.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import polars as pl

from fxlab import instruments
from fxlab.bridge import build_bridge
from fxlab.data import load_bars
from fxlab.setups import SetupConfig, detect_setups
from fxlab.zones.builder import ZoneConfig, build_zones

LOOKBACK_YEARS = 2  # enough for active zones; keeps the scan fast

# Rules the detector GUARANTEES (shown pre-verified) vs the discretionary reads
# it can't judge (the human confirms these before taking the trade).
VERIFIED_RULES = {
    "wammie": [
        "Support zone already established (>=2 prior touches)",
        "Price touched the zone twice",
        "Second touch is shallower (slightly higher) than the first",
        "At least 6 candles between the touches",
        "Bullish reversal candle closed back above the zone",
        "Entry above the trigger, stop below the first touch, target the next zone (>=1.5R)",
    ],
    "moolah": [
        "Resistance zone already established (>=2 prior touches)",
        "Price touched the zone twice",
        "Second touch is shallower (slightly lower) than the first",
        "At least 6 candles between the touches",
        "Bearish reversal candle closed back below the zone",
        "Entry below the trigger, stop above the first touch, target the next zone (>=1.5R)",
    ],
}
DISCRETIONARY_RULES = [
    "Room to the left (the level was abandoned, not grinding)",
    "Not just ranging into the level",
    "Reversal candle body is convincing (closed well past the prior body)",
    "This fits my read of the higher-timeframe context",
]

_cache: dict = {"signals": [], "by_id": {}, "pairs": {}, "scanned_at": None,
                "latest": None, "data_through": None}


def _sid(s) -> str:
    return hashlib.md5(f"{s.symbol}{s.pattern}{s.zone_id}{s.signal_bar}".encode()).hexdigest()[:10]


def scan(recent_days: int = 45, symbols=None) -> dict:
    symbols = symbols or [i.symbol for i in instruments.BASKET]
    zc = ZoneConfig(swing_window=3, max_untouched_bars=250)
    sc = SetupConfig()
    cutoff = dt.date.today().year - LOOKBACK_YEARS
    by_id, pairs, found = {}, {}, []
    latest = None        # newest setup
    data_through = None  # newest bar in the data (true freshness)

    for sym in symbols:
        h4 = load_bars(sym, "H4").filter(pl.col("ts_open").dt.year() >= cutoff)
        d1 = load_bars(sym, "D1").filter(pl.col("ts_open").dt.year() >= cutoff)
        if h4.is_empty() or d1.is_empty():
            continue
        bar_max = h4["ts_open"].max()
        data_through = bar_max if data_through is None or bar_max > data_through else data_through
        h4b = build_zones(h4, sym, "H4", zc)
        d1b = build_zones(d1, sym, "D1", zc)
        setups = detect_setups(
            sym, h4, [h4b, d1b],
            query_by_book=[h4["bar"].to_numpy(), build_bridge(h4, d1)], config=sc,
        )
        pairs[sym] = {"h4": h4, "books": [h4b, d1b]}
        for s in setups:
            latest = s.ts if latest is None or s.ts > latest else latest
            by_id[_sid(s)] = (s, sym)
            found.append(s)

    if latest is not None:
        cut = latest - dt.timedelta(days=recent_days)
        found = [s for s in found if s.ts >= cut]
    found.sort(key=lambda s: s.ts, reverse=True)

    _cache.update(signals=found, by_id=by_id, pairs=pairs,
                  scanned_at=dt.datetime.now(dt.timezone.utc), latest=latest,
                  data_through=data_through)
    return _cache


def signal_dicts() -> list[dict]:
    out = []
    for s in _cache["signals"]:
        inst = instruments.get(s.symbol)
        dec = 3 if inst.is_jpy else 5
        out.append({
            "id": _sid(s), "ts": s.ts.isoformat(), "symbol": s.symbol, "setup": s.pattern,
            "direction": s.direction, "zone_tier": s.zone_tier, "confluent": s.confluent,
            "entry": round(s.entry, dec), "stop": round(s.stop, dec), "target": round(s.target, dec),
            "rr": round(s.reward_risk, 2), "risk_pips": round(abs(s.entry - s.stop) / inst.pip, 1),
            "verified": VERIFIED_RULES.get(s.pattern, []), "discretionary": DISCRETIONARY_RULES,
        })
    return out


def render_chart(sid: str, path) -> None:
    from fxlab.viz import render

    s, sym = _cache["by_id"][sid]
    p = _cache["pairs"][sym]
    touch_col = "#2a78d6" if s.direction == "long" else "#eb6834"
    render(
        p["h4"], p["books"][0], s.signal_bar, path,
        symbol=f"{sym} H4", window=150,
        levels={"entry": s.entry, "stop": s.stop, "target": s.target},
        markers=[
            {"bar": s.first_touch_bar, "label": "touch 1", "color": touch_col},
            {"bar": s.second_touch_bar, "label": "touch 2", "color": touch_col},
            {"bar": s.signal_bar, "label": "trigger", "color": "#0b0b0b"},
        ],
        title=(f"{sym}  ·  {s.pattern.upper()} ({s.direction})  ·  "
               f"{s.zone_tier}{' +confluence' if s.confluent else ''}  ·  {s.reward_risk:.2f}R"),
    )


def cache() -> dict:
    return _cache
