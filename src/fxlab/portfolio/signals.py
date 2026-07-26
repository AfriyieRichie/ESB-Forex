"""Time-series momentum signal and volatility-target weights.

All point-in-time: every value at row t uses only closes up to and including t.
The engine then applies the weight from t to the return from t to t+1, so the
lookahead discipline lives at the boundary between this module and the engine.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252
MOMENTUM_LOOKBACK = 252  # ~12 months, the canonical TSMOM horizon
VOL_LOOKBACK = 60
VOL_TARGET = 0.10  # annualised portfolio vol
GROSS_CAP = 3.0


def log_returns(closes: np.ndarray) -> np.ndarray:
    """Row t = log(close_t / close_{t-1}); row 0 is NaN. Shape preserved."""
    out = np.full_like(closes, np.nan, dtype=float)
    out[1:] = np.log(closes[1:] / closes[:-1])
    return out


def momentum_sign(closes: np.ndarray, lookback: int = MOMENTUM_LOOKBACK) -> np.ndarray:
    """+1 / -1 from the trailing `lookback`-day return; NaN during warmup.

    Knowable at t: compares close_t to close_{t-lookback}, both in the past.
    """
    out = np.full_like(closes, np.nan, dtype=float)
    out[lookback:] = np.sign(closes[lookback:] / closes[:-lookback] - 1.0)
    # A dead-flat trailing window gives sign 0 = flat, which is correct.
    return out


def realized_vol(
    closes: np.ndarray, lookback: int = VOL_LOOKBACK, trading_days: int = TRADING_DAYS
) -> np.ndarray:
    """Annualised stdev of daily log returns over the trailing window.

    Uses returns strictly up to t. NaN until a full window exists.
    """
    r = log_returns(closes)
    out = np.full_like(closes, np.nan, dtype=float)
    for t in range(lookback, len(closes)):
        window = r[t - lookback + 1 : t + 1]
        if not np.isnan(window).any():
            out[t] = window.std(ddof=1) * np.sqrt(trading_days)
    return out


def scale_by_vol(
    sign: np.ndarray,
    closes: np.ndarray,
    *,
    n_assets: int,
    vol_lookback: int = VOL_LOOKBACK,
    vol_target: float = VOL_TARGET,
    gross_cap: float = GROSS_CAP,
) -> np.ndarray:
    """Turn a signed signal into vol-scaled weights for one asset.

    w_t = sign_t * (vol_target / n_assets) / vol_t, clipped so a single asset
    cannot demand the whole gross budget on a collapsed vol estimate. NaN
    wherever sign or vol is undefined -> the engine reads it as flat.

    Shared by momentum and carry so both sleeves are sized identically; only
    the sign differs between them.
    """
    vol = realized_vol(closes, vol_lookback)
    per_asset_budget = vol_target / n_assets
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = sign * per_asset_budget / vol

    cap = gross_cap / n_assets
    weights = np.clip(raw, -cap, cap)
    weights[np.isnan(sign) | np.isnan(vol)] = np.nan
    return weights


def target_weights(
    closes: np.ndarray,
    *,
    n_assets: int,
    momentum_lookback: int = MOMENTUM_LOOKBACK,
    vol_lookback: int = VOL_LOOKBACK,
    vol_target: float = VOL_TARGET,
    gross_cap: float = GROSS_CAP,
) -> np.ndarray:
    """Vol-scaled time-series-momentum weights for one asset."""
    sign = momentum_sign(closes, momentum_lookback)
    return scale_by_vol(
        sign,
        closes,
        n_assets=n_assets,
        vol_lookback=vol_lookback,
        vol_target=vol_target,
        gross_cap=gross_cap,
    )
