"""Central-bank policy rates from the BIS, for the carry signal.

Authoritative and free (BIS WS_CBPOL, monthly). Policy rates are a standard
proxy for the short-rate differential that drives FX carry; they are coarse
(they step a few times a year) but the carry *ranking* they imply is what the
signal uses, and that ranking is stable and well-identified.

Cached to disk; the BIS file is small and updates monthly.
"""

from __future__ import annotations

import io

import polars as pl
import requests

from fxlab.data.store import RAW_DIR

API = "https://stats.bis.org/api/v1/data/WS_CBPOL/M.{area}/all?format=csv"

# Currency -> BIS reference area. EUR is the euro area (XM).
CURRENCY_AREA = {
    "USD": "US",
    "EUR": "XM",
    "GBP": "GB",
    "JPY": "JP",
    "AUD": "AU",
    "CAD": "CA",
    "CHF": "CH",
    "NZD": "NZ",
}

_RATES_DIR = RAW_DIR / "rates"


def _fetch_area(area: str) -> str:
    path = _RATES_DIR / f"{area}.csv"
    if path.exists():
        return path.read_text(encoding="utf-8")
    resp = requests.get(
        API.format(area=area),
        timeout=90,
        headers={"User-Agent": "Mozilla/5.0 (fxlab research)"},
    )
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text, encoding="utf-8")
    return resp.text


def load_policy_rates(currencies: list[str] | None = None) -> pl.DataFrame:
    """Tidy monthly policy rates: columns (month, currency, rate_pct).

    `month` is the first day of the month as a date; `rate_pct` is the annual
    policy rate in percent.
    """
    currencies = currencies or list(CURRENCY_AREA)
    frames = []
    for currency in currencies:
        area = CURRENCY_AREA[currency]
        raw = _fetch_area(area)
        frame = (
            pl.read_csv(io.StringIO(raw))
            .select(
                pl.col("TIME_PERIOD").alias("period"),
                pl.col("OBS_VALUE").cast(pl.Float64).alias("rate_pct"),
            )
            .with_columns(
                # Monthly periods are "YYYY-MM"; pin to the first of the month.
                pl.col("period").str.strptime(pl.Date, "%Y-%m", strict=False).alias("month"),
                pl.lit(currency).alias("currency"),
            )
            .drop_nulls("month")
            .select("month", "currency", "rate_pct")
        )
        frames.append(frame)
    return pl.concat(frames).sort("currency", "month")
