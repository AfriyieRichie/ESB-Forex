"""Parquet-backed bar storage.

Layout:
    data/raw/<SYMBOL>/<YYYY>/<MM>_hour.bi5   cached Dukascopy payloads
    data/bars/<SYMBOL>_<TF>.parquet          decoded, resampled bars
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
BARS_DIR = DATA_DIR / "bars"


def bars_path(symbol: str, timeframe: str, *, bars_dir: Path | None = None) -> Path:
    return (bars_dir or BARS_DIR) / f"{symbol}_{timeframe}.parquet"


def save_bars(
    frame: pl.DataFrame, symbol: str, timeframe: str, *, bars_dir: Path | None = None
) -> Path:
    path = bars_path(symbol, timeframe, bars_dir=bars_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def load_bars(
    symbol: str, timeframe: str, *, bars_dir: Path | None = None
) -> pl.DataFrame:
    """Load bars with a positional `bar` index attached.

    `bar` is the integer index everything downstream refers to - swing pivots,
    zone creation, breaks. It is assigned after sorting so it is stable for a
    given stored file.
    """
    path = bars_path(symbol, timeframe, bars_dir=bars_dir)
    if not path.exists():
        raise FileNotFoundError(f"no bars for {symbol} {timeframe} at {path}")
    return (
        pl.read_parquet(path)
        .sort("ts_open")
        .with_row_index("bar")
        .with_columns(pl.col("bar").cast(pl.Int64))
    )


def available(*, bars_dir: Path | None = None) -> list[tuple[str, str]]:
    directory = bars_dir or BARS_DIR
    if not directory.exists():
        return []
    out = []
    for path in sorted(directory.glob("*.parquet")):
        symbol, _, timeframe = path.stem.rpartition("_")
        out.append((symbol, timeframe))
    return out
