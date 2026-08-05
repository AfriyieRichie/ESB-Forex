"""Dukascopy datafeed client.

Format, established empirically (see scratch/probe_format.py):

  - LZMA-alone compressed, 24-byte big-endian records: >IIIIIf
  - Fields are (time_offset_seconds, open, close, low, high, volume)
    -- note the O/C/L/H ordering, which is not OHLC
  - Integer prices divide by the instrument's price_scale
  - Hourly bars ship as one file per month, offsets from the month start
  - Month in the URL path is ZERO-INDEXED: January is 00

Bars with volume == 0 are synthetic weekend/holiday fills where OHLC are all
equal. They are dropped: left in, they produce flat runs that the swing
detector reads as pivots, which would fabricate zones out of market closures.
"""

from __future__ import annotations

import calendar
import datetime as dt
import lzma
import struct
import time
from pathlib import Path

import polars as pl
import requests

from fxlab.instruments import Instrument

BASE = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">IIIIIf")
RECORD_SIZE = 24

# Tick records: 20 bytes big-endian, (ms_from_hour, ask, bid, ask_vol, bid_vol).
# Tick files publish in near-real-time, unlike the monthly candle file which the
# current month lacks until it is aggregated. They are the live-edge source.
TICK = struct.Struct(">IIIff")
TICK_SIZE = 20

# Dukascopy throttles hard and answers with a small JSON error body rather than
# a distinct status in some cases, so both are treated as "back off".
_THROTTLE_BODY = b'{"error"'


class DukascopyError(RuntimeError):
    pass


class Client:
    """Polite, caching client. Raw .bi5 files are cached on disk forever."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        min_interval: float = 0.4,
        max_attempts: int = 6,
        initial_backoff: float = 15.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.initial_backoff = initial_backoff
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (fxlab research)"})
        self._last_request = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str) -> bytes | None:
        """Returns payload, or None when the file legitimately does not exist.

        Retries transient failures with exponential backoff — including connection
        timeouts and dropped connections, which Dukascopy throws under load. A short
        connect timeout fails fast so a dropped connection retries quickly rather
        than hanging for the full read timeout.
        """
        backoff = self.initial_backoff
        for _ in range(self.max_attempts):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=(10, 60))
            except requests.exceptions.RequestException:
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code == 200 and not resp.content.startswith(_THROTTLE_BODY):
                return resp.content
            time.sleep(backoff)
            backoff *= 2
        raise DukascopyError(f"gave up after {self.max_attempts} attempts: {url}")

    def _cache_path(self, symbol: str, year: int, month: int) -> Path:
        return self.cache_dir / symbol / str(year) / f"{month:02d}_hour.bi5"

    def fetch_h1_month(self, inst: Instrument, year: int, month: int) -> bytes | None:
        """Raw payload for one month of H1 bars. `month` is 1-indexed here."""
        path = self._cache_path(inst.symbol, year, month)
        if path.exists():
            return path.read_bytes() or None

        url = f"{BASE}/{inst.symbol}/{year}/{month - 1:02d}/BID_candles_hour_1.bi5"
        payload = self._get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        # An empty file records "checked, nothing there" so reruns skip it.
        path.write_bytes(payload or b"")
        return payload

    def _tick_cache_path(self, symbol: str, year: int, month: int, day: int, hour: int) -> Path:
        return (
            self.cache_dir / symbol / str(year) / f"{month:02d}" / f"{day:02d}" / f"{hour:02d}h_ticks.bi5"
        )

    def fetch_hour_ticks(
        self, inst: Instrument, year: int, month: int, day: int, hour: int
    ) -> bytes | None:
        """Raw tick payload for one hour. A completed hour is immutable, so the
        on-disk cache is safe; callers just avoid fetching the in-progress hour."""
        path = self._tick_cache_path(inst.symbol, year, month, day, hour)
        if path.exists():
            return path.read_bytes() or None
        # Month in the path is ZERO-indexed (Jan is 00); day is the literal day.
        url = f"{BASE}/{inst.symbol}/{year}/{month - 1:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
        payload = self._get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload or b"")
        return payload


def decode_h1(payload: bytes, inst: Instrument, year: int, month: int) -> pl.DataFrame:
    """Decode one monthly H1 payload into a bar frame."""
    body = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(payload)
    if len(body) % RECORD_SIZE:
        raise DukascopyError(
            f"{inst.symbol} {year}-{month:02d}: {len(body)} bytes is not a multiple of {RECORD_SIZE}"
        )

    month_start = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
    rows = []
    for offset, o, c, lo, hi, vol in RECORD.iter_unpack(body):
        rows.append(
            (
                month_start + dt.timedelta(seconds=offset),
                o / inst.price_scale,
                hi / inst.price_scale,
                lo / inst.price_scale,
                c / inst.price_scale,
                vol,
            )
        )

    frame = pl.DataFrame(
        rows,
        schema={
            "ts_open": pl.Datetime(time_unit="us", time_zone="UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
        orient="row",
    )

    # Drop synthetic market-closed fills before anything else sees them.
    return frame.filter(pl.col("volume") > 0)


def months(start: dt.date, end: dt.date):
    """Yield (year, month) pairs inclusive of both endpoints."""
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def download_h1(
    client: Client,
    inst: Instrument,
    start: dt.date,
    end: dt.date,
    *,
    progress: bool = True,
) -> pl.DataFrame:
    """Fetch and decode H1 bars across a date range."""
    frames = []
    todo = list(months(start, end))
    for i, (year, month) in enumerate(todo, 1):
        payload = client.fetch_h1_month(inst, year, month)
        if payload:
            frames.append(decode_h1(payload, inst, year, month))
        if progress and (i % 24 == 0 or i == len(todo)):
            print(f"  {inst.symbol}: {i}/{len(todo)} months", flush=True)

    if not frames:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    last_day = calendar.monthrange(end.year, end.month)[1]
    return (
        pl.concat(frames)
        .filter(
            pl.col("ts_open").is_between(
                dt.datetime(start.year, start.month, 1, tzinfo=dt.timezone.utc),
                dt.datetime(end.year, end.month, last_day, 23, 59, tzinfo=dt.timezone.utc),
            )
        )
        .sort("ts_open")
        .unique(subset="ts_open", keep="first")
    )


_EMPTY_SCHEMA = {
    "ts_open": pl.Datetime(time_unit="us", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def decode_ticks(
    payload: bytes, inst: Instrument, year: int, month: int, day: int, hour: int
) -> pl.DataFrame:
    """Decode one hour of ticks into (ts, bid, volume) rows."""
    body = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(payload)
    if len(body) % TICK_SIZE:
        raise DukascopyError(
            f"{inst.symbol} ticks {year}-{month:02d}-{day:02d} {hour:02d}h: "
            f"{len(body)} bytes is not a multiple of {TICK_SIZE}"
        )
    hour_start = dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc)
    rows = [
        (hour_start + dt.timedelta(milliseconds=ms), bid / inst.price_scale, bvol)
        for ms, _ask, bid, _avol, bvol in TICK.iter_unpack(body)
    ]
    return pl.DataFrame(
        rows,
        schema={"ts": pl.Datetime("us", "UTC"), "bid": pl.Float64, "volume": pl.Float64},
        orient="row",
    )


def download_h1_from_ticks(
    client: Client, inst: Instrument, start: dt.date, end: dt.date, *, progress: bool = True
) -> pl.DataFrame:
    """Build H1 BID bars from tick files across [start, end] inclusive.

    The current month's monthly candle file is not published yet, so recent bars
    come from tick files (near-real-time). Only fully-completed hours are used, so
    every bar returned is closed; the in-progress hour is skipped.
    """
    now = dt.datetime.now(dt.timezone.utc)
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    frames = []
    day = start
    while day <= end:
        weekday = day.weekday()  # Mon=0 .. Sat=5, Sun=6
        for hour in range(24):
            # FX is closed from ~Fri 21:00 UTC to ~Sun 21:00 UTC. Those hours have
            # no ticks, so skip them rather than paying a throttled request each.
            if weekday == 5:  # Saturday — closed all day
                break
            if weekday == 6 and hour < 20:  # Sunday — before the ~21:00 UTC reopen
                continue
            slot = dt.datetime(day.year, day.month, day.day, hour, tzinfo=dt.timezone.utc)
            if slot >= current_hour:
                break
            payload = client.fetch_hour_ticks(inst, day.year, day.month, day.day, hour)
            if payload:
                frames.append(decode_ticks(payload, inst, day.year, day.month, day.day, hour))
        if progress:
            print(f"  {inst.symbol}: ticks through {day}", flush=True)
        day += dt.timedelta(days=1)

    if not frames:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    return (
        pl.concat(frames)
        .sort("ts")
        .group_by_dynamic("ts", every="1h", closed="left")
        .agg(
            [
                pl.col("bid").first().alias("open"),
                pl.col("bid").max().alias("high"),
                pl.col("bid").min().alias("low"),
                pl.col("bid").last().alias("close"),
                pl.col("volume").sum(),
            ]
        )
        .rename({"ts": "ts_open"})
        .filter(pl.col("volume") > 0)
        .select(["ts_open", "open", "high", "low", "close", "volume"])
        .sort("ts_open")
    )
