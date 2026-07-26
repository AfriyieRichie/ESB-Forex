"""Instrument definitions.

`price_scale` is the divisor Dukascopy applies to its integer prices; `pip` is
one pip in price terms. JPY crosses differ from everything else on both counts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    price_scale: float
    pip: float
    spread_pips: float

    @property
    def is_jpy(self) -> bool:
        return self.symbol.endswith("JPY")

    @property
    def spread(self) -> float:
        return self.spread_pips * self.pip


# Typical retail spreads, set slightly wider than today's tightest quotes.
# Spreads through 2005-2016 were materially worse than modern ones, so these
# remain optimistic for the research window - a strategy that only works at
# these costs is not a strategy.
_SPREAD_PIPS = {
    "EURUSD": 1.2,
    "GBPUSD": 1.5,
    "USDJPY": 1.2,
    "AUDUSD": 1.4,
    "USDCAD": 1.7,
    "USDCHF": 1.7,
    "NZDUSD": 2.0,
    "EURJPY": 1.8,
    "GBPJPY": 2.8,
    "EURGBP": 1.6,
}


def _make(symbol: str) -> Instrument:
    spread = _SPREAD_PIPS[symbol]
    if symbol.endswith("JPY"):
        return Instrument(symbol, price_scale=1e3, pip=0.01, spread_pips=spread)
    return Instrument(symbol, price_scale=1e5, pip=0.0001, spread_pips=spread)


# The basket: majors plus the crosses that trade cleanly. Note these are far
# from independent - USD appears in seven of them - so a large trade count
# across this basket does not mean a large effective sample.
BASKET = [
    _make(s)
    for s in (
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "NZDUSD",
        "EURJPY",
        "GBPJPY",
        "EURGBP",
    )
]

BY_SYMBOL = {i.symbol: i for i in BASKET}


def get(symbol: str) -> Instrument:
    try:
        return BY_SYMBOL[symbol]
    except KeyError:
        raise KeyError(f"{symbol} not in basket: {sorted(BY_SYMBOL)}") from None
