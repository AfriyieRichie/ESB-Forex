"""Timeframe bridge: which D1 bar is knowable from a given H4 bar.

The only place D1 and H4 are allowed to meet. Bar indices are per-timeframe -
D1 bar 400 and H4 bar 400 are different moments - so any cross-timeframe lookup
has to go through here.

The trap this exists to close: a D1 bar is not knowable until it closes. H4
logic running at 09:00 that consults today's still-forming D1 bar is reading
the future, and because entries are triggered off levels derived from that bar,
it inflates results invisibly rather than crashing.

Simultaneous closes are treated as knowable. An H4 bar closing at 17:00 NY ends
at the same instant as that day's D1 bar, so at that moment both are final.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def build_bridge(h4: pl.DataFrame, d1: pl.DataFrame) -> np.ndarray:
    """Map each H4 bar position to the latest D1 bar index closed by then.

    Returns an array parallel to `h4`, holding -1 where no D1 bar has closed
    yet (the warmup at the very start of history).
    """
    joined = (
        h4.select("bar", "ts_close")
        .rename({"bar": "h4_bar"})
        .sort("ts_close")
        .join_asof(
            d1.select("bar", "ts_close").rename({"bar": "d1_bar"}).sort("ts_close"),
            on="ts_close",
            strategy="backward",
        )
        .sort("h4_bar")
    )
    return joined["d1_bar"].fill_null(-1).to_numpy().astype(np.int64)
