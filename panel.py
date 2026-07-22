# -*- coding: utf-8 -*-
"""Build wide (date x stock) field panels, forward returns, and group masks."""
import numpy as np
import pandas as pd

import config as C
import data_loader as dl


FIELDS = ["open", "high", "low", "close", "vwap", "volume", "amount"]


def load_fields():
    """Return (F, meta) where F[field] is a wide frame (date x stock)."""
    df = dl.get_daily_panel()
    df = df.sort_values(["trade_date", "ts_code"])
    F = {}
    for f in FIELDS:
        F[f] = df.pivot(index="trade_date", columns="ts_code", values=f).sort_index()
    total_mv = df.pivot(index="trade_date", columns="ts_code", values="total_mv").sort_index()
    meta = {
        "total_mv": total_mv,
        "trade_dates": list(F["close"].index),
        "ts_codes": list(F["close"].columns),
    }
    return F, meta


def forward_returns(close, periods):
    return {k: close.shift(-k) / close - 1.0 for k in periods}


def mktcap_group_masks(total_mv, n=3):
    """Time-varying market-cap buckets by daily cross-sectional quantile.
    Returns dict label -> boolean wide mask. label ordered small..large."""
    ranks = total_mv.rank(axis=1, pct=True)
    if n == 3:
        labels = ["Small", "Mid", "Large"]
        edges = [0.0, 1 / 3, 2 / 3, 1.0]
    else:
        # zero-padded so labels sort small->large lexicographically (D01..D10)
        labels = [f"D{i+1:02d}" for i in range(n)]
        edges = list(np.linspace(0, 1, n + 1))
    masks = {}
    for i, lab in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            masks[lab] = (ranks > lo) & (ranks <= hi) | (ranks == 0)
        else:
            masks[lab] = (ranks > lo) & (ranks <= hi)
    return masks


def index_masks(trade_dates, ts_codes):
    """Boolean wide mask per index using time-varying membership."""
    mm = dl.membership_matrix(trade_dates)      # dict index -> {date: set(codes)}
    codes = pd.Index(ts_codes)
    out = {}
    for name, per_date in mm.items():
        M = pd.DataFrame(False, index=trade_dates, columns=ts_codes)
        for d in trade_dates:
            s = per_date.get(d, set())
            if s:
                cols = codes.intersection(s)
                M.loc[d, cols] = True
        out[name] = M
    return out
