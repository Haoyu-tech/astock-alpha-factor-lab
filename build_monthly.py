# -*- coding: utf-8 -*-
"""Build the monthly ML dataset for the LightGBM / Alpha158 strategy.

Rows = (rebalance_date, stock) for the investable universe
       = 中证1000 constituents that month, minus the smallest 10% by total_mv.
Columns = 158 Alpha158 features (as of the rebalance date) + y (next-month return).

Alpha158 is computed daily via the streaming sink and immediately reduced to the
month-end rows, so the full daily factor panel is never held in memory.
"""
import os, time, warnings
from itertools import groupby
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import config as C
import panel, data_loader as dl, alpha158

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

DROP = float(os.environ.get("FR_DROP", "0.10"))   # drop smallest DROP by mv (0 = keep all)
TAG = os.environ.get("FR_TAG", "")                # dataset name suffix
log(f"universe filter: drop smallest {DROP:.0%} by mv   TAG='{TAG}'")

log("loading fields ...")
F, meta = panel.load_fields()
close = F["close"]
dates = meta["trade_dates"]
codes = meta["ts_codes"]
total_mv = meta["total_mv"]

# ---- rebalance dates (month-end or week-end) -------------------------------
import datetime
FREQ = os.environ.get("FR_FREQ", "M")            # 'M' monthly, 'W' weekly
if FREQ == "W":
    def _key(d): return datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])).isocalendar()[:2]
else:
    def _key(d): return d[:6]
rebal = [list(g)[-1] for _, g in groupby(dates, key=_key)]
log(f"{len(rebal)} {FREQ}-freq rebalance dates: {rebal[0]} -> {rebal[-1]}")

# ---- membership base at each rebalance date --------------------------------
#   FR_UNIV = ZZ1000 (default) | BROAD (union of the 4 indices) | ALL (full market)
UNIV = os.environ.get("FR_UNIV", "ZZ1000")
mm_all = dl.membership_matrix(dates)
codeset = set(codes)
mclose = close.loc[rebal]
mmv = total_mv.loc[rebal]
present = mclose.notna() & mmv.notna()

if UNIV == "ALL":
    zz = present.copy()                                   # every listed stock
else:
    idx_names = list(mm_all.keys()) if UNIV == "BROAD" else [UNIV]
    zz = pd.DataFrame(False, index=rebal, columns=codes)
    for d in rebal:
        members = set()
        for name in idx_names:
            members |= codeset & mm_all[name].get(d, set())
        if members:
            zz.loc[d, list(members)] = True
log(f"universe = {UNIV}")

# ---- apply presence + drop smallest DROP by mv -----------------------------
mv_in = mmv.where(zz & present)
mv_rank = mv_in.rank(axis=1, pct=True)
uni = (zz & present & (mv_rank > DROP)).fillna(False)
log(f"avg universe size / period: {int(uni.sum(axis=1).mean())}")

# ---- next-month return (label) --------------------------------------------
fwd_ret = mclose.shift(-1) / mclose - 1.0

# ---- Alpha158 features at month-ends (streaming) --------------------------
monthly = {}
def sink(name, fr):
    monthly[name] = fr.reindex(index=rebal).astype("float32")
log("computing Alpha158 (monthly slice) ...")
alpha158.compute(F, sink=sink)
log(f"Alpha158 features: {len(monthly)}")

# ---- assemble long dataset over the universe ------------------------------
uni_long = uni.stack()
idx = uni_long[uni_long].index.set_names(["date", "stock"])   # robust to source index names
data = {name: monthly[name].stack().reindex(idx) for name in monthly}
df = pd.DataFrame(data)
df["y"] = fwd_ret.stack().reindex(idx)
df["mv"] = mmv.stack().reindex(idx)
df = df.reset_index()                                # -> columns 'date','stock'
df = df.dropna(subset=["y"]).reset_index(drop=True)   # need a realised next-month return

feat_cols = list(monthly.keys())
meta_out = {"feat_cols": feat_cols, "rebal": rebal}
df.to_parquet(os.path.join(C.DATA_DIR, f"monthly_dataset{TAG}.parquet"))
pd.Series(feat_cols).to_csv(os.path.join(C.DATA_DIR, f"feat_cols{TAG}.csv"), index=False, header=False)
log(f"DONE dataset {df.shape} (rows x cols) saved. months={df['date'].nunique()}")
print(df[["date", "stock", "y"]].head(3).to_string(index=False))
print("y describe:", df["y"].describe()[["mean", "std", "min", "max"]].round(4).to_dict())
