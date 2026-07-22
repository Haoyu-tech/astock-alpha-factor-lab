# -*- coding: utf-8 -*-
"""Save the daily rank-IC series (overall universe, 5d) for every kept factor.

Output: results/daily_ic.parquet  (index = trade_date, columns = factor).
Cumulative-IC curves are just cumsum() of this; also enables rolling-IC later.
"""
import os, time, warnings, gc
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import config as C
import panel, alpha158, gtja191
import factor_test as ft

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
PRIMARY = 5

log("loading fields ...")
F, meta = panel.load_fields()
for k in list(F):
    F[k] = F[k].astype("float32")
close = F["close"]
dates = meta["trade_dates"]
fwd = panel.forward_returns(close, [PRIMARY])
fwd_rank = ft.rank_cs(fwd[PRIMARY]).to_numpy("float64")

kept_df = pd.read_csv(os.path.join(C.RESULTS_DIR, "factor_ic_overall.csv"))
kept = set(kept_df["factor"])

cols = {}
counter = [0]


def sink(name, fr, fam):
    if name not in kept:
        return
    frv = fr.replace([np.inf, -np.inf], np.nan)
    Rf = frv.rank(axis=1).to_numpy("float64")
    ic, n = ft._masked_pearson_rows(Rf, fwd_rank)
    ic = np.where(n >= C.IC_MIN_STOCKS, ic, np.nan)
    cols[name] = ic.astype("float32")
    counter[0] += 1
    if counter[0] % 40 == 0:
        log(f"  {counter[0]} factors")
    del frv, Rf
    gc.collect()


log("streaming Alpha158 ...")
alpha158.compute(F, sink=lambda n, v: sink(n, v, "Alpha158"))
log("streaming GTJA191 ...")
gtja191.compute(F, sink=lambda n, v: sink(n, v, "GTJA191"))

out = pd.DataFrame(cols, index=pd.Index(dates, name="trade_date"))
out.to_parquet(os.path.join(C.RESULTS_DIR, "daily_ic.parquet"))
log(f"DONE saved daily_ic.parquet {out.shape} in {time.time()-t0:.0f}s")
