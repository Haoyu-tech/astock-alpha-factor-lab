# -*- coding: utf-8 -*-
"""IC vs holding horizon (rebalance-frequency analysis) + daily IC (h=5).

For every kept factor computes mean rank-IC at forward horizons
[1,2,3,5,10,15,20,30,40,60] days.  The horizon where IC peaks, and where
IC/sqrt(h) (cost-free per-period efficiency) peaks, informs the natural
rebalancing frequency.  Also dumps the daily IC series at h=5 for cumulative-IC.
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
HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 60]
DAILY_H = 5

log("loading fields ...")
F, meta = panel.load_fields()
for k in list(F):
    F[k] = F[k].astype("float32")
close = F["close"]
dates = meta["trade_dates"]
fwd = panel.forward_returns(close, HORIZONS)
fwd_rank = {h: ft.rank_cs(fwd[h]).to_numpy("float64") for h in HORIZONS}
log("forward-return ranks ready")

kept_df = pd.read_csv(os.path.join(C.RESULTS_DIR, "factor_ic_overall.csv"))
kept = set(kept_df["factor"])
family = kept_df.set_index("factor")["family"].to_dict()

rows, daily_cols = [], {}
counter = [0]


def _summ(ic, n):
    s = ic[np.isfinite(ic) & (n >= C.IC_MIN_STOCKS)]
    if len(s) < 20:
        return np.nan, np.nan
    m, sd = s.mean(), s.std()
    return m, (m / sd if sd > 0 else np.nan)


def sink(name, fr, fam):
    if name not in kept:
        return
    frv = fr.replace([np.inf, -np.inf], np.nan)
    Rf = frv.rank(axis=1).to_numpy("float64")
    row = {"factor": name, "family": fam}
    for h in HORIZONS:
        ic, n = ft._masked_pearson_rows(Rf, fwd_rank[h])
        m, icir = _summ(ic, n)
        row[f"IC_{h}"] = m; row[f"ICIR_{h}"] = icir
        if h == DAILY_H:
            daily_cols[name] = np.where(n >= C.IC_MIN_STOCKS, ic, np.nan).astype("float32")
    rows.append(row)
    counter[0] += 1
    if counter[0] % 40 == 0:
        log(f"  {counter[0]} factors")
    del frv, Rf
    gc.collect()


log("streaming Alpha158 ...")
alpha158.compute(F, sink=lambda n, v: sink(n, v, family.get(n, "Alpha158")))
log("streaming GTJA191 ...")
gtja191.compute(F, sink=lambda n, v: sink(n, v, family.get(n, "GTJA191")))

out = pd.DataFrame(rows)
# peak horizon (by |IC|) and cost-free-efficiency peak (|IC|/sqrt(h))
ic_cols = [f"IC_{h}" for h in HORIZONS]
abs_ic = out[ic_cols].abs()
out["peak_h"] = [HORIZONS[i] for i in abs_ic.to_numpy().argmax(axis=1)]
eff = abs_ic.to_numpy() / np.sqrt(HORIZONS)
out["eff_peak_h"] = [HORIZONS[i] for i in eff.argmax(axis=1)]
out.to_csv(os.path.join(C.RESULTS_DIR, "factor_ic_horizon.csv"), index=False)

pd.DataFrame(daily_cols, index=pd.Index(dates, name="trade_date")).to_parquet(
    os.path.join(C.RESULTS_DIR, "daily_ic.parquet"))

log(f"DONE {len(out)} factors in {time.time()-t0:.0f}s")
# family-average IC by horizon
print("\n=== mean |IC| by holding horizon (days) ===")
hdr = "family     " + "".join(f"{h:>7}" for h in HORIZONS)
print(hdr)
for famname in ["Alpha158", "GTJA191"]:
    sub = out[out.family == famname]
    vals = "".join(f"{sub[f'IC_{h}'].abs().mean():7.4f}" for h in HORIZONS)
    print(f"{famname:10s} {vals}")
allv = "".join(f"{out[f'IC_{h}'].abs().mean():7.4f}" for h in HORIZONS)
print(f"{'ALL':10s} {allv}")
print("\npeak-horizon vote (by |IC|):", out["peak_h"].value_counts().sort_index().to_dict())
print("eff-peak (|IC|/sqrt h) vote:", out["eff_peak_h"].value_counts().sort_index().to_dict())
