# -*- coding: utf-8 -*-
"""Per-year and recent-window rank-IC for every factor (overall universe, 5d).

Streams factors, builds each factor's daily IC series once, then derives
calendar-year means and a 'recent' (>= RECENT_START) summary so we can rank
which factors work *lately* vs over the full 2014-2026 sample.
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
RECENT_START = "20230101"          # "近几年" = last ~3.5 years
RECENT2_START = "20240101"         # last ~2.5 years

log("loading fields ...")
F, meta = panel.load_fields()
for k in list(F):
    F[k] = F[k].astype("float32")
close = F["close"]
dates = np.array(meta["trade_dates"])
years = np.array([d[:4] for d in meta["trade_dates"]])
uniq_years = sorted(set(years.tolist()))
fwd = panel.forward_returns(close, [PRIMARY])
fwd_rank = ft.rank_cs(fwd[PRIMARY]).to_numpy("float64")

kept_df = pd.read_csv(os.path.join(C.RESULTS_DIR, "factor_ic_overall.csv"))
kept = set(kept_df["factor"])
family = kept_df.set_index("factor")["family"].to_dict()

rec_mask = dates >= RECENT_START
rec2_mask = dates >= RECENT2_START
rows = []
counter = [0]


def _summ(s):
    s = s[np.isfinite(s)]
    if len(s) < 20:
        return np.nan, np.nan, np.nan
    m, sd = s.mean(), s.std()
    icir = m / sd if sd > 0 else np.nan
    t = icir * np.sqrt(len(s)) if sd > 0 else np.nan
    return m, icir, t


def sink(name, fr, fam):
    if name not in kept:
        return
    frv = fr.replace([np.inf, -np.inf], np.nan)
    Rf = frv.rank(axis=1).to_numpy("float64")
    ic, n = ft._masked_pearson_rows(Rf, fwd_rank)
    ic = np.where(n >= C.IC_MIN_STOCKS, ic, np.nan)
    row = {"factor": name, "family": fam}
    # full
    fm, ficir, ft_ = _summ(ic)
    row["IC_full"] = fm; row["ICIR_full"] = ficir
    # per year
    for y in uniq_years:
        row[f"IC_{y}"] = np.nanmean(np.where(years == y, ic, np.nan))
    # recent windows
    rm, ricir, rt = _summ(ic[rec_mask])
    row["IC_recent3y"] = rm; row["ICIR_recent3y"] = ricir; row["t_recent3y"] = rt
    r2m, r2icir, r2t = _summ(ic[rec2_mask])
    row["IC_recent2y"] = r2m; row["ICIR_recent2y"] = r2icir
    rows.append(row)
    counter[0] += 1
    if counter[0] % 20 == 0:
        log(f"  {counter[0]} factors ({name})")
    del frv, Rf
    gc.collect()


log("streaming Alpha158 ...")
alpha158.compute(F, sink=lambda n, v: sink(n, v, family.get(n, "Alpha158")))
log("streaming GTJA191 ...")
gtja191.compute(F, sink=lambda n, v: sink(n, v, family.get(n, "GTJA191")))

out = pd.DataFrame(rows)
out["absIC_recent3y"] = out["IC_recent3y"].abs()
out["absICIR_recent3y"] = out["ICIR_recent3y"].abs()
out = out.sort_values("absICIR_recent3y", ascending=False)
out.to_csv(os.path.join(C.RESULTS_DIR, "factor_ic_yearly.csv"), index=False)
log(f"DONE {len(out)} factors in {time.time()-t0:.0f}s")

show = ["factor", "family", "IC_recent3y", "ICIR_recent3y", "t_recent3y", "IC_full", "ICIR_full"]
print("\n=== Top 20 by RECENT (>=2023) |ICIR| ===")
print(out[show].head(20).round(4).to_string(index=False))
print("\n=== Top 15 by RECENT |IC| ===")
print(out.sort_values("absIC_recent3y", ascending=False)[show].head(15).round(4).to_string(index=False))
# factors that improved most recently vs full
out["delta"] = out["IC_recent3y"].abs() - out["IC_full"].abs()
print("\n=== biggest 'recent > full' improvers ===")
print(out.sort_values("delta", ascending=False)[
    ["factor", "family", "IC_full", "IC_recent3y", "IC_recent2y"]].head(12).round(4).to_string(index=False))
