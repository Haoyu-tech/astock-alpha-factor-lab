# -*- coding: utf-8 -*-
"""Memory-safe streaming driver for the max-window study.

Each factor is computed, fully tested (overall + market-cap + index rank-IC),
sampled for the correlation matrix, then discarded — so the 2,859 x 3,497 panel
never holds all ~344 factor frames at once.
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

# --------------------------------------------------------------------------- #
log("loading fields ...")
F, meta = panel.load_fields()
for k in list(F):
    F[k] = F[k].astype("float32")
close = F["close"]
dates, codes = meta["trade_dates"], meta["ts_codes"]
valid = close.notna().to_numpy()
total_valid = int(valid.sum())
log(f"panel {len(dates)} x {len(codes)}  (valid cells {total_valid:,})")

fwd = panel.forward_returns(close, C.FWD_PERIODS)
mktcap_masks = panel.mktcap_group_masks(meta["total_mv"], C.N_MKTCAP_GROUPS)
index_masks = panel.index_masks(dates, codes)
log("index masks avg sizes: " +
    ", ".join(f"{k}={int(v.sum(axis=1).mean())}" for k, v in index_masks.items()))

# precompute forward-return cross-sectional ranks (float64 for the IC math)
fwd_rank_overall = {k: ft.rank_cs(fwd[k]).to_numpy("float64") for k in C.FWD_PERIODS}
fwd_rank_mc = {lab: ft.rank_cs(fwd[PRIMARY], mktcap_masks[lab]).to_numpy("float64")
               for lab in mktcap_masks}
fwd_rank_idx = {lab: ft.rank_cs(fwd[PRIMARY], index_masks[lab]).to_numpy("float64")
                for lab in index_masks}

# correlation sample positions (fixed across factors)
rng = np.random.default_rng(0)
valid_flat = np.flatnonzero(valid.ravel())
samp = rng.choice(valid_flat, min(50000, len(valid_flat)), replace=False)

overall_rows, mc_rows, idx_rows, screen_rows = [], [], [], []
family, corr_cols = {}, {}
counter = [0]


def _ic(Rf_np, Rr_np):
    ic, n = ft._masked_pearson_rows(Rf_np, Rr_np)
    return pd.Series(np.where(n >= C.IC_MIN_STOCKS, ic, np.nan), index=dates)


def sink(name, fr, fam):
    frv = fr.replace([np.inf, -np.inf], np.nan)
    arr = frv.to_numpy()
    fin = np.isfinite(arr)
    cov = (fin & valid).sum() / total_valid
    day_std = np.nanstd(np.where(fin, arr, np.nan), axis=1)
    good = np.isfinite(day_std) & (day_std > 0)
    ok = (cov >= 0.30) and (good.mean() >= 0.5)
    screen_rows.append((name, fam, round(float(cov), 3),
                        "kept" if ok else ("low_coverage" if cov < 0.30 else "degenerate")))
    if not ok:
        return
    family[name] = fam
    # overall IC (3 horizons)
    Rf = frv.rank(axis=1).to_numpy("float64")
    row = {"factor": name, "family": fam, "coverage": float(cov)}
    for k in C.FWD_PERIODS:
        st = ft.ic_summary(_ic(Rf, fwd_rank_overall[k]))
        row[f"IC{k}"] = st["IC_mean"]; row[f"ICIR{k}"] = st["ICIR"]
        row[f"t{k}"] = st["t_stat"]; row[f"posrate{k}"] = st["IC_pos_rate"]
    overall_rows.append(row)
    # market-cap buckets
    mrow = {"factor": name, "family": fam}; best, ba = None, -1
    for lab in mktcap_masks:
        Rfg = frv.where(mktcap_masks[lab]).rank(axis=1).to_numpy("float64")
        st = ft.ic_summary(_ic(Rfg, fwd_rank_mc[lab]))
        mrow[f"IC_{lab}"] = st["IC_mean"]; mrow[f"ICIR_{lab}"] = st["ICIR"]
        if np.isfinite(st["IC_mean"]) and abs(st["IC_mean"]) > ba:
            ba, best = abs(st["IC_mean"]), lab
    mrow["best_mktcap"] = best; mc_rows.append(mrow)
    # index membership
    irow = {"factor": name, "family": fam}; best, ba = None, -1
    for lab in index_masks:
        Rfg = frv.where(index_masks[lab]).rank(axis=1).to_numpy("float64")
        st = ft.ic_summary(_ic(Rfg, fwd_rank_idx[lab]))
        irow[f"IC_{lab}"] = st["IC_mean"]; irow[f"ICIR_{lab}"] = st["ICIR"]
        if np.isfinite(st["IC_mean"]) and abs(st["IC_mean"]) > ba:
            ba, best = abs(st["IC_mean"]), lab
    irow["best_index"] = best; idx_rows.append(irow)
    # correlation sample
    corr_cols[name] = arr.ravel()[samp].astype("float32")

    counter[0] += 1
    if counter[0] % 40 == 0:
        log(f"  tested {counter[0]} kept factors ({name})")
        pd.DataFrame(overall_rows).to_csv(os.path.join(C.RESULTS_DIR, "factor_ic_overall.csv"), index=False)
    del frv, arr, Rf
    gc.collect()


# --------------------------------------------------------------------------- #
log("streaming Alpha158 ...")
alpha158.compute(F, sink=lambda n, v: sink(n, v, "Alpha158"))
log(f"  Alpha158 done ({counter[0]} kept so far)")
log("streaming GTJA191 ...")
_, g_fails = gtja191.compute(F, sink=lambda n, v: sink(n, v, "GTJA191"))
log(f"  GTJA191 done ({len(g_fails)} raised errors)")

# --------------------------------------------------------------------------- #
log("saving IC tables ...")
overall = pd.DataFrame(overall_rows)
overall["absIC5"] = overall["IC5"].abs()
overall = overall.sort_values("absIC5", ascending=False)
overall.to_csv(os.path.join(C.RESULTS_DIR, "factor_ic_overall.csv"), index=False)
pd.DataFrame(mc_rows).to_csv(os.path.join(C.RESULTS_DIR, "factor_ic_by_mktcap.csv"), index=False)
pd.DataFrame(idx_rows).to_csv(os.path.join(C.RESULTS_DIR, "factor_ic_by_index.csv"), index=False)
pd.DataFrame(screen_rows, columns=["factor", "family", "coverage", "status"]).to_csv(
    os.path.join(C.RESULTS_DIR, "screened_factors.csv"), index=False)
n158 = sum(1 for v in family.values() if v == "Alpha158")
n191 = sum(1 for v in family.values() if v == "GTJA191")
log(f"kept {len(family)} factors (Alpha158={n158}, GTJA191={n191})")

import json
meta_out = {
    "n_stocks": len(codes), "n_dates": len(dates),
    "start": dates[0], "end": dates[-1],
    "n158": n158, "n191": n191, "n_total": len(family),
    "gtja_computed": counter[0], "gtja_errors": len(g_fails),
}
with open(os.path.join(C.RESULTS_DIR, "meta.json"), "w") as f:
    json.dump(meta_out, f, indent=2)

# --------------------------------------------------------------------------- #
log("factor correlation ...")
names = list(corr_cols.keys())
M = pd.DataFrame({n: corr_cols[n] for n in names})
corr = M.rank().corr()          # Spearman
corr.to_csv(os.path.join(C.RESULTS_DIR, "factor_correlation.csv"))
cn = corr.columns; cm = corr.to_numpy(); pairs = []
for i in range(len(cn)):
    for j in range(i + 1, len(cn)):
        pairs.append((cn[i], cn[j], family[cn[i]], family[cn[j]], cm[i, j]))
pdf = pd.DataFrame(pairs, columns=["f1", "f2", "fam1", "fam2", "corr"])
pdf["abscorr"] = pdf["corr"].abs()
pdf.sort_values("abscorr", ascending=False).head(60).to_csv(
    os.path.join(C.RESULTS_DIR, "top_correlated_pairs.csv"), index=False)

log(f"DONE in {time.time()-t0:.0f}s")
print("\nTop 15 by |IC@5|:\n",
      overall[["factor", "family", "IC5", "ICIR5", "t5"]].head(15).round(4).to_string(index=False))
