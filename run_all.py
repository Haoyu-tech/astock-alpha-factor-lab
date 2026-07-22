# -*- coding: utf-8 -*-
"""End-to-end driver:
  1. build panels
  2. construct Alpha158 + GTJA191 factors
  3. screen degenerate factors
  4. rank-IC (overall, by market-cap bucket, by index membership)
  5. factor correlation
  6. save all results to results/
"""
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C
import panel
import alpha158
import gtja191
import factor_test as ft

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1. panels
# --------------------------------------------------------------------------- #
log("loading fields ...")
F, meta = panel.load_fields()
close = F["close"]
dates, codes = meta["trade_dates"], meta["ts_codes"]
log(f"panel: {len(dates)} dates x {len(codes)} stocks")

valid = close.notna()                                    # tradable cells
total_cells = valid.to_numpy().sum()

fwd = panel.forward_returns(close, C.FWD_PERIODS)
mktcap_masks = panel.mktcap_group_masks(meta["total_mv"], C.N_MKTCAP_GROUPS)
index_masks = panel.index_masks(dates, codes)
log(f"index masks per-day avg sizes: " +
    ", ".join(f"{k}={int(v.sum(axis=1).mean())}" for k, v in index_masks.items()))


# --------------------------------------------------------------------------- #
# 2. construct factors
# --------------------------------------------------------------------------- #
log("computing Alpha158 ...")
a158 = alpha158.compute(F)
log(f"Alpha158: {len(a158)} raw factors")

log("computing GTJA191 ...")
g191, g_fails = gtja191.compute(F)
g191 = {k: v for k, v in g191.items() if v is not None}
log(f"GTJA191: {len(g191)} computed, {len(g_fails)} raised errors")

family = {**{k: "Alpha158" for k in a158}, **{k: "GTJA191" for k in g191}}
raw = {**a158, **g191}


# --------------------------------------------------------------------------- #
# 3. screen: coverage + non-degenerate
# --------------------------------------------------------------------------- #
log("screening factors ...")
factors, screen_rows = {}, []
for name, fr in raw.items():
    if fr is None:
        screen_rows.append((name, family[name], 0.0, "none")); continue
    fr = fr.reindex(index=dates, columns=codes)
    fr = fr.replace([np.inf, -np.inf], np.nan)
    fin = np.isfinite(fr.to_numpy())
    cov = (fin & valid.to_numpy()).sum() / total_cells
    # cross-sectional variability: median daily std > 0 on enough days
    day_std = fr.std(axis=1)
    ok = (cov >= 0.30) and (day_std.replace(0, np.nan).notna().mean() >= 0.5)
    reason = "kept" if ok else ("low_coverage" if cov < 0.30 else "degenerate")
    screen_rows.append((name, family[name], round(float(cov), 3), reason))
    if ok:
        factors[name] = fr.astype("float32")
screen_df = pd.DataFrame(screen_rows, columns=["factor", "family", "coverage", "status"])
screen_df.to_csv(os.path.join(C.RESULTS_DIR, "screened_factors.csv"), index=False)
n158 = sum(1 for k in factors if family[k] == "Alpha158")
n191 = sum(1 for k in factors if family[k] == "GTJA191")
log(f"kept {len(factors)} factors  (Alpha158={n158}, GTJA191={n191})")


# --------------------------------------------------------------------------- #
# 4a. overall rank-IC for each horizon
# --------------------------------------------------------------------------- #
log("overall IC ...")
fwd_ranks = {k: ft.rank_cs(fwd[k]) for k in C.FWD_PERIODS}
overall_rows = []
ic_store = {}                       # (name) -> daily IC series at primary horizon
PRIMARY = 5
for i, (name, fr) in enumerate(factors.items()):
    Rf = ft.rank_cs(fr)
    row = {"factor": name, "family": family[name],
           "coverage": float((np.isfinite(fr.to_numpy()) & valid.to_numpy()).sum() / total_cells)}
    for k in C.FWD_PERIODS:
        ic, n = ft._masked_pearson_rows(Rf.to_numpy(float), fwd_ranks[k].to_numpy(float))
        s = pd.Series(np.where(n >= C.IC_MIN_STOCKS, ic, np.nan), index=dates)
        st = ft.ic_summary(s)
        row[f"IC{k}"] = st["IC_mean"]; row[f"ICIR{k}"] = st["ICIR"]
        row[f"t{k}"] = st["t_stat"]; row[f"posrate{k}"] = st["IC_pos_rate"]
        if k == PRIMARY:
            ic_store[name] = s
    overall_rows.append(row)
    if (i + 1) % 50 == 0:
        log(f"  overall IC {i+1}/{len(factors)}")
overall = pd.DataFrame(overall_rows)
overall["absIC5"] = overall["IC5"].abs()
overall = overall.sort_values("absIC5", ascending=False)
overall.to_csv(os.path.join(C.RESULTS_DIR, "factor_ic_overall.csv"), index=False)
log("saved factor_ic_overall.csv")


# --------------------------------------------------------------------------- #
# 4b. IC by market-cap bucket (primary horizon)
# --------------------------------------------------------------------------- #
log("IC by market-cap bucket ...")
mc_labels = list(mktcap_masks.keys())
fwd_ranks_mc = {lab: ft.rank_cs(fwd[PRIMARY], mktcap_masks[lab]) for lab in mc_labels}
mc_rows = []
for name, fr in factors.items():
    row = {"factor": name, "family": family[name]}
    best_lab, best_abs = None, -1
    for lab in mc_labels:
        Rf = ft.rank_cs(fr, mktcap_masks[lab])
        ic, n = ft._masked_pearson_rows(Rf.to_numpy(float), fwd_ranks_mc[lab].to_numpy(float))
        s = pd.Series(np.where(n >= C.IC_MIN_STOCKS, ic, np.nan), index=dates)
        st = ft.ic_summary(s)
        row[f"IC_{lab}"] = st["IC_mean"]; row[f"ICIR_{lab}"] = st["ICIR"]
        if abs(st["IC_mean"]) > best_abs and np.isfinite(st["IC_mean"]):
            best_abs, best_lab = abs(st["IC_mean"]), lab
    row["best_mktcap"] = best_lab
    mc_rows.append(row)
mc_df = pd.DataFrame(mc_rows)
mc_df.to_csv(os.path.join(C.RESULTS_DIR, "factor_ic_by_mktcap.csv"), index=False)
log("saved factor_ic_by_mktcap.csv")


# --------------------------------------------------------------------------- #
# 4c. IC by index membership (primary horizon)
# --------------------------------------------------------------------------- #
log("IC by index membership ...")
idx_labels = list(index_masks.keys())
fwd_ranks_idx = {lab: ft.rank_cs(fwd[PRIMARY], index_masks[lab]) for lab in idx_labels}
idx_rows = []
for name, fr in factors.items():
    row = {"factor": name, "family": family[name]}
    best_lab, best_abs = None, -1
    for lab in idx_labels:
        Rf = ft.rank_cs(fr, index_masks[lab])
        ic, n = ft._masked_pearson_rows(Rf.to_numpy(float), fwd_ranks_idx[lab].to_numpy(float))
        s = pd.Series(np.where(n >= C.IC_MIN_STOCKS, ic, np.nan), index=dates)
        st = ft.ic_summary(s)
        row[f"IC_{lab}"] = st["IC_mean"]; row[f"ICIR_{lab}"] = st["ICIR"]
        if abs(st["IC_mean"]) > best_abs and np.isfinite(st["IC_mean"]):
            best_abs, best_lab = abs(st["IC_mean"]), lab
    row["best_index"] = best_lab
    idx_rows.append(row)
idx_df = pd.DataFrame(idx_rows)
idx_df.to_csv(os.path.join(C.RESULTS_DIR, "factor_ic_by_index.csv"), index=False)
log("saved factor_ic_by_index.csv")


# --------------------------------------------------------------------------- #
# 5. factor correlation
# --------------------------------------------------------------------------- #
log("factor correlation matrix ...")
corr = ft.factor_correlation(factors, sample_obs=50000)
corr.to_csv(os.path.join(C.RESULTS_DIR, "factor_correlation.csv"))
# top correlated cross-family pairs
pairs = []
cn = corr.columns
cm = corr.to_numpy()
for i in range(len(cn)):
    for j in range(i + 1, len(cn)):
        pairs.append((cn[i], cn[j], family[cn[i]], family[cn[j]], cm[i, j]))
pdf = pd.DataFrame(pairs, columns=["f1", "f2", "fam1", "fam2", "corr"])
pdf["abscorr"] = pdf["corr"].abs()
pdf.sort_values("abscorr", ascending=False).head(60).to_csv(
    os.path.join(C.RESULTS_DIR, "top_correlated_pairs.csv"), index=False)
log("saved factor_correlation.csv")


# --------------------------------------------------------------------------- #
# 5b. alphalens cross-validation on the strongest factors
#     (uses the open-source alphalens-reloaded package to independently
#      recompute IC, confirming our vectorized IC is correct)
# --------------------------------------------------------------------------- #
log("alphalens validation on top factors ...")
try:
    import alphalens as al
    prices = close.copy()
    prices.index = pd.to_datetime(prices.index)
    top_names = overall.head(6)["factor"].tolist()
    al_rows = []
    for name in top_names:
        fr = factors[name].copy()
        fr.index = pd.to_datetime(fr.index)
        fser = fr.stack(dropna=True)
        fser.index = fser.index.set_names(["date", "asset"])
        try:
            data = al.utils.get_clean_factor_and_forward_returns(
                fser, prices, periods=(1, 5, 10),
                quantiles=5, max_loss=0.6)
            ic = al.performance.factor_information_coefficient(data)
            al_rows.append({
                "factor": name,
                "al_IC1": ic["1D"].mean(), "al_IC5": ic["5D"].mean(),
                "al_IC10": ic["10D"].mean(),
                "our_IC5": float(overall.loc[overall.factor == name, "IC5"].iloc[0]),
            })
        except Exception as e:
            al_rows.append({"factor": name, "error": repr(e)[:100]})
    al_df = pd.DataFrame(al_rows)
    al_df.to_csv(os.path.join(C.RESULTS_DIR, "alphalens_validation.csv"), index=False)
    print("\nalphalens vs our IC (top factors):\n", al_df.to_string(index=False))
except Exception as e:
    log(f"alphalens step skipped: {repr(e)[:120]}")


# --------------------------------------------------------------------------- #
# 6. family-level aggregates
# --------------------------------------------------------------------------- #
log("family aggregates ...")
agg_rows = []
for fam in ["Alpha158", "GTJA191"]:
    sub = overall[overall["family"] == fam]
    agg_rows.append({
        "family": fam, "n_factors": len(sub),
        "mean_absIC5": sub["absIC5"].mean(),
        "mean_ICIR5": sub["ICIR5"].abs().mean(),
        "frac_|t|>2": (sub["t5"].abs() > 2).mean(),
    })
pd.DataFrame(agg_rows).to_csv(os.path.join(C.RESULTS_DIR, "family_summary.csv"), index=False)

# mktcap group means by family
mc_group_mean = mc_df.assign(**{f"abs_{l}": mc_df[f"IC_{l}"].abs() for l in mc_labels}) \
                     .groupby("family")[[f"abs_{l}" for l in mc_labels]].mean()
mc_group_mean.to_csv(os.path.join(C.RESULTS_DIR, "mktcap_group_family_mean.csv"))
idx_group_mean = idx_df.assign(**{f"abs_{l}": idx_df[f"IC_{l}"].abs() for l in idx_labels}) \
                       .groupby("family")[[f"abs_{l}" for l in idx_labels]].mean()
idx_group_mean.to_csv(os.path.join(C.RESULTS_DIR, "index_group_family_mean.csv"))

# best-group vote counts
mc_votes = mc_df["best_mktcap"].value_counts()
idx_votes = idx_df["best_index"].value_counts()

log("=== SUMMARY ===")
print("\nFamily summary:\n", pd.DataFrame(agg_rows).to_string(index=False))
print("\nMean |IC@5| by market-cap bucket (by family):\n", mc_group_mean.to_string())
print("\nBest market-cap bucket vote counts:\n", mc_votes.to_string())
print("\nMean |IC@5| by index (by family):\n", idx_group_mean.to_string())
print("\nBest index vote counts:\n", idx_votes.to_string())
print("\nTop 15 factors by |IC@5|:\n",
      overall[["factor", "family", "IC5", "ICIR5", "t5"]].head(15).to_string(index=False))
log(f"DONE in {time.time()-t0:.0f}s")
