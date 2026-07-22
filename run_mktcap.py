# -*- coding: utf-8 -*-
"""Recompute ONLY the market-cap-bucket IC (config.N_MKTCAP_GROUPS deciles).

Streams factors exactly like run_stream but computes within-bucket rank-IC at the
primary horizon only, then overwrites factor_ic_by_mktcap.csv.  Overall / index /
correlation results are left untouched (they do not depend on the cap grouping).
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
dates, codes = meta["trade_dates"], meta["ts_codes"]
valid = close.notna().to_numpy()
total_valid = int(valid.sum())
log(f"panel {len(dates)} x {len(codes)}")

fwd = panel.forward_returns(close, [PRIMARY])
masks = panel.mktcap_group_masks(meta["total_mv"], C.N_MKTCAP_GROUPS)
labels = list(masks.keys())
log(f"{len(labels)} market-cap buckets: {labels}")
log("avg bucket sizes: " + ", ".join(f"{l}={int(masks[l].sum(axis=1).mean())}" for l in labels))

fwd_rank = {l: ft.rank_cs(fwd[PRIMARY], masks[l]).to_numpy("float64") for l in labels}

# only score factors that survived screening in the main run (keeps tables aligned)
kept = set(pd.read_csv(os.path.join(C.RESULTS_DIR, "factor_ic_overall.csv"))["factor"])
family = pd.read_csv(os.path.join(C.RESULTS_DIR, "factor_ic_overall.csv")).set_index("factor")["family"].to_dict()

rows = []
counter = [0]


def _ic(Rf, Rr):
    ic, n = ft._masked_pearson_rows(Rf, Rr)
    return pd.Series(np.where(n >= C.IC_MIN_STOCKS, ic, np.nan), index=dates)


def sink(name, fr, fam):
    if name not in kept:
        return
    frv = fr.replace([np.inf, -np.inf], np.nan)
    row = {"factor": name, "family": fam}
    best, ba = None, -1
    for lab in labels:
        Rfg = frv.where(masks[lab]).rank(axis=1).to_numpy("float64")
        st = ft.ic_summary(_ic(Rfg, fwd_rank[lab]))
        row[f"IC_{lab}"] = st["IC_mean"]; row[f"ICIR_{lab}"] = st["ICIR"]
        if np.isfinite(st["IC_mean"]) and abs(st["IC_mean"]) > ba:
            ba, best = abs(st["IC_mean"]), lab
    row["best_mktcap"] = best
    rows.append(row)
    counter[0] += 1
    if counter[0] % 40 == 0:
        log(f"  {counter[0]} factors ({name})")
    del frv
    gc.collect()


log("streaming Alpha158 ...")
alpha158.compute(F, sink=lambda n, v: sink(n, v, family.get(n, "Alpha158")))
log("streaming GTJA191 ...")
gtja191.compute(F, sink=lambda n, v: sink(n, v, family.get(n, "GTJA191")))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(C.RESULTS_DIR, "factor_ic_by_mktcap.csv"), index=False)
log(f"DONE {len(out)} factors x {len(labels)} deciles in {time.time()-t0:.0f}s")
# quick pooled gradient
pooled = {l: out[f"IC_{l}"].abs().mean() for l in labels}
print("\npooled mean |IC@5| by decile:")
for l in labels:
    print(f"  {l}: {pooled[l]:.4f}")
print("best-bucket votes:", out["best_mktcap"].value_counts().to_dict())
