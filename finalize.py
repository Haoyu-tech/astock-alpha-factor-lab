# -*- coding: utf-8 -*-
"""Regenerate family/group aggregates from the saved IC CSVs (fast, no recompute)."""
import os
import numpy as np
import pandas as pd
import config as C

R = C.RESULTS_DIR
ov = pd.read_csv(os.path.join(R, "factor_ic_overall.csv"))
mc = pd.read_csv(os.path.join(R, "factor_ic_by_mktcap.csv"))
ix = pd.read_csv(os.path.join(R, "factor_ic_by_index.csv"))
ov["absIC5"] = ov["IC5"].abs()

# derive bucket labels from the CSV columns (order preserved) -> supports N deciles
mc_labels = [c[3:] for c in mc.columns if c.startswith("IC_")]
idx_labels = list(C.INDEX_CODES.keys())

# ---- family summary --------------------------------------------------------
fam = ov.groupby("family").agg(
    n_factors=("factor", "size"),
    mean_absIC5=("absIC5", "mean"),
    mean_absICIR5=("ICIR5", lambda s: s.abs().mean()),
    frac_t_gt2=("t5", lambda s: (s.abs() > 2).mean()),
).reset_index()
fam.to_csv(os.path.join(R, "family_summary.csv"), index=False)

# ---- market-cap group means by family -------------------------------------
for l in mc_labels:
    mc[f"abs_{l}"] = mc[f"IC_{l}"].abs()
mc_fm = mc.groupby("family")[[f"abs_{l}" for l in mc_labels]].mean()
mc_fm.to_csv(os.path.join(R, "mktcap_group_family_mean.csv"))

for l in idx_labels:
    ix[f"abs_{l}"] = ix[f"IC_{l}"].abs()
ix_fm = ix.groupby("family")[[f"abs_{l}" for l in idx_labels]].mean()
ix_fm.to_csv(os.path.join(R, "index_group_family_mean.csv"))

# ---- votes -----------------------------------------------------------------
mc_votes = mc["best_mktcap"].value_counts()
ix_votes = ix["best_index"].value_counts()
mc_votes.to_csv(os.path.join(R, "mktcap_best_votes.csv"))
ix_votes.to_csv(os.path.join(R, "index_best_votes.csv"))

# ---- overall means across ALL factors (both families pooled) --------------
mc_all = pd.Series({l: mc[f"abs_{l}"].mean() for l in mc_labels})
ix_all = pd.Series({l: ix[f"abs_{l}"].mean() for l in idx_labels})

print("=== FAMILY SUMMARY ===")
print(fam.round(4).to_string(index=False))
print("\n=== mean |IC@5| by MARKET-CAP bucket ===")
print("  by family:\n", mc_fm.round(4).to_string())
print("  pooled (all factors):", mc_all.round(4).to_dict())
print("  best-bucket vote counts:", mc_votes.to_dict())
print("\n=== mean |IC@5| by INDEX ===")
print("  by family:\n", ix_fm.round(4).to_string())
print("  pooled (all factors):", ix_all.round(4).to_dict())
print("  best-index vote counts:", ix_votes.to_dict())

# strongest factors within each group (for the report)
print("\n=== strongest single-factor IC in each market-cap bucket ===")
for l in mc_labels:
    top = mc.reindex(mc[f"IC_{l}"].abs().sort_values(ascending=False).index).head(3)
    print(f"  {l}:", [(r.factor, round(r[f'IC_{l}'], 3)) for _, r in top.iterrows()])
print("=== strongest single-factor IC in each index ===")
for l in idx_labels:
    top = ix.reindex(ix[f"IC_{l}"].abs().sort_values(ascending=False).index).head(3)
    print(f"  {C.INDEX_LABELS[l]}:", [(r.factor, round(r[f'IC_{l}'], 3)) for _, r in top.iterrows()])
