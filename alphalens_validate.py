# -*- coding: utf-8 -*-
"""Independent IC cross-validation using alphalens-reloaded (open-source pkg).

Recomputes IC for the strongest factors and compares to our vectorized IC.
Each factor is processed in isolation with incremental CSV writes so a crash
cannot lose earlier results.
"""
import os, warnings, faulthandler
faulthandler.enable()
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import config as C
import panel, alpha158, gtja191

print("loading panel ...", flush=True)
F, meta = panel.load_fields()
close = F["close"]
prices = close.copy(); prices.index = pd.to_datetime(prices.index)

# recompute ONLY the top factors we need (filtering sink -> memory-safe)
ov = pd.read_csv(os.path.join(C.RESULTS_DIR, "factor_ic_overall.csv"))
top = ov.head(6)["factor"].tolist()
want = set(top)
allf = {}
def _grab(n, v):
    if n in want:
        allf[n] = v
alpha158.compute(F, sink=_grab)
gtja191.compute(F, sink=_grab)

import alphalens as al
rows = []
out = os.path.join(C.RESULTS_DIR, "alphalens_validation.csv")
for name in top:
    print(f"alphalens: {name}", flush=True)
    try:
        fr = allf[name].astype("float64").copy()
        fr.index = pd.to_datetime(fr.index)
        fser = fr.stack(dropna=True)
        fser.index = fser.index.set_names(["date", "asset"])
        data = al.utils.get_clean_factor_and_forward_returns(
            fser, prices, periods=(1, 5, 10), quantiles=5, max_loss=0.6)
        ic = al.performance.factor_information_coefficient(data)
        our = float(ov.loc[ov.factor == name, "IC5"].iloc[0])
        rows.append({"factor": name,
                     "alphalens_IC1": round(ic["1D"].mean(), 4),
                     "alphalens_IC5": round(ic["5D"].mean(), 4),
                     "alphalens_IC10": round(ic["10D"].mean(), 4),
                     "our_IC5": round(our, 4)})
    except Exception as e:
        rows.append({"factor": name, "error": repr(e)[:120]})
    pd.DataFrame(rows).to_csv(out, index=False)     # incremental
    print("  ->", rows[-1], flush=True)

print("\nDONE alphalens validation")
print(pd.DataFrame(rows).to_string(index=False))
