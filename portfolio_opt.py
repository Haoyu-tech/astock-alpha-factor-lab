# -*- coding: utf-8 -*-
"""Portfolio weight optimization on the ML long book (G20).

For each monthly rebalance we take the selected long book, estimate a
Ledoit-Wolf shrinkage covariance from trailing daily returns, and compare
weighting schemes — all long-only, fully invested, per-name cap:

  EW      equal weight (baseline)
  InvVol  inverse-volatility (risk weighting)
  MinVar  minimum variance
  MVO     maximize predicted alpha s.t. variance <= equal-weight variance
  MVO_TC  MVO with a monthly one-way turnover cap

Outputs results/ml{TAG}/opt/: returns, metrics, weight diagnostics, charts.
"""
import os, time, warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import panel

TAG = os.environ.get("FR_TAG", "")
ML = os.path.join(C.RESULTS_DIR, f"ml{TAG}")
OPT = os.path.join(ML, "opt"); os.makedirs(OPT, exist_ok=True)
CAP = 0.08            # per-name weight cap
LOOKBACK = 252        # trailing trading days for the risk model
TC_CAP = 0.30         # MVO_TC one-way turnover cap per rebalance
PPY = int(os.environ.get("FR_PPY", "12"))   # periods/yr for annualization (12 monthly, 52 weekly)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# --------------------------------------------------------------------------- #
log("loading daily panel + holdings ...")
F, meta = panel.load_fields()
close = F["close"]
dret = close.pct_change()
dpos = {d: i for i, d in enumerate(close.index)}

hold = pd.read_parquet(os.path.join(ML, "holdings.parquet"))
TOP = hold["group"].max()
book = hold[hold.group == TOP].copy()
rebal = sorted(book["date"].unique())
log(f"{len(rebal)} rebalances, long book avg {int(book.groupby('date').size().mean())} names")


def solve(kind, Sigma, alpha, w_prev=None, var_budget=None):
    n = len(alpha)
    w = cp.Variable(n)
    S = cp.psd_wrap(Sigma)
    cons = [w >= 0, cp.sum(w) == 1, w <= CAP]
    if kind == "minvar":
        obj = cp.Minimize(cp.quad_form(w, S))
    elif kind == "mvo":
        cons.append(cp.quad_form(w, S) <= var_budget)
        obj = cp.Maximize(alpha @ w)
    elif kind == "mvo_tc":
        cons += [cp.quad_form(w, S) <= var_budget]
        if w_prev is not None:
            cons.append(0.5 * cp.norm1(w - w_prev) <= TC_CAP)
        obj = cp.Maximize(alpha @ w)
    prob = cp.Problem(obj, cons)
    for solver in (cp.CLARABEL, cp.SCS):        # CLARABEL handles QP + the SOC risk constraint
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None and np.isfinite(w.value).all():
                wv = np.clip(w.value, 0, None)
                return wv / wv.sum() if wv.sum() > 0 else None
        except Exception:
            continue
    return None


schemes = ["EW", "InvVol", "MinVar", "MVO", "MVO_TC"]
rets = {s: {} for s in schemes}
prev_w = {s: None for s in schemes}     # stock->weight from previous month
turn = {s: [] for s in schemes}
effN = {s: [] for s in schemes}
cover = []

for t in rebal:
    sub = book[book.date == t]
    names = sub["stock"].tolist()
    i = dpos[t]
    win = dret.iloc[max(0, i - LOOKBACK + 1):i + 1][names]
    win = win.dropna(axis=1, thresh=int(0.8 * len(win)))
    kept = list(win.columns)
    if len(kept) < 5:
        continue
    R = win[kept].fillna(0.0).to_numpy()
    Sigma = LedoitWolf().fit(R).covariance_
    a = sub.set_index("stock").loc[kept, "pred"].to_numpy(float)
    a = (a - a.mean()) / (a.std() + 1e-9)               # standardize alpha
    y = sub.set_index("stock").loc[kept, "y"].to_numpy(float)
    cover.append(len(kept) / len(names))
    n = len(kept)

    w_ew = np.full(n, 1.0 / n)
    vol = np.sqrt(np.diag(Sigma))
    w_iv = (1 / vol); w_iv = np.minimum(w_iv / w_iv.sum(), CAP); w_iv /= w_iv.sum()
    var_ew = float(w_ew @ Sigma @ w_ew)
    w_mv = solve("minvar", Sigma, a)
    w_opt = solve("mvo", Sigma, a, var_budget=var_ew)
    wprev_vec = None
    if prev_w["MVO_TC"] is not None:
        wprev_vec = np.array([prev_w["MVO_TC"].get(s, 0.0) for s in kept])
        wprev_vec = wprev_vec / wprev_vec.sum() if wprev_vec.sum() > 0 else None
    w_tc = solve("mvo_tc", Sigma, a, w_prev=wprev_vec, var_budget=var_ew)

    W = {"EW": w_ew, "InvVol": w_iv, "MinVar": w_mv, "MVO": w_opt, "MVO_TC": w_tc}
    for s in schemes:
        w = W[s]
        if w is None:
            w = w_ew                                    # fallback
        rets[s][t] = float(w @ y)
        effN[s].append(1.0 / np.sum(w ** 2))
        wd = dict(zip(kept, w))
        if prev_w[s] is not None:
            allk = set(wd) | set(prev_w[s])
            turn[s].append(0.5 * sum(abs(wd.get(k, 0) - prev_w[s].get(k, 0)) for k in allk))
        prev_w[s] = wd

log("optimization done, computing metrics ...")
ret_df = pd.DataFrame(rets).sort_index()
ret_df.index = ret_df.index.astype(str)
ret_df.to_csv(os.path.join(OPT, "portfolio_returns.csv"))


def metrics(r):
    r = r.dropna(); n = len(r); cum = (1 + r).prod()
    ann = cum ** (PPY / n) - 1
    sharpe = r.mean() / r.std() * np.sqrt(PPY) if r.std() > 0 else np.nan
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return dict(ann_return=ann, ann_vol=r.std() * np.sqrt(PPY), sharpe=sharpe,
                max_dd=mdd, win_rate=(r > 0).mean())

met = pd.DataFrame({s: metrics(ret_df[s]) for s in schemes}).T
met["turnover"] = [np.mean(turn[s]) if turn[s] else np.nan for s in schemes]
met["eff_N"] = [np.mean(effN[s]) for s in schemes]
met.to_csv(os.path.join(OPT, "metrics.csv"))

# --------------------------------------------------------------------------- #
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
palette = {"EW": "#94a3b8", "InvVol": "#f59e0b", "MinVar": "#0f766e",
           "MVO": "#2563eb", "MVO_TC": "#7c3aed"}
x = pd.to_datetime(ret_df.index)
fig, ax = plt.subplots(figsize=(9, 4.8))
for s in schemes:
    ax.plot(x, (1 + ret_df[s]).cumprod(), color=palette[s], lw=1.9,
            label=f"{s}  (Sharpe {met.loc[s,'sharpe']:.2f})")
ax.set_yscale("log"); ax.set_ylabel("cumulative net value (log)")
ax.set_title(f"Long-book weighting schemes — OOS ({'full univ.' if TAG else 'ex smallest 10%'})")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OPT, "fig_opt_equity.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.bar(schemes, met["eff_N"], color=[palette[s] for s in schemes])
ax.set_ylabel("avg effective #names (1/Σw²)"); ax.set_title("Diversification by scheme")
fig.tight_layout(); fig.savefig(os.path.join(OPT, "fig_opt_effn.png")); plt.close(fig)

log(f"DONE. avg book coverage {np.mean(cover):.0%}")
print(met.round(4).to_string())
