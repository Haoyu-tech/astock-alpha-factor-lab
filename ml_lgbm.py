# -*- coding: utf-8 -*-
"""Walk-forward LightGBM on Alpha158 -> 20-group long-short / long-only backtest.

Scheme (all out-of-sample):
  * rolling 36-month train  ->  predict next 12 months  ->  step 12 months
  * within each train window the last 6 months are a validation set used for
    early stopping and a small hyper-parameter grid (chosen by validation IC)
  * label y = next-month return (winsorised per date for training only)
  * features cross-sectionally z-scored per date
  * each test month: sort predictions into 20 equal groups; Long = G20, Short = G1
Outputs (results/ml/): predictions+holdings, monthly returns, group means,
metrics, and charts.
"""
import os, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

import config as C

TAG = os.environ.get("FR_TAG", "")
ML = os.path.join(C.RESULTS_DIR, f"ml{TAG}"); os.makedirs(ML, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

TRAIN = int(os.environ.get("FR_TRAIN", "36"))    # periods in the training window
TEST = int(os.environ.get("FR_TEST", "12"))
STEP = int(os.environ.get("FR_STEP", "12"))
VALID = int(os.environ.get("FR_VALID", "6"))
PPY = int(os.environ.get("FR_PPY", "12"))        # periods per year (12 monthly, 52 weekly)
N_GROUPS = 20
A158C, LSC, BMC = "#2563eb", "#0f766e", "#94a3b8"

# --------------------------------------------------------------------------- #
log("loading monthly dataset ...")
df = pd.read_parquet(os.path.join(C.DATA_DIR, f"monthly_dataset{TAG}.parquet"))
feat = pd.read_csv(os.path.join(C.DATA_DIR, f"feat_cols{TAG}.csv"), header=None)[0].tolist()
df = df.sort_values(["date", "stock"]).reset_index(drop=True)
log(f"rows={len(df)}, feats={len(feat)}, months={df['date'].nunique()}")

# ---- cross-sectional feature normalisation (per date, robust z, clip) ------
g = df.groupby("date")[feat]
df[feat] = ((df[feat] - g.transform("mean")) / g.transform("std").replace(0, np.nan)) \
    .clip(-3, 3).fillna(0.0)
# training label: CROSS-SECTIONAL z-score of y per date (learn relative alpha,
# not market timing); clip tails. Backtest still uses the raw return y.
df["y_cs"] = df.groupby("date")["y"].transform(
    lambda x: ((x - x.mean()) / x.std()) if x.std() > 0 else x * 0).clip(-4, 4)

rebal = sorted(df["date"].unique())
log(f"usable months: {len(rebal)} ({rebal[0]}..{rebal[-1]})")

BASE = dict(objective="regression", metric="None", feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, min_child_samples=100,
            lambda_l2=1.0, verbose=-1, num_threads=4, seed=7)
GRID = [dict(num_leaves=nl, learning_rate=lr)
        for nl in (15, 31, 63) for lr in (0.03, 0.05)]


def date_ic(sub, pcol="pred"):
    ics = []
    for _, s in sub.groupby("date"):
        if len(s) > 20 and s[pcol].std() > 0:
            ics.append(spearmanr(s[pcol], s["y"]).correlation)
    return np.nanmean(ics) if ics else np.nan


def make_ic_feval(dates_arr, y_raw):
    """LightGBM eval: mean per-date Spearman IC of preds vs raw return (maximize)."""
    tmp = pd.DataFrame({"d": dates_arr, "y": y_raw})
    groups = [g.index.to_numpy() for _, g in tmp.groupby("d")]

    def feval(preds, dset):
        ics = []
        for idx in groups:
            p = preds[idx]
            if p.std() > 0:
                ics.append(spearmanr(p, tmp["y"].to_numpy()[idx]).correlation)
        ic = np.nanmean(ics) if ics else 0.0
        return "IC", (ic if np.isfinite(ic) else 0.0), True
    return feval


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
preds = []
fold_log = []
i = 0
while i + TRAIN < len(rebal):
    tr_dates = rebal[i:i + TRAIN]
    te_dates = rebal[i + TRAIN:i + TRAIN + TEST]
    if not te_dates:
        break
    fit_d, val_d = tr_dates[:-VALID], tr_dates[-VALID:]
    tr = df[df.date.isin(fit_d)]
    val = df[df.date.isin(val_d)].reset_index(drop=True)
    te = df[df.date.isin(te_dates)]
    dtr = lgb.Dataset(tr[feat], tr["y_cs"])
    dval = lgb.Dataset(val[feat], val["y_cs"])
    feval = make_ic_feval(val["date"].to_numpy(), val["y"].to_numpy())
    best = None
    for gp in GRID:
        params = {**BASE, **gp}
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dval], feval=feval,
                      callbacks=[lgb.early_stopping(80, verbose=False),
                                 lgb.log_evaluation(0)])
        vpred = val.assign(pred=m.predict(val[feat], num_iteration=m.best_iteration))
        vic = date_ic(vpred)
        if best is None or (np.isfinite(vic) and vic > best[0]):
            best = (vic, m, gp)
    vic, model, gp = best
    tp = te.assign(pred=model.predict(te[feat], num_iteration=model.best_iteration))
    preds.append(tp[["date", "stock", "pred", "y"]])
    fold_log.append(dict(train=f"{tr_dates[0]}..{tr_dates[-1]}",
                         test=f"{te_dates[0]}..{te_dates[-1]}",
                         val_IC=round(vic, 4), num_leaves=gp["num_leaves"],
                         lr=gp["learning_rate"], best_iter=model.best_iteration,
                         test_months=len(te_dates)))
    log(f"  fold {len(fold_log)}: test {te_dates[0]}..{te_dates[-1]}  "
        f"valIC={vic:.3f}  nl={gp['num_leaves']} lr={gp['learning_rate']} it={model.best_iteration}")
    i += STEP

pred = pd.concat(preds, ignore_index=True)
pd.DataFrame(fold_log).to_csv(os.path.join(ML, "folds.csv"), index=False)
log(f"OOS predictions: {len(pred)} rows over {pred['date'].nunique()} months")

# --------------------------------------------------------------------------- #
# 20-group backtest
# --------------------------------------------------------------------------- #
def assign_groups(s):
    r = s["pred"].rank(method="first")
    return pd.qcut(r, N_GROUPS, labels=False)

pred["group"] = pred.groupby("date", group_keys=False).apply(
    lambda s: pd.Series(assign_groups(s), index=s.index))
pred["group"] = pred["group"].astype(int)

# per-date group mean returns
gm = pred.groupby(["date", "group"])["y"].mean().unstack("group")
monthly = pd.DataFrame(index=gm.index)
monthly["long"] = gm[N_GROUPS - 1]
monthly["short"] = gm[0]
monthly["ls"] = monthly["long"] - monthly["short"]
monthly["bench"] = pred.groupby("date")["y"].mean()
monthly = monthly.sort_index()

# IC
ic_series = pred.groupby("date").apply(
    lambda s: spearmanr(s["pred"], s["y"]).correlation if s["pred"].std() > 0 else np.nan)
IC_mean = ic_series.mean(); IC_ir = IC_mean / ic_series.std()
IC_t = IC_ir * np.sqrt(ic_series.notna().sum())

# turnover of the long book (G20)
long_book = {d: set(s.loc[s.group == N_GROUPS - 1, "stock"]) for d, s in pred.groupby("date")}
ds = sorted(long_book)
turn = [1 - len(long_book[ds[k]] & long_book[ds[k-1]]) / max(len(long_book[ds[k]]), 1)
        for k in range(1, len(ds))]
turnover = float(np.mean(turn))


def metrics(r):
    r = r.dropna(); n = len(r)
    cum = (1 + r).prod()
    ann = cum ** (PPY / n) - 1
    vol = r.std() * np.sqrt(PPY)
    sharpe = r.mean() / r.std() * np.sqrt(PPY) if r.std() > 0 else np.nan
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return dict(ann_return=ann, ann_vol=vol, sharpe=sharpe, max_dd=dd,
                win_rate=(r > 0).mean(), months=n)

met = {k: metrics(monthly[k]) for k in ["ls", "long", "bench"]}
met_df = pd.DataFrame(met).T
met_df["IC_mean"] = [IC_mean, IC_mean, np.nan]
met_df["ICIR"] = [IC_ir, IC_ir, np.nan]
met_df["turnover"] = [turnover, turnover, np.nan]
met_df.to_csv(os.path.join(ML, "metrics.csv"))

group_means = pred.groupby("group")["y"].mean()
group_means.to_csv(os.path.join(ML, "group_means.csv"))
monthly.to_csv(os.path.join(ML, "monthly_returns.csv"))
pred.to_parquet(os.path.join(ML, "holdings.parquet"))          # per-period holdings
ic_series.rename("IC").to_csv(os.path.join(ML, "ic_series.csv"))

# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
x = pd.to_datetime(monthly.index.astype(str))

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(x, (1 + monthly["ls"]).cumprod(), color=LSC, lw=2, label="Long-Short (G20−G1)")
ax.plot(x, (1 + monthly["long"]).cumprod(), color=A158C, lw=2, label="Long-only (G20)")
ax.plot(x, (1 + monthly["bench"]).cumprod(), color=BMC, lw=1.6, label="Universe (EW)")
ax.set_yscale("log"); ax.set_ylabel("cumulative net value (log)")
ax.set_title("LightGBM · Alpha158 · 中证1000 — out-of-sample equity")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(ML, "fig_equity.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, N_GROUPS))
ax.bar(range(N_GROUPS), group_means.values * 100, color=colors)
ax.set_xlabel("prediction group (0 = lowest → 19 = highest)")
ax.set_ylabel("mean monthly return (%)")
ax.set_title("Monotonicity — mean next-month return by predicted group")
fig.tight_layout(); fig.savefig(os.path.join(ML, "fig_groups.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.4))
eq = (1 + monthly["ls"]).cumprod(); dd = eq / eq.cummax() - 1
ax.fill_between(x, dd * 100, 0, color=LSC, alpha=.35)
ax.set_ylabel("Long-Short drawdown (%)"); ax.set_title("Long-Short drawdown")
fig.tight_layout(); fig.savefig(os.path.join(ML, "fig_drawdown.png")); plt.close(fig)

log("DONE. metrics:")
print(met_df.round(4).to_string())
print(f"\nIC_mean={IC_mean:.4f}  ICIR={IC_ir:.3f}  t={IC_t:.1f}  turnover={turnover:.2%}")
print("group monotonicity (mean monthly %):",
      (group_means.values * 100).round(2).tolist())
