# -*- coding: utf-8 -*-
"""Generate charts + a data bundle from the results CSVs for the report."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import config as C

# try to render Chinese labels if a CJK font exists, else fall back silently
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

R = C.RESULTS_DIR
A158, G191 = "#2563eb", "#f59e0b"          # family colors
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def _read(name):
    return pd.read_csv(os.path.join(R, name))


def fig_family_ic_hist(overall):
    fig, ax = plt.subplots(figsize=(7, 4))
    for fam, col in [("Alpha158", A158), ("GTJA191", G191)]:
        v = overall.loc[overall.family == fam, "IC5"].dropna()
        ax.hist(v, bins=40, alpha=0.6, color=col, label=f"{fam} (n={len(v)})")
    ax.axvline(0, color="#334155", lw=1)
    ax.set_xlabel("Rank IC @ 5-day horizon"); ax.set_ylabel("factor count")
    ax.set_title("Distribution of factor IC by family")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_ic_hist.png")); plt.close(fig)


def fig_mktcap(mc_family_mean):
    labels = list(mc_family_mean.columns)
    disp = [l.replace("abs_", "") for l in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    if len(labels) > 4:
        # deciles -> line/marker gradient reads cleaner than 20 bars
        ax.plot(x, mc_family_mean.loc["Alpha158"], "-o", color=A158, lw=2, ms=5, label="Alpha158")
        ax.plot(x, mc_family_mean.loc["GTJA191"], "-o", color=G191, lw=2, ms=5, label="GTJA191")
        ax.set_xticks(x); ax.set_xticklabels([d.replace("D", "").lstrip("0") or "0" for d in disp])
        ax.set_xlabel(f"market-cap decile   (1 = smallest  →  {len(labels)} = largest)")
        ax.margins(x=0.02)
    else:
        w = 0.38
        ax.bar(x - w/2, mc_family_mean.loc["Alpha158"], w, color=A158, label="Alpha158")
        ax.bar(x + w/2, mc_family_mean.loc["GTJA191"], w, color=G191, label="GTJA191")
        ax.set_xticks(x); ax.set_xticklabels(disp)
    ax.set_ylabel("mean |IC@5| across factors")
    ax.set_ylim(bottom=0)
    ax.set_title("Factor effectiveness by market-cap decile")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_mktcap.png")); plt.close(fig)


def fig_index(idx_family_mean):
    labels = list(idx_family_mean.columns)
    disp = [C.INDEX_LABELS.get(l.replace("abs_", ""), l) for l in labels]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w/2, idx_family_mean.loc["Alpha158"], w, color=A158, label="Alpha158")
    ax.bar(x + w/2, idx_family_mean.loc["GTJA191"], w, color=G191, label="GTJA191")
    ax.set_xticks(x); ax.set_xticklabels(disp)
    ax.set_ylabel("mean |IC@5| across factors")
    ax.set_title("Factor effectiveness by index constituent group")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_index.png")); plt.close(fig)


def fig_corr(corr):
    # order by family then reduce to a readable subset (every k-th factor)
    names = list(corr.columns)
    k = max(1, len(names) // 80)
    sel = names[::k]
    m = corr.loc[sel, sel].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"Factor correlation (subsample {len(sel)} of {len(names)})")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman corr")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_corr.png")); plt.close(fig)


def fig_ic_horizon(hz, horizons=(1, 2, 3, 5, 10, 15, 20, 30, 40, 60)):
    """IC decay vs holding horizon + cost-free efficiency |IC|/sqrt(h)."""
    H = list(horizons)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for famname, c in [("Alpha158", A158), ("GTJA191", G191)]:
        sub = hz[hz.family == famname]
        y = np.array([sub[f"IC_{h}"].abs().mean() for h in H])
        ax1.plot(H, y, "-o", color=c, ms=5, label=famname)
        eff = y / np.sqrt(H)
        ax2.plot(H, eff / eff[0], "-o", color=c, ms=5, label=famname)
    yall = np.array([hz[f"IC_{h}"].abs().mean() for h in H])
    ax1.plot(H, yall, "--", color="#64748b", lw=1.3, label="all")
    peak = H[int(yall.argmax())]
    ax1.axvline(peak, color="#94a3b8", ls=":", lw=1)
    ax1.annotate(f"|IC| peaks ~{peak}d", (peak, yall.max()), fontsize=9,
                 xytext=(6, -4), textcoords="offset points", color="#475569")
    effall = (yall / np.sqrt(H)); effall = effall / effall[0]
    epeak = H[int(effall.argmax())]
    ax2.axhline(1.0, color="#cbd5e1", lw=1)
    ax2.annotate(f"efficiency peaks ~{epeak}d", (epeak, effall.max()), fontsize=9,
                 xytext=(6, 4), textcoords="offset points", color="#475569")
    ax1.set_xlabel("holding horizon (days)"); ax1.set_ylabel("mean |IC|")
    ax1.set_title("Signal persistence — |IC| vs horizon"); ax1.grid(alpha=.25); ax1.legend()
    ax2.set_xlabel("holding horizon (days)"); ax2.set_ylabel("|IC| / √h   (rel. to 1-day)")
    ax2.set_title("Cost-free per-period efficiency"); ax2.grid(alpha=.25); ax2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_ic_horizon.png")); plt.close(fig)


def fig_cumic(overall, n=10):
    """Cumulative sign-aligned rank-IC curves for the top-|IC| factors."""
    dic = pd.read_parquet(os.path.join(R, "daily_ic.parquet"))
    x = pd.to_datetime(dic.index.astype(str))
    fam = dict(zip(overall.factor, overall.family))
    top = overall.reindex(overall["IC5"].abs().sort_values(ascending=False).index)
    top = [f for f in top.factor if f in dic.columns][:n]
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, f in enumerate(top):
        s = dic[f].to_numpy(float)
        s = s * np.sign(np.nanmean(s))                 # align so all curves rise
        cum = np.nancumsum(np.where(np.isfinite(s), s, 0.0))
        tag = "158" if fam.get(f) == "Alpha158" else "GTJA"
        ax.plot(x, cum, lw=1.6, color=palette[i], label=f"{f} ({tag})")
    ax.set_ylabel("cumulative sign-aligned rank-IC")
    ax.set_title("Cumulative IC — steeper = stronger, straighter = steadier")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=.9)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_cumic.png")); plt.close(fig)


def fig_cumic_family(overall):
    """Family-average cumulative sign-aligned IC (breadth of each library)."""
    dic = pd.read_parquet(os.path.join(R, "daily_ic.parquet"))
    x = pd.to_datetime(dic.index.astype(str))
    fam = dict(zip(overall.factor, overall.family))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for family_name, col in [("Alpha158", A158), ("GTJA191", G191)]:
        cols = [f for f in dic.columns if fam.get(f) == family_name]
        aligned = []
        for f in cols:
            s = dic[f].to_numpy(float)
            aligned.append(np.where(np.isfinite(s), s * np.sign(np.nanmean(s)), 0.0))
        mean_daily = np.nanmean(np.vstack(aligned), axis=0)
        ax.plot(x, np.cumsum(mean_daily), lw=2, color=col, label=f"{family_name} (n={len(cols)})")
    ax.set_ylabel("cumulative mean |aligned IC|")
    ax.set_title("Cumulative IC per family — average factor, sign-aligned")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_cumic_family.png")); plt.close(fig)


def fig_yearly_heatmap(yearly, n=22):
    years = [str(y) for y in range(2015, 2027)]
    cols = [f"IC_{y}" for y in years if f"IC_{y}" in yearly.columns]
    top = yearly.reindex(yearly["IC_recent3y"].abs().sort_values(ascending=False).index).head(n)
    M = top[cols].to_numpy(float)
    labels = top["factor"].tolist()
    fams = top["family"].tolist()
    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 3.2, 0.40 * len(labels) + 1.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.10, vmax=0.10, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[3:] for c in cols])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, family="monospace")
    for tick, fam in zip(ax.get_yticklabels(), fams):
        tick.set_color(A158 if fam == "Alpha158" else G191)
    for i in range(len(labels)):
        for j in range(len(cols)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v*100:+.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.055 else "#30363d")
    ax.set_title("Yearly rank-IC ×100  ·  top factors by recent |IC|  (blue = +, red = −)")
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0); ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label="rank-IC")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_yearly_heatmap.png")); plt.close(fig)


def fig_recent_top(yearly, n=15):
    top = yearly.reindex(yearly["IC_recent3y"].abs().sort_values(ascending=False).index).head(n).iloc[::-1]
    cols = [A158 if f == "Alpha158" else G191 for f in top.family]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.factor, top.IC_recent3y, color=cols)
    ax.axvline(0, color="#334155", lw=1)
    ax.set_xlabel("rank IC, recent 3y (2023–2026)")
    ax.set_title(f"Strongest factors lately — top {n} by recent |IC|")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_recent_top.png")); plt.close(fig)


def fig_top_factors(overall, n=15):
    top = overall.head(n).iloc[::-1]
    cols = [A158 if f == "Alpha158" else G191 for f in top.family]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.factor, top.IC5, color=cols)
    ax.axvline(0, color="#334155", lw=1)
    ax.set_xlabel("Rank IC @ 5-day"); ax.set_title(f"Top {n} factors by |IC@5|")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_top_factors.png")); plt.close(fig)


def main():
    overall = _read("factor_ic_overall.csv")
    mc_fm = _read("mktcap_group_family_mean.csv").set_index("family")
    idx_fm = _read("index_group_family_mean.csv").set_index("family")
    corr = pd.read_csv(os.path.join(R, "factor_correlation.csv"), index_col=0)

    fig_family_ic_hist(overall)
    fig_mktcap(mc_fm)
    fig_index(idx_fm)
    fig_corr(corr)
    fig_top_factors(overall)
    yf = os.path.join(R, "factor_ic_yearly.csv")
    if os.path.exists(yf):
        yearly = pd.read_csv(yf)
        fig_yearly_heatmap(yearly)
        fig_recent_top(yearly)
    if os.path.exists(os.path.join(R, "daily_ic.parquet")):
        fig_cumic(overall)
        fig_cumic_family(overall)
    hzf = os.path.join(R, "factor_ic_horizon.csv")
    if os.path.exists(hzf):
        fig_ic_horizon(pd.read_csv(hzf))
    print("charts written to", R)


if __name__ == "__main__":
    main()
