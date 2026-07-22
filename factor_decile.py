# -*- coding: utf-8 -*-
"""Pure single-factor decile (10-group) backtest — NO machine learning.

Each month, sort stocks by a single factor value into 10 groups, hold each
group equal-weighted for the next month, and cumulate. A strong factor makes
the 10 net-value lines fan out monotonically (D1 lowest factor value ..
D10 highest). Purely factor-driven layered backtest of the strongest Alpha158
and GTJA191 factors.
"""
import os, base64, time
from itertools import groupby
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

import config as C
import panel, alpha158, gtja191

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
NG = 10

# factors to layer-test (from the max-window IC ranking)
A158_F = ["MIN60", "KLEN", "STD10", "QTLD60"]
G191_F = ["alpha070", "alpha095", "alpha042", "alpha159"]
WANT = set(A158_F + G191_F)
FAM = {**{f: "Alpha158" for f in A158_F}, **{f: "GTJA191" for f in G191_F}}
DESC = {"MIN60": "60日最低价/收盘", "KLEN": "(高−低)/开 振幅", "STD10": "10日收益波动",
        "QTLD60": "60日20%分位/收盘", "alpha070": "STD(成交额,6)", "alpha095": "STD(成交额,20)",
        "alpha042": "−RANK(STD(高,10))·CORR(高,量,10)", "alpha159": "KDJ式反转"}

# --------------------------------------------------------------------------- #
log("loading fields ...")
F, meta = panel.load_fields()
close = F["close"]
dates = meta["trade_dates"]
rebal = [list(g)[-1] for _, g in groupby(dates, key=lambda d: d[:6])]   # month-ends
mclose = close.loc[rebal]
fwd = (mclose.shift(-1) / mclose - 1.0)                                 # next-month return
present = mclose.notna()
log(f"{len(rebal)} monthly rebalances, ~{int(present.sum(axis=1).mean())} stocks/mo")

# ---- compute the wanted factors at month-ends (streaming, keep targets) ----
monthly = {}
def sink(name, fr):
    if name in WANT:
        monthly[name] = fr.reindex(index=rebal).astype("float32")
log("computing Alpha158 (monthly slice) ...")
alpha158.compute(F, sink=sink)
log("computing GTJA191 (monthly slice) ...")
gtja191.compute(F, sink=sink)
log(f"got factors: {sorted(monthly.keys())}")

# ---- decile backtest per factor -------------------------------------------
fwd_np = fwd.to_numpy()
def decile_curves(fac_wide):
    fv = fac_wide.where(present).to_numpy(float)
    grp_ret = np.full((len(rebal), NG), np.nan)
    for i in range(len(rebal)):
        row = fv[i]; y = fwd_np[i]
        m = np.isfinite(row) & np.isfinite(y)
        if m.sum() < NG * 10:
            continue
        r = pd.Series(row[m]).rank(method="first")
        q = pd.qcut(r, NG, labels=False).to_numpy()
        yy = y[m]
        for g in range(NG):
            grp_ret[i, g] = yy[q == g].mean()
    curves = np.nancumprod(1 + np.nan_to_num(grp_ret), axis=0)
    return grp_ret, curves

results = {name: decile_curves(monthly[name]) for name in (A158_F + G191_F) if name in monthly}

# ---- plot 2 x 4 grid -------------------------------------------------------
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .22,
                     "axes.spines.top": False, "axes.spines.right": False})
order = A158_F + G191_F
fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
x = pd.to_datetime(pd.Series(rebal).astype(str))
cmap = plt.cm.RdYlGn(np.linspace(0.08, 0.95, NG))       # D1 red .. D10 green
for ax, name in zip(axes.ravel(), order):
    if name not in results:
        ax.axis("off"); continue
    grp_ret, curves = results[name]
    for g in range(NG):
        ax.plot(x, curves[:, g], color=cmap[g], lw=1.3)
    # long-short D10-D1 monthly returns -> annualized & monotonicity
    ls = np.nanmean(grp_ret[:, NG - 1] - grp_ret[:, 0]) * 12
    finals = curves[-1]
    mono = pd.Series(finals).corr(pd.Series(range(NG)), method="spearman")
    ax.set_yscale("log")
    ax.set_title(f"{name}  ({FAM[name]})\n{DESC.get(name,'')}", fontsize=10)
    ax.text(0.03, 0.97, f"D10−D1 年化 {ls*100:+.0f}%\n单调性 ρ={mono:+.2f}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=.85))
axes[0, 0].set_ylabel("累计净值（对数）"); axes[1, 0].set_ylabel("累计净值（对数）")
fig.suptitle("单因子 10 分层回测（月度，全成分股，等权，D1低因子值→D10高因子值）  上排 Alpha158 · 下排 GTJA191",
             fontsize=13, y=1.01)
fig.tight_layout()
figpath = os.path.join(C.RESULTS_DIR, "fig_factor_deciles.png")
fig.savefig(figpath, bbox_inches="tight"); plt.close(fig)
log("saved " + figpath)

# ---- HTML page -------------------------------------------------------------
img = "data:image/png;base64," + base64.b64encode(open(figpath, "rb").read()).decode()
HTML = f"""
<style>
:root{{--bg:#f6f7f9;--panel:#fff;--ink:#0f1720;--muted:#5b6675;--line:#e3e7ec;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0c1119;--panel:#131b26;--ink:#e8edf3;--muted:#93a1b3;--line:#22303f;}}}}
:root[data-theme="dark"]{{--bg:#0c1119;--panel:#131b26;--ink:#e8edf3;--muted:#93a1b3;--line:#22303f;}}
:root[data-theme="light"]{{--bg:#f6f7f9;--panel:#fff;--ink:#0f1720;--muted:#5b6675;--line:#e3e7ec;}}
body{{margin:0;background:var(--bg);color:var(--ink);line-height:1.6;
font-family:-apple-system,"Segoe UI",system-ui,"Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1180px;margin:0 auto;padding:clamp(20px,4vw,44px) clamp(14px,3vw,32px);}}
h1{{font-size:clamp(20px,3vw,28px);font-weight:800;margin:0 0 6px;}}
.sub{{color:var(--muted);margin:0 0 20px;max-width:80ch;}}
img{{width:100%;height:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel);}}
.note{{color:var(--muted);font-size:13.5px;margin-top:16px;max-width:82ch}}.note b{{color:var(--ink)}}
</style>
<div class="wrap">
<h1>单因子 10 分层回测（纯因子，无模型）</h1>
<p class="sub">每月按单个因子值把全部成分股分成 10 组（D1=因子值最低，D10=最高），每组等权、持有到下月，
累计净值。<b>好因子 → 10 条线单调张成扇形</b>。上排 Alpha158，下排 GTJA191，取各族最有效的 4 个。</p>
<img src="{img}" alt="factor deciles">
<p class="note"><b>怎么看：</b>每格 10 条线（红=D1 低因子值 → 绿=D10 高因子值）。
<b>D10−D1 年化</b>=顶组减底组的多空年化；<b>单调性 ρ</b>=10 组终值与组号的秩相关（越接近 ±1 越单调）。
正 IC 因子（MIN60/QTLD60/alpha042）绿在上；负 IC 因子（KLEN/STD10/alpha070/095/159）红在上。
税前、等权、未计成本。</p>
</div>
"""
open(os.path.join(C.RESULTS_DIR, "factor_deciles.html"), "w", encoding="utf-8").write(HTML)
log("wrote factor_deciles.html")
# print monotonicity summary
for name in order:
    if name in results:
        gr, cv = results[name]
        ls = np.nanmean(gr[:, NG-1] - gr[:, 0]) * 12
        mono = pd.Series(cv[-1]).corr(pd.Series(range(NG)), method="spearman")
        print(f"{name:9s} {FAM[name]:8s}  D10-D1 ann={ls*100:+5.1f}%  mono={mono:+.2f}  D10/D1={cv[-1,NG-1]/cv[-1,0]:.1f}x")
