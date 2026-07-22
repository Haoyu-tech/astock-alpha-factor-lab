# -*- coding: utf-8 -*-
"""Assemble the LightGBM / Alpha158 strategy tear-sheet (body-only HTML)."""
import os, base64
import numpy as np
import pandas as pd
import config as C

TAG = os.environ.get("FR_TAG", "")
ML = os.path.join(C.RESULTS_DIR, f"ml{TAG}")
OUT = os.path.join(ML, "report_ml.html")


def b64(p):
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


met = pd.read_csv(os.path.join(ML, "metrics.csv"), index_col=0)
folds = pd.read_csv(os.path.join(ML, "folds.csv"))
monthly = pd.read_csv(os.path.join(ML, "monthly_returns.csv"))
hold = pd.read_parquet(os.path.join(ML, "holdings.parquet"))
imgs = {k: b64(os.path.join(ML, f"fig_{k}.png")) for k in ["equity", "groups", "drawdown"]}

# ---- portfolio weight optimization (optional) ------------------------------
OPT_SECTION = ""
_optm = os.path.join(ML, "opt", "metrics.csv")
if os.path.exists(_optm):
    om = pd.read_csv(_optm, index_col=0)
    oimg = {k: b64(os.path.join(ML, "opt", f"fig_{k}.png")) for k in ["opt_equity", "opt_effn"]}
    order = ["EW", "InvVol", "MinVar", "MVO", "MVO_TC"]
    olab = {"EW": "Equal weight (baseline)", "InvVol": "Inverse-volatility",
            "MinVar": "Minimum variance", "MVO": "MVO — α-tilt @ EW risk",
            "MVO_TC": "MVO + turnover cap"}
    orows = ""
    for s in order:
        if s in om.index:
            r = om.loc[s]
            hl = ' style="background:color-mix(in srgb,var(--accent) 8%,transparent)"' if s == "MVO" else ""
            orows += (f'<tr{hl}><td>{olab[s]}</td><td class="num">{r.ann_return*100:+.1f}%</td>'
                      f'<td class="num">{r.ann_vol*100:.1f}%</td><td class="num">{r.sharpe:.2f}</td>'
                      f'<td class="num neg">{r.max_dd*100:.1f}%</td>'
                      f'<td class="num">{r.turnover*100:.0f}%</td><td class="num">{r.eff_N:.0f}</td></tr>')
    EWR = om.loc["EW", "ann_return"]; MVR = om.loc["MVO", "ann_return"]
    EWS = om.loc["EW", "sharpe"]; MVS = om.loc["MVO", "sharpe"]
    OPT_SECTION = f"""
<section>
<h2><span class="idx">06</span>Portfolio weight optimization</h2>
<p class="sub">The long book above is equal-weighted. Here we instead optimize the weights of
the selected names each rebalance — a <b>Ledoit-Wolf shrinkage covariance</b> from 252 trailing daily
returns as the risk model, and the ML score as expected alpha. All schemes are <b>long-only,
fully invested, per-name cap 8%</b>.</p>
<figure><img src="{oimg['opt_equity']}" alt="weighting schemes equity">
<figcaption>Long-only cumulative net value under each weighting scheme (log).</figcaption></figure>
<div class="panel" style="margin-top:20px"><div class="panel-h">Long-only by weighting scheme (OOS, same selected names)</div>
<div class="tbl-scroll"><table>
<thead><tr><th>Scheme</th><th>Ann. return</th><th>Ann. vol</th><th>Sharpe</th><th>Max DD</th><th>Turnover</th><th>Eff. #</th></tr></thead>
<tbody>{orows}</tbody></table></div></div>
<div class="callout"><b>MVO adds ~{(MVR-EWR)*100:.0f}pp/yr</b> ({EWR*100:.1f}% → {MVR*100:.1f}%, Sharpe
{EWS:.2f} → {MVS:.2f}) by tilting into the highest-alpha names <i>at the same risk budget as
equal-weight</i> — concentrating to ~{om.loc['MVO','eff_N']:.0f} effective names. Minimum-variance
throws the alpha away (return halves); inverse-vol ≈ equal-weight. The turnover-capped MVO keeps
most of the gain while staying diversified — the practical choice.</div>
</section>
"""

oos_start = str(monthly["date"].min()); oos_end = str(monthly["date"].max())
n_months = len(monthly)
FREQ = os.environ.get("FR_FREQ", "M")
PSING = "week" if FREQ == "W" else "month"          # singular
PLABEL = "weeks" if FREQ == "W" else "months"       # plural
FREQADV = "weekly" if FREQ == "W" else "monthly"
PABBR = "wk" if FREQ == "W" else "mo"
TRAIN_ = os.environ.get("FR_TRAIN", "36"); TEST_ = os.environ.get("FR_TEST", "12")
VALID_ = os.environ.get("FR_VALID", "6")
PPY_ = int(os.environ.get("FR_PPY", "12"))
avg_long = int(hold[hold.group == hold.group.max()].groupby("date").size().mean())
IC_mean = met.loc["ls", "IC_mean"]; ICIR = met.loc["ls", "ICIR"]; TURN = met.loc["ls", "turnover"]
TURN_ANN = TURN * PPY_                               # annualized one-way turnover
FREQ_CN = "周频" if FREQ == "W" else "月频"
UNIV_LABEL = {
    "": "中证1000（剔最小10%市值）",
    "_full": "中证1000（全，不剔微盘）",
    "_broad": "宽基（沪深300+中证500+中证1000+创业板）",
    "_all": "全市场 / 中证全指（含微盘）",
    "_wk": "中证1000（剔最小10%市值）",
}.get(TAG, "中证1000")
UNIV_EN = {
    "": "CSI1000 ex the smallest 10% by market cap",
    "_full": "the full CSI1000 (no market-cap filter)",
    "_broad": "the union of CSI300 / CSI500 / CSI1000 / ChiNext",
    "_all": "the entire A-share market (CSI All Share, micro-caps included)",
    "_wk": "CSI1000 ex the smallest 10% by market cap",
}.get(TAG, "CSI1000")


def _pct(x): return f"{x*100:+.1f}%"
def mrow(name, key, label):
    r = met.loc[key]
    return (f'<tr><td>{label}</td><td class="num">{_pct(r.ann_return)}</td>'
            f'<td class="num">{r.ann_vol*100:.1f}%</td>'
            f'<td class="num">{r.sharpe:.2f}</td>'
            f'<td class="num neg">{r.max_dd*100:.1f}%</td>'
            f'<td class="num">{r.win_rate*100:.0f}%</td></tr>')

metrics_rows = (mrow("ls", "ls", "Long-Short (G20−G1)") +
                mrow("long", "long", "Long-only (G20)") +
                mrow("bench", "bench", "Universe (equal-weight)"))

fold_rows = ""
for _, r in folds.iterrows():
    fold_rows += (f'<tr><td class="mono">{r.train}</td><td class="mono">{r.test}</td>'
                  f'<td class="num">{r.val_IC:+.3f}</td><td class="num">{int(r.num_leaves)}</td>'
                  f'<td class="num">{r.lr}</td><td class="num">{int(r.best_iter)}</td></tr>\n')

# latest-month long book sample
last_d = hold["date"].max()
lastbook = hold[(hold.date == last_d) & (hold.group == hold.group.max())].sort_values("pred", ascending=False)
book_names = ", ".join(lastbook["stock"].head(15).tolist())

LS = met.loc["ls"]; LO = met.loc["long"]

CSS = """
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#0f1720;--muted:#5b6675;--line:#e3e7ec;--line2:#eef1f5;
--long:#2563eb;--ls:#0f766e;--bench:#94a3b8;--pos:#15803d;--neg:#c02636;--accent:#0f766e;
--shadow:0 1px 2px rgba(16,23,32,.04),0 8px 24px rgba(16,23,32,.06);}
@media (prefers-color-scheme:dark){:root{--bg:#0c1119;--panel:#131b26;--ink:#e8edf3;--muted:#93a1b3;
--line:#22303f;--line2:#1a2531;--long:#5b9bff;--ls:#2dd4bf;--bench:#64748b;--pos:#43c96f;--neg:#ff6b7a;--accent:#2dd4bf;
--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}
:root[data-theme="dark"]{--bg:#0c1119;--panel:#131b26;--ink:#e8edf3;--muted:#93a1b3;--line:#22303f;
--line2:#1a2531;--long:#5b9bff;--ls:#2dd4bf;--bench:#64748b;--pos:#43c96f;--neg:#ff6b7a;--accent:#2dd4bf;
--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}
:root[data-theme="light"]{--bg:#f6f7f9;--panel:#fff;--ink:#0f1720;--muted:#5b6675;--line:#e3e7ec;
--line2:#eef1f5;--long:#2563eb;--ls:#0f766e;--bench:#94a3b8;--pos:#15803d;--neg:#c02636;--accent:#0f766e;
--shadow:0 1px 2px rgba(16,23,32,.04),0 8px 24px rgba(16,23,32,.06);}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.6;
font-family:-apple-system,"Segoe UI",system-ui,"Microsoft YaHei",sans-serif;-webkit-font-smoothing:antialiased;}
.wrap{max-width:980px;margin:0 auto;padding:clamp(20px,5vw,60px) clamp(16px,4vw,40px);}
.mono{font-family:"Cascadia Code","SF Mono",Consolas,ui-monospace,monospace;}
.num{font-variant-numeric:tabular-nums;text-align:right;}
.eyebrow{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);font-weight:700;
font-family:"Cascadia Code",monospace;}
h1{font-size:clamp(28px,4.6vw,42px);line-height:1.1;font-weight:800;letter-spacing:-.02em;margin:.3em 0 .3em;text-wrap:balance;}
.lede{font-size:clamp(15px,2vw,18px);color:var(--muted);max-width:66ch;margin:0;}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px;}
.chip{font-size:12.5px;padding:5px 11px;border:1px solid var(--line);border-radius:100px;color:var(--muted);background:var(--panel);}
.chip b{color:var(--ink);}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:26px 0;}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow);}
.stat .k{font-size:12px;color:var(--muted);margin-bottom:6px;}
.stat .v{font-size:29px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;}
.stat .s{font-size:12.5px;color:var(--muted);margin-top:3px;}
section{margin:44px 0;}
h2{font-size:22px;font-weight:750;margin:0 0 4px;display:flex;gap:11px;align-items:baseline;}
h2 .idx{font-family:"Cascadia Code",monospace;font-size:13px;color:var(--accent);font-weight:700;}
.sub{color:var(--muted);margin:0 0 16px;max-width:70ch;}
p{max-width:70ch;}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);overflow:hidden;}
.panel-h{padding:13px 18px;border-bottom:1px solid var(--line);font-size:13px;color:var(--muted);}
.tbl-scroll{overflow-x:auto;}
table{width:100%;border-collapse:collapse;font-size:14px;}
th,td{padding:10px 14px;border-bottom:1px solid var(--line2);white-space:nowrap;}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600;text-align:right;}
th:first-child,td:first-child{text-align:left;}
tbody tr:last-child td{border-bottom:0;}
td.pos{color:var(--pos);}td.neg{color:var(--neg);}
figure{margin:18px 0 0;}figure img{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:12px;background:var(--panel);}
figcaption{font-size:12.5px;color:var(--muted);margin-top:8px;text-align:center;}
.callout{border-left:3px solid var(--accent);background:color-mix(in srgb,var(--accent) 8%,transparent);
padding:12px 16px;border-radius:0 10px 10px 0;margin:16px 0;font-size:14.5px;}
ul.tight{padding-left:20px;max-width:70ch;}ul.tight li{margin:5px 0;}
code{font-family:"Cascadia Code",Consolas,monospace;font-size:.9em;background:var(--line2);padding:1px 6px;border-radius:5px;}
.foot{color:var(--muted);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:18px;}
</style>
"""

HTML = f"""
{CSS}
<div class="wrap">
<div class="eyebrow">机器学习选股 · WALK-FORWARD BACKTEST</div>
<h1>LightGBM × Alpha158 — {UNIV_LABEL} {FREQ_CN}选股</h1>
<p class="lede">A 158-factor LightGBM model, walk-forward trained (3-year window, early stopping)
and tested fully out-of-sample: {FREQADV} rebalancing, 20 prediction groups, long-short and
long-only, on {UNIV_EN}.</p>
<div class="meta">
<span class="chip">OOS <b>{oos_start[:6]}–{oos_end[:6]}</b></span>
<span class="chip"><b>{n_months}</b> {PLABEL}</span>
<span class="chip">Universe ≈ <b>{avg_long*20}</b>/{PABBR}</span>
<span class="chip">Model <b>LightGBM</b></span>
<span class="chip">Rebalance <b>{FREQADV}</b></span>
</div>

<div class="stats">
<div class="stat"><div class="k">Long-Short ann. return</div><div class="v">{_pct(LS.ann_return)}</div>
<div class="s">Sharpe {LS.sharpe:.2f} · MDD {LS.max_dd*100:.0f}%</div></div>
<div class="stat"><div class="k">Long-only ann. return</div><div class="v">{_pct(LO.ann_return)}</div>
<div class="s">Sharpe {LO.sharpe:.2f} · vs universe {_pct(LO.ann_return-met.loc['bench','ann_return'])}</div></div>
<div class="stat"><div class="k">Prediction IC</div><div class="v">{IC_mean:.3f}</div>
<div class="s">ICIR {ICIR:.2f} · {FREQADV} rank-IC</div></div>
<div class="stat"><div class="k">Monthly turnover</div><div class="v">{TURN*100:.0f}%</div>
<div class="s">long book, one-way</div></div>
</div>

<section>
<h2><span class="idx">01</span>Out-of-sample equity</h2>
<p class="sub">Every point is out-of-sample: the model that trades {PSING} <i>t</i> was trained only
on data ending before <i>t</i>. Long-Short = top group − bottom group (dollar-neutral);
Long-only = top group; benchmark = equal-weight universe.</p>
<figure><img src="{imgs['equity']}" alt="equity curves"><figcaption>Cumulative net value (log scale).</figcaption></figure>
<div class="panel" style="margin-top:20px"><div class="panel-h">Performance ({oos_start[:6]}–{oos_end[:6]}, {n_months} {PLABEL}, out-of-sample)</div>
<div class="tbl-scroll"><table>
<thead><tr><th>Strategy</th><th>Ann. return</th><th>Ann. vol</th><th>Sharpe</th><th>Max DD</th><th>Win rate</th></tr></thead>
<tbody>{metrics_rows}</tbody></table></div></div>
</section>

<section>
<h2><span class="idx">02</span>Group monotonicity &amp; signal quality</h2>
<p class="sub">Sorting stocks into 20 groups by predicted score — a good model makes realised
next-month returns rise monotonically from group 0 to group 19.</p>
<figure><img src="{imgs['groups']}" alt="group returns"><figcaption>Mean realised next-{PSING}
return by predicted group. Monthly cross-sectional rank-IC = {IC_mean:.3f} (ICIR {ICIR:.2f}).</figcaption></figure>
</section>

<section>
<h2><span class="idx">03</span>Risk — Long-Short drawdown</h2>
<figure style="margin-top:0"><img src="{imgs['drawdown']}" alt="drawdown"><figcaption>Long-Short drawdown.
Max {LS.max_dd*100:.1f}%.</figcaption></figure>
</section>

<section>
<h2><span class="idx">04</span>Walk-forward folds</h2>
<p class="sub">Each fold trains on {TRAIN_} {PLABEL} (last {VALID_} = validation for early stopping and a small
LightGBM grid, chosen by validation IC) and predicts the next {TEST_} {PLABEL}.</p>
<div class="panel"><div class="tbl-scroll"><table>
<thead><tr><th>Train window</th><th>Test (OOS)</th><th>Val IC</th><th>num_leaves</th><th>lr</th><th>trees</th></tr></thead>
<tbody>{fold_rows}</tbody></table></div></div>
</section>

<section>
<h2><span class="idx">05</span>Holdings</h2>
<p class="sub">Every rebalance records each stock's group, prediction and realised return
(<code>results/ml/holdings.parquet</code>) — the full long/short book for all {n_months} {PLABEL}.
Long book ≈ <b>{avg_long}</b> names/{PSING}; one-way turnover ≈ {TURN*100:.0f}%.</p>
<div class="callout"><b>Latest long book ({last_d}), top by score:</b><br>
<span class="mono" style="font-size:12.5px">{book_names}</span></div>
</section>

{OPT_SECTION}
<section>
<h2><span class="idx">07</span>Notes &amp; caveats</h2>
<ul class="tight">
<li><b>No transaction costs / slippage</b> in these curves — at ~{TURN*100:.0f}% {FREQADV} (~{TURN_ANN*100:.0f}%/yr) turnover,
realistic A-share costs (~0.1–0.3% one-way) would materially reduce the long-short return.</li>
<li>Equal-weighted groups; universe is a mild survivorship-tilted 中证1000 snapshot; short leg
assumes shortability (not always available for A-shares — the long-only line is the practical one).</li>
<li>Label = next-{PSING} return, cross-sectionally z-scored per date (learn relative alpha); features cross-sectionally z-scored. Fully walk-forward, no lookahead.</li>
</ul>
</section>

<div class="foot">Generated from <code>factor_research/ml_lgbm.py</code> · features: Alpha158 ·
model: LightGBM · data: Tushare Pro (hfq). Research/education only — not investment advice.</div>
</div>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
