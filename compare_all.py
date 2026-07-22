# -*- coding: utf-8 -*-
"""One-figure comparison of every LightGBM/Alpha158 configuration
(universe x frequency), plus a summary metrics table -> a compact HTML page."""
import os, base64
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

# (tag, label, freq, periods/yr, color)
CFG = [
    ("",       "中证1000 剔10% (月)",     "M", 12, "#2563eb"),
    ("_full",  "中证1000 不剔 (月)",       "M", 12, "#0f766e"),
    ("_broad", "宽基 1800+创业板 (月)",    "M", 12, "#94a3b8"),
    ("_all",   "全市场/中证全指 (月)",     "M", 12, "#dc2626"),
    ("_wk",    "中证1000 剔10% (周)",     "W", 52, "#f59e0b"),
]
R = C.RESULTS_DIR


def load(tag):
    d = os.path.join(R, f"ml{tag}")
    mr = pd.read_csv(os.path.join(d, "monthly_returns.csv"))
    mr["dt"] = pd.to_datetime(mr["date"].astype(str))
    met = pd.read_csv(os.path.join(d, "metrics.csv"), index_col=0)
    return mr, met


plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
rows = []
for tag, label, freq, ppy, col in CFG:
    mr, met = load(tag)
    axes[0].plot(mr["dt"], (1 + mr["ls"]).cumprod(), color=col, lw=1.9, label=label)
    axes[1].plot(mr["dt"], (1 + mr["long"]).cumprod(), color=col, lw=1.9, label=label)
    ls, lo, be = met.loc["ls"], met.loc["long"], met.loc["bench"]
    rows.append(dict(cfg=label,
                     ls_ann=ls.ann_return, ls_sh=ls.sharpe, ls_dd=ls.max_dd,
                     lo_ann=lo.ann_return, lo_sh=lo.sharpe,
                     bench=be.ann_return, ic=ls.IC_mean, turn_ann=ls.turnover * ppy))

for ax, title in zip(axes, ["Long-Short（多空 G20−G1）", "Long-only（纯多头 G20）"]):
    ax.set_yscale("log"); ax.set_title(title)
    ax.set_ylabel("累计净值（对数）"); ax.legend(fontsize=8.5, loc="upper left")
fig.suptitle("LightGBM × Alpha158 — 不同股票池 / 频率 样本外对比（税前，2018–2026）",
             fontsize=13, y=1.01)
fig.tight_layout()
figpath = os.path.join(R, "fig_compare_all.png")
fig.savefig(figpath, bbox_inches="tight"); plt.close(fig)

cmp = pd.DataFrame(rows)
cmp.to_csv(os.path.join(R, "compare_all.csv"), index=False)
print(cmp.round(3).to_string(index=False))

# ---- compact HTML page -----------------------------------------------------
with open(figpath, "rb") as f:
    img = "data:image/png;base64," + base64.b64encode(f.read()).decode()

trows = ""
best_ls = cmp["ls_sh"].max()
for _, r in cmp.iterrows():
    hl = ' style="background:color-mix(in srgb,#2563eb 8%,transparent)"' if r.ls_sh == best_ls else ""
    trows += (f'<tr{hl}><td>{r.cfg}</td>'
              f'<td class="num">{r.ls_ann*100:+.1f}%</td><td class="num">{r.ls_sh:.2f}</td>'
              f'<td class="num neg">{r.ls_dd*100:.0f}%</td>'
              f'<td class="num">{r.lo_ann*100:+.1f}%</td><td class="num">{r.lo_sh:.2f}</td>'
              f'<td class="num">{r.bench*100:+.1f}%</td><td class="num">{r.ic:.3f}</td>'
              f'<td class="num">{r.turn_ann:.0f}×</td></tr>')

HTML = f"""
<style>
:root{{--bg:#f6f7f9;--panel:#fff;--ink:#0f1720;--muted:#5b6675;--line:#e3e7ec;--line2:#eef1f5;--neg:#c02636;--accent:#2563eb;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0c1119;--panel:#131b26;--ink:#e8edf3;--muted:#93a1b3;--line:#22303f;--line2:#1a2531;--neg:#ff6b7a;--accent:#5b9bff;}}}}
:root[data-theme="dark"]{{--bg:#0c1119;--panel:#131b26;--ink:#e8edf3;--muted:#93a1b3;--line:#22303f;--line2:#1a2531;--neg:#ff6b7a;--accent:#5b9bff;}}
:root[data-theme="light"]{{--bg:#f6f7f9;--panel:#fff;--ink:#0f1720;--muted:#5b6675;--line:#e3e7ec;--line2:#eef1f5;--neg:#c02636;--accent:#2563eb;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);line-height:1.6;
font-family:-apple-system,"Segoe UI",system-ui,"Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1040px;margin:0 auto;padding:clamp(20px,4vw,48px) clamp(16px,4vw,36px);}}
h1{{font-size:clamp(22px,3.4vw,30px);font-weight:800;letter-spacing:-.01em;margin:0 0 6px;}}
.sub{{color:var(--muted);margin:0 0 22px;max-width:74ch;}}
figure{{margin:0 0 26px}}img{{width:100%;height:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel);}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 1px 2px rgba(16,23,32,.04),0 8px 24px rgba(16,23,32,.06);}}
.tbl-scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th,td{{padding:9px 12px;border-bottom:1px solid var(--line2);white-space:nowrap}}
th{{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);text-align:right;font-weight:600}}
th:first-child,td:first-child{{text-align:left}}tbody tr:last-child td{{border-bottom:0}}
.num{{font-variant-numeric:tabular-nums;text-align:right}}td.neg{{color:var(--neg)}}
.note{{color:var(--muted);font-size:13px;margin-top:16px;max-width:78ch}}
.note b{{color:var(--ink)}}
</style>
<div class="wrap">
<h1>股票池 × 频率 — 一图对比</h1>
<p class="sub">同一套 LightGBM × Alpha158（walk-forward、20组、样本外 2018–2026、等权、<b>税前</b>），
只改股票池与调仓频率。左=多空，右=纯多头，对数净值。</p>
<figure><img src="{img}" alt="comparison"></figure>
<div class="panel"><div class="tbl-scroll"><table>
<thead><tr><th>配置</th><th>多空年化</th><th>多空夏普</th><th>多空回撤</th>
<th>多头年化</th><th>多头夏普</th><th>基准年化</th><th>IC</th><th>年化换手</th></tr></thead>
<tbody>{trows}</tbody></table></div></div>
<p class="note"><b>读法：</b>票池越下沉，税前越强（全市场 &gt; 中证1000不剔 &gt; 剔10% &gt; 宽基），
但超额越来自不可交易的微盘；频率越高（周&gt;月）税前夏普越高，但换手成倍上升。
<b>这些都是税前</b>——真正可比要看扣费后，尤其微盘/高换手最吃成本。</p>
</div>
"""
out = os.path.join(R, "compare_all.html")
open(out, "w", encoding="utf-8").write(HTML)
print("wrote", out)
