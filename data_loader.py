# -*- coding: utf-8 -*-
"""Data acquisition (tushare Pro) with parquet caching — threaded + resumable.

Provides:
  * trading calendar
  * time-varying index membership for the 4 target indices  -> stock universe
  * a daily panel (adjusted OHLCV + vwap + market cap) for that universe
Fetches are parallelised (config.WORKERS threads) and cached per-year so a long
max-window pull can resume after interruption.
"""
import os
import time
import bisect
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import tushare as ts

import config as C

pro = ts.pro_api()


def _retry(fn, *args, tries=10, pause=0.7, **kwargs):
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            msg = str(e)
            # tushare per-minute rate limit -> wait out most of the minute window
            if ("频率" in msg) or ("每分钟" in msg) or ("抱歉" in msg):
                time.sleep(8 + i * 2)
            else:
                time.sleep(pause * (i + 1))
    raise last


def _pmap(fn, items, workers=None):
    workers = workers or C.WORKERS
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


# --------------------------------------------------------------------------- #
# trading calendar
# --------------------------------------------------------------------------- #
def get_trade_dates():
    fp = os.path.join(C.DATA_DIR, "trade_dates.parquet")
    if os.path.exists(fp):
        return pd.read_parquet(fp)["cal_date"].tolist()
    cal = _retry(pro.trade_cal, exchange="SSE",
                 start_date=C.START_DATE, end_date=C.END_DATE)
    dates = sorted(cal.loc[cal["is_open"] == 1, "cal_date"].tolist())
    pd.DataFrame({"cal_date": dates}).to_parquet(fp)
    return dates


# --------------------------------------------------------------------------- #
# index membership  ->  universe
# --------------------------------------------------------------------------- #
def get_index_membership():
    """Long df [index, trade_date, con_code] of monthly constituent snapshots."""
    fp = os.path.join(C.DATA_DIR, "index_membership.parquet")
    if os.path.exists(fp):
        return pd.read_parquet(fp)

    months = list(pd.date_range(C.START_DATE, C.END_DATE, freq="MS").strftime("%Y%m%d"))
    months = months + [C.END_DATE]
    tasks = []
    for name, code in C.INDEX_CODES.items():
        for i in range(len(months) - 1):
            tasks.append((name, code, months[i], months[i + 1]))

    def fetch(t):
        name, code, s, e = t
        df = _retry(pro.index_weight, index_code=code, start_date=s, end_date=e)
        if df is not None and len(df):
            df = df[["trade_date", "con_code"]].copy()
            df["index"] = name
            return df
        return None

    t0 = time.time()
    # index_weight is capped at 200 calls/min -> throttle to ~3 concurrent
    frames = [f for f in _pmap(fetch, tasks, workers=3) if f is not None]
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    out.to_parquet(fp)
    print(f"  membership: {len(tasks)} calls in {time.time()-t0:.0f}s, "
          f"{out['con_code'].nunique()} unique stocks")
    return out


def get_universe(membership=None):
    if membership is None:
        membership = get_index_membership()
    return sorted(membership["con_code"].unique().tolist())


def membership_matrix(trade_dates, membership=None):
    """dict[index] -> {trade_date: set(con_code)} via forward-filled snapshots."""
    if membership is None:
        membership = get_index_membership()
    trade_dates = sorted(trade_dates)
    out = {}
    for name in C.INDEX_CODES:
        sub = membership[membership["index"] == name]
        snaps = sorted(sub["trade_date"].unique())
        snap_sets = {d: set(sub.loc[sub["trade_date"] == d, "con_code"]) for d in snaps}
        rows = {}
        for d in trade_dates:
            j = bisect.bisect_right(snaps, d) - 1
            if j < 0:
                j = 0
            rows[d] = snap_sets[snaps[j]] if snaps else set()
        out[name] = rows
    return out


# --------------------------------------------------------------------------- #
# daily panel  (threaded per-date, cached per-year, then concatenated)
# --------------------------------------------------------------------------- #
def _fetch_one_day(d, uni):
    px = _retry(pro.daily, trade_date=d)
    if px is None or not len(px):
        return None
    adj = _retry(pro.adj_factor, trade_date=d)
    bas = _retry(pro.daily_basic, trade_date=d,
                 fields="ts_code,total_mv,circ_mv,turnover_rate")
    if uni is not None:                       # uni=None -> keep the whole market
        px = px[px["ts_code"].isin(uni)]
    m = px.merge(adj[["ts_code", "adj_factor"]], on="ts_code", how="left")
    m = m.merge(bas, on="ts_code", how="left")
    return m


def _shape_panel(raw):
    adjf = raw["adj_factor"].fillna(1.0)
    out = pd.DataFrame({
        "ts_code": raw["ts_code"],
        "trade_date": raw["trade_date"],
        "open":  raw["open"] * adjf,
        "high":  raw["high"] * adjf,
        "low":   raw["low"] * adjf,
        "close": raw["close"] * adjf,
        "vwap":  (raw["amount"] * 10.0 / raw["vol"].replace(0, pd.NA)) * adjf,
        "volume": raw["vol"],
        "amount": raw["amount"],
        "total_mv": raw["total_mv"],
        "circ_mv": raw["circ_mv"],
        "turnover_rate": raw["turnover_rate"],
    })
    out["vwap"] = out["vwap"].fillna(out["close"])
    return out


def get_daily_panel(universe=None):
    fp = os.path.join(C.DATA_DIR, "daily_panel.parquet")
    if os.path.exists(fp):
        return pd.read_parquet(fp)

    if os.environ.get("FR_FULLMKT", "0") == "1":
        uni = None                            # keep every stock (full market)
    else:
        if universe is None:
            universe = get_universe()
        uni = set(universe)
    dates = get_trade_dates()
    by_year = {}
    for d in dates:
        by_year.setdefault(d[:4], []).append(d)

    year_frames = []
    t0 = time.time()
    for yr in sorted(by_year):
        ypath = os.path.join(C.DATA_DIR, f"panel_{yr}.parquet")
        if os.path.exists(ypath):
            year_frames.append(pd.read_parquet(ypath))
            print(f"  year {yr}: cached")
            continue
        ds = by_year[yr]
        res = _pmap(lambda d: _fetch_one_day(d, uni), ds)
        frames = [r for r in res if r is not None]
        yraw = pd.concat(frames, ignore_index=True)
        yshaped = _shape_panel(yraw)
        yshaped.to_parquet(ypath)
        year_frames.append(yshaped)
        print(f"  year {yr}: {len(ds)} days, {len(yshaped)} rows, "
              f"elapsed {time.time()-t0:.0f}s", flush=True)

    out = pd.concat(year_frames, ignore_index=True)
    out = out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    out.to_parquet(fp)
    print(f"  PANEL total {len(out)} rows in {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    d = get_trade_dates()
    print("trade dates:", len(d), d[0], "->", d[-1])
    mem = get_index_membership()
    uni = get_universe(mem)
    print("universe size:", len(uni))
