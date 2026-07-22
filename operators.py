# -*- coding: utf-8 -*-
"""Vectorized factor operators over WIDE panels (index=trade_date, cols=ts_code).

Time-series operators act along axis 0 (dates), per stock.
Cross-sectional operators act along axis 1 (stocks), per date.

This mirrors the operator sets used by qlib's Alpha158 and by the common
open-source GTJA-191 implementations, but implemented once here so both
factor families share the same, tested primitives.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import config as C

# --------------------------------------------------------------------------- #
# element-wise helpers
# --------------------------------------------------------------------------- #
def log(df):        return np.log(df)
def sign(df):       return np.sign(df)
def abs_(df):       return df.abs()
def sqrt(df):       return np.sqrt(df)


def emax(a, b):
    """element-wise max of two aligned frames (or frame & scalar) = GTJA MAX."""
    if np.isscalar(a):            # scalar first arg (e.g. MAX(0, frame))
        a, b = b, a
    return np.maximum(a, b) if np.isscalar(b) else a.where(a >= b, b)


def emin(a, b):
    if np.isscalar(a):
        a, b = b, a
    return np.minimum(a, b) if np.isscalar(b) else a.where(a <= b, b)


# --------------------------------------------------------------------------- #
# simple time-series operators (pandas rolling, NaN-aware via min_periods)
# --------------------------------------------------------------------------- #
def delay(df, n):       return df.shift(n)                     # Ref
def delta(df, n):       return df - df.shift(n)
def ts_sum(df, n):      return df.rolling(n, min_periods=n).sum()
def ts_mean(df, n):     return df.rolling(n, min_periods=n).mean()
def ts_std(df, n):      return df.rolling(n, min_periods=n).std()
def ts_max(df, n):      return df.rolling(n, min_periods=n).max()
def ts_min(df, n):      return df.rolling(n, min_periods=n).min()
def ts_quantile(df, n, q):
    return df.rolling(n, min_periods=n).quantile(q)


def ts_corr(a, b, n):
    return a.rolling(n, min_periods=n).corr(b).replace([np.inf, -np.inf], np.nan)


def ts_cov(a, b, n):
    return a.rolling(n, min_periods=n).cov(b)


# --------------------------------------------------------------------------- #
# window helper (numpy) for custom-weighted / regression operators
# --------------------------------------------------------------------------- #
def _apply_win(df, n, func):
    """Apply a window-level func over trailing windows, chunked over columns to
    bound memory.  func(w) takes w of shape (rows, block, n) and returns
    (rows, block).  Result is tail-aligned into a full (T, N) frame."""
    T, N = df.shape
    if T < n:
        return pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    arr = df.to_numpy("float64")
    blk = getattr(C, "COL_CHUNK", 300)
    outs = []
    for i in range(0, N, blk):
        w = sliding_window_view(arr[:, i:i + blk], n, axis=0)   # (T-n+1, block, n)
        outs.append(func(w))
    res = np.concatenate(outs, axis=1)                          # (T-n+1, N)
    full = np.full((T, N), np.nan)
    full[n - 1:] = res
    return pd.DataFrame(full, index=df.index, columns=df.columns)


def _apply_win2(a, b, n, func):
    """Paired-window version for two aligned frames (chunked over columns)."""
    T, N = a.shape
    if T < n:
        return pd.DataFrame(np.nan, index=a.index, columns=a.columns)
    aa = a.to_numpy("float64"); bb = b.to_numpy("float64")
    blk = getattr(C, "COL_CHUNK", 300)
    outs = []
    for i in range(0, N, blk):
        wa = sliding_window_view(aa[:, i:i + blk], n, axis=0)
        wb = sliding_window_view(bb[:, i:i + blk], n, axis=0)
        outs.append(func(wa, wb))
    res = np.concatenate(outs, axis=1)
    full = np.full((T, N), np.nan)
    full[n - 1:] = res
    return pd.DataFrame(full, index=a.index, columns=a.columns)


# ---- window-level kernels --------------------------------------------------
def _k_rank(w):
    last = w[:, :, -1:]
    valid = ~np.isnan(w)
    cnt = valid.sum(axis=2)
    le = ((w <= last) & valid).sum(axis=2)
    return np.where(cnt > 0, le / cnt, np.nan)


def _k_argmax(w):
    res = np.nanargmax(np.where(np.isnan(w), -np.inf, w), axis=2).astype(float)
    res[np.isnan(w).all(axis=2)] = np.nan
    return res


def _k_argmin(w):
    res = np.nanargmin(np.where(np.isnan(w), np.inf, w), axis=2).astype(float)
    res[np.isnan(w).all(axis=2)] = np.nan
    return res


def _make_reg(kind):
    def k(w):
        n = w.shape[-1]
        x = np.arange(n, dtype="float64"); xm = x.mean()
        cx = x - xm; Sxx = (cx ** 2).sum()
        ym = np.nanmean(w, axis=2, keepdims=True)
        cy = w - ym
        Sxy = np.nansum(cx * cy, axis=2)
        slope = Sxy / Sxx
        if kind == "slope":
            return slope
        if kind == "rsq":
            SST = np.nansum(cy ** 2, axis=2)
            SSR = (slope ** 2) * Sxx
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(SST > 0, SSR / SST, np.nan)
        pred_last = ym[:, :, 0] + slope * (x[-1] - xm)
        return w[:, :, -1] - pred_last
    return k


def _make_wsum(weights):
    def k(w):
        res = np.nansum(w * weights, axis=2)
        res[np.isnan(w).all(axis=2)] = np.nan
        return res
    return k


def _k_beta(wa, wb):
    xm = np.nanmean(wb, axis=2, keepdims=True)
    ym = np.nanmean(wa, axis=2, keepdims=True)
    cx = wb - xm; cy = wa - ym
    Sxx = np.nansum(cx ** 2, axis=2)
    Sxy = np.nansum(cx * cy, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Sxx > 0, Sxy / Sxx, np.nan)


def _k_resi_xy(wa, wb):
    xm = np.nanmean(wb, axis=2, keepdims=True)
    ym = np.nanmean(wa, axis=2, keepdims=True)
    cx = wb - xm; cy = wa - ym
    Sxx = np.nansum(cx ** 2, axis=2)
    Sxy = np.nansum(cx * cy, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(Sxx > 0, Sxy / Sxx, np.nan)
    pred_last = ym[:, :, 0] + beta * (wb[:, :, -1] - xm[:, :, 0])
    return wa[:, :, -1] - pred_last


def ts_rank(df, n):     return _apply_win(df, n, _k_rank)
def ts_argmax(df, n):   return _apply_win(df, n, _k_argmax)
def ts_argmin(df, n):   return _apply_win(df, n, _k_argmin)
def ts_slope(df, n):    return _apply_win(df, n, _make_reg("slope"))
def ts_rsquare(df, n):  return _apply_win(df, n, _make_reg("rsq"))
def ts_resi(df, n):     return _apply_win(df, n, _make_reg("resi"))
def reg_beta(a, b, n):  return _apply_win2(a, b, n, _k_beta)
def reg_resi(a, b, n):  return _apply_win2(a, b, n, _k_resi_xy)


def highday(df, n):
    """Days since the window maximum (GTJA HIGHDAY)."""
    return (n - 1) - ts_argmax(df, n)


def lowday(df, n):
    return (n - 1) - ts_argmin(df, n)


# ---- weighted moving averages --------------------------------------------- #
def decay_linear(df, n):
    w = np.arange(1, n + 1, dtype="float64")        # newest gets weight n
    return _apply_win(df, n, _make_wsum(w / w.sum()))


def wma(df, n):
    """GTJA WMA: weight 0.9^i, i=0 is the most recent observation."""
    w = 0.9 ** np.arange(n, dtype="float64")        # [1, .9, .81, ...] newest first
    w = w[::-1]                                      # align oldest->newest for window
    return _apply_win(df, n, _make_wsum(w / w.sum()))


def sma(df, n, m):
    """GTJA SMA(A,n,m): Y_t = (m*A_t + (n-m)*Y_{t-1}) / n  == EWM(alpha=m/n)."""
    return df.ewm(alpha=m / n, adjust=False, min_periods=1).mean()


# --------------------------------------------------------------------------- #
# cross-sectional operators (per date, across stocks)
# --------------------------------------------------------------------------- #
def cs_rank(df):
    """Cross-sectional percentile rank in [0,1] (GTJA RANK)."""
    return df.rank(axis=1, pct=True)


def sequence(n):
    return np.arange(1, n + 1, dtype="float64")


def count_up(df, n):
    """fraction of days close>prev close etc. — pass a boolean frame."""
    return df.rolling(n, min_periods=n).mean()
