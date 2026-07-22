# -*- coding: utf-8 -*-
"""Factor effectiveness testing: rank-IC, IC summary stats, factor correlation.

Rank IC = daily cross-sectional Spearman correlation between the factor and
forward returns.  Implemented vectorized (rank once, then a masked Pearson
across the row) so it also supports restricting to a sub-universe (a market-cap
bucket or an index's constituents) -> within-group IC.
"""
import numpy as np
import pandas as pd


def rank_cs(df, mask=None):
    """Cross-sectional rank per date. If mask given, rank only within the mask
    (values outside the mask become NaN and are excluded from the ranking)."""
    if mask is not None:
        df = df.where(mask)
    return df.rank(axis=1)


def _masked_pearson_rows(X, Y):
    """Per-row Pearson corr of two aligned (T,N) arrays, NaN-pairwise."""
    M = np.isfinite(X) & np.isfinite(Y)
    Xm = np.where(M, X, 0.0)
    Ym = np.where(M, Y, 0.0)
    n = M.sum(axis=1).astype(float)
    sx = Xm.sum(1); sy = Ym.sum(1)
    sxx = (Xm * Xm).sum(1); syy = (Ym * Ym).sum(1)
    sxy = (Xm * Ym).sum(1)
    cov = sxy - sx * sy / np.where(n > 0, n, np.nan)
    vx = sxx - sx * sx / np.where(n > 0, n, np.nan)
    vy = syy - sy * sy / np.where(n > 0, n, np.nan)
    denom = np.sqrt(vx * vy)
    ic = np.where(denom > 0, cov / denom, np.nan)
    ic[n < 5] = np.nan
    return ic, n


def daily_ic(factor, fwd, mask=None, min_stocks=30):
    """Return a Series of daily rank-IC (index=date)."""
    Rf = rank_cs(factor, mask)
    Rr = rank_cs(fwd, mask)
    idx = Rf.index
    ic, n = _masked_pearson_rows(Rf.to_numpy(float), Rr.to_numpy(float))
    ic = np.where(n >= min_stocks, ic, np.nan)
    return pd.Series(ic, index=idx)


def ic_summary(ic_series):
    s = ic_series.dropna()
    if len(s) < 5:
        return dict(IC_mean=np.nan, IC_std=np.nan, ICIR=np.nan, t_stat=np.nan,
                    IC_pos_rate=np.nan, absIC=np.nan, n_days=len(s))
    mean, std = s.mean(), s.std()
    icir = mean / std if std > 0 else np.nan
    t = icir * np.sqrt(len(s)) if std > 0 else np.nan
    return dict(IC_mean=mean, IC_std=std, ICIR=icir, t_stat=t,
                IC_pos_rate=(s > 0).mean(), absIC=abs(mean), n_days=len(s))


def factor_correlation(factor_frames, sample_obs=60000, method="spearman", seed=0):
    """Pooled correlation matrix across factors on a random sub-sample of
    (date, stock) observations.  Returns a DataFrame (factors x factors)."""
    names = list(factor_frames.keys())
    stacked = {name: factor_frames[name].to_numpy(float).ravel() for name in names}
    n_total = len(next(iter(stacked.values())))
    rng = np.random.default_rng(seed)
    # rows where most factors are present -> use close-family presence proxy
    finite_count = np.zeros(n_total)
    for v in stacked.values():
        finite_count += np.isfinite(v)
    good = np.where(finite_count > 0.8 * len(names))[0]
    if len(good) > sample_obs:
        good = rng.choice(good, sample_obs, replace=False)
    M = np.column_stack([stacked[n][good] for n in names])
    dfm = pd.DataFrame(M, columns=names)
    if method == "spearman":
        dfm = dfm.rank()
    corr = dfm.corr()      # Pearson on ranks == Spearman
    return corr
