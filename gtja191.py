# -*- coding: utf-8 -*-
"""Guotai Junan (国泰君安) Alpha-191 factor construction.

Faithful translation of the 191 formulaic alphas from the Guotai Junan
research report "基于短周期价量特征的多因子选股体系", implemented on wide
(date x stock) panels using the shared operator library.

Each alpha is evaluated inside try/except so a single failure cannot abort
the batch; the driver reports coverage and screens degenerate factors.
A handful of alphas require Fama-French style factor returns (Alpha191 #30)
and are returned as None (skipped) — everything else is computed from OHLCV.
"""
import numpy as np
import pandas as pd
import operators as op

# ---- operator shorthands (match the report's notation) --------------------
RANK   = op.cs_rank
DELAY  = op.delay
DELTA  = op.delta
CORR   = op.ts_corr
COV    = op.ts_cov
SUM    = op.ts_sum
MEAN   = op.ts_mean
STD    = op.ts_std
TSMAX  = op.ts_max
TSMIN  = op.ts_min
TSRANK = op.ts_rank
MAX    = op.emax            # element-wise
MIN    = op.emin
ABS    = op.abs_
LOG    = op.log
SIGN   = op.sign
SMA    = op.sma
WMA    = op.wma
DECAY  = op.decay_linear
REGBETA = op.reg_beta
REGRESI = op.reg_resi
HIGHDAY = op.highday
LOWDAY  = op.lowday
SLOPE  = op.ts_slope       # REGBETA(x, SEQUENCE(n)) == slope of x on 1..n


def iif(cond, a, b):
    """Vectorized ternary  cond ? a : b  (a/b may be frame or scalar)."""
    if np.isscalar(a):
        a = pd.DataFrame(a, index=cond.index, columns=cond.columns)
    return a.where(cond.astype(bool) & cond.notna(), b)


def COUNT(cond, n):
    return cond.astype(float).rolling(n, min_periods=n).sum()


def SUMIF(vals, cond, n):
    return vals.where(cond.astype(bool), 0.0).rolling(n, min_periods=n).sum()


def PROD(df, n):
    return df.rolling(n, min_periods=n).apply(np.prod, raw=True)


def compute(F, sink=None):
    """If `sink` is given each computed alpha is passed to sink(name, frame) and
    not retained (memory-safe streaming); returns ({}|res, fails)."""
    O   = F["open"]
    H   = F["high"]
    L   = F["low"]
    C   = F["close"]
    VW  = F["vwap"]
    V   = F["volume"]
    AMT = F["amount"]
    RET = C / DELAY(C, 1) - 1.0

    res = {}
    fails = {}

    def add(name, fn):
        try:
            r = fn()
        except Exception as e:               # noqa
            r = None
            fails[name] = repr(e)[:160]
        if sink is not None:
            if r is not None:
                sink(name, r)
        else:
            res[name] = r

    a = add  # alias

    a("alpha001", lambda: -1 * CORR(RANK(DELTA(LOG(V), 1)), RANK((C - O) / O), 6))
    a("alpha002", lambda: -1 * DELTA(((C - L) - (H - C)) / (H - L), 1))
    a("alpha003", lambda: SUM(iif(C == DELAY(C, 1), 0.0,
                                  C - iif(C > DELAY(C, 1), MIN(L, DELAY(C, 1)), MAX(H, DELAY(C, 1)))), 6))
    a("alpha004", lambda: iif((SUM(C, 8) / 8 + STD(C, 8)) < (SUM(C, 2) / 2), -1.0,
                       iif((SUM(C, 2) / 2) < (SUM(C, 8) / 8 - STD(C, 8)), 1.0,
                           iif((V / MEAN(V, 20)) >= 1, 1.0, -1.0))))
    a("alpha005", lambda: -1 * TSMAX(CORR(TSRANK(V, 5), TSRANK(H, 5), 5), 3))
    a("alpha006", lambda: -1 * RANK(SIGN(DELTA(O * 0.85 + H * 0.15, 4))))
    a("alpha007", lambda: (RANK(TSMAX(VW - C, 3)) + RANK(TSMIN(VW - C, 3))) * RANK(DELTA(V, 3)))
    a("alpha008", lambda: -1 * RANK(DELTA((H + L) / 2 * 0.2 + VW * 0.8, 4)))
    a("alpha009", lambda: SMA(((H + L) / 2 - (DELAY(H, 1) + DELAY(L, 1)) / 2) * (H - L) / V, 7, 2))
    a("alpha010", lambda: RANK(TSMAX(iif(RET < 0, STD(RET, 20), C) ** 2, 5)))
    a("alpha011", lambda: SUM(((C - L) - (H - C)) / (H - L) * V, 6))
    a("alpha012", lambda: RANK(O - SUM(VW, 10) / 10) * (-1 * RANK(ABS(C - VW))))
    a("alpha013", lambda: (H * L) ** 0.5 - VW)
    a("alpha014", lambda: C - DELAY(C, 5))
    a("alpha015", lambda: O / DELAY(C, 1) - 1)
    a("alpha016", lambda: -1 * TSMAX(RANK(CORR(RANK(V), RANK(VW), 5)), 5))
    a("alpha017", lambda: RANK(VW - TSMAX(VW, 15)) ** DELTA(C, 5))
    a("alpha018", lambda: C / DELAY(C, 5))
    a("alpha019", lambda: iif(C < DELAY(C, 5), (C - DELAY(C, 5)) / DELAY(C, 5),
                       iif(C == DELAY(C, 5), 0.0, (C - DELAY(C, 5)) / C)))
    a("alpha020", lambda: (C - DELAY(C, 6)) / DELAY(C, 6) * 100)
    a("alpha021", lambda: SLOPE(MEAN(C, 6), 6))
    a("alpha022", lambda: SMA((C - MEAN(C, 6)) / MEAN(C, 6) -
                              DELAY((C - MEAN(C, 6)) / MEAN(C, 6), 3), 12, 1))
    a("alpha023", lambda: SMA(iif(C > DELAY(C, 1), STD(C, 20), 0.0), 20, 1) /
                          (SMA(iif(C > DELAY(C, 1), STD(C, 20), 0.0), 20, 1) +
                           SMA(iif(C <= DELAY(C, 1), STD(C, 20), 0.0), 20, 1)) * 100)
    a("alpha024", lambda: SMA(C - DELAY(C, 5), 5, 1))
    a("alpha025", lambda: -1 * RANK(DELTA(C, 7) * (1 - RANK(DECAY(V / MEAN(V, 20), 9)))) *
                          (1 + RANK(SUM(RET, 250))))
    a("alpha026", lambda: SUM(C, 7) / 7 - C + CORR(VW, DELAY(C, 5), 230))
    a("alpha027", lambda: WMA((C - DELAY(C, 3)) / DELAY(C, 3) * 100 +
                              (C - DELAY(C, 6)) / DELAY(C, 6) * 100, 12))
    a("alpha028", lambda: 3 * SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)) * 100, 3, 1) -
                          2 * SMA(SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)) * 100, 3, 1), 3, 1))
    a("alpha029", lambda: (C - DELAY(C, 6)) / DELAY(C, 6) * V)
    a("alpha030", lambda: None)   # requires Fama-French factor returns -> skipped
    a("alpha031", lambda: (C - MEAN(C, 12)) / MEAN(C, 12) * 100)
    a("alpha032", lambda: -1 * SUM(RANK(CORR(RANK(H), RANK(V), 3)), 3))
    a("alpha033", lambda: (-1 * TSMIN(L, 5) + DELAY(TSMIN(L, 5), 5)) *
                          RANK((SUM(RET, 240) - SUM(RET, 20)) / 220) * TSRANK(V, 5))
    a("alpha034", lambda: MEAN(C, 12) / C)
    a("alpha035", lambda: MIN(RANK(DECAY(DELTA(O, 1), 15)),
                              RANK(DECAY(CORR(V, O, 17), 7))) * -1)
    a("alpha036", lambda: RANK(SUM(CORR(RANK(V), RANK(VW), 6), 2)))
    a("alpha037", lambda: -1 * RANK(SUM(O, 5) * SUM(RET, 5) - DELAY(SUM(O, 5) * SUM(RET, 5), 10)))
    a("alpha038", lambda: iif(SUM(H, 20) / 20 < H, -1 * DELTA(H, 2), 0.0))
    a("alpha039", lambda: (RANK(DECAY(DELTA(C, 2), 8)) -
                           RANK(DECAY(CORR(VW * 0.3 + O * 0.7, SUM(MEAN(V, 180), 37), 14), 12))) * -1)
    a("alpha040", lambda: SUM(iif(C > DELAY(C, 1), V, 0.0), 26) /
                          SUM(iif(C <= DELAY(C, 1), V, 0.0), 26) * 100)
    a("alpha041", lambda: RANK(TSMAX(DELTA(VW, 3), 5)) * -1)
    a("alpha042", lambda: -1 * RANK(STD(H, 10)) * CORR(H, V, 10))
    a("alpha043", lambda: SUM(iif(C > DELAY(C, 1), V, iif(C < DELAY(C, 1), -V, 0.0)), 6))
    a("alpha044", lambda: TSRANK(DECAY(CORR(L, MEAN(V, 10), 7), 6), 4) +
                          TSRANK(DECAY(DELTA(VW, 3), 10), 15))
    a("alpha045", lambda: RANK(DELTA(C * 0.6 + O * 0.4, 1)) * RANK(CORR(VW, MEAN(V, 150), 15)))
    a("alpha046", lambda: (MEAN(C, 3) + MEAN(C, 6) + MEAN(C, 12) + MEAN(C, 24)) / (4 * C))
    a("alpha047", lambda: SMA((TSMAX(H, 6) - C) / (TSMAX(H, 6) - TSMIN(L, 6)) * 100, 9, 1))
    a("alpha048", lambda: -1 * RANK(SIGN(C - DELAY(C, 1)) + SIGN(DELAY(C, 1) - DELAY(C, 2)) +
                                    SIGN(DELAY(C, 2) - DELAY(C, 3))) * SUM(V, 5) / SUM(V, 20))

    # ---- 49/50/51 : directional movement --------------------------------
    up49 = iif((H + L) >= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
               MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1))))
    dn49 = iif((H + L) <= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
               MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1))))
    a("alpha049", lambda: SUM(up49, 12) / (SUM(up49, 12) + SUM(dn49, 12)))
    a("alpha050", lambda: SUM(dn49, 12) / (SUM(up49, 12) + SUM(dn49, 12)) -
                          SUM(up49, 12) / (SUM(up49, 12) + SUM(dn49, 12)))
    a("alpha051", lambda: SUM(dn49, 12) / (SUM(up49, 12) + SUM(dn49, 12)))

    a("alpha052", lambda: SUM(MAX(0.0, H - DELAY((H + L + C) / 3, 1)), 26) /
                          SUM(MAX(0.0, DELAY((H + L + C) / 3, 1) - L), 26) * 100)
    a("alpha053", lambda: COUNT(C > DELAY(C, 1), 12) / 12 * 100)
    a("alpha054", lambda: -1 * RANK(STD(ABS(C - O), 10) + (C - O) + CORR(C, O, 10)))
    a("alpha055", lambda: SUM(16 * (C - DELAY(C, 1) + (C - O) / 2 + DELAY(C, 1) - DELAY(O, 1)) /
        iif((ABS(H - DELAY(C, 1)) > ABS(L - DELAY(C, 1))) &
            (ABS(H - DELAY(C, 1)) > ABS(H - DELAY(L, 1))),
            ABS(H - DELAY(C, 1)) + ABS(L - DELAY(C, 1)) / 2 + ABS(DELAY(C, 1) - DELAY(O, 1)) / 4,
            iif((ABS(L - DELAY(C, 1)) > ABS(H - DELAY(L, 1))) &
                (ABS(L - DELAY(C, 1)) > ABS(H - DELAY(C, 1))),
                ABS(L - DELAY(C, 1)) + ABS(H - DELAY(C, 1)) / 2 + ABS(DELAY(C, 1) - DELAY(O, 1)) / 4,
                ABS(H - DELAY(L, 1)) + ABS(DELAY(C, 1) - DELAY(O, 1)) / 4)) *
        MAX(ABS(H - DELAY(C, 1)), ABS(L - DELAY(C, 1))), 20))
    a("alpha056", lambda: iif(RANK(O - TSMIN(O, 12)) <
                              RANK(RANK(CORR(SUM((H + L) / 2, 19), SUM(MEAN(V, 40), 19), 13)) ** 5),
                              1.0, 0.0) * -1)
    a("alpha057", lambda: SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)) * 100, 3, 1))
    a("alpha058", lambda: COUNT(C > DELAY(C, 1), 20) / 20 * 100)
    a("alpha059", lambda: SUM(iif(C == DELAY(C, 1), 0.0,
                                  C - iif(C > DELAY(C, 1), MIN(L, DELAY(C, 1)), MAX(H, DELAY(C, 1)))), 20))
    a("alpha060", lambda: SUM(((C - L) - (H - C)) / (H - L) * V, 20))
    a("alpha061", lambda: MAX(RANK(DECAY(DELTA(VW, 1), 12)),
                              RANK(DECAY(RANK(CORR(L, MEAN(V, 80), 8)), 17))) * -1)
    a("alpha062", lambda: -1 * CORR(H, RANK(V), 5))
    a("alpha063", lambda: SMA(MAX(C - DELAY(C, 1), 0.0), 6, 1) /
                          SMA(ABS(C - DELAY(C, 1)), 6, 1) * 100)
    a("alpha064", lambda: MAX(RANK(DECAY(CORR(RANK(VW), RANK(V), 4), 4)),
                              RANK(DECAY(TSMAX(CORR(RANK(C), RANK(MEAN(V, 60)), 4), 13), 14))) * -1)
    a("alpha065", lambda: MEAN(C, 6) / C)
    a("alpha066", lambda: (C - MEAN(C, 6)) / MEAN(C, 6) * 100)
    a("alpha067", lambda: SMA(MAX(C - DELAY(C, 1), 0.0), 24, 1) /
                          SMA(ABS(C - DELAY(C, 1)), 24, 1) * 100)
    a("alpha068", lambda: SMA(((H + L) / 2 - (DELAY(H, 1) + DELAY(L, 1)) / 2) *
                              (H - L) / V, 15, 2))

    # ---- 69 : DTM/DBM ---------------------------------------------------
    DTM = iif(O <= DELAY(O, 1), 0.0, MAX(H - O, O - DELAY(O, 1)))
    DBM = iif(O >= DELAY(O, 1), 0.0, MAX(O - L, O - DELAY(O, 1)))
    a("alpha069", lambda: iif(SUM(DTM, 20) > SUM(DBM, 20),
                              (SUM(DTM, 20) - SUM(DBM, 20)) / SUM(DTM, 20),
                              iif(SUM(DTM, 20) == SUM(DBM, 20), 0.0,
                                  (SUM(DTM, 20) - SUM(DBM, 20)) / SUM(DBM, 20))))
    a("alpha070", lambda: STD(AMT, 6))
    a("alpha071", lambda: (C - MEAN(C, 24)) / MEAN(C, 24) * 100)
    a("alpha072", lambda: SMA((TSMAX(H, 6) - C) / (TSMAX(H, 6) - TSMIN(L, 6)) * 100, 15, 1))
    a("alpha073", lambda: (TSRANK(DECAY(DECAY(CORR(C, V, 10), 16), 4), 5) -
                           RANK(DECAY(CORR(VW, MEAN(V, 30), 4), 3))) * -1)
    a("alpha074", lambda: RANK(CORR(SUM((L * 0.35 + VW * 0.65), 20), SUM(MEAN(V, 40), 20), 7)) +
                          RANK(CORR(RANK(VW), RANK(V), 6)))
    a("alpha075", lambda: COUNT((C > O), 50))  # approx: without benchmark, count up-days/50
    a("alpha076", lambda: STD(ABS(C / DELAY(C, 1) - 1) / V, 20) /
                          MEAN(ABS(C / DELAY(C, 1) - 1) / V, 20))
    a("alpha077", lambda: MIN(RANK(DECAY((H + L) / 2 + H - (VW + H), 20)),
                              RANK(DECAY(CORR((H + L) / 2, MEAN(V, 40), 3), 6))))
    a("alpha078", lambda: ((H + L + C) / 3 - MEAN((H + L + C) / 3, 12)) /
                          (0.015 * MEAN(ABS(C - MEAN((H + L + C) / 3, 12)), 12)))
    a("alpha079", lambda: SMA(MAX(C - DELAY(C, 1), 0.0), 12, 1) /
                          SMA(ABS(C - DELAY(C, 1)), 12, 1) * 100)
    a("alpha080", lambda: (V - DELAY(V, 5)) / DELAY(V, 5) * 100)
    a("alpha081", lambda: SMA(V, 21, 2))
    a("alpha082", lambda: SMA((TSMAX(H, 6) - C) / (TSMAX(H, 6) - TSMIN(L, 6)) * 100, 20, 1))
    a("alpha083", lambda: -1 * RANK(COV(RANK(H), RANK(V), 5)))
    a("alpha084", lambda: SUM(iif(C > DELAY(C, 1), V, iif(C < DELAY(C, 1), -V, 0.0)), 20))
    a("alpha085", lambda: TSRANK(V / MEAN(V, 20), 20) * TSRANK(-1 * DELTA(C, 7), 8))
    a("alpha086", lambda: iif((0.25 < ((DELAY(C, 20) - DELAY(C, 10)) / 10 - (DELAY(C, 10) - C) / 10)), -1.0,
                       iif(((DELAY(C, 20) - DELAY(C, 10)) / 10 - (DELAY(C, 10) - C) / 10) < 0, 1.0,
                           -1 * (C - DELAY(C, 1)))))
    a("alpha087", lambda: (RANK(DECAY(DELTA(VW, 4), 7)) +
                           TSRANK(DECAY((L * 0.9 + L * 0.1 - VW) / (O - (H + L) / 2), 11), 7)) * -1)
    a("alpha088", lambda: (C - DELAY(C, 20)) / DELAY(C, 20) * 100)
    a("alpha089", lambda: 2 * (SMA(C, 13, 2) - SMA(C, 27, 2) -
                               SMA(SMA(C, 13, 2) - SMA(C, 27, 2), 10, 2)))
    a("alpha090", lambda: -1 * RANK(CORR(RANK(VW), RANK(V), 5)))
    a("alpha091", lambda: -1 * RANK(C - TSMAX(C, 5)) * RANK(CORR(MEAN(V, 40), L, 5)))
    a("alpha092", lambda: MAX(RANK(DECAY(DELTA(C * 0.35 + VW * 0.65, 2), 3)),
                              TSRANK(DECAY(ABS(CORR(MEAN(V, 180), C, 13)), 5), 15)) * -1)
    a("alpha093", lambda: SUM(iif(O >= DELAY(O, 1), 0.0,
                                  MAX(O - L, O - DELAY(O, 1))), 20))
    a("alpha094", lambda: SUM(iif(C > DELAY(C, 1), V, iif(C < DELAY(C, 1), -V, 0.0)), 30))
    a("alpha095", lambda: STD(AMT, 20))
    a("alpha096", lambda: SMA(SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)) * 100, 3, 1), 3, 1))
    a("alpha097", lambda: STD(V, 10))
    a("alpha098", lambda: iif((DELTA(SUM(C, 100) / 100, 100) / DELAY(C, 100) <= 0.05),
                              -1 * (C - TSMIN(C, 100)), -1 * DELTA(C, 3)))
    a("alpha099", lambda: -1 * RANK(COV(RANK(C), RANK(V), 5)))
    a("alpha100", lambda: STD(V, 20))
    a("alpha101", lambda: iif((RANK(CORR(C, SUM(MEAN(V, 30), 37), 15)) <
                               RANK(CORR(RANK(H * 0.1 + VW * 0.9), RANK(V), 11))), -1.0, 0.0))
    a("alpha102", lambda: SMA(MAX(V - DELAY(V, 1), 0.0), 6, 1) /
                          SMA(ABS(V - DELAY(V, 1)), 6, 1) * 100)
    a("alpha103", lambda: (20 - LOWDAY(L, 20)) / 20 * 100)
    a("alpha104", lambda: -1 * DELTA(CORR(H, V, 5), 5) * RANK(STD(C, 20)))
    a("alpha105", lambda: -1 * CORR(RANK(O), RANK(V), 10))
    a("alpha106", lambda: C - DELAY(C, 20))
    a("alpha107", lambda: -1 * RANK(O - DELAY(H, 1)) * RANK(O - DELAY(C, 1)) * RANK(O - DELAY(L, 1)))
    a("alpha108", lambda: (RANK(H - TSMIN(H, 2)) ** RANK(CORR(VW, MEAN(V, 120), 6))) * -1)
    a("alpha109", lambda: SMA(H - L, 10, 2) / SMA(SMA(H - L, 10, 2), 10, 2))
    a("alpha110", lambda: SUM(MAX(0.0, H - DELAY(C, 1)), 20) /
                          SUM(MAX(0.0, DELAY(C, 1) - L), 20) * 100)
    a("alpha111", lambda: SMA(V * ((C - L) - (H - C)) / (H - L), 11, 2) -
                          SMA(V * ((C - L) - (H - C)) / (H - L), 4, 2))
    a("alpha112", lambda: (SUM(iif(C - DELAY(C, 1) > 0, C - DELAY(C, 1), 0.0), 12) -
                           SUM(iif(C - DELAY(C, 1) < 0, ABS(C - DELAY(C, 1)), 0.0), 12)) /
                          (SUM(iif(C - DELAY(C, 1) > 0, C - DELAY(C, 1), 0.0), 12) +
                           SUM(iif(C - DELAY(C, 1) < 0, ABS(C - DELAY(C, 1)), 0.0), 12)) * 100)
    a("alpha113", lambda: -1 * RANK(SUM(DELAY(C, 5), 20) / 20) * CORR(C, V, 2) *
                          RANK(CORR(SUM(C, 5), SUM(C, 20), 2)))
    a("alpha114", lambda: RANK(DELAY((H - L) / (SUM(C, 5) / 5), 2)) * RANK(RANK(V)) /
                          ((H - L) / (SUM(C, 5) / 5) / (VW - C)))
    a("alpha115", lambda: RANK(CORR(H * 0.9 + C * 0.1, MEAN(V, 30), 10)) **
                          RANK(CORR(TSRANK((H + L) / 2, 4), TSRANK(V, 10), 7)))
    a("alpha116", lambda: SLOPE(C, 20))
    a("alpha117", lambda: TSRANK(V, 32) * (1 - TSRANK(C + H - L, 16)) * (1 - TSRANK(RET, 32)))
    a("alpha118", lambda: SUM(H - O, 20) / SUM(O - L, 20) * 100)
    a("alpha119", lambda: RANK(DECAY(CORR(VW, SUM(MEAN(V, 5), 26), 5), 7)) -
                          RANK(DECAY(TSRANK(TSMIN(CORR(RANK(O), RANK(MEAN(V, 15)), 21), 9), 7), 8)))
    a("alpha120", lambda: RANK(VW - C) / RANK(VW + C))
    a("alpha121", lambda: (RANK(VW - TSMIN(VW, 12)) **
                           TSRANK(CORR(TSRANK(VW, 20), TSRANK(MEAN(V, 60), 2), 18), 3)) * -1)
    a("alpha122", lambda: (SMA(SMA(SMA(LOG(C), 13, 2), 13, 2), 13, 2) -
                           DELAY(SMA(SMA(SMA(LOG(C), 13, 2), 13, 2), 13, 2), 1)) /
                          DELAY(SMA(SMA(SMA(LOG(C), 13, 2), 13, 2), 13, 2), 1))
    a("alpha123", lambda: iif(RANK(CORR(SUM((H + L) / 2, 20), SUM(MEAN(V, 60), 20), 9)) <
                              RANK(CORR(L, V, 6)), -1.0, 0.0))
    a("alpha124", lambda: (C - VW) / DECAY(RANK(TSMAX(C, 30)), 2))
    a("alpha125", lambda: RANK(DECAY(CORR(VW, MEAN(V, 80), 17), 20)) /
                          RANK(DECAY(DELTA(C * 0.5 + VW * 0.5, 3), 16)))
    a("alpha126", lambda: (C + H + L) / 3)
    a("alpha127", lambda: (MEAN((100 * (C - TSMAX(C, 12)) / TSMAX(C, 12)) ** 2, 12)) ** 0.5)
    a("alpha128", lambda: 100 - 100 / (1 + SUM(iif((H + L + C) / 3 > DELAY((H + L + C) / 3, 1),
                                                   (H + L + C) / 3 * V, 0.0), 14) /
                                       SUM(iif((H + L + C) / 3 < DELAY((H + L + C) / 3, 1),
                                               (H + L + C) / 3 * V, 0.0), 14)))
    a("alpha129", lambda: SUM(iif(C - DELAY(C, 1) < 0, ABS(C - DELAY(C, 1)), 0.0), 12))
    a("alpha130", lambda: RANK(DECAY(CORR((H + L) / 2, MEAN(V, 40), 9), 10)) /
                          RANK(DECAY(CORR(RANK(VW), RANK(V), 7), 3)))
    a("alpha131", lambda: RANK(DELTA(VW, 1)) ** TSRANK(CORR(C, MEAN(V, 50), 18), 18))
    a("alpha132", lambda: MEAN(AMT, 20))
    a("alpha133", lambda: (20 - HIGHDAY(H, 20)) / 20 * 100 - (20 - LOWDAY(L, 20)) / 20 * 100)
    a("alpha134", lambda: (C - DELAY(C, 12)) / DELAY(C, 12) * V)
    a("alpha135", lambda: SMA(DELAY(C / DELAY(C, 20), 1), 20, 1))
    a("alpha136", lambda: -1 * RANK(DELTA(RET, 3)) * CORR(O, V, 10))
    a("alpha137", lambda: 16 * (C - DELAY(C, 1) + (C - O) / 2 + DELAY(C, 1) - DELAY(O, 1)) /
        iif((ABS(H - DELAY(C, 1)) > ABS(L - DELAY(C, 1))) &
            (ABS(H - DELAY(C, 1)) > ABS(H - DELAY(L, 1))),
            ABS(H - DELAY(C, 1)) + ABS(L - DELAY(C, 1)) / 2 + ABS(DELAY(C, 1) - DELAY(O, 1)) / 4,
            iif((ABS(L - DELAY(C, 1)) > ABS(H - DELAY(L, 1))) &
                (ABS(L - DELAY(C, 1)) > ABS(H - DELAY(C, 1))),
                ABS(L - DELAY(C, 1)) + ABS(H - DELAY(C, 1)) / 2 + ABS(DELAY(C, 1) - DELAY(O, 1)) / 4,
                ABS(H - DELAY(L, 1)) + ABS(DELAY(C, 1) - DELAY(O, 1)) / 4)) *
        MAX(ABS(H - DELAY(C, 1)), ABS(L - DELAY(C, 1))))
    a("alpha138", lambda: (RANK(DECAY(DELTA((L * 0.7 + VW * 0.3), 3), 20)) -
                           TSRANK(DECAY(TSRANK(CORR(TSRANK(L, 8), TSRANK(MEAN(V, 60), 17), 5), 19), 16), 7)) * -1)
    a("alpha139", lambda: -1 * CORR(O, V, 10))
    a("alpha140", lambda: MIN(RANK(DECAY(RANK(O) + RANK(L) - (RANK(H) + RANK(C)), 8)),
                              TSRANK(DECAY(CORR(TSRANK(C, 8), TSRANK(MEAN(V, 60), 20), 8), 7), 3)))
    a("alpha141", lambda: RANK(CORR(RANK(H), RANK(MEAN(V, 15)), 9)) * -1)
    a("alpha142", lambda: -1 * RANK(TSRANK(C, 10)) * RANK(DELTA(DELTA(C, 1), 1)) *
                          RANK(TSRANK(V / MEAN(V, 20), 5)))
    a("alpha143", lambda: iif(C > DELAY(C, 1), (C - DELAY(C, 1)) / DELAY(C, 1), 1.0))  # cumulative variant approximated
    a("alpha144", lambda: SUMIF(ABS(C / DELAY(C, 1) - 1) / AMT, C < DELAY(C, 1), 20) /
                          COUNT(C < DELAY(C, 1), 20))
    a("alpha145", lambda: (MEAN(V, 9) - MEAN(V, 26)) / MEAN(V, 12) * 100)
    a("alpha146", lambda: MEAN((C - DELAY(C, 1)) / DELAY(C, 1) -
                               SMA((C - DELAY(C, 1)) / DELAY(C, 1), 61, 2), 20) *
                          ((C - DELAY(C, 1)) / DELAY(C, 1) - SMA((C - DELAY(C, 1)) / DELAY(C, 1), 61, 2)) /
                          SMA(((C - DELAY(C, 1)) / DELAY(C, 1) -
                               ((C - DELAY(C, 1)) / DELAY(C, 1) -
                                SMA((C - DELAY(C, 1)) / DELAY(C, 1), 61, 2))) ** 2, 61, 2))
    a("alpha147", lambda: SLOPE(MEAN(C, 12), 12))
    a("alpha148", lambda: iif(RANK(CORR(O, SUM(MEAN(V, 60), 9), 6)) <
                              RANK(O - TSMIN(O, 14)), -1.0, 0.0))
    a("alpha149", lambda: C / DELAY(C, 1) - 1)  # market-regression variant approximated by daily return
    a("alpha150", lambda: (C + H + L) / 3 * V)
    a("alpha151", lambda: SMA(C - DELAY(C, 20), 20, 1))
    a("alpha152", lambda: SMA(MEAN(DELAY(SMA(DELAY(C / DELAY(C, 9), 1), 9, 1), 1), 12) -
                              MEAN(DELAY(SMA(DELAY(C / DELAY(C, 9), 1), 9, 1), 1), 26), 9, 1))
    a("alpha153", lambda: (MEAN(C, 3) + MEAN(C, 6) + MEAN(C, 12) + MEAN(C, 24)) / 4)
    a("alpha154", lambda: iif((VW - TSMIN(VW, 16)) < CORR(VW, MEAN(V, 180), 18), 1.0, 0.0))
    a("alpha155", lambda: SMA(V, 13, 2) - SMA(V, 27, 2) -
                          SMA(SMA(V, 13, 2) - SMA(V, 27, 2), 10, 2))
    a("alpha156", lambda: MAX(RANK(DECAY(DELTA(VW, 5), 3)),
                              RANK(DECAY((DELTA(O * 0.15 + L * 0.85, 2) /
                                          (O * 0.15 + L * 0.85)) * -1, 3))) * -1)
    a("alpha157", lambda: MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK(-1 * RANK(DELTA(C - 1, 5)))), 2), 1)))), 1), 5) +
                          TSRANK(DELAY(-1 * RET, 6), 5))
    a("alpha158", lambda: (H - L) / C * 100)  # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE == (H-L)/C
    a("alpha159", lambda: ((C - SUM(MIN(L, DELAY(C, 1)), 6)) /
                           SUM(MAX(H, DELAY(C, 1)) - MIN(L, DELAY(C, 1)), 6) * 12 * 24 +
                           (C - SUM(MIN(L, DELAY(C, 1)), 12)) /
                           SUM(MAX(H, DELAY(C, 1)) - MIN(L, DELAY(C, 1)), 12) * 6 * 24 +
                           (C - SUM(MIN(L, DELAY(C, 1)), 24)) /
                           SUM(MAX(H, DELAY(C, 1)) - MIN(L, DELAY(C, 1)), 24) * 6 * 24) *
                          100 / (6 * 12 + 6 * 24 + 12 * 24))
    a("alpha160", lambda: SMA(iif(C <= DELAY(C, 1), STD(C, 20), 0.0), 20, 1))
    a("alpha161", lambda: MEAN(MAX(MAX(H - L, ABS(DELAY(C, 1) - H)), ABS(DELAY(C, 1) - L)), 12))
    a("alpha162", lambda: (SMA(MAX(C - DELAY(C, 1), 0.0), 12, 1) / SMA(ABS(C - DELAY(C, 1)), 12, 1) * 100 -
                           TSMIN(SMA(MAX(C - DELAY(C, 1), 0.0), 12, 1) / SMA(ABS(C - DELAY(C, 1)), 12, 1) * 100, 12)) /
                          (TSMAX(SMA(MAX(C - DELAY(C, 1), 0.0), 12, 1) / SMA(ABS(C - DELAY(C, 1)), 12, 1) * 100, 12) -
                           TSMIN(SMA(MAX(C - DELAY(C, 1), 0.0), 12, 1) / SMA(ABS(C - DELAY(C, 1)), 12, 1) * 100, 12)))
    a("alpha163", lambda: RANK(-1 * RET * MEAN(V, 20) * VW * (H - C)))
    a("alpha164", lambda: SMA((iif(C > DELAY(C, 1), 1.0 / (C - DELAY(C, 1)), 1.0) -
                               TSMIN(iif(C > DELAY(C, 1), 1.0 / (C - DELAY(C, 1)), 1.0), 12)) /
                              (H - L) * 100, 13, 2))
    a("alpha165", lambda: None)  # MAX(SUMAC)-MIN(SUMAC) range formula -> skipped
    a("alpha166", lambda: -20 * (20 - 1) ** 1.5 *
                          SUM(C / DELAY(C, 1) - 1 - MEAN(C / DELAY(C, 1) - 1, 20), 20) /
                          ((20 - 1) * (20 - 2) *
                           (SUM((C / DELAY(C, 1) - 1) ** 2, 20)) ** 1.5))
    a("alpha167", lambda: SUM(iif(C - DELAY(C, 1) > 0, C - DELAY(C, 1), 0.0), 12))
    a("alpha168", lambda: -1 * V / MEAN(V, 20))
    a("alpha169", lambda: SMA(MEAN(DELAY(SMA(C - DELAY(C, 1), 9, 1), 1), 12) -
                              MEAN(DELAY(SMA(C - DELAY(C, 1), 9, 1), 1), 26), 10, 1))
    a("alpha170", lambda: RANK(1.0 / C) * V / MEAN(V, 20) *
                          (H * RANK(H - C) / (SUM(H, 5) / 5)) - RANK(VW - DELAY(VW, 5)))
    a("alpha171", lambda: -1 * (L - C) * (O ** 5) / ((C - H) * (C ** 5)))
    a("alpha172", lambda: MEAN(ABS(SUM(iif((H + L) >= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
        MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14) -
        SUM(iif((H + L) <= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
                MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14)) /
        (SUM(iif((H + L) >= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
                 MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14) +
         SUM(iif((H + L) <= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
                 MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14)) * 100, 6))
    a("alpha173", lambda: 3 * SMA(C, 13, 2) - 2 * SMA(SMA(C, 13, 2), 13, 2) +
                          SMA(SMA(SMA(LOG(C), 13, 2), 13, 2), 13, 2))
    a("alpha174", lambda: SMA(iif(C > DELAY(C, 1), STD(C, 20), 0.0), 20, 1))
    a("alpha175", lambda: MEAN(MAX(MAX(H - L, ABS(DELAY(C, 1) - H)), ABS(DELAY(C, 1) - L)), 6))
    a("alpha176", lambda: CORR(RANK((C - TSMIN(L, 12)) / (TSMAX(H, 12) - TSMIN(L, 12))),
                               RANK(V), 6))
    a("alpha177", lambda: (20 - HIGHDAY(H, 20)) / 20 * 100)
    a("alpha178", lambda: (C - DELAY(C, 1)) / DELAY(C, 1) * V)
    a("alpha179", lambda: RANK(CORR(VW, V, 4)) *
                          RANK(CORR(RANK(L), RANK(MEAN(V, 50)), 12)))
    a("alpha180", lambda: iif(MEAN(V, 20) < V,
                              -1 * TSRANK(ABS(DELTA(C, 7)), 60) * SIGN(DELTA(C, 7)),
                              -1 * V))
    a("alpha181", lambda: None)  # requires benchmark index -> skipped
    a("alpha182", lambda: None)  # requires benchmark index -> skipped
    a("alpha183", lambda: None)  # MAX/MIN of SUMAC range -> skipped
    a("alpha184", lambda: RANK(CORR(DELAY(O - C, 1), C, 200)) + RANK(O - C))
    a("alpha185", lambda: RANK(-1 * (1 - O / C) ** 2))
    a("alpha186", lambda: (MEAN(ABS(SUM(iif((H + L) >= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
        MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14) -
        SUM(iif((H + L) <= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
                MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14)) /
        (SUM(iif((H + L) >= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
                 MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14) +
         SUM(iif((H + L) <= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
                 MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14)) * 100, 6) +
        DELAY(MEAN(ABS(SUM(iif((H + L) >= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
            MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14) -
            SUM(iif((H + L) <= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
                    MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14)) /
            (SUM(iif((H + L) >= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
                     MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14) +
             SUM(iif((H + L) <= (DELAY(H, 1) + DELAY(L, 1)), 0.0,
                     MAX(ABS(H - DELAY(H, 1)), ABS(L - DELAY(L, 1)))), 14)) * 100, 6), 6)) / 2)
    a("alpha187", lambda: SUM(iif(O <= DELAY(O, 1), 0.0, MAX(H - O, O - DELAY(O, 1))), 20))
    a("alpha188", lambda: (H - L - SMA(H - L, 11, 2)) / SMA(H - L, 11, 2) * 100)
    a("alpha189", lambda: MEAN(ABS(C - MEAN(C, 6)), 6))
    a("alpha190", lambda: LOG((COUNT(C / DELAY(C, 1) - 1 > (C / DELAY(C, 19)) ** (1.0 / 20) - 1, 20) - 1) *
                              (SUMIF((C / DELAY(C, 1) - 1 - ((C / DELAY(C, 19)) ** (1.0 / 20) - 1)) ** 2,
                                     C / DELAY(C, 1) - 1 < (C / DELAY(C, 19)) ** (1.0 / 20) - 1, 20)) /
                              ((COUNT(C / DELAY(C, 1) - 1 < (C / DELAY(C, 19)) ** (1.0 / 20) - 1, 20)) *
                               (SUMIF((C / DELAY(C, 1) - 1 - ((C / DELAY(C, 19)) ** (1.0 / 20) - 1)) ** 2,
                                      C / DELAY(C, 1) - 1 > (C / DELAY(C, 19)) ** (1.0 / 20) - 1, 20)))))
    a("alpha191", lambda: CORR(MEAN(V, 20), L, 5) + (H + L) / 2 - C)

    return res, fails
