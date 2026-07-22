# -*- coding: utf-8 -*-
"""Alpha158 factor construction.

Faithful re-implementation of qlib's standard Alpha158 feature set
(9 KBAR + 4 price + 29 rolling operators x 5 windows = 158 factors),
computed on wide (date x stock) panels.
"""
import numpy as np
import operators as op

EPS = 1e-12
WINDOWS = [5, 10, 20, 30, 60]


def compute(F, sink=None):
    """F: dict of wide frames with keys open/high/low/close/vwap/volume.
    If `sink` is given, each factor is passed to sink(name, frame) and NOT
    retained (memory-safe streaming); otherwise a dict {name: frame} is returned."""
    O, H, L, C, VW, V = F["open"], F["high"], F["low"], F["close"], F["vwap"], F["volume"]
    hl = (H - L)
    out = None if sink is not None else {}

    def emit(name, val):
        if sink is not None:
            sink(name, val)
        else:
            out[name] = val

    # ---- KBAR (9) ----------------------------------------------------------
    emit("KMID",  (C - O) / O)
    emit("KLEN",  hl / O)
    emit("KMID2", (C - O) / (hl + EPS))
    emit("KUP",   (H - op.emax(O, C)) / O)
    emit("KUP2",  (H - op.emax(O, C)) / (hl + EPS))
    emit("KLOW",  (op.emin(O, C) - L) / O)
    emit("KLOW2", (op.emin(O, C) - L) / (hl + EPS))
    emit("KSFT",  (2 * C - H - L) / O)
    emit("KSFT2", (2 * C - H - L) / (hl + EPS))

    # ---- price (4) ---------------------------------------------------------
    emit("OPEN0", O / C)
    emit("HIGH0", H / C)
    emit("LOW0",  L / C)
    emit("VWAP0", VW / C)

    # helpers for rolling block
    dC = C - op.delay(C, 1)
    dV = V - op.delay(V, 1)
    logV1 = op.log(V + 1)
    ret1 = C / op.delay(C, 1)
    dlogV = op.log((V / op.delay(V, 1)) + 1)
    absret_vol = (C / op.delay(C, 1) - 1).abs() * V
    up_c = (dC > 0).astype(float);   up_c[dC.isna()] = np.nan
    dn_c = (dC < 0).astype(float);   dn_c[dC.isna()] = np.nan

    for d in WINDOWS:
        emit(f"ROC{d}",  op.delay(C, d) / C)
        emit(f"MA{d}",   op.ts_mean(C, d) / C)
        emit(f"STD{d}",  op.ts_std(C, d) / C)
        emit(f"BETA{d}", op.ts_slope(C, d) / C)
        emit(f"RSQR{d}", op.ts_rsquare(C, d))
        emit(f"RESI{d}", op.ts_resi(C, d) / C)
        emit(f"MAX{d}",  op.ts_max(H, d) / C)
        emit(f"MIN{d}",  op.ts_min(L, d) / C)
        emit(f"QTLU{d}", op.ts_quantile(C, d, 0.8) / C)
        emit(f"QTLD{d}", op.ts_quantile(C, d, 0.2) / C)
        emit(f"RANK{d}", op.ts_rank(C, d))
        mn = op.ts_min(L, d)
        emit(f"RSV{d}",  (C - mn) / (op.ts_max(H, d) - mn + EPS))
        emit(f"IMAX{d}", op.ts_argmax(H, d) / d)
        emit(f"IMIN{d}", op.ts_argmin(L, d) / d)
        emit(f"IMXD{d}", (op.ts_argmax(H, d) - op.ts_argmin(L, d)) / d)
        emit(f"CORR{d}", op.ts_corr(C, logV1, d))
        emit(f"CORD{d}", op.ts_corr(ret1, dlogV, d))
        cntp = up_c.rolling(d, min_periods=d).mean()
        cntn = dn_c.rolling(d, min_periods=d).mean()
        emit(f"CNTP{d}", cntp)
        emit(f"CNTN{d}", cntn)
        emit(f"CNTD{d}", cntp - cntn)
        sabs = op.ts_sum(dC.abs(), d)
        sump = op.ts_sum(op.emax(dC, 0), d) / (sabs + EPS)
        sumn = op.ts_sum(op.emax(-dC, 0), d) / (sabs + EPS)
        emit(f"SUMP{d}", sump)
        emit(f"SUMN{d}", sumn)
        emit(f"SUMD{d}", sump - sumn)
        emit(f"VMA{d}",  op.ts_mean(V, d) / (V + EPS))
        emit(f"VSTD{d}", op.ts_std(V, d) / (V + EPS))
        emit(f"WVMA{d}", op.ts_std(absret_vol, d) / (op.ts_mean(absret_vol, d) + EPS))
        vsabs = op.ts_sum(dV.abs(), d)
        vsump = op.ts_sum(op.emax(dV, 0), d) / (vsabs + EPS)
        vsumn = op.ts_sum(op.emax(-dV, 0), d) / (vsabs + EPS)
        emit(f"VSUMP{d}", vsump)
        emit(f"VSUMN{d}", vsumn)
        emit(f"VSUMD{d}", vsump - vsumn)

    return out


ALPHA158_NAMES = None  # filled after first compute for reference
