# -*- coding: utf-8 -*-
"""Global configuration for the Alpha158 / GTJA-191 factor study."""
import os

# ---- paths -----------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
# paths can be redirected via env vars (used for isolated testing)
DATA_DIR = os.environ.get("FR_DATA_DIR", os.path.join(ROOT, "data"))
RESULTS_DIR = os.environ.get("FR_RESULTS_DIR", os.path.join(ROOT, "results"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- study window ----------------------------------------------------------
# Maximum window supported by the 4-index universe: 中证1000 (000852.SH) has
# constituent data from its 2014-10 inception; latest published daily bar is
# 2026-07-21 (today's bars are not yet available).
START_DATE = "20141017"
END_DATE = "20260721"

# ---- fetch / compute tuning ------------------------------------------------
WORKERS = 12          # tushare concurrent request threads (~160 calls/min/endpoint)
COL_CHUNK = 300       # stock-column block size for memory-bounded window ops

# ---- index universe --------------------------------------------------------
# tushare index_weight codes for the four target indices.
INDEX_CODES = {
    "HS300":  "399300.SZ",   # 沪深300  (large cap)
    "ZZ500":  "000905.SH",   # 中证500  (mid cap)
    "ZZ1000": "000852.SH",   # 中证1000 (small cap)
    "CYB":    "399006.SZ",   # 创业板指  (growth board)
}
INDEX_LABELS = {
    "HS300":  "沪深300",
    "ZZ500":  "中证500",
    "ZZ1000": "中证1000",
    "CYB":    "创业板指",
}

# ---- factor-test parameters ------------------------------------------------
FWD_PERIODS = [1, 5, 10]      # forward-return horizons (trading days) for IC
N_MKTCAP_GROUPS = 10          # market-cap deciles (D01 = smallest .. D10 = largest)
IC_MIN_STOCKS = 30            # min cross-section size to compute a daily IC
