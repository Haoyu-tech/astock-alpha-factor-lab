<div align="center">

# A-Share Alpha Factor Lab

From formulaic factors to walk-forward stock selection and portfolio optimization.

![Python: 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square) ![Factors: Alpha158 + GTJA191](https://img.shields.io/badge/Factors-Alpha158%20%2B%20GTJA191-0F766E?style=flat-square) ![Models: LightGBM](https://img.shields.io/badge/Models-LightGBM-7C3AED?style=flat-square)

[Quick start](#quick-start) · [Project map](#project-map) · [Reproduction guide](docs/REPRODUCIBILITY.md)

</div>

---

## Overview

Construct Alpha158 and Guotai Junan Alpha191 factors from daily A-share price and volume data. Study information coefficients, correlations, market-cap groups and signal horizons, then evaluate LightGBM portfolios under walk-forward validation.

## Research workflow

```text
Tushare panel → Factor libraries → IC and decile tests → Walk-forward ML → Portfolio weights
```

| Component | Coverage |
|---|---|
| Factor construction | 158 Alpha158 factors; 186 price-volume GTJA191 factors |
| Diagnostics | Rank IC, correlations, market-cap groups, index groups and yearly trends |
| Portfolio study | Single-factor deciles, LightGBM stock selection and weight optimization |
| Data | Tushare Pro daily data; original study covers October 2014–July 2026 |

## Selected recorded results

| Configuration | Annualized long–short return | Sharpe | Annualized long-only return |
|---|---:|---:|---:|
| CSI 1000, bottom 10% by size excluded, monthly | 24.6% | 1.06 | 9.5% |
| CSI 1000, no size exclusion, monthly | 30.3% | 1.50 | 10.1% |
| Broad index universe, monthly | 14.6% | 0.73 | 5.3% |
| Full market, monthly | 57.7% | 2.27 | 24.7% |
| CSI 1000, bottom 10% excluded, weekly | 91.5% | 3.01 | 18.7% |

These are previously reported results **before trading costs**. The smallest stocks and high-turnover configurations can be difficult to trade; full-market and weekly results should be read as signal studies. Universe selection and survivorship assumptions also affect interpretation.

## Visual results

| Factor effectiveness | Walk-forward portfolios |
|---|---|
| ![Factor IC distributions](assets/01_factor_ic_distribution.png) | ![LightGBM equity curves](assets/09_ml_equity.png) |
| ![IC by market-cap group](assets/02_mktcap_decile.png) | ![Portfolio weight comparison](assets/11_weight_optimization.png) |

![Universe and rebalancing comparison](assets/12_universe_comparison.png)

## Quick start

Install dependencies and configure a Tushare Pro token with the required permissions:

```powershell
python -m pip install -r requirements.txt
python -c "import tushare as ts; ts.set_token('YOUR_TUSHARE_TOKEN')"
python -c "import data_loader as dl; dl.get_daily_panel()"
python run_stream.py
python finalize.py
python factor_decile.py
python build_monthly.py
python ml_lgbm.py
python portfolio_opt.py
```

Run commands from the repository root. Keep `FR_UNIV`, `FR_DROP`, `FR_FREQ` and `FR_TAG` consistent across dependent steps.

## Project map

| Module | Role |
|---|---|
| [data_loader.py](data_loader.py) | Tushare ingestion and cached panels |
| [alpha158.py](alpha158.py) / [gtja191.py](gtja191.py) | Factor construction |
| [operators.py](operators.py) / [factor_test.py](factor_test.py) | Operators and statistical diagnostics |
| [run_stream.py](run_stream.py) | Streaming factor workflow |
| [build_monthly.py](build_monthly.py) / [ml_lgbm.py](ml_lgbm.py) | Model data and walk-forward training |
| [portfolio_opt.py](portfolio_opt.py) | Portfolio weight optimization |
| [make_report.py](make_report.py) / [compare_all.py](compare_all.py) | Reports and cross-configuration comparisons |

## Study notes

The original study reports stronger signals in smaller stocks and useful cross-library redundancy checks. Monthly CSI 1000 portfolios provide a more restrained comparison than full-market weekly tests. Industry/size neutrality and realistic financing and trading constraints require additional evaluation.

The complete earlier narrative and figure catalog are retained in [the original study notes, in Chinese](docs/PROJECT_REFERENCE.md).

## Documentation

- [Reproduction guide](docs/REPRODUCIBILITY.md): inputs, execution order and validation limits.
- [Code and artifact map](docs/REPOSITORY_MAP.md): source and configuration paths.

<sub>Research artifacts reflect the recorded experiment settings. Full results were not regenerated during documentation cleanup.</sub>
