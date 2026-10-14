# Reproduction guide

## Environment and data

Run from the repository root with the Python 3.10 environment described in [README](../README.md). Install [requirements.txt](../requirements.txt) and configure a Tushare Pro token with the required data access.

```powershell
python -m pip install -r requirements.txt
```

Raw/cached panels and per-run outputs must be prepared locally. The committed figures are selected historical artifacts, not a replacement for the input data.

## Execution order

Follow the full commands in the README:

1. Fetch and cache the daily panel using `data_loader.get_daily_panel()`.
2. Run `run_stream.py`, then `finalize.py`.
3. Run `factor_decile.py` for single-factor portfolios.
4. Run `build_monthly.py`, `ml_lgbm.py`, then `portfolio_opt.py`.
5. Generate reports using `make_report.py` and `compare_all.py`.

Use the same `FR_UNIV`, `FR_DROP`, `FR_FREQ` and `FR_TAG` settings consistently across dependent steps. Otherwise monthly-panel, model and portfolio artifacts can represent different configurations.

## Results and limits

The README presents historical results before trading costs and discusses universe selection. Record the downloaded data cutoff, historical membership, missing-data handling and software versions before comparing reruns. Live data revisions and access permissions can change the generated panel.

## Run record

For every result you intend to compare or share, retain the repository commit, exact command/configuration, input-data identity, environment versions and output location. Save a dependency snapshot after a successful reference run; newly added requirements lists are intentionally unpinned until that validation exists.

## Validation status

Cleanup review: 14 October 2026. File paths and documented entry points were checked against the checkout. Full empirical results were not regenerated. Project-specific external prerequisites and known blockers are listed above.
