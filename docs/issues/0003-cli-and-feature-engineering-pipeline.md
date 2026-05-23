# Issue 0003: CLI & Tabular Feature Engineering Pipeline

## What to build

Build the `feature_engineering` module to transform time-series metrics into a tabular structure with lag features (e.g., $t-1$, $t-7$, $t-30$), rolling averages (7-day, 30-day), and seasonal indicators (day of week, month). Set up the main CLI entrypoint (`main.py`) to run data generation and log feature engineering pipeline diagnostics. Validate the feature engineering module with pytest to ensure correctness and prevent look-ahead bias (data leakage).

## Acceptance criteria

- [ ] Feature engineering module constructs lag features, rolling means, and seasonal indicators for a given aggregated metric DataFrame.
- [ ] No future data leakage is introduced (i.e., feature values at date $t$ are derived exclusively from observations on or before $t-1$).
- [ ] Running `python main.py` triggers the data generation pipeline and prints feature extraction shapes and summary stats in the terminal.
- [ ] Pytest suite contains tests validating that feature engineering generates correct lags and rolling averages (e.g., verifying math on simple dummy arrays) and that there are no NaNs in the output outside of the expected initial warm-up period.

## Blocked by

- #2
