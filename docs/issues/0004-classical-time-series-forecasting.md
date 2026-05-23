# Issue 0004: Classical Time-Series Forecasting

## What to build

Integrate classical time-series forecasting models (Facebook Prophet and statsmodels ARIMA) in the backend forecasting module. Implement a temporal validation split (using the first 2.75 years for training and the final 90 days as the held-out quarter). Compute validation metrics (MAE and RMSE) comparing forecasts against the clean, un-anomalous baseline metrics. Update Tab 2 (Forecasting) of the Streamlit dashboard to display actuals, forecasts, confidence intervals, and a table comparing Prophet and ARIMA MAE/RMSE scores. Implement CLI support for `--mode forecast` with `--bu` and `--metric` flags.

## Acceptance criteria

- [ ] Forecasting engine successfully trains Prophet and ARIMA models on historical training data.
- [ ] Model forecasts generate predictions and confidence intervals for the 90-day forecast horizon.
- [ ] MAE and RMSE are computed on the held-out quarter validation set against clean baseline metrics.
- [ ] Dashboard Tab 2 displays actuals, Prophet forecast, ARIMA forecast, confidence intervals, and a model comparison table showing MAE and RMSE.
- [ ] Running `python main.py --mode forecast --bu saas --metric revenue` runs the forecasting pipeline and prints the MAE/RMSE validation scores in the terminal.
- [ ] Pytest suite contains tests validating model fitting, inference output shapes, and MAE/RMSE calculation correctness. All tests pass.

## Blocked by

- #3
