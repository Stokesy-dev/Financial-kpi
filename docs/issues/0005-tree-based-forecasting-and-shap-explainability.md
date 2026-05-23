# Issue 0005: Tabular Machine Learning Forecast & Interactive SHAP Explanations

## What to build

Implement the machine learning forecasting and explainability layer. Train a Tabular Tree-based Regressor (Random Forest) on the engineered lag, rolling, and seasonal features. Compute validation MAE and RMSE on the held-out quarter, displaying them alongside Prophet and ARIMA in Tab 2. Integrate local prediction explanations using SHAP's `TreeExplainer` on the Random Forest. Add Tab 4 (SHAP Explanations) to the Streamlit UI, allowing the user to select any forecast date in the horizon and view a dynamically updated SHAP waterfall plot showing positive and negative feature contributions.

## Acceptance criteria

- [ ] Tabular Tree-based Regressor (Random Forest) trains on engineered features and generates predictions for the 90-day forecast horizon.
- [ ] Random Forest MAE and RMSE scores are computed and displayed in the Tab 2 model comparison table.
- [ ] SHAP values are computed using `TreeExplainer` on the Random Forest model.
- [ ] Streamlit Tab 4 displays an interactive SHAP waterfall plot. When the user selects a forecast date (via dropdown or slider), the plot dynamically updates to show feature attributions for that specific day's prediction.
- [ ] Pytest suite contains tests validating Random Forest training, forecast output shapes, and SHAP value matrices correctness. All tests pass.

## Blocked by

- [Issue 0004: Classical Time-Series Forecasting](file:///Users/sohamwarad/Financial%20KPI/docs/issues/0004-classical-time-series-forecasting.md)
