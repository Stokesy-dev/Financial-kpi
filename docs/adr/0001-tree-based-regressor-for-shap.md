# ADR 0001: Tabular Tree-Based Regressor for SHAP Explainability

## Status
Accepted

## Context
The project brief requires a forecasting dashboard displaying SHAP waterfall plots showing the top contributing features for individual predictions. The primary time-series forecasting models specified are Facebook Prophet and ARIMA.

However, Prophet and ARIMA are statistical time-series models that do not natively support SHAP (which is designed for supervised machine learning models with a clear feature-response structure). 

Using SHAP's general `KernelExplainer` to wrap Prophet or ARIMA predictions is possible but has major disadvantages:
1. **High Computational Latency**: KernelExplainer is extremely slow, which makes interactive rendering in the Streamlit dashboard sluggish.
2. **Integration Instability**: Wrapping statsmodels' ARIMA or Prophet API inside KernelExplainer is brittle and prone to boundary condition issues (e.g., date indexing and exogenous variable alignments).

## Decision
We will:
1. Introduce a **Tabular Tree-based Regressor** (specifically a `RandomForestRegressor`) into the forecasting pipeline alongside Prophet and ARIMA.
2. Perform **Feature Engineering** (generating lag features, rolling averages, and seasonal indicators) to transform the time-series forecasting task into a supervised regression task.
3. Use this Tree-based Regressor as the exclusive target for generating SHAP values and waterfall plots.
4. Compare the performance (MAE, RMSE) of all three models (Prophet, ARIMA, and the Tree-based model) on the held-out validation quarter in the dashboard.

## Consequences

### Positive
- **Interactive Performance**: Native Tree SHAP (via `TreeExplainer`) runs in milliseconds, allowing users to dynamically select forecast dates and view explanation plots instantly.
- **Robust Local Explanations**: Features like lag values and rolling averages are directly interpretable and map logically to business questions (e.g., "how did last week's performance impact today's forecast?").
- **Model Comparison**: Stakeholders get a comparison of classical statistical methods (ARIMA), modern additive models (Prophet), and a machine learning regression model (Random Forest).

### Negative
- **Pipeline Complexity**: We need to engineer tabular features specifically for the tree-based model and maintain three distinct forecasting pipelines.
- **Warm-up Period**: Feature engineering (e.g., using a 30-day lag) means the tree-based model cannot make predictions for the very first 30 days of the dataset, slightly reducing the available training timeline.
