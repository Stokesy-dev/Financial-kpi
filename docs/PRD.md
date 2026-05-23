# Product Requirements Document (PRD)
## Financial KPI Anomaly Detector & Revenue Forecasting Dashboard

## Problem Statement

Financial operations teams currently rely on static reports that fail to highlight early indicators of financial risks or opportunities. When revenue dips, costs spike, or transaction volumes change unexpectedly, these issues are often noticed too late because standard reports lack proactive outlier detection and forward-looking forecasts. Additionally, non-technical stakeholders struggle to trust automated forecasts because they cannot see or understand the driving factors behind the model's predictions.

## Solution

Build an interactive, intelligent dashboard that proactively detects anomalies, forecasts key financial metrics, and provides clear, explainable AI explanations for non-technical users. 
The solution will:
1. Generate daily transaction-level records across three distinct business units and store them in a local relational database.
2. Provide a forecasting layer using three models (Prophet, ARIMA, and a Tabular Tree-based model) compared on a held-out validation quarter.
3. Detect anomalies using a combined approach (multivariate Isolation Forest and univariate Z-score thresholding) mapped to High, Medium, and Low confidence ratings.
4. Integrate SHAP explainability to explain individual daily forecast predictions interactively.
5. Provide a Streamlit dashboard for exploration, forecasting comparison, anomaly flagging, and interactive forecast explanations.

## User Stories

1. As a Financial Operations Manager, I want to explore raw daily transactions and view automated EDA summary statistics, so that I can understand the distribution and summary profiles of our data.
2. As a Business Analyst, I want to filter transaction data by Business Unit (SaaS, E-commerce, Enterprise Services) and Metric (Revenue, Cost, Volume), so that I can focus my analysis on specific business areas.
3. As a Finance Lead, I want to view a 90-day forecast of Revenue, Cost, and Volume, so that I can plan budgets and resource allocation for the upcoming quarter.
4. As a Finance Lead, I want to see actual metrics plotted alongside forecasted values and confidence intervals, so that I can visualize the range of expected outcomes.
5. As an Analytics Engineer, I want to compare the forecasting performance (MAE and RMSE) of Prophet, ARIMA, and the Tabular Tree-based Regressor on a held-out validation quarter, so that I can understand which model is performing best.
6. As a Risk Analyst, I want the system to flag anomalies in our financial metrics and categorize them by confidence levels (High, Medium, Low), so that I can prioritize reviewing the most critical issues.
7. As a Support Specialist, I want to spot E-commerce Checkout Gateway Outages early, so that we can coordinate with engineering to fix payment bugs.
8. As a SaaS Product Manager, I want to detect Infrastructure Cost Leaks immediately, so that we can shut down runaway cloud jobs before they exceed budget.
9. As an Auditor, I want to flag E-commerce Pricing Glitches (volume spikes with low revenue), so that we can cancel orders placed under incorrect promotional prices.
10. As an Enterprise Account Director, I want to see Contract Delays flagged in my business unit, so that I can follow up on large missing enterprise deals.
11. As a Stakeholder, I want to select a specific forecast date in the future and view a SHAP waterfall plot showing the top features (e.g., lags, rolling means, day of week) that drove that day's prediction, so that I can understand the drivers behind the forecast and build trust in the model.
12. As a Systems Operator, I want to run the pipeline via CLI to generate forecasts or anomalies for specific business units and metrics, so that I can automate running the backend models in background scripts.

## Implementation Decisions

### Core Modules
- **`data_generation`**: Generates synthetic transaction-level data (3 years of daily transactions) across E-commerce, SaaS, and Enterprise Services, injecting controlled anomalies (Checkout Outage, Infrastructure Cost Leak, Pricing Glitch, Contract Delay).
- **`db_interface`**: Manages the SQLite database connection, table schemas, and queries. It stores raw transaction records in a `transactions` table and performs metric aggregation dynamically using SQL `GROUP BY` and date functions.
- **`feature_engineering`**: Engineers input variables (lag features, rolling averages, seasonal indicators) from the aggregated metrics to train the Tree-based model.
- **`forecasting_engine`**: Manages model training, evaluation, and inference. It trains Prophet, ARIMA (SARIMAX), and a Tabular Tree-based Regressor (Random Forest) using a temporal split (first 2.75 years for training, final 90 days for validation).
- **`anomaly_detector`**: Runs multivariate Isolation Forest and univariate Z-score thresholding on the aggregated metrics, outputting combined confidence ratings based on whether both, one, or neither flag the data.
- **`shap_explainability`**: Generates local prediction explanations using SHAP's `TreeExplainer` on the Tabular Tree-based model.
- **`dashboard`**: Multi-tab Streamlit dashboard:
  - **Tab 1: Raw Data & EDA**: Summary statistics and raw table filters.
  - **Tab 2: Forecasting**: Interactive Plotly plots showing actuals vs forecast with confidence intervals, model selector, and MAE/RMSE comparisons.
  - **Tab 3: Anomalies**: Time-series plots with color-coded points indicating anomaly confidence levels.
  - **Tab 4: SHAP Explanations**: Interactive waterfall plot for a selected forecast date.
- **`cli`**: Parser for command execution modes (`forecast`, `anomaly`) taking `--business-unit` (saas, ecommerce, enterprise) and `--metric` (revenue, cost, volume) inputs.

### Key Architectural Choices
- **Tree-based Regressor for SHAP**: Due to Prophet/ARIMA not natively supporting SHAP explanations, a supervised Tabular Tree-based model (Random Forest) is introduced. It is trained on engineered features (lags and rolling averages) and explained using fast, native Tree SHAP.
- **Dynamic Database Aggregation**: Relational database integrity is maintained by storing only raw transactions. All metrics are aggregated dynamically on-the-fly using SQL queries, preventing data duplicity and stale tables.
- **Clean Validation Splitting**: To evaluate forecasting performance, the final 90 days are held out. The error metrics are calculated by comparing forecasts against the clean, un-anomalous baseline metrics to ensure anomalies do not artificially skew the evaluation.

## Testing Decisions

### Test Strategy
- We will focus entirely on **functional, behavior-driven testing** of core backend modules in isolation rather than testing internal states or front-end rendering.
- All tests will be written using `pytest`.

### Tested Modules
- **`data_generation`**: Verify that transaction datasets are created with the correct shapes, ranges, columns, and that controlled anomalies are successfully injected at the targeted dates.
- **`db_interface`**: Verify table creation, transaction writing, and correctness of custom SQL dynamic aggregation queries (e.g., verifying that aggregated values match expected math).
- **`feature_engineering`**: Verify that lag features, rolling windows, and seasonal indicators are correctly aligned and do not introduce look-ahead bias (data leakage).
- **`forecasting_engine`**: Verify that all three models train without error, predict over the 90-day forecast horizon, and output reasonable MAE/RMSE scores.
- **`anomaly_detector`**: Verify that Isolation Forest and Z-score thresholding run correctly, and that combined confidence rules are accurately mapped.
- **`shap_explainability`**: Verify that Tree SHAP computes values with correct shapes and formats.

## Out of Scope

- **Production-grade Database Integration**: Multi-user, cloud-hosted relational databases (e.g., PostgreSQL or MySQL) are out of scope; local SQLite is sufficient.
- **Real-Time Data Ingestion**: The database ingestion pipeline runs as a batch script rather than streaming real-time metrics.
- **Fine-Tuning & Hyperparameter Search**: Models will use standard, reasonable default parameters; complex auto-tuning pipelines are out of scope.
- **User Authentication**: There is no login screen or user authorization level system.
- **Cloud Deployment**: Deployment to hosting services (e.g., Streamlit Community Cloud, AWS, GCP) is out of scope.

## Further Notes

- **Model Latency**: Feature engineering and forecasting models must be designed to run fast. The pre-computation of SHAP values and forecasts should ideally happen during pipeline runs, allowing the dashboard to read pre-calculated tables or run rapid cached inferences for a smooth user experience.
