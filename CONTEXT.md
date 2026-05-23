# Domain Context Glossary

This document defines the key domain concepts and terminology for the Financial KPI Anomaly Dashboard.

## Core Domain Terms

### Transaction
A raw, daily transaction-level record containing:
- **TransactionID**: Unique identifier for each transaction.
- **Date**: The calendar date when the transaction occurred.
- **Business Unit**: The operating division to which the transaction belongs.
- **Type**: The financial category (e.g., Revenue or Cost).
- **Amount**: The monetary value of the transaction.
- **Volume**: The physical or unit quantity associated with the transaction.

### Business Unit (BU)
A distinct operating division of the company. The dashboard models three specific business units, each representing a unique business model:
1. **SaaS (Software as a Service)**: Characterized by steady monthly recurring revenue (MRR) growth, high profit margins (low relative costs), and predictable, medium-frequency subscription transactions.
2. **E-commerce**: Characterized by high transaction volume, lower profit margins (higher relative variable costs), strong weekly and annual seasonality (with peaks during holidays/Q4), and high transaction frequency.
3. **Enterprise Services**: Characterized by low transaction volume, very high transaction amounts (lumpy revenue from large project contracts), and irregular contract win intervals.

### Metric
An aggregated financial indicator calculated from raw Transactions over a specific time granularity (daily, weekly, or monthly). The primary metrics are:
- **Revenue**: The sum of transaction amounts for revenue-generating events.
- **Cost**: The sum of transaction amounts for cost-incurring events.
- **Volume**: The sum of transaction volumes.

### Transaction Store
The persistent registry of raw Transaction records, storing transaction-level details chronologically.

### Metric Aggregation
The process of grouping and summing raw Transactions from the Transaction Store over a specified time frequency (daily, weekly, or monthly) and Business Unit to compute comparative Metrics.

## Anomalies

### Anomaly
An unexpected observation or sequence of observations in one or more Metrics that significantly deviates from historical patterns, seasonality, and trends.

### Checkout Gateway Outage
A business-critical anomaly where customers cannot complete purchases, characterized by a concurrent, sudden dip in both **Revenue** and **Volume** (specific to E-commerce).

### Infrastructure Cost Leak
An anomaly where operational expenses spike due to inefficient infrastructure scaling or errors, characterized by a sudden increase in **Cost** without a matching change in **Revenue** or **Volume** (specific to SaaS).

### Pricing Glitch / Promo Exploit
An anomaly where items are sold at incorrect prices, characterized by a sudden, massive increase in **Volume** accompanied by a dip or stagnation in **Revenue** (specific to E-commerce).

### Contract Delay
An anomaly where a large enterprise deal fails to close on time, characterized by a significant, unexpected dip in monthly **Revenue** with a low transaction **Volume** (specific to Enterprise Services).

### Isolation Forest Anomaly Score
A metric representing the anomaly score from the Isolation Forest model based on multivariate inputs (Revenue, Cost, Volume). Values closer to -1 indicate highly anomalous behavior.

### Z-Score Anomaly Score
A metric representing the number of standard deviations a specific observation is from the moving historical mean of that metric.

### Anomaly Confidence Score
A categorical rating representing the likelihood that a flagged observation is a true anomaly:
- **High Confidence**: An anomaly detected by both the multivariate Isolation Forest model and the univariate Z-score threshold.
- **Medium Confidence**: An anomaly detected by the multivariate Isolation Forest model only, signifying broken relationships between metrics (e.g., pricing glitch).
- **Low Confidence**: An anomaly detected by the Z-score threshold only, indicating a simple single-metric outlier that did not disrupt overall business patterns.

### Anomaly Detection Threshold
A business policy configuration that controls the sensitivity of the anomaly detection system, balancing precision (minimizing false alarms) and recall (minimizing missed anomalies):
- **Strict (High Precision)**: Configured to flag only the most severe, high-confidence anomalies (e.g., Z-score > 3.5, Isolation Forest contamination = 1%). Focuses on minimizing false alerts.
- **Standard (Balanced)**: The default configuration balancing sensitivity and alert volume (e.g., Z-score > 3.0, Isolation Forest contamination = 3%).
- **Lenient (High Recall)**: Configured to capture all potential anomalies, including minor deviations (e.g., Z-score > 2.0, Isolation Forest contamination = 5%). Intended for deep manual audits.

## Forecasting and Explainability

### Forecasting Model
A mathematical or statistical model trained on historical Metric values to predict future values of that Metric.

### Prophet
A additive time-series forecasting model that decomposes a time series into trend, seasonal effects (yearly, weekly, daily), and holiday effects.

### ARIMA (Autoregressive Integrated Moving Average)
A classic statistical time-series model that forecasts future values using linear combinations of past values (autoregressive terms), past errors (moving average terms), and differences of the time series to achieve stationarity.

### Tabular Tree-based Regressor
A supervised machine learning model (e.g., Random Forest or XGBoost) that predicts future Metric values using structured features engineered from the time series. This model serves as the primary target for SHAP explainability.

### Feature Engineering
The process of constructing explanatory features from the raw time series to train the Tabular Tree-based Regressor.
- **Lag Feature**: The value of a Metric at a set number of days in the past (e.g., $t-1$, $t-7$, $t-30$).
- **Rolling Mean**: The moving average of a Metric over a defined historical window (e.g., 7-day or 30-day moving average).
- **Seasonal Indicator**: Calendar-based feature extraction such as day of the week, day of the month, or month of the year.

### SHAP (SHapley Additive exPlanations)
A cooperative game theory approach to explain the output of a machine learning model, quantifying the positive or negative contribution of each engineered feature (e.g., lags, rolling means) to a specific prediction.

### Local Prediction Explanation
An explanation of a single forecasted data point, showing how much each individual input feature (e.g., specific lags or rolling means) increased or decreased that specific prediction relative to the base value (average forecast).

## Model Evaluation and Validation

### Temporal Train/Test Split
A partition of time-series data into training and validation sets where all training observations chronologically precede the validation observations, preventing future data leakage.

### Held-out Quarter
The final 90-day period of the 3-year dataset reserved exclusively as the validation set for model performance evaluation.

### Baseline Metric
The theoretical, un-anomalous value of a Metric representing normal business operations, used to calculate true forecasting accuracy.

### Mean Absolute Error (MAE)
The average magnitude of errors between forecasted values and baseline values, where all individual errors are weighted equally.

### Root Mean Squared Error (RMSE)
The square root of the average of squared differences between forecasted values and baseline values, giving higher weight to larger forecasting errors.
