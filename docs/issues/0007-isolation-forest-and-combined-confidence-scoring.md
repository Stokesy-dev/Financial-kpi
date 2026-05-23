# Issue 0007: Multivariate Isolation Forest & Combined Confidence Scores

## What to build

Implement Scikit-learn's multivariate Isolation Forest in the anomaly detection backend module. Combine Isolation Forest and univariate Z-score flags to compute combined Anomaly Confidence Scores: High (detected by both), Medium (detected by Isolation Forest only), and Low (detected by Z-score only). Map CLI thresholds to Isolation Forest contamination settings (Strict: 1%, Standard: 3%, Lenient: 5%). Update Tab 3 (Anomalies) in the Streamlit UI to color-code anomaly markers by confidence levels (High = Red, Medium = Orange, Low = Yellow).

## Acceptance criteria

- [ ] Backend anomaly module runs multivariate Isolation Forest on aggregated metric features (Revenue, Cost, Volume).
- [ ] Isolation Forest thresholding is correctly mapped to CLI strict/standard/lenient contamination configurations.
- [ ] Combined anomaly logic correctly evaluates and assigns High, Medium, and Low confidence ratings based on multivariate and univariate flagging rules.
- [ ] Streamlit Tab 3 anomaly plot renders flagged points with color-coding corresponding to High (Red), Medium (Orange), and Low (Yellow) confidence.
- [ ] CLI `python main.py --mode anomaly` outputs tables with flagged anomalies including their confidence scores.
- [ ] Pytest suite contains tests validating Isolation Forest execution, contamination thresholds, and correctness of combined confidence score mapping rules. All tests pass.

## Blocked by

- [Issue 0005: Tabular Machine Learning Forecast & Interactive SHAP Explanations](file:///Users/sohamwarad/Financial%20KPI/docs/issues/0005-tree-based-forecasting-and-shap-explainability.md)
- [Issue 0006: Anomaly Injection & Univariate Z-Score Flagging](file:///Users/sohamwarad/Financial%20KPI/docs/issues/0006-anomaly-injection-and-univariate-zscore.md)
