# Issue 0006: Anomaly Injection & Univariate Z-Score Flagging

## What to build

Update the synthetic data generation script to inject controlled, domain-specific anomalies: E-commerce Checkout Outage (revenue/volume dip), SaaS Infrastructure Cost Leak (cost spike), E-commerce Pricing Glitch (volume spike, revenue dip), and Enterprise Services Contract Delay (monthly revenue dip). Implement univariate Z-score thresholding in the anomaly detection backend module. Update the CLI `--mode anomaly` to support strict/standard/lenient thresholds mapped to Z-score limits (Strict: >3.5, Standard: >3.0, Lenient: >2.0). Display univariate flagged anomalies on the Tab 3 (Anomalies) Plotly time-series chart.

## Acceptance criteria

- [ ] Data generation script injects E-commerce Checkout Outage, SaaS Infrastructure Cost Leak, E-commerce Pricing Glitch, and Enterprise Services Contract Delay anomalies at specific, pre-determined date ranges.
- [ ] Backend anomaly module calculates moving Z-scores on aggregated metrics and flags anomalies using the user-specified threshold.
- [ ] Running `python main.py --mode anomaly --threshold strict` executes the anomaly detection pipeline and prints a list of flagged dates.
- [ ] Streamlit Tab 3 (Anomalies) displays the metric time series with univariate flagged points highlight-colored.
- [ ] Pytest suite contains tests validating anomaly injection dates and Z-score flagging accuracy. All tests pass.

## Blocked by

- [Issue 0002: Multi-BU & Multi-Metric Extension](file:///Users/sohamwarad/Financial%20KPI/docs/issues/0002-multi-bu-multi-metric-extension.md)
