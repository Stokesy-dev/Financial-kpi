# Issue 0001: Baseline E-commerce Revenue Dashboard

## What to build

Implement the foundational data pipeline and visualization vertical slice. This includes generating a baseline (clean, un-anomalous) daily transactions dataset for the E-commerce business unit, storing it in a local SQLite database, writing a SQL Metric Aggregation query to compute the `Revenue` Metric dynamically, and rendering a baseline Streamlit dashboard containing a raw data explorer (Tab 1) and historical line chart (Tab 2). Set up a pytest suite to validate data generation and database extraction.

## Acceptance criteria

- [ ] Data generation script generates 3 years of daily transaction-level records for the E-commerce business unit.
- [ ] SQLite database file is created, containing a `transactions` table populated with the generated E-commerce transaction records.
- [ ] Relational data extraction works via SQL `GROUP BY` and date queries, dynamically aggregating daily transaction amounts into daily, weekly, and monthly `Revenue` metrics.
- [ ] Running `streamlit run app.py` launches the dashboard displaying E-commerce Revenue in Tab 1 (Raw Data Explorer) and Tab 2 (Historical actuals line chart).
- [ ] Pytest suite contains tests verifying that the generated dataset matches expected columns, non-empty shapes, and that SQL aggregation results are mathematically correct. All tests pass.

## Blocked by

None - can start immediately.
