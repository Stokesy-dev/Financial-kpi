# Issue 0002: Multi-BU & Multi-Metric Extension

## What to build

Extend the data generation and database layers to support SaaS (characterized by steady MRR growth and low relative costs) and Enterprise Services (lumpy contracts with high transaction amounts) business units. Update SQL queries to dynamically aggregate all three Metrics (Revenue, Cost, Volume) across all three Business Units. Expand the Streamlit dashboard (Tab 1 and Tab 2 baseline) with filters to let users interactively select and plot metrics for specific business units.

## Acceptance criteria

- [ ] Data generation script generates 3 years of daily transaction-level records for SaaS and Enterprise Services business units matching their distinct domain behaviors.
- [ ] SQLite database is successfully populated with SaaS and Enterprise Services transaction records in the `transactions` table.
- [ ] SQL aggregation queries correctly extract and aggregate daily, weekly, and monthly Revenue, Cost, and Volume metrics for all three business units.
- [ ] Streamlit UI (Tab 1 and Tab 2) includes dropdown filters allowing users to select a Business Unit (SaaS, E-commerce, Enterprise Services) and a Metric (Revenue, Cost, Volume). Selecting a combination dynamically updates the raw data table and baseline line chart.
- [ ] Pytest suite is expanded with tests validating multi-BU properties (e.g., higher margins for SaaS, low contract frequency for Enterprise) and verification of metric calculations. All tests pass.

## Blocked by

- #1
