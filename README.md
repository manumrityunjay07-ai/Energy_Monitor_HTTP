# Energy_Monitor_HTTP

HTTP energy monitoring, automated data collection, and AI inference for **Device 153**.

---

## Overview

This repository runs automated data pipelines that query Device 153's telemetry endpoints, process daily totals and hourly consumption profiles, compute predictions and anomaly metrics, and publish results to the `live-data` branch for consumption by the [Energy_Monitor_Dashboard](https://github.com/manumrityunjay07-ai/Energy_Monitor_Dashboard).

---

## Key Features

- **Automated Collection**: Regularly queries the upstream HTTP endpoint for device telemetry.
- **Data Validation**: Strict validation ([`validate_outputs.py`](validate_outputs.py)) verifying day-wise continuity and profile completeness.
- **AI Forecasting & Anomaly Detection**: Generates hourly predictions and evaluates error distributions against learned bounds.
- **Contract Integrity**: Prevents fabricated timestamps by honoring the data contract documented in [`docs/data-contract.md`](docs/data-contract.md).
- **Zero-Backend Client Serving**: Results are committed and published to GitHub (`live-data` branch) as lightweight JSON and CSV files.

---

## Project Structure

```text
Energy_Monitor_HTTP/
├── collect_once.py          # Main execution script for telemetry collection and model inference
├── validate_outputs.py      # Contract and data quality validation runner
├── requirements.txt         # Minimal Python dependencies
├── esp32_parameters.json    # Hardware and device connection settings
├── data/                    # State files (state.json, holidays.json)
├── results/                 # Output artifacts (dashboard_data.json, ai_results.csv, daily_totals.csv, health.json)
├── storage/                 # Database schema and persistence documentation
├── tests/                   # Unit test suite (test_collector.py)
└── docs/                    # Architecture and data contract specifications
```

---

## Verification & Testing

Run unit tests and validation locally:

```bash
# Run unit tests
python -m unittest discover tests -v

# Run output contract validation
python validate_outputs.py
```
