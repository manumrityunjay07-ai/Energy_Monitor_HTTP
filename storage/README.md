# Scalable historical storage contract

The collector currently publishes Git-compatible CSV and JSON artifacts because the dashboard is static. `schema.sql` defines the versioned PostgreSQL contract for moving high-volume history into a database without changing the dashboard’s semantic fields. The schema is PostgreSQL-compatible and includes optional TimescaleDB hypertable guidance.

The migration sequence is deliberately additive. First, provision PostgreSQL and apply `schema.sql`. Next, backfill `daily_energy` from `results/ai_results.csv`, `hourly_energy` from `results/hourly_predictions.csv`, and `health_events` from `results/health_history.json`. Compare row counts, date ranges, daily totals, and a sample of prediction errors. Keep the Git artifacts as a read-only fallback until two consecutive scheduled runs write and validate both destinations. Finally, switch the dashboard data endpoint to a read-only API that returns the existing `dashboard_data.json` shape.

The database is not made a runtime dependency by this change. Scheduled collection remains functional when the database is unavailable, which preserves the current public dashboard while migration is verified separately.
