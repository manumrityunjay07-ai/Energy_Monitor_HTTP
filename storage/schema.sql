-- Energy Monitor storage schema v1
-- Compatible with PostgreSQL; optionally run CREATE EXTENSION timescaledb before this file.
CREATE TABLE IF NOT EXISTS daily_energy (
  device_id integer NOT NULL,
  actual_date date NOT NULL,
  forecast_date date,
  processed_at timestamptz NOT NULL,
  evaluated_at timestamptz,
  calendar_profile text NOT NULL,
  status text NOT NULL,
  data_status text NOT NULL,
  actual_kwh double precision,
  previous_prediction_kwh double precision,
  prediction_error_kwh double precision,
  prediction_kwh double precision,
  prediction_lower_kwh double precision,
  prediction_upper_kwh double precision,
  base_prediction_kwh double precision,
  correction_kwh double precision,
  confidence_inside boolean,
  model_version text,
  PRIMARY KEY (device_id, actual_date)
);

CREATE TABLE IF NOT EXISTS hourly_energy (
  device_id integer NOT NULL,
  reading_date date NOT NULL,
  hour smallint NOT NULL CHECK (hour BETWEEN 0 AND 23),
  predicted_kwh double precision,
  actual_kwh double precision,
  error_kwh double precision,
  feedback_samples integer,
  collected_at timestamptz,
  model_version text,
  PRIMARY KEY (device_id, reading_date, hour)
);

CREATE TABLE IF NOT EXISTS health_events (
  device_id integer NOT NULL,
  observed_at timestamptz NOT NULL,
  collector_status text NOT NULL,
  data_status text,
  hourly_latency_ms double precision,
  daily_latency_ms double precision,
  missing_hours smallint[],
  confidence_coverage double precision,
  payload jsonb NOT NULL,
  PRIMARY KEY (device_id, observed_at)
);

CREATE INDEX IF NOT EXISTS daily_energy_forecast_date_idx ON daily_energy (device_id, forecast_date);
CREATE INDEX IF NOT EXISTS hourly_energy_date_idx ON hourly_energy (device_id, reading_date);
CREATE INDEX IF NOT EXISTS health_events_observed_idx ON health_events (device_id, observed_at DESC);

-- Optional TimescaleDB optimization after extension installation:
-- SELECT create_hypertable('hourly_energy', 'reading_date', if_not_exists => TRUE);
-- SELECT create_hypertable('health_events', 'observed_at', if_not_exists => TRUE);
