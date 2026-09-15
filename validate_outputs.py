from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DEVICE_ID = "153"
EXPECTED_AI = {"device_id", "date", "actual_date", "forecast_date", "status", "data_status", "prediction_kwh", "model_version"}
EXPECTED_HOURLY = {"device_id", "date", "hour", "predicted_kwh", "model_version"}


def fail(message: str) -> None:
    raise SystemExit(message)


def read_json(path: Path) -> object:
    if not path.exists():
        fail(f"missing JSON artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        fail(f"missing CSV artifact: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            fail(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def finite_number(value: str, label: str, allow_blank: bool = True) -> None:
    if value in (None, ""):
        if allow_blank:
            return
        fail(f"missing numeric value: {label}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        fail(f"invalid numeric value: {label}={value!r}")
    if not math.isfinite(number) or number < 0:
        fail(f"invalid non-finite/negative value: {label}={value!r}")


def main() -> None:
    state = read_json(DATA / "state.json")
    health = read_json(RESULTS / "health.json")
    payload = read_json(RESULTS / "dashboard_data.json")
    if not isinstance(state, dict) or not isinstance(health, dict) or not isinstance(payload, dict):
        fail("state, health, and dashboard payload must be JSON objects")
    if health.get("collector_status") != "healthy":
        fail(f"collector is not healthy: {health.get('error')}")
    if payload.get("device_id") != int(DEVICE_ID):
        fail(f"payload device mismatch: {payload.get('device_id')}")
    quality = health.get("data_quality", {})
    if quality.get("status") not in ("complete", "partial", None):
        fail(f"data quality is degraded: {quality}")

    ai_fields, ai_rows = read_csv(RESULTS / "ai_results.csv")
    hourly_fields, hourly_rows = read_csv(RESULTS / "hourly_predictions.csv")
    if not EXPECTED_AI.issubset(ai_fields):
        fail(f"AI schema is missing columns: {EXPECTED_AI - set(ai_fields)}")
    if not EXPECTED_HOURLY.issubset(hourly_fields):
        fail(f"hourly schema is missing columns: {EXPECTED_HOURLY - set(hourly_fields)}")

    daily_keys = [(row.get("device_id"), row.get("date")) for row in ai_rows]
    if len(daily_keys) != len(set(daily_keys)):
        fail("duplicate daily result keys detected")
    hourly_keys = [(row.get("device_id"), row.get("date"), row.get("hour")) for row in hourly_rows]
    if len(hourly_keys) != len(set(hourly_keys)):
        fail("duplicate hourly result keys detected")
    for row in ai_rows:
        if row.get("device_id") != DEVICE_ID:
            fail(f"daily row has wrong device: {row}")
        finite_number(row.get("prediction_kwh"), f"daily prediction {row.get('date')}")
        finite_number(row.get("actual_kwh"), f"daily actual {row.get('date')}")
    for row in hourly_rows:
        if row.get("device_id") != DEVICE_ID:
            fail(f"hourly row has wrong device: {row}")
        try:
            hour = int(row.get("hour", ""))
        except ValueError:
            fail(f"invalid hourly key: {row.get('hour')!r}")
        if not 0 <= hour <= 23:
            fail(f"hour outside 0..23: {hour}")
        finite_number(row.get("predicted_kwh"), f"hourly prediction {row.get('date')}:{hour}")
        finite_number(row.get("actual_kwh"), f"hourly actual {row.get('date')}:{hour}")

    processed = state.get("processed", {})
    if not isinstance(processed, dict):
        fail("state.processed must be an object")
    state_dates = set(processed)
    csv_dates = {row.get("date") for row in ai_rows if row.get("date")}
    if state_dates != csv_dates:
        fail(f"state/CSV daily dates differ: state-only={sorted(state_dates - csv_dates)}, csv-only={sorted(csv_dates - state_dates)}")

    for key in ("device_id", "ai_results", "hourly_predictions", "state", "health"):
        if key not in payload:
            fail(f"dashboard payload is missing {key}")
    if payload["ai_results"] != ai_rows:
        fail("dashboard payload ai_results does not exactly match ai_results.csv")
    if payload["hourly_predictions"] != hourly_rows:
        fail("dashboard payload hourly_predictions does not exactly match hourly_predictions.csv")
    if payload["state"] != state:
        fail("dashboard payload state does not exactly match state.json")
    if payload["health"] != health:
        fail("dashboard payload health does not exactly match health.json")
    if not isinstance(payload.get("health_history", []), list):
        fail("dashboard payload health_history must be a list")
    versions = {row.get("model_version") for row in ai_rows + hourly_rows if row.get("model_version")}
    reported_versions = set(health.get("model_versions", []))
    if versions and not versions.issubset(reported_versions | {health.get("model_version")}):
        fail(f"health model versions do not cover result versions: {sorted(versions - reported_versions)}")
    print(f"validated exactly {len(ai_rows)} daily rows, {len(hourly_rows)} hourly rows, quality={quality.get('score', 'n/a')}")


if __name__ == "__main__":
    main()
