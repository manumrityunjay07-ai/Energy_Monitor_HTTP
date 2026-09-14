from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

EXPECTED_AI = {"device_id", "date", "status", "data_status", "prediction_kwh", "model_version"}
EXPECTED_HOURLY = {"device_id", "date", "hour", "predicted_kwh", "model_version"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    state = json.loads((ROOT / "data/state.json").read_text(encoding="utf-8"))
    health = json.loads((RESULTS / "health.json").read_text(encoding="utf-8"))
    if health.get("collector_status") != "healthy":
        raise SystemExit(f"collector is not healthy: {health.get('error')}")
    quality = health.get("data_quality", {})
    if quality.get("status") not in ("complete", "partial", None):
        raise SystemExit(f"data quality is degraded: {quality}")
    ai_rows = read_csv(RESULTS / "ai_results.csv")
    hourly_rows = read_csv(RESULTS / "hourly_predictions.csv")
    if ai_rows and not EXPECTED_AI.issubset(ai_rows[0]):
        raise SystemExit(f"AI schema is missing columns: {EXPECTED_AI - set(ai_rows[0])}")
    if hourly_rows and not EXPECTED_HOURLY.issubset(hourly_rows[0]):
        raise SystemExit(f"hourly schema is missing columns: {EXPECTED_HOURLY - set(hourly_rows[0])}")
    keys = [(row.get("device_id"), row.get("date"), row.get("hour")) for row in hourly_rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate hourly result keys detected")
    state_dates = set(state.get("processed", {}).keys())
    csv_dates = {row.get("date") for row in ai_rows if row.get("date")}
    if not state_dates.issubset(csv_dates):
        raise SystemExit(f"state/CSV mismatch; missing CSV dates: {sorted(state_dates - csv_dates)}")
    payload = json.loads((RESULTS / "dashboard_data.json").read_text(encoding="utf-8"))
    for key in ("device_id", "ai_results", "hourly_predictions", "state", "health"):
        if key not in payload:
            raise SystemExit(f"dashboard payload is missing {key}")
    payload_dates = {row.get("date") for row in payload.get("ai_results", []) if row.get("date")}
    if not csv_dates.issubset(payload_dates):
        raise SystemExit(f"CSV/payload mismatch; missing payload dates: {sorted(csv_dates - payload_dates)}")
    print(f"validated {len(ai_rows)} daily rows, {len(hourly_rows)} hourly rows, quality={quality.get('score', 'n/a')}")


if __name__ == "__main__":
    main()
