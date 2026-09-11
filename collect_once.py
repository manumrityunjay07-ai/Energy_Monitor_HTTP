from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

DEVICE_ID = 153
TZ = ZoneInfo("Asia/Kolkata")
HOURLY_URL = "https://adc.bitsathy.ac.in/2024/ems_dashboard/api/fetch_hourly_energy_consumption.php"
DAYWISE_URL = "https://adc.bitsathy.ac.in/2024/ems_dashboard/api/fetch_day_energy_consumption.php"
STATE = Path("data/state.json")
RESULTS = Path("results/ai_results.csv")
PARAMS = json.loads(Path("esp32_parameters.json").read_text(encoding="utf-8"))

STATE.parent.mkdir(exist_ok=True)
RESULTS.parent.mkdir(exist_ok=True)


def fetch_daily_totals(end_date: date) -> dict[str, float]:
    response = requests.post(
        DAYWISE_URL,
        files={
            "deviceId": (None, str(DEVICE_ID)),
            "fromDate": (None, (end_date - timedelta(days=60)).isoformat()),
            "toDate": (None, end_date.isoformat()),
        },
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("data", [])
    totals: dict[str, float] = {}
    for row in rows:
        raw_date = row.get("custom_day") or row.get("date")
        raw_total = row.get("total_consumption")
        if raw_date is None or raw_total is None:
            continue
        try:
            day = str(raw_date)[:10]
            value = float(raw_total)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            totals[day] = value
    return totals


def anomaly_result(values: list[float], now: datetime, processed_date: str) -> dict:
    scaled = [
        (x - mean) / scale if scale else 0
        for x, mean, scale in zip(
            values,
            PARAMS["cluster_scaler_mean"],
            PARAMS["cluster_scaler_scale"],
        )
    ]
    distances = [
        math.sqrt(sum((x - centre) ** 2 for x, centre in zip(scaled, centroid)))
        for centroid in PARAMS["cluster_centres_scaled"]
    ]
    cluster_index = min(range(len(distances)), key=distances.__getitem__)
    score = distances[cluster_index]
    return {
        "processed_at": now.isoformat(),
        "device_id": DEVICE_ID,
        "date": processed_date,
        "status": "anomaly" if score > PARAMS["anomaly_threshold"] else "normal",
        "cluster": cluster_index + 1,
        "anomaly_score": round(score, 6),
        "anomaly_threshold": PARAMS["anomaly_threshold"],
    }


def adaptive_forecast(
    daily_totals: dict[str, float],
    forecast_history: dict[str, dict],
    target_date: str,
    source_date: str,
    now: datetime,
) -> tuple[float, float, float, int] | None:
    ordered = sorted(
        ((day, total) for day, total in daily_totals.items() if day <= source_date),
        key=lambda item: item[0],
    )
    if len(ordered) < 3:
        return None

    recent = [total for _, total in ordered[-3:]]
    features = [recent[-1], recent[-2], recent[-3], sum(recent) / 3]
    scaled_features = [
        (value - mean) / scale if scale else 0
        for value, mean, scale in zip(
            features,
            PARAMS["prediction_scaler_mean"],
            PARAMS["prediction_scaler_scale"],
        )
    ]
    base_prediction = PARAMS["prediction_intercept"] + sum(
        coefficient * value
        for coefficient, value in zip(
            PARAMS["prediction_coefficients"], scaled_features
        )
    )

    # Only completed forecast errors influence the correction. Predictions are
    # never inserted as if they were measured energy values.
    errors = [
        float(item["error_kwh"])
        for item in forecast_history.values()
        if isinstance(item, dict) and item.get("error_kwh") not in (None, "")
    ]
    recent_errors = errors[-14:]
    correction = sum(recent_errors) / len(recent_errors) if recent_errors else 0.0
    correction = max(
        -0.5 * abs(base_prediction),
        min(0.5 * abs(base_prediction), correction),
    )
    prediction = max(0.0, base_prediction + correction)

    forecast_history[target_date] = {
        "generated_at": now.isoformat(),
        "source_date": source_date,
        "base_prediction_kwh": round(base_prediction, 6),
        "correction_kwh": round(correction, 6),
        "predicted_kwh": round(prediction, 6),
    }
    return round(prediction, 6), round(base_prediction, 6), round(correction, 6), len(recent_errors)


now = datetime.now(TZ)
today = now.date().isoformat()
state = (
    json.loads(STATE.read_text(encoding="utf-8"))
    if STATE.exists()
    else {"profiles": {}, "processed": {}, "forecast_history": {}}
)
profiles = state.setdefault("profiles", {})
processed = state.setdefault("processed", {})
forecast_history = state.setdefault("forecast_history", {})
profile = profiles.setdefault(today, {})

response = requests.post(HOURLY_URL, json={"deviceId": DEVICE_ID}, timeout=30)
response.raise_for_status()
rows = response.json().get("data", [])
if not isinstance(rows, list) or len(rows) < 24:
    raise RuntimeError(f"Hourly response did not contain 24 rows: {response.text[:500]}")

# Future hours are returned as zero. Store only the current and completed hours.
for row in rows:
    hour = int(row["hour"])
    value = float(row["total_energy"])
    if 0 <= hour <= now.hour and math.isfinite(value) and value >= 0:
        profile[str(hour)] = value

previous_date = (now.date() - timedelta(days=1)).isoformat()
completed_profile = profiles.get(previous_date, {})
status = f"collected {len(profile)}/24 hours for {today}"
result = None

if len(completed_profile) == 24 and previous_date not in processed:
    values = [float(completed_profile[str(hour)]) for hour in range(24)]
    result = anomaly_result(values, now, previous_date)
    daily_totals = fetch_daily_totals(now.date())

    # Evaluate the prior forecast against the actual completed day.
    prior_forecast = forecast_history.get(previous_date)
    actual_total = daily_totals.get(previous_date)
    if actual_total is not None and isinstance(prior_forecast, dict):
        predicted_total = float(prior_forecast["predicted_kwh"])
        error = actual_total - predicted_total
        prior_forecast.update(
            {
                "actual_kwh": round(actual_total, 6),
                "error_kwh": round(error, 6),
                "evaluated_at": now.isoformat(),
            }
        )
        result.update(
            {
                "actual_kwh": round(actual_total, 6),
                "previous_prediction_kwh": round(predicted_total, 6),
                "prediction_error_kwh": round(error, 6),
            }
        )

    forecast = adaptive_forecast(
        daily_totals,
        forecast_history,
        target_date=today,
        source_date=previous_date,
        now=now,
    )
    if forecast is None:
        result.update(
            {
                "prediction_kwh": "",
                "base_prediction_kwh": "",
                "correction_kwh": "",
                "feedback_samples": 0,
            }
        )
    else:
        prediction, base_prediction, correction, feedback_samples = forecast
        result.update(
            {
                "prediction_kwh": prediction,
                "base_prediction_kwh": base_prediction,
                "correction_kwh": correction,
                "feedback_samples": feedback_samples,
            }
        )

    processed[previous_date] = result
    fields = [
        "processed_at",
        "device_id",
        "date",
        "status",
        "cluster",
        "anomaly_score",
        "anomaly_threshold",
        "actual_kwh",
        "previous_prediction_kwh",
        "prediction_error_kwh",
        "prediction_kwh",
        "base_prediction_kwh",
        "correction_kwh",
        "feedback_samples",
    ]
    file_exists = RESULTS.exists() and RESULTS.stat().st_size > 0
    with RESULTS.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in fields})

state["last_run"] = now.isoformat()
STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

print(f"Device {DEVICE_ID}; {status}; local time {now.isoformat()}")
if result:
    print(json.dumps(result, indent=2))
else:
    print(f"AI analysis waiting; completed profile for {previous_date}: {len(completed_profile)}/24 hours")
