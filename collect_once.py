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
HOURLY_RESULTS = Path("results/hourly_predictions.csv")
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


def adaptive_daily_forecast(
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
        for coefficient, value in zip(PARAMS["prediction_coefficients"], scaled_features)
    )
    errors = [
        float(item["error_kwh"])
        for item in forecast_history.values()
        if isinstance(item, dict) and item.get("error_kwh") not in (None, "")
    ]
    recent_errors = errors[-14:]
    correction = sum(recent_errors) / len(recent_errors) if recent_errors else 0.0
    correction = max(-0.5 * abs(base_prediction), min(0.5 * abs(base_prediction), correction))
    prediction = max(0.0, base_prediction + correction)
    forecast_history[target_date] = {
        "generated_at": now.isoformat(),
        "source_date": source_date,
        "base_prediction_kwh": round(base_prediction, 6),
        "correction_kwh": round(correction, 6),
        "predicted_kwh": round(prediction, 6),
    }
    return round(prediction, 6), round(base_prediction, 6), round(correction, 6), len(recent_errors)


def adaptive_hourly_forecast(
    profiles: dict[str, dict],
    hourly_history: dict[str, dict],
    target_date: str,
    source_date: str,
    now: datetime,
) -> tuple[list[float], list[float], list[float], int] | None:
    completed = sorted(
        (day, profile)
        for day, profile in profiles.items()
        if day <= source_date and all(str(hour) in profile for hour in range(24))
    )
    if not completed:
        return None
    recent = completed[-7:]
    base = [
        sum(float(profile[str(hour)]) for _, profile in recent) / len(recent)
        for hour in range(24)
    ]
    corrections = [0.0] * 24
    feedback_samples = 0
    for entry in hourly_history.values():
        if not isinstance(entry, dict) or "error_kwh" not in entry:
            continue
        errors = entry["error_kwh"]
        if isinstance(errors, list) and len(errors) == 24:
            for hour, error in enumerate(errors):
                corrections[hour] += float(error)
            feedback_samples += 1
    if feedback_samples:
        corrections = [value / feedback_samples for value in corrections]
    predicted = [max(0.0, base[hour] + corrections[hour]) for hour in range(24)]
    hourly_history[target_date] = {
        "generated_at": now.isoformat(),
        "source_date": source_date,
        "base_prediction_kwh": [round(value, 6) for value in base],
        "correction_kwh": [round(value, 6) for value in corrections],
        "predicted_kwh": [round(value, 6) for value in predicted],
    }
    return predicted, base, corrections, feedback_samples


def write_hourly_result(target_date: str, now: datetime, result: dict) -> None:
    fields = ["processed_at", "device_id", "date", "hour", "predicted_kwh", "actual_kwh", "error_kwh", "feedback_samples"]
    exists = HOURLY_RESULTS.exists() and HOURLY_RESULTS.stat().st_size > 0
    with HOURLY_RESULTS.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if not exists:
            writer.writeheader()
        forecast = result.get("hourly_forecast", {})
        actual = result.get("hourly_actual", {})
        errors = result.get("hourly_error", {})
        for hour in range(24):
            writer.writerow({
                "processed_at": now.isoformat(), "device_id": DEVICE_ID,
                "date": target_date, "hour": hour,
                "predicted_kwh": forecast.get(str(hour), ""),
                "actual_kwh": actual.get(str(hour), ""),
                "error_kwh": errors.get(str(hour), ""),
                "feedback_samples": result.get("hourly_feedback_samples", 0),
            })


now = datetime.now(TZ)
today = now.date().isoformat()
state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"profiles": {}, "processed": {}, "forecast_history": {}, "hourly_forecast_history": {}}
profiles = state.setdefault("profiles", {})
processed = state.setdefault("processed", {})
forecast_history = state.setdefault("forecast_history", {})
hourly_history = state.setdefault("hourly_forecast_history", {})
profile = profiles.setdefault(today, {})

response = requests.post(HOURLY_URL, json={"deviceId": DEVICE_ID}, timeout=30)
response.raise_for_status()
rows = response.json().get("data", [])
if not isinstance(rows, list) or len(rows) < 24:
    raise RuntimeError(f"Hourly response did not contain 24 rows: {response.text[:500]}")
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

    prior_daily = forecast_history.get(previous_date)
    actual_total = daily_totals.get(previous_date)
    if actual_total is not None and isinstance(prior_daily, dict):
        predicted_total = float(prior_daily["predicted_kwh"])
        error = actual_total - predicted_total
        prior_daily.update({"actual_kwh": round(actual_total, 6), "error_kwh": round(error, 6), "evaluated_at": now.isoformat()})
        result.update({"actual_kwh": round(actual_total, 6), "previous_prediction_kwh": round(predicted_total, 6), "prediction_error_kwh": round(error, 6)})

    prior_hourly = hourly_history.get(previous_date)
    if isinstance(prior_hourly, dict) and all(str(hour) in completed_profile for hour in range(24)):
        predicted_hours = prior_hourly.get("predicted_kwh", [])
        if len(predicted_hours) == 24:
            actual_hours = [float(completed_profile[str(hour)]) for hour in range(24)]
            errors = [actual_hours[hour] - float(predicted_hours[hour]) for hour in range(24)]
            prior_hourly.update({"actual_kwh": actual_hours, "error_kwh": [round(value, 6) for value in errors], "evaluated_at": now.isoformat()})
            result["hourly_error_mae"] = round(sum(abs(value) for value in errors) / 24, 6)

    daily_forecast = adaptive_daily_forecast(daily_totals, forecast_history, today, previous_date, now)
    if daily_forecast:
        prediction, base_prediction, correction, feedback_samples = daily_forecast
        result.update({"prediction_kwh": prediction, "base_prediction_kwh": base_prediction, "correction_kwh": correction, "feedback_samples": feedback_samples})
    else:
        result.update({"prediction_kwh": "", "base_prediction_kwh": "", "correction_kwh": "", "feedback_samples": 0})

    hourly_forecast = adaptive_hourly_forecast(profiles, hourly_history, today, previous_date, now)
    if hourly_forecast:
        predicted, base, corrections, feedback_samples = hourly_forecast
        result.update({
            "hourly_forecast": {str(hour): round(predicted[hour], 6) for hour in range(24)},
            "hourly_base_prediction": {str(hour): round(base[hour], 6) for hour in range(24)},
            "hourly_correction": {str(hour): round(corrections[hour], 6) for hour in range(24)},
            "hourly_feedback_samples": feedback_samples,
        })
        write_hourly_result(today, now, result)
    else:
        result.update({"hourly_forecast": {}, "hourly_feedback_samples": 0})

    processed[previous_date] = result
    fields = ["processed_at", "device_id", "date", "status", "cluster", "anomaly_score", "anomaly_threshold", "actual_kwh", "previous_prediction_kwh", "prediction_error_kwh", "prediction_kwh", "base_prediction_kwh", "correction_kwh", "feedback_samples", "hourly_error_mae"]
    exists = RESULTS.exists() and RESULTS.stat().st_size > 0
    with RESULTS.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in fields})

state["last_run"] = now.isoformat()
STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
print(f"Device {DEVICE_ID}; {status}; local time {now.isoformat()}")
if result:
    print(json.dumps(result, indent=2))
else:
    print(f"AI analysis waiting; completed profile for {previous_date}: {len(completed_profile)}/24 hours")
