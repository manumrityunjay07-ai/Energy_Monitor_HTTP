from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median, pstdev
from zoneinfo import ZoneInfo

import requests

DEVICE_ID = int(os.getenv("DEVICE_ID", "153"))
TZ = ZoneInfo("Asia/Kolkata")
HOURLY_URL = os.getenv("HOURLY_URL", "https://adc.bitsathy.ac.in/2024/ems_dashboard/api/fetch_hourly_energy_consumption.php")
DAYWISE_URL = os.getenv("DAYWISE_URL", "https://adc.bitsathy.ac.in/2024/ems_dashboard/api/fetch_day_energy_consumption.php")
STATE = Path("data/state.json")
RESULTS = Path("results/ai_results.csv")
HOURLY_RESULTS = Path("results/hourly_predictions.csv")
HEALTH = Path("results/health.json")
HEALTH_HISTORY = Path("results/health_history.json")
DASHBOARD_DATA = Path("results/dashboard_data.json")
IMPROVEMENT_REPORT = Path("results/improvement_report.json")
PARAMS = json.loads(Path("esp32_parameters.json").read_text(encoding="utf-8"))
HOLIDAYS = json.loads(Path("data/holidays.json").read_text(encoding="utf-8")) if Path("data/holidays.json").exists() else {"holidays": {}}
MODEL_VERSION = "device153-adaptive-v2"
MAX_RETRIES = 4
HOURLY_REPAIR_ATTEMPTS = 2

STATE.parent.mkdir(exist_ok=True)
RESULTS.parent.mkdir(exist_ok=True)


def now_ist() -> datetime:
    return datetime.now(TZ)


def calendar_profile(day: str) -> str:
    if day in HOLIDAYS.get("holidays", {}):
        return "holiday"
    return "weekend" if date.fromisoformat(day).weekday() >= 5 else "weekday"


def request_json(method: str, url: str, **kwargs) -> tuple[dict, float]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        started = time.perf_counter()
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("API response was not a JSON object")
            return payload, round((time.perf_counter() - started) * 1000, 2)
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}: {last_error}") from last_error


def fetch_daily_totals(end_date: date) -> tuple[dict[str, float], float]:
    start_date = end_date - timedelta(days=60)
    payload, latency = request_json(
        "POST",
        DAYWISE_URL,
            files={
                "deviceId": (None, str(DEVICE_ID)),
                "fromDate": (None, start_date.isoformat()),
            "toDate": (None, end_date.isoformat()),
        },
    )
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise ValueError("Day-wise API data field is not a list")
    totals: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_date = row.get("custom_day") or row.get("date")
        raw_total = row.get("total_consumption")
        try:
            day, value = str(raw_date)[:10], float(raw_total)
            date.fromisoformat(day)
        except (TypeError, ValueError):
            continue
        if start_date.isoformat() <= day <= end_date.isoformat() and len(day) == 10 and math.isfinite(value) and value >= 0:
            totals[day] = value
    return totals, latency


def fetch_hourly(now: datetime) -> tuple[dict[str, float], float, list[int]]:
    by_hour: dict[int, float] = {}
    latency = 0.0
    last_missing: list[int] = []
    for attempt in range(HOURLY_REPAIR_ATTEMPTS + 1):
        payload, latency = request_json("POST", HOURLY_URL, json={"deviceId": DEVICE_ID})
        rows = payload.get("data", [])
        if not isinstance(rows, list):
            raise ValueError("Hourly API data field is not a list")
        candidate: dict[int, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                hour = int(row["hour"])
                value = float(row["total_energy"])
            except (KeyError, TypeError, ValueError):
                continue
            if hour in candidate:
                raise ValueError(f"Hourly API returned duplicate hour {hour}")
            if not 0 <= hour <= 23 or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid hourly row: hour={hour}, value={value}")
            candidate[hour] = value
        by_hour.update(candidate)
        last_missing = sorted(set(range(now.hour)) - set(by_hour))
        if not last_missing:
            break
        if attempt < HOURLY_REPAIR_ATTEMPTS:
            time.sleep(1)
    # The current hour and future hours are intentionally excluded from completeness.
    completed = {str(hour): by_hour[hour] for hour in range(now.hour) if hour in by_hour}
    return completed, latency, last_missing


def anomaly_result(values: list[float], now: datetime, processed_date: str, threshold: float) -> dict:
    if len(values) != len(PARAMS["cluster_scaler_mean"]) or len(values) != len(PARAMS["cluster_scaler_scale"]):
        raise ValueError("Anomaly input length does not match scaler parameters")
    if any(len(centroid) != len(values) for centroid in PARAMS["cluster_centres_scaled"]):
        raise ValueError("Anomaly centroid length does not match input length")
    scaled = [(x - mean_value) / scale if scale else 0 for x, mean_value, scale in zip(values, PARAMS["cluster_scaler_mean"], PARAMS["cluster_scaler_scale"])]
    distances = [math.sqrt(sum((x - centre) ** 2 for x, centre in zip(scaled, centroid))) for centroid in PARAMS["cluster_centres_scaled"]]
    cluster_index = min(range(len(distances)), key=distances.__getitem__)
    score = distances[cluster_index]
    return {"processed_at": now.isoformat(), "device_id": DEVICE_ID, "date": processed_date, "status": "anomaly" if score > threshold else "normal", "cluster": cluster_index + 1, "anomaly_score": round(score, 6), "anomaly_threshold": round(threshold, 6), "model_version": MODEL_VERSION}


def explain_profile(values: list[float], profiles: dict[str, dict], processed_date: str) -> str:
    target_weekday = date.fromisoformat(processed_date).weekday()
    peers = []
    for day, profile in profiles.items():
        if day >= processed_date or date.fromisoformat(day).weekday() != target_weekday:
            continue
        if all(str(hour) in profile for hour in range(24)):
            peers.append([float(profile[str(hour)]) for hour in range(24)])
    if len(peers) < 2:
        return "Not enough comparable historical days for a detailed explanation."
    averages = [sum(row[hour] for row in peers) / len(peers) for hour in range(24)]
    deviations = [(values[hour] - averages[hour]) / max(averages[hour], 1.0) for hour in range(24)]
    peak_hour = max(range(24), key=lambda hour: abs(deviations[hour]))
    direction = "higher" if deviations[peak_hour] >= 0 else "lower"
    return f"{peak_hour:02d}:00 is {abs(deviations[peak_hour]) * 100:.1f}% {direction} than comparable {'weekday' if target_weekday < 5 else 'weekend'} profiles."


def evaluate_model_guard(forecast_history: dict[str, dict]) -> dict:
    evaluated = [item for _, item in sorted(forecast_history.items(), key=lambda pair: pair[0]) if isinstance(item, dict) and item.get("actual_kwh") not in (None, "") and item.get("base_prediction_kwh") not in (None, "") and item.get("predicted_kwh") not in (None, "")]
    if len(evaluated) < 5:
        return {"adaptation_enabled": True, "reason": "insufficient history for rollback decision", "evaluated_cycles": len(evaluated)}
    base_mae = mean(abs(float(item["actual_kwh"]) - float(item["base_prediction_kwh"])) for item in evaluated[-14:])
    adapted_mae = mean(abs(float(item["actual_kwh"]) - float(item["predicted_kwh"])) for item in evaluated[-14:])
    enabled = adapted_mae <= base_mae * 1.10
    return {"adaptation_enabled": enabled, "reason": "adapted model retained" if enabled else "rollback to base model: adapted MAE exceeded baseline by more than 10%", "evaluated_cycles": len(evaluated), "base_mae": round(base_mae, 6), "adapted_mae": round(adapted_mae, 6)}


def build_improvement_report(state: dict, now: datetime) -> dict:
    guard = state.get("model_guard", {})
    samples = sum(1 for item in state.get("processed", {}).values() if isinstance(item, dict) and item.get("actual_kwh") not in (None, ""))
    return {
        "generated_at": now.isoformat(),
        "model_version": MODEL_VERSION,
        "mode": "bounded_adaptation",
        "status": "adaptive" if guard.get("adaptation_enabled", True) and samples >= 5 else "calibrating",
        "learning_samples": samples,
        "adaptation_enabled": guard.get("adaptation_enabled", True),
        "guard_reason": guard.get("reason", "insufficient history for rollback decision"),
        "evaluated_cycles": guard.get("evaluated_cycles", 0),
        "base_mae_kwh": guard.get("base_mae"),
        "adapted_mae_kwh": guard.get("adapted_mae"),
        "safety_policy": {
            "minimum_feedback_samples": 5,
            "maximum_correction_fraction": 0.25,
            "rollback_tolerance_fraction": 0.10,
            "unreviewed_code_changes": False,
        },
        "next_action": "continue calibration" if samples < 5 else ("retain bounded adaptation" if guard.get("adaptation_enabled", True) else "use baseline until adaptation improves"),
    }


def adaptive_daily_forecast(daily_totals: dict[str, float], forecast_history: dict[str, dict], target_date: str, source_date: str, now: datetime, adaptation_enabled: bool = True) -> tuple | None:
    ordered = sorted(((day, total) for day, total in daily_totals.items() if day <= source_date), key=lambda item: item[0])
    if len(ordered) < 3:
        return None
    target_weekday = date.fromisoformat(target_date).weekday()
    matching = [(day, total) for day, total in ordered if date.fromisoformat(day).weekday() == target_weekday]
    selected = matching[-7:] if len(matching) >= 3 else ordered[-7:]
    mode = "weekday profile" if len(matching) >= 3 else "recent profile"
    weights = [0.72 ** (len(selected) - index - 1) for index in range(len(selected))]
    weight_total = sum(weights)
    recent = [total for _, total in selected[-3:]]
    features = [recent[-1], recent[-2], recent[-3], sum(recent) / 3]
    scaled = [(value - m) / s if s else 0 for value, m, s in zip(features, PARAMS["prediction_scaler_mean"], PARAMS["prediction_scaler_scale"])]
    base_prediction = PARAMS["prediction_intercept"] + sum(c * v for c, v in zip(PARAMS["prediction_coefficients"], scaled))
    errors = [float(item["error_kwh"]) for _, item in sorted(forecast_history.items(), key=lambda pair: pair[0]) if isinstance(item, dict) and item.get("error_kwh") not in (None, "")]
    recent_errors = errors[-14:]
    mean_error = mean(recent_errors) if recent_errors else 0.0
    robust_error = median(recent_errors) if recent_errors else 0.0
    # One or two feedback samples are too noisy to safely change the forecast.
    # Keep the baseline until the rollback guard has enough evidence.
    correction = 0.6 * mean_error + 0.4 * robust_error if adaptation_enabled and len(recent_errors) >= 5 else 0.0
    correction = max(-0.25 * abs(base_prediction), min(0.25 * abs(base_prediction), correction))
    prediction = max(0.0, base_prediction + correction)
    weighted_mean = sum(weight * total for weight, (_, total) in zip(weights, selected)) / weight_total
    spread = math.sqrt(sum(weight * (total - weighted_mean) ** 2 for weight, (_, total) in zip(weights, selected)) / weight_total)
    lower, upper = max(0.0, prediction - 1.28 * spread), prediction + 1.28 * spread
    learning_guard = "robust median/mean blend, 25% cap" if adaptation_enabled and len(recent_errors) >= 5 else "baseline held: fewer than 5 feedback samples"
    forecast_history[target_date] = {"generated_at": now.isoformat(), "source_date": source_date, "base_prediction_kwh": round(base_prediction, 6), "correction_kwh": round(correction, 6), "predicted_kwh": round(prediction, 6), "lower_kwh": round(lower, 6), "upper_kwh": round(upper, 6), "profile_mode": mode, "model_version": MODEL_VERSION, "learning_guard": learning_guard}
    return round(prediction, 6), round(base_prediction, 6), round(correction, 6), len(recent_errors), round(lower, 6), round(upper, 6), mode


def adaptive_hourly_forecast(profiles: dict[str, dict], hourly_history: dict[str, dict], target_date: str, source_date: str, now: datetime, adaptation_enabled: bool = True) -> tuple | None:
    completed = sorted((day, profile) for day, profile in profiles.items() if day <= source_date and all(str(hour) in profile for hour in range(24)))
    if not completed:
        return None
    target_weekday = date.fromisoformat(target_date).weekday()
    matching = [(day, profile) for day, profile in completed if date.fromisoformat(day).weekday() == target_weekday]
    recent = matching[-7:] if len(matching) >= 3 else completed[-7:]
    mode = "weekday profile" if len(matching) >= 3 else "recent profile"
    weights = [0.72 ** (len(recent) - index - 1) for index in range(len(recent))]
    total_weight = sum(weights)
    base = [sum(weight * float(profile[str(hour)]) for weight, (_, profile) in zip(weights, recent)) / total_weight for hour in range(24)]
    spread = [math.sqrt(sum(weight * (float(profile[str(hour)]) - base[hour]) ** 2 for weight, (_, profile) in zip(weights, recent)) / total_weight) for hour in range(24)]
    corrections, samples = [0.0] * 24, 0
    for entry in hourly_history.values():
        errors = entry.get("error_kwh") if isinstance(entry, dict) else None
        if isinstance(errors, list) and len(errors) == 24:
            for hour, error in enumerate(errors):
                corrections[hour] += float(error)
            samples += 1
    if samples:
        corrections = [value / samples for value in corrections]
    corrections = [max(-0.25 * abs(base[hour]), min(0.25 * abs(base[hour]), corrections[hour])) for hour in range(24)] if adaptation_enabled else [0.0] * 24
    predicted = [max(0.0, base[hour] + corrections[hour]) for hour in range(24)]
    lower = [max(0.0, predicted[hour] - 1.28 * spread[hour]) for hour in range(24)]
    upper = [predicted[hour] + 1.28 * spread[hour] for hour in range(24)]
    hourly_history[target_date] = {"generated_at": now.isoformat(), "source_date": source_date, "base_prediction_kwh": [round(v, 6) for v in base], "correction_kwh": [round(v, 6) for v in corrections], "predicted_kwh": [round(v, 6) for v in predicted], "lower_kwh": [round(v, 6) for v in lower], "upper_kwh": [round(v, 6) for v in upper], "profile_mode": mode, "model_version": MODEL_VERSION, "learning_guard": "hourly correction capped at 25% of profile" if adaptation_enabled else "rollback guard: base model"}
    return predicted, base, corrections, samples, lower, upper, mode


def upsert_csv(path: Path, fields: list[str], rows: list[dict], key_fields: list[str]) -> None:
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    else:
        existing = []
    merged: dict[tuple, dict] = {}
    for row in existing + rows:
        normalized = {field: row.get(field, "") for field in fields}
        if "data_status" in fields and not normalized.get("data_status"):
            normalized["data_status"] = "daily_total_only" if row.get("status") in ("data_incomplete", "daily_total_only") else "complete"
        if "actual_date" in fields and not normalized.get("actual_date"):
            normalized["actual_date"] = row.get("date", "")
        if "forecast_date" in fields and not normalized.get("forecast_date") and row.get("date"):
            try:
                normalized["forecast_date"] = (date.fromisoformat(str(row["date"])) + timedelta(days=1)).isoformat()
            except ValueError:
                normalized["forecast_date"] = ""
        if "calendar_profile" in fields and not normalized.get("calendar_profile") and row.get("date"):
            normalized["calendar_profile"] = calendar_profile(str(row["date"]))
        if "backfill_status" in fields and not normalized.get("backfill_status"):
            normalized["backfill_status"] = "measured_hours_only" if row.get("status") in ("data_incomplete", "daily_total_only") else "measured_24_hours"
        if "missing_hours" in fields and not normalized.get("missing_hours"):
            normalized["missing_hours"] = "unknown" if normalized.get("backfill_status") == "measured_hours_only" else ""
        merged[tuple(str(row.get(k, "")) for k in key_fields)] = normalized
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(merged.values(), key=lambda row: tuple(str(row.get(k, "")) for k in key_fields)))


def ensure_csv_schema(path: Path, fields: list[str], key_fields: list[str]) -> None:
    """Migrate an existing CSV even when this run has no new forecast rows."""
    if not path.exists() or not path.stat().st_size:
        return
    existing = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    upsert_csv(path, fields, existing, key_fields)


def data_quality(profile: dict[str, float], now: datetime, hourly_latency: float | None) -> dict:
    expected = set(str(hour) for hour in range(now.hour))
    received = set(profile)
    completeness = len(received & expected) / len(expected) if expected else 1.0
    return {
        "score": round(completeness * 100, 2),
        "completed_hours": len(received & expected),
        "expected_completed_hours": len(expected),
        "completeness": round(completeness, 4),
        "api_latency_ms": hourly_latency,
        "status": "complete" if completeness >= 0.99 else "partial",
        "data_status": "complete" if completeness >= 0.99 else "partial",
    }


def write_health(state: dict, now: datetime, status: str, hourly_latency: float | None, daily_latency: float | None, missing_hours: list[int], quality: dict | None = None, error: str | None = None) -> dict:
    samples = sum(1 for item in state.get("processed", {}).values() if isinstance(item, dict) and item.get("actual_kwh") not in (None, ""))
    intervals = [item for item in state.get("processed", {}).values() if isinstance(item, dict) and all(item.get(key) not in (None, "") for key in ("actual_kwh", "prediction_lower_kwh", "prediction_upper_kwh"))]
    inside = sum(float(item["prediction_lower_kwh"]) <= float(item["actual_kwh"]) <= float(item["prediction_upper_kwh"]) for item in intervals)
    evaluated = [item for item in state.get("processed", {}).values() if isinstance(item, dict) and item.get("actual_kwh") not in (None, "") and item.get("previous_prediction_kwh") not in (None, "")]
    errors = [float(item["actual_kwh"]) - float(item["previous_prediction_kwh"]) for item in evaluated]
    versions = sorted({str(item.get("model_version")) for item in state.get("processed", {}).values() if isinstance(item, dict) and item.get("model_version")} | {MODEL_VERSION})
    guard = state.get("model_guard", {"adaptation_enabled": True, "reason": "insufficient history for rollback decision"})
    health = {"collector_status": status, "last_successful_collection": now.isoformat() if status == "healthy" else state.get("health", {}).get("last_successful_collection"), "last_run": now.isoformat(), "device_id": DEVICE_ID, "api_latency_ms": {"hourly": hourly_latency, "daily": daily_latency}, "missing_hours": missing_hours, "data_quality": quality or state.get("health", {}).get("data_quality", {}), "confidence_coverage": {"inside_interval": inside, "evaluated": len(intervals), "rate": round(inside / len(intervals), 4) if intervals else None}, "model_evaluation": {"evaluated_days": len(evaluated), "mae_kwh": round(sum(abs(v) for v in errors) / len(errors), 6) if errors else None, "rmse_kwh": round(math.sqrt(sum(v * v for v in errors) / len(errors)), 6) if errors else None, "mape_percent": round(sum(abs(v / float(item["actual_kwh"])) for v, item in zip(errors, evaluated) if float(item["actual_kwh"]) != 0) / max(1, sum(float(item["actual_kwh"]) != 0 for item in evaluated)) * 100, 6) if errors else None, "bias_kwh": round(sum(errors) / len(errors), 6) if errors else None}, "model_versions": versions, "error": error, "model_version": MODEL_VERSION, "learning_samples": samples, "learning_status": "calibrating" if samples < 14 else "adaptive", "rollback_guard": guard}
    HEALTH.write_text(json.dumps(health, indent=2), encoding="utf-8")
    history = []
    if HEALTH_HISTORY.exists():
        try:
            history = json.loads(HEALTH_HISTORY.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []
    history.append(health)
    HEALTH_HISTORY.write_text(json.dumps(history[-500:], indent=2), encoding="utf-8")
    state["health"] = health
    return health


def publish_dashboard_data(state: dict, health: dict) -> None:
    def read_csv(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    health_history = []
    if HEALTH_HISTORY.exists():
        try:
            health_history = json.loads(HEALTH_HISTORY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            health_history = []
    report = json.loads(IMPROVEMENT_REPORT.read_text(encoding="utf-8")) if IMPROVEMENT_REPORT.exists() else {}
    DASHBOARD_DATA.write_text(json.dumps({"generated_at": now_ist().isoformat(), "device_id": DEVICE_ID, "ai_results": read_csv(RESULTS), "hourly_predictions": read_csv(HOURLY_RESULTS), "state": state, "health": health, "improvement_report": report, "health_history": health_history[-500:]}, indent=2), encoding="utf-8")


def main() -> None:
    now = now_ist()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"profiles": {}, "processed": {}, "forecast_history": {}, "hourly_forecast_history": {}}
    profiles = state.setdefault("profiles", {})
    processed = state.setdefault("processed", {})
    forecast_history = state.setdefault("forecast_history", {})
    hourly_history = state.setdefault("hourly_forecast_history", {})
    state["model_guard"] = evaluate_model_guard(forecast_history)
    today = now.date().isoformat()
    ai_fields = ["processed_at", "device_id", "date", "actual_date", "forecast_date", "evaluated_at", "calendar_profile", "backfill_status", "missing_hours", "status", "data_status", "cluster", "anomaly_score", "anomaly_threshold", "anomaly_explanation", "actual_kwh", "previous_prediction_kwh", "prediction_error_kwh", "prediction_kwh", "prediction_lower_kwh", "prediction_upper_kwh", "profile_mode", "base_prediction_kwh", "correction_kwh", "feedback_samples", "hourly_error_mae", "model_version"]
    hourly_fields = ["processed_at", "device_id", "date", "hour", "predicted_kwh", "actual_kwh", "error_kwh", "feedback_samples", "model_version"]
    ensure_csv_schema(RESULTS, ai_fields, ["device_id", "date"])
    ensure_csv_schema(HOURLY_RESULTS, hourly_fields, ["device_id", "date", "hour"])
    # Recover any state records that were persisted before a CSV write or push failed.
    if processed:
        upsert_csv(RESULTS, ai_fields, list(processed.values()), ["device_id", "date"])
    try:
        profile, hourly_latency, missing_hours = fetch_hourly(now)
        profiles.setdefault(today, {}).update(profile)
        quality = data_quality(profiles[today], now, hourly_latency)
        previous_date = os.getenv("REPROCESS_DATE") or (now.date() - timedelta(days=1)).isoformat()
        result = None
        daily_latency = None
        completed_profile = profiles.get(previous_date, {})
        force_reprocess = bool(os.getenv("REPROCESS_DATE"))
        if previous_date not in processed or force_reprocess:
            daily_totals, daily_latency = fetch_daily_totals(now.date())
            state["last_daily_total_collection"] = now.isoformat()
            actual_total = daily_totals.get(previous_date)
            values = [float(completed_profile[str(hour)]) for hour in range(24) if str(hour) in completed_profile]
            historical_scores = [float(item.get("anomaly_score")) for item in processed.values() if isinstance(item, dict) and item.get("anomaly_score") not in (None, "")]
            adaptive_threshold = float(state.get("adaptive_anomaly_threshold", PARAMS["anomaly_threshold"]))
            if len(historical_scores) >= 7:
                adaptive_threshold = max(float(PARAMS["anomaly_threshold"]), mean(historical_scores) + 2 * pstdev(historical_scores))
                state["adaptive_anomaly_threshold"] = round(adaptive_threshold, 6)
            if len(values) == 24:
                result = anomaly_result(values, now, previous_date, adaptive_threshold)
                result["anomaly_explanation"] = explain_profile(values, profiles, previous_date)
            else:
                result = {"processed_at": now.isoformat(), "device_id": DEVICE_ID, "date": previous_date, "status": "daily_total_only", "data_status": "daily_total_only", "backfill_status": "measured_hours_only", "missing_hours": [hour for hour in range(24) if str(hour) not in completed_profile], "cluster": "", "anomaly_score": "", "anomaly_threshold": adaptive_threshold, "anomaly_explanation": f"Daily total available; hourly profile is missing {24 - len(values)} hour(s); no hourly estimate was invented."}
            result.setdefault("data_status", "complete" if len(values) == 24 else "daily_total_only")
            result.setdefault("actual_date", previous_date)
            result.setdefault("forecast_date", today)
            result.setdefault("calendar_profile", calendar_profile(previous_date))
            result.setdefault("model_version", MODEL_VERSION)
            result.setdefault("backfill_status", "measured_24_hours" if len(values) == 24 else "measured_hours_only")
            result.setdefault("missing_hours", [] if len(values) == 24 else [hour for hour in range(24) if str(hour) not in completed_profile])
            prior_daily = forecast_history.get(previous_date)
            if actual_total is not None:
                if len(values) == 24:
                    hourly_total = sum(values)
                    difference = abs(hourly_total - actual_total)
                    quality["daily_hourly_consistency"] = {"hourly_total_kwh": round(hourly_total, 6), "daily_total_kwh": round(actual_total, 6), "difference_kwh": round(difference, 6), "relative_difference": round(difference / max(abs(actual_total), 1.0), 6), "status": "good" if difference / max(abs(actual_total), 1.0) <= 0.05 else "review"}
                else:
                    quality["daily_hourly_consistency"] = {"status": "review", "reason": "hourly profile incomplete", "available_hours": len(values), "missing_hours": [hour for hour in range(24) if str(hour) not in completed_profile], "daily_total_kwh": round(actual_total, 6), "hourly_residual_not_allocated_kwh": None}
            if actual_total is not None and isinstance(prior_daily, dict):
                predicted_total = float(prior_daily["predicted_kwh"])
                error = actual_total - predicted_total
                prior_daily.update({"actual_kwh": round(actual_total, 6), "error_kwh": round(error, 6), "evaluated_at": now.isoformat()})
                result.update({"actual_kwh": round(actual_total, 6), "previous_prediction_kwh": round(predicted_total, 6), "prediction_error_kwh": round(error, 6), "evaluated_at": now.isoformat()})
            prior_hourly = hourly_history.get(previous_date)
            if isinstance(prior_hourly, dict) and len(values) == 24 and len(prior_hourly.get("predicted_kwh", [])) == 24:
                actual_hours = [float(completed_profile[str(hour)]) for hour in range(24)]
                errors = [actual_hours[h] - float(prior_hourly["predicted_kwh"][h]) for h in range(24)]
                prior_hourly.update({"actual_kwh": actual_hours, "error_kwh": [round(v, 6) for v in errors], "evaluated_at": now.isoformat()})
                result["hourly_error_mae"] = round(sum(abs(v) for v in errors) / 24, 6)
            state["model_guard"] = evaluate_model_guard(forecast_history)
            adaptation_enabled = state["model_guard"].get("adaptation_enabled", True)
            daily_forecast = adaptive_daily_forecast(daily_totals, forecast_history, today, previous_date, now, adaptation_enabled)
            if daily_forecast:
                prediction, base_prediction, correction, samples, lower, upper, mode = daily_forecast
                result.update({"prediction_kwh": prediction, "base_prediction_kwh": base_prediction, "correction_kwh": correction, "feedback_samples": samples, "prediction_lower_kwh": lower, "prediction_upper_kwh": upper, "profile_mode": mode})
            hourly_forecast = adaptive_hourly_forecast(profiles, hourly_history, today, previous_date, now, adaptation_enabled)
            if hourly_forecast:
                predicted, base, corrections, samples, lower, upper, mode = hourly_forecast
                result.update({"hourly_forecast": {str(h): round(predicted[h], 6) for h in range(24)}, "hourly_actual": {}, "hourly_error": {}, "hourly_feedback_samples": samples, "hourly_lower_kwh": {str(h): round(lower[h], 6) for h in range(24)}, "hourly_upper_kwh": {str(h): round(upper[h], 6) for h in range(24)}, "hourly_profile_mode": mode})
            state.setdefault("daily_quality_history", {})[previous_date] = {"data_status": result.get("data_status"), "backfill_status": result.get("backfill_status"), "available_hours": len(values), "missing_hours": result.get("missing_hours", []), "consistency": quality.get("daily_hourly_consistency", {}), "recorded_at": now.isoformat()}
            processed[previous_date] = result
            upsert_csv(RESULTS, ai_fields, [result], ["device_id", "date"])
            if hourly_forecast:
                upsert_csv(HOURLY_RESULTS, hourly_fields, [{"processed_at": now.isoformat(), "device_id": DEVICE_ID, "date": today, "hour": h, "predicted_kwh": result["hourly_forecast"].get(str(h), ""), "actual_kwh": "", "error_kwh": "", "feedback_samples": result.get("hourly_feedback_samples", 0), "model_version": MODEL_VERSION} for h in range(24)], ["device_id", "date", "hour"])
        state["last_run"] = now.isoformat()
        state["model_version"] = MODEL_VERSION
        IMPROVEMENT_REPORT.write_text(json.dumps(build_improvement_report(state, now), indent=2), encoding="utf-8")
        health = write_health(state, now, "healthy", hourly_latency, daily_latency, missing_hours, quality)
        health.update({"hourly_collection_at": now.isoformat(), "daily_total_collection_at": state.get("last_daily_total_collection"), "dashboard_payload_generated_at": now.isoformat(), "dashboard_build": os.getenv("DASHBOARD_BUILD", "2026-09-14-r5")})
        HEALTH.write_text(json.dumps(health, indent=2), encoding="utf-8")
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        publish_dashboard_data(state, health)
        print(f"Device {DEVICE_ID}; collected {len(profiles.get(today, {}))}/24 hours for {today}; model {MODEL_VERSION}")
    except Exception as exc:
        health = write_health(state, now, "degraded", None, None, [], None, str(exc))
        state["last_run"] = now.isoformat()
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        publish_dashboard_data(state, health)
        raise


if __name__ == "__main__":
    main()
