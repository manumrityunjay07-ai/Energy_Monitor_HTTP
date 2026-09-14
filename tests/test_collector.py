import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import collect_once


class CollectorSafetyTests(unittest.TestCase):
    def test_data_quality_is_complete_for_completed_hours(self):
        now = datetime(2026, 9, 13, 5, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
        profile = {str(hour): float(hour) for hour in range(5)}
        quality = collect_once.data_quality(profile, now, 100.0)
        self.assertEqual(quality["score"], 100.0)
        self.assertEqual(quality["completed_hours"], 5)
        self.assertEqual(quality["status"], "complete")

    def test_upsert_migrates_legacy_schema_without_duplicates(self):
        fields = ["device_id", "date", "model_version"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text("device_id,date\n153,2026-09-12\n", encoding="utf-8")
            collect_once.upsert_csv(path, fields, [{"device_id": 153, "date": "2026-09-12", "model_version": "v2"}], ["device_id", "date"])
            rows = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0], "device_id,date,model_version")
            self.assertEqual(len(rows), 2)
            self.assertIn("v2", rows[1])

    def test_daily_adaptation_is_capped(self):
        daily = {f"2026-09-{day:02d}": 1000.0 for day in range(1, 8)}
        history = {f"2026-09-{day:02d}": {"error_kwh": 10000.0} for day in range(1, 8)}
        forecast = collect_once.adaptive_daily_forecast(daily, history, "2026-09-08", "2026-09-07", datetime.now(ZoneInfo("Asia/Kolkata")))
        self.assertIsNotNone(forecast)
        prediction, base, correction, *_ = forecast
        self.assertLessEqual(abs(correction), abs(base) * 0.25 + 1e-6)
        self.assertGreaterEqual(prediction, 0)

    def test_model_guard_rolls_back_when_adaptation_is_worse(self):
        history = {
            str(day): {
                "actual_kwh": 1000,
                "base_prediction_kwh": 1000,
                "predicted_kwh": 1300,
            }
            for day in range(5)
        }
        guard = collect_once.evaluate_model_guard(history)
        self.assertFalse(guard["adaptation_enabled"])
        self.assertIn("rollback", guard["reason"])

    def test_early_feedback_holds_the_baseline(self):
        daily_totals = {f"2026-09-{day:02d}": 1000.0 for day in range(10, 13)}
        forecast = collect_once.adaptive_daily_forecast(
            daily_totals,
            {"2026-09-12": {"error_kwh": 500.0}},
            "2026-09-13",
            "2026-09-12",
            datetime.now(ZoneInfo("Asia/Kolkata")),
            True,
        )
        self.assertIsNotNone(forecast)
        self.assertEqual(forecast[2], 0.0)


if __name__ == "__main__":
    unittest.main()
