# Device 153 data contract

The collector publishes two kinds of historical information. `results/daily_totals.csv` contains every valid daily total returned by the day-wise Device 153 API within its available range. `data/state.json` stores hourly profiles captured by the hourly endpoint under the IST date on which they were collected.

The hourly endpoint currently returns the latest available 24-hour profile even when a historical date is included in the request. The collector therefore never relabels that response as yesterday's data. A historical day is marked `daily_total_only` when its daily total exists but its measured hourly profile is unavailable. This prevents fabricated hourly values and keeps date alignment honest.

On every scheduled run, the collector fetches the available daily-total range, persists it, captures the current day's available hours, updates predictions and health metadata, validates all artifacts, and publishes the snapshot to the `live-data` branch. As new hours arrive, the current day's profile is updated; once the upstream service exposes historical hourly data, a future backfill routine can safely fill historical profiles without changing the data contract.
