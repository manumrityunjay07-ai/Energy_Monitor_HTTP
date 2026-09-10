from __future__ import annotations
import csv, json, math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

DEVICE_ID = 153
TZ = ZoneInfo('Asia/Kolkata')
HOURLY_URL = 'https://adc.bitsathy.ac.in/2024/ems_dashboard/api/fetch_hourly_energy_consumption.php'
DAYWISE_URL = 'https://adc.bitsathy.ac.in/2024/ems_dashboard/api/fetch_day_energy_consumption.php'
STATE = Path('data/state.json')
RESULTS = Path('results/ai_results.csv')
PARAMS = json.loads(Path('esp32_parameters.json').read_text(encoding='utf-8'))
STATE.parent.mkdir(exist_ok=True)
RESULTS.parent.mkdir(exist_ok=True)

now = datetime.now(TZ)
today = now.date().isoformat()
state = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {'profiles': {}, 'processed': {}}
profiles = state.setdefault('profiles', {})
processed = state.setdefault('processed', {})
profile = profiles.setdefault(today, {})

r = requests.post(HOURLY_URL, json={'deviceId': DEVICE_ID}, timeout=30)
r.raise_for_status()
rows = r.json().get('data', [])
if not isinstance(rows, list) or len(rows) < 24:
    raise RuntimeError(f'Hourly response did not contain 24 rows: {r.text[:500]}')
# The dashboard shows future hours as zero. At the current hour, keep the
# displayed value because it is the latest server value; never keep later hours.
for row in rows:
    hour = int(row['hour'])
    value = float(row['total_energy'])
    if 0 <= hour <= now.hour and math.isfinite(value) and value >= 0:
        profile[str(hour)] = value

previous_date = (now.date() - timedelta(days=1)).isoformat()
completed_profile = profiles.get(previous_date, {})
status = f'collected {len(profile)}/24 hours for {today}'
result = None
if len(completed_profile) == 24 and previous_date not in processed:
    values = [float(completed_profile[str(h)]) for h in range(24)]
    scaled = [(x-m)/s if s else 0 for x,m,s in zip(values, PARAMS['cluster_scaler_mean'], PARAMS['cluster_scaler_scale'])]
    distances = [math.sqrt(sum((x-c)**2 for x,c in zip(scaled, centre))) for centre in PARAMS['cluster_centres_scaled']]
    cluster = min(range(len(distances)), key=distances.__getitem__) + 1
    score = distances[cluster-1]
    is_anomaly = score > PARAMS['anomaly_threshold']
    result = {'processed_at': now.isoformat(), 'device_id': DEVICE_ID, 'date': previous_date, 'status': 'anomaly' if is_anomaly else 'normal', 'cluster': cluster, 'anomaly_score': round(score, 6), 'anomaly_threshold': PARAMS['anomaly_threshold']}
    # Retrieve recent daily totals for the prediction features.
    end = datetime.strptime(previous_date, '%Y-%m-%d').date()
    dr = requests.post(DAYWISE_URL, files={'deviceId': (None, str(DEVICE_ID)), 'fromDate': (None, (end-timedelta(days=45)).isoformat()), 'toDate': (None, end.isoformat())}, timeout=30)
    dr.raise_for_status()
    daily = sorted([x for x in dr.json().get('data', []) if x.get('total_consumption') is not None], key=lambda x: x.get('custom_day',''))
    totals = [float(x['total_consumption']) for x in daily[-3:]]
    if len(totals) == 3:
        features = [totals[-1], totals[-2], totals[-3], sum(totals)/3]
        scaled_features = [(x-m)/s if s else 0 for x,m,s in zip(features, PARAMS['prediction_scaler_mean'], PARAMS['prediction_scaler_scale'])]
        result['prediction_kwh'] = round(PARAMS['prediction_intercept'] + sum(c*x for c,x in zip(PARAMS['prediction_coefficients'], scaled_features)), 6)
    else:
        result['prediction_kwh'] = ''
    processed[previous_date] = result
    with RESULTS.open('a', newline='', encoding='utf-8') as f:
        fields = ['processed_at','device_id','date','status','cluster','anomaly_score','anomaly_threshold','prediction_kwh']
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0: writer.writeheader()
        writer.writerow({k: result.get(k, '') for k in fields})

state['last_run'] = now.isoformat()
STATE.write_text(json.dumps(state, indent=2), encoding='utf-8')
print(f'Device {DEVICE_ID}; {status}; local time {now.isoformat()}')
if result:
    print(json.dumps(result, indent=2))
else:
    print(f'AI analysis waiting; completed profile for {previous_date}: {len(completed_profile)}/24 hours')
