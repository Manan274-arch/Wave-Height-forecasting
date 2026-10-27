"""Read-only Vercel Python API; python api/index.py also serves it locally."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlparse, parse_qs
import json
import sys
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1] / 'artifacts'
FILES = {'Linear Regression':'linear_regression','Decision Tree':'decision_tree','Random Forest':'random_forest','XGBoost':'xgboost'}

@lru_cache(maxsize=1)
def data():
    source = pd.read_csv(ROOT/'observations.csv', parse_dates=['timestamp']).set_index('timestamp')
    predictions = pd.read_csv(ROOT/'test_predictions.csv', parse_dates=['timestamp','target_timestamp']).set_index('timestamp')
    return source, predictions

@lru_cache(maxsize=4)
def model(name):
    fitted = joblib.load(ROOT/(FILES[name]+'.joblib'))
    if hasattr(fitted, 'n_jobs'): fitted.n_jobs = 1
    return fitted

def features(source):
    # Same causal formulas and column order as src/03_feature_engineering.py.
    names = json.loads((ROOT/'model_metadata.json').read_text())['feature_names']
    f = source[names[:8]].copy()
    f['VMDR_sin'] = np.sin(np.deg2rad(source.VMDR))
    f['VMDR_cos'] = np.cos(np.deg2rad(source.VMDR))
    f['VHM0_lag_6h'] = source.VHM0.shift(2)
    f['VHM0_lag_24h'] = source.VHM0.shift(8)
    f['VHM0_change_3h'] = source.VHM0.diff()
    f['VHM0_roll_mean_12h'] = source.VHM0.rolling(4).mean()
    f['VHM0_roll_std_12h'] = source.VHM0.rolling(4).std(ddof=1)
    f['wind_speed_roll_mean_12h'] = source.wind_speed.rolling(4).mean()
    f['month_sin'] = np.sin(2*np.pi*source.index.month/12)
    f['month_cos'] = np.cos(2*np.pi*source.index.month/12)
    return f[names]

def bootstrap():
    source, p = data()
    r = pd.read_csv(ROOT/'regression_metrics.csv')
    c = pd.read_csv(ROOT/'classification_metrics.csv')
    metrics = r.merge(c, on=['model','sample_count']).to_dict('records')
    dates = [t.isoformat() for t in p.target_timestamp]
    calm = p.actual_VHM0.idxmin()
    elevated = p.actual_VHM0.idxmax()
    difficult = (p.XGBoost-p.actual_VHM0).abs().idxmax()
    monthly = source.groupby(source.index.month).agg(height=('VHM0','mean'),wind=('wind_speed','mean')).reset_index().rename(columns={'timestamp':'month'}).to_dict('records')
    return {'dates':dates,'metrics':metrics,'monthly':monthly,'presets':{'Calm seas':p.loc[calm,'target_timestamp'].isoformat(),'Elevated event':p.loc[elevated,'target_timestamp'].isoformat(),'Largest XGBoost error':p.loc[difficult,'target_timestamp'].isoformat()}}

def forecast(timestamp):
    source, p = data()
    try:
        target = pd.Timestamp(timestamp)
        t = target - pd.Timedelta(hours=12)
    except (ValueError,TypeError): raise ValueError('Choose a valid test timestamp.')
    if t not in p.index: raise ValueError('Choose a prediction time in the held-out test set.')
    # Compute only from observations available at the forecast origin.
    history = source.loc[:t]
    x = features(history).iloc[[-1]]
    outputs = {name:float(model(name).predict(x)[0]) for name in FILES}
    record = source.loc[t]
    curve = source.loc[t-pd.Timedelta(hours=48):t+pd.Timedelta(hours=12)]
    return {'origin':t.isoformat(),'target':p.loc[t,'target_timestamp'].isoformat(),'actual':float(p.loc[t,'actual_VHM0']),'current':float(record.VHM0),'wind':float(record.wind_speed),'direction':float(record.VMDR),'predictions':outputs,'features':{k:float(v) for k,v in x.iloc[0].items()},'history':[{'time':i.isoformat(),'height':float(row.VHM0),'future':bool(i>t)} for i,row in curve.iterrows()]}

def timeline(start, period):
    _, p = data()
    try:
        begin = pd.Timestamp(start)
        if begin.tzinfo is None: begin = begin.tz_localize('UTC')
        else: begin = begin.tz_convert('UTC')
        if pd.isna(begin): raise ValueError('Missing date')
        begin = begin.normalize()
    except (ValueError, TypeError): raise ValueError('Choose a valid start date.')
    if period not in ('week', 'month'): raise ValueError('Choose week or month.')
    if period == 'month': begin = begin.replace(day=1)
    end = begin + (pd.DateOffset(months=1) if period == 'month' else pd.Timedelta(days=7))
    rows = p[(p.target_timestamp >= begin) & (p.target_timestamp < end)]
    if rows.empty: raise ValueError('No test results for this period. Choose dates in 2023 or 2024.')
    return {'start':begin.isoformat(), 'end':end.isoformat(), 'records':[
        {'time':row.target_timestamp.isoformat(), 'actual':float(row.actual_VHM0),
         'predictions':{name:float(row[name]) for name in FILES}}
        for _, row in rows.iterrows()
    ]}

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in ('/api/bootstrap','/api/bootstrap/'): payload=bootstrap()
            elif parsed.path in ('/api/forecast','/api/forecast/'):
                payload=forecast(parse_qs(parsed.query).get('timestamp',[''])[0])
            elif parsed.path == '/api/timeline':
                query = parse_qs(parsed.query)
                payload=timeline(query.get('start',[''])[0], query.get('period',['month'])[0])
            elif parsed.path == '/api/health': payload={'status':'ok','inference':'saved-models','horizon_hours':12}
            else: return self.send_json({'error':'Not found'},404)
            self.send_json(payload)
        except ValueError as exc: self.send_json({'error':str(exc)},400)
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.send_json({'error':'Inference service unavailable. Please retry.'},500)
    def send_json(self,payload,status=200):
        body=json.dumps(payload,allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == '__main__':
    print('CoastCast API: http://127.0.0.1:8000',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8000),handler).serve_forever()

