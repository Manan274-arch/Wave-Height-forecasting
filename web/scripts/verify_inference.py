"""Compare deployed inference with canonical predictions and causal feature pipeline."""
import sys
from pathlib import Path
import importlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'web'/'api'))
sys.path.insert(0,str(ROOT/'src'))
import index
canonical=importlib.import_module('03_feature_engineering')
source,p=index.data()
expected=canonical.make_features(source)
b=index.bootstrap()
origins=list(b['presets'].values())+[b['dates'][0],b['dates'][-1],b['dates'][len(b['dates'])//2]]
for origin in origins:
    result=index.forecast(origin)
    row=p.loc[origin]
    np.testing.assert_allclose(list(result['features'].values()),expected.loc[origin].to_numpy(),rtol=1e-10,atol=1e-10)
    for name,value in result['predictions'].items():
        np.testing.assert_allclose(value,row[name],rtol=1e-6,atol=1e-6)
for invalid in ['bad-date','2020-01-01T00:00:00+00:00','']:
    try:index.forecast(invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid origin accepted')
print(f'PASS: {len(origins)} origins, all four models, 18 canonical features, and invalid inputs.')
