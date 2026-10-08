"""Verify serialized deployment inference against independent fitting-time numerics."""
from pathlib import Path
import json
import numpy as np
from calibration_math import predict
from deployment_readout import Readout
ROOT=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(20261008);x=rng.normal(size=(12,1024));z=np.linspace(-1000,1000,12)
models=[dict(kind='positive_affine',slope=.7,intercept=-.9),
        dict(kind='frozen_feature_linear',feature_mean=rng.normal(size=1024).tolist(),
             feature_std=rng.uniform(.1,2,1024).tolist(),weight=rng.normal(size=1024).tolist(),intercept=-.2,temperature=1.7)]
checks=[]
for model in models:
    adapter=Readout(json.loads(json.dumps(model)));reference=predict(model,z,x)
    actual=[adapter.predict(float(delta),feature) for delta,feature in zip(z,x)]
    assert np.max(abs(reference-np.array([r['probabilities']['replan'] for r in actual])))<1e-13
    assert all(r['choice']==('replan' if p>=.5 else 'continue') for r,p in zip(actual,reference))
    checks.append(dict(name=model['kind']+' serialized adapter, independent sigmoid and normalization',passed=True))
tie=Readout(dict(kind='positive_affine',slope=1.,intercept=0.)).predict(0.)
assert tie['choice']=='replan' and tie['probabilities']=={'replan':.5,'continue':.5}
checks.append(dict(name='documented exact tie preserves first option',passed=True))
(ROOT/'checks/deployment_readout.json').write_text(json.dumps(checks,indent=2),newline='\n')
print(json.dumps(dict(checks=len(checks),failures=0)))
