"""Known miscalibration and label/correctness boundaries, without research test data."""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit
from calibration_math import metrics,fit_temperature,fit_affine,fit_linear,predict,paired_score_intervals
ROOT=Path(__file__).resolve().parents[1]
checks=[]
perfect=metrics([0,0,1,1],[.1,.2,.8,.9]);assert perfect['auc']==perfect['accuracy']==1
constant=metrics([0,0,1,1],[.5]*4);assert constant['auc']==.5 and constant['above_07']==0
single=metrics([0,0],[.1,.2]);assert single['auc'] is None and single['replan_recall'] is None
checks.append(dict(name='AUC ties, single class, coverage and decision polarity',passed=True))
rng=np.random.default_rng(20261008);z=rng.normal(0,3,4000);y=rng.binomial(1,expit(z/2))
temp=fit_temperature(z[:2000],y[:2000]);assert 1.6<temp['temperature']<2.4
raw=metrics(y[2000:],expit(z[2000:]));cal=metrics(y[2000:],predict(temp,z[2000:]))
assert cal['log_loss']<raw['log_loss'] and cal['brier']<raw['brier']
checks.append(dict(name='Recover known temperature and improve independent synthetic proper scores',passed=True))
labels=rng.binomial(1,expit(.7*z-1.2));affine=fit_affine(z,labels)
assert abs(affine['slope']-.7)<.1 and abs(affine['intercept']+1.2)<.2
checks.append(dict(name='Recover positive slope and class offset',passed=True))
x=rng.normal(size=(600,6));target=(x[:,0]+x[:,1]*.4>0).astype(int)
model=fit_linear(x[:300],target[:300],.1)
evaluation=metrics(target[300:],predict(model,np.zeros(300),x[300:]))
assert evaluation['accuracy']>.92
checks.append(dict(name='Frozen-feature readout learns an independent synthetic signal',passed=True))
result=paired_score_intervals(y[2000:],predict(temp,z[2000:]),expit(z[2000:]),np.arange(2000)%10)
assert result['log_loss_difference_95_task_bootstrap'][1]<0
checks.append(dict(name='Task bootstrap preserves improvement sign',passed=True))
(ROOT/'checks/calibration_math.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(dict(checks=len(checks),failures=0)))
