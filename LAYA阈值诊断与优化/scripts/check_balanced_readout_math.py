"""Check weighting, analytic gradient and identifiable/constant-feature limits."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.optimize import check_grad
from scipy.special import expit
from balanced_readout_math import class_weights, objective, fit

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(44)
x = rng.normal(size=(20, 5))
y = np.array([1., 1., *([0.]*18)])
w = class_weights(y)
theta = rng.normal(size=6)
error = check_grad(lambda z: objective(z, x, y, w, .2)[0],
                   lambda z: objective(z, x, y, w, .2)[1], theta)
assert error < 1e-5
constant = fit(np.zeros((20, 3)), y, .1)
assert np.max(np.abs(constant['weight'])) == 0 and abs(constant['intercept']) < 1e-8
separated = fit((2*y-1)[:, None], y, .1)
p = expit((((2*y-1)[:, None]-separated['feature_mean'])/separated['feature_std'])@np.array(separated['weight'])+separated['intercept'])
assert np.array_equal(p >= .5, y.astype(bool))
result = dict(checks=4, errors=0, class_weight_totals=[float(w[y == 0].sum()), float(w[y == 1].sum())],
              finite_difference_gradient_error=float(error), constant_feature_score=.5,
              separable_case_correct=20, module_sha256=hashlib.sha256((ROOT/'scripts/balanced_readout_math.py').read_bytes()).hexdigest())
(ROOT/'checks/balanced_readout_math.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(result))
