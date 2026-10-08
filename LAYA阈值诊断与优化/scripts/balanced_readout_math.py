"""Class-balanced linear probe on fixed features; scores are not population-calibrated."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def class_weights(y):
    y = np.asarray(y, dtype=float)
    assert len(np.unique(y)) == 2 and set(y) == {0., 1.}
    positive = y.sum()
    weights = np.where(y == 1, len(y)/(2*positive), len(y)/(2*(len(y)-positive)))
    assert abs(weights[y == 1].sum()-len(y)/2) < 1e-9
    assert abs(weights[y == 0].sum()-len(y)/2) < 1e-9
    return weights


def objective(theta, x, y, weights, regularization):
    z = x@theta[:-1]+theta[-1]
    error = weights*(expit(z)-y)
    loss = np.mean(weights*(np.logaddexp(0, z)-y*z))+regularization*np.dot(theta[:-1], theta[:-1])/2
    gradient = np.r_[x.T@error/len(y)+regularization*theta[:-1], error.mean()]
    return loss, gradient


def fit(features, y, regularization):
    features = np.asarray(features, dtype=float)
    y = np.asarray(y, dtype=float)
    assert features.ndim == 2 and len(features) == len(y) and np.isfinite(features).all()
    mean = features.mean(0)
    std = features.std(0)
    std[std < 1e-4] = 1.
    x = (features-mean)/std
    weights = class_weights(y)
    result = minimize(lambda theta: objective(theta, x, y, weights, regularization),
                      np.zeros(x.shape[1]+1), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-11, 'gtol': 1e-7})
    assert result.success and np.isfinite(result.fun), result.message
    return dict(kind='frozen_feature_linear', regularization=float(regularization),
                feature_mean=mean.tolist(), feature_std=std.tolist(), weight=result.x[:-1].tolist(),
                intercept=float(result.x[-1]), temperature=1., iterations=int(result.nit),
                objective='Equal total weight for each utility class; mean weighted log loss plus lambda L2/2',
                population_probability_calibration=False, replan_training_count=int(y.sum()),
                continue_training_count=int(len(y)-y.sum()))
