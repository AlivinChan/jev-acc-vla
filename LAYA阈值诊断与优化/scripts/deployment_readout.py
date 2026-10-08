"""Small NumPy-only inference adapter for separately saved frozen-LAYA readouts."""
import math
import numpy as np


class Readout:
    def __init__(self,model):
        self.model=model;self.kind=model['kind']
        assert self.kind in {'positive_affine','frozen_feature_linear'}
        if self.kind=='frozen_feature_linear':
            self.mean=np.asarray(model['feature_mean'],dtype=np.float64)
            self.std=np.asarray(model['feature_std'],dtype=np.float64)
            self.weight=np.asarray(model['weight'],dtype=np.float64)
            assert self.mean.shape==self.std.shape==self.weight.shape==(1024,)
            assert np.isfinite(self.mean).all() and np.isfinite(self.std).all() and np.isfinite(self.weight).all()
            assert (self.std>0).all()
            self.temperature=float(model.get('temperature',1.));self.intercept=float(model['intercept'])
            assert .5<=self.temperature<=5 and math.isfinite(self.intercept)
        else:
            self.slope=float(model['slope']);self.intercept=float(model['intercept'])
            assert .2<=self.slope<=2 and math.isfinite(self.intercept)

    def predict(self,raw_delta,features=None):
        assert math.isfinite(raw_delta)
        if self.kind=='positive_affine':logit=self.slope*raw_delta+self.intercept
        else:
            features=np.asarray(features,dtype=np.float64)
            assert features.shape==(1024,) and np.isfinite(features).all()
            logit=float(((features-self.mean)/self.std)@self.weight+self.intercept)/self.temperature
        assert math.isfinite(logit)
        exponential=math.exp(-abs(logit))
        probability=1/(1+exponential) if logit>=0 else exponential/(1+exponential)
        return dict(logit=logit,probabilities={'replan':probability,'continue':1-probability},
                    choice='replan' if probability>=.5 else 'continue')
