"""CPU-only proper-score fitting; no VLA or LAYA weights are changed here."""
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from scipy.special import expit
from scipy.stats import rankdata


def nll_from_logits(y,z):return float(np.mean(np.logaddexp(0,z)-np.asarray(y)*z))


def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    assert y.ndim==p.ndim==1 and len(y)==len(p) and len(y)>0
    assert set(y).issubset({0,1}) and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    pred=(p>=.5).astype(int);pos=int(y.sum());neg=len(y)-pos
    tp=int(np.sum((y==1)&(pred==1)));tn=int(np.sum((y==0)&(pred==0)))
    covered=np.maximum(p,1-p)>=.7
    safe=np.clip(p,1e-12,1-1e-12)
    auc=(float(np.sum(rankdata(p)[y==1]))-pos*(pos+1)/2)/(pos*neg) if pos and neg else None
    ece=0.
    for index in range(5):
        mask=(p>=index/5)&((p<(index+1)/5) if index<4 else (p<=1))
        if np.any(mask):ece+=float(np.mean(mask)*abs(np.mean(p[mask])-np.mean(y[mask])))
    return dict(n=len(y),replan_labels=pos,continue_labels=neg,replan_predictions=int(pred.sum()),
        accuracy=float(np.mean(pred==y)),balanced_accuracy=(tp/pos+tn/neg)/2 if pos and neg else None,
        replan_recall=tp/pos if pos else None,continue_recall=tn/neg if neg else None,
        auc=auc,brier=float(np.mean((p-y)**2)),
        log_loss=float(np.mean(-y*np.log(safe)-(1-y)*np.log(1-safe))),ece_5_equal_bins=ece,
        above_07=int(covered.sum()),coverage_07=float(np.mean(covered)),
        covered_accuracy_07=float(np.mean(pred[covered]==y[covered])) if covered.any() else None)


def fit_temperature(z,y):
    z=np.asarray(z,dtype=float);y=np.asarray(y,dtype=float)
    result=minimize_scalar(lambda t:nll_from_logits(y,z/t),bounds=(.5,5.),method='bounded',
                           options={'xatol':1e-8,'maxiter':200})
    assert result.success and np.isfinite(result.fun)
    return dict(kind='temperature',temperature=float(result.x),bounds=[.5,5.],fit_log_loss=float(result.fun))


def fit_affine(z,y):
    z=np.asarray(z,dtype=float);y=np.asarray(y,dtype=float)
    def objective(theta):
        slope,intercept=theta;logits=slope*z+intercept;error=expit(logits)-y
        loss=nll_from_logits(y,logits)+.0001*intercept**2/2
        gradient=np.array([np.mean(error*z),np.mean(error)+.0001*intercept])
        return loss,gradient
    result=minimize(objective,np.array([1.,0.]),jac=True,method='L-BFGS-B',
                    bounds=[(.2,2.),(-8.,8.)],options={'maxiter':1000,'ftol':1e-12,'gtol':1e-9})
    assert result.success and np.isfinite(result.fun),result.message
    return dict(kind='positive_affine',slope=float(result.x[0]),intercept=float(result.x[1]),
                slope_bounds=[.2,2.],intercept_bounds=[-8.,8.],intercept_penalty=.0001)


def fit_linear(features,y,regularization):
    features=np.asarray(features,dtype=float);y=np.asarray(y,dtype=float)
    assert features.ndim==2 and len(features)==len(y) and len(np.unique(y))==2
    mean=features.mean(0);std=features.std(0);std[std<1e-4]=1.
    x=(features-mean)/std
    initial=np.zeros(x.shape[1]+1)
    prior=(y.sum()+1)/(len(y)+2);initial[-1]=np.log(prior/(1-prior))
    def objective(theta):
        logits=x@theta[:-1]+theta[-1];error=expit(logits)-y
        loss=nll_from_logits(y,logits)+regularization*np.dot(theta[:-1],theta[:-1])/2
        gradient=np.r_[x.T@error/len(y)+regularization*theta[:-1],error.mean()]
        return loss,gradient
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',
                    options={'maxiter':1000,'ftol':1e-11,'gtol':1e-7})
    assert result.success and np.isfinite(result.fun),result.message
    return dict(kind='frozen_feature_linear',regularization=float(regularization),
                feature_mean=mean.tolist(),feature_std=std.tolist(),weight=result.x[:-1].tolist(),
                intercept=float(result.x[-1]),temperature=1.,iterations=int(result.nit),
                objective='mean binary log loss + lambda * squared L2(weight)/2; intercept unpenalized')


def predict(model,z,features=None,horizons=None,tasks=None):
    z=np.asarray(z,dtype=float);kind=model['kind']
    if kind=='raw':return expit(z/model['temperature'])
    if kind=='temperature':return expit(z/model['temperature'])
    if kind=='positive_affine':return expit(model['slope']*z+model['intercept'])
    if kind=='constant':return np.full(len(z),model['probability'],dtype=float)
    if kind=='horizon_prior':return np.array([model['probabilities'][str(h)] for h in horizons],dtype=float)
    if kind=='task_horizon_prior':
        assert tasks is not None and len(tasks)==len(horizons)==len(z)
        return np.array([model['probabilities'][f'{t}:{h}'] for t,h in zip(tasks,horizons)],dtype=float)
    if kind=='frozen_feature_linear':
        logits=((np.asarray(features,dtype=float)-model['feature_mean'])/model['feature_std'])@np.array(model['weight'])+model['intercept']
        return expit(logits/model.get('temperature',1.))
    raise ValueError(kind)


def paired_score_intervals(y,a,b,tasks,repeats=10000):
    """Task-cluster bootstrap of proper-score differences; no fitting inside resampling."""
    y=np.asarray(y);a=np.clip(a,1e-12,1-1e-12);b=np.clip(b,1e-12,1-1e-12);tasks=np.asarray(tasks)
    dnll=(-y*np.log(a)-(1-y)*np.log(1-a))-(-y*np.log(b)-(1-y)*np.log(1-b))
    dbrier=(a-y)**2-(b-y)**2
    groups=np.array([[dnll[tasks==task].sum(),dbrier[tasks==task].sum(),np.sum(tasks==task)] for task in sorted(set(tasks))])
    sample=groups[np.random.default_rng(20261008).integers(0,len(groups),(repeats,len(groups)))].sum(1)
    return dict(log_loss_difference=float(dnll.mean()),brier_difference=float(dbrier.mean()),
        log_loss_difference_95_task_bootstrap=np.quantile(sample[:,0]/sample[:,2],[.025,.975]).tolist(),
        brier_difference_95_task_bootstrap=np.quantile(sample[:,1]/sample[:,2],[.025,.975]).tolist(),tasks=len(groups))
