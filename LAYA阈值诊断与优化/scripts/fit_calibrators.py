"""Fit only state0, choose only on state1, write frozen coefficients before state2 evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
import scipy
from scipy.special import logit
from calibration_data import ROOT,VARIANTS,load_runs,select
from calibration_math import metrics,fit_temperature,fit_affine,fit_linear,predict,paired_score_intervals
from local_policy import allowed


def evaluate_model(model,data):
    p=predict(model,data['z'],data['features'],data['horizons'],data['tasks'])
    return metrics(data['y'],p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--name',required=True)
    p.add_argument('--variants',default=','.join(VARIANTS))
    args=p.parse_args();assert args.name.isascii() and args.name.replace('_','').isalnum()
    variants=args.variants.split(',');assert len(set(variants))==len(variants) and set(variants)<=set(VARIANTS+['binary_window','binary_window3'])
    allowed()
    out=ROOT/'models'/args.name;out.mkdir(parents=True,exist_ok=False)
    rows,features,provenance=load_runs(args.runs.split(','),{0,1},variants)
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),source=provenance,
        numpy=np.__version__,scipy=scipy.__version__,training_states=[0],selection_and_calibration_states=[1],
        diagnostic_evaluation_states=[2],state2_loaded=False,variants=variants,
        label_policy='Exclude both-failure cases from fitting and score evaluation; report coverage explicitly',
        temperature_bounds=[.5,5.],affine_slope_bounds=[.2,2.],affine_intercept_bounds=[-8,8],
        readout_l2_grid=[.01,.1,1.,10.],VLA_training_steps=0,base_LAYA_training_steps=0,
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('calibration*.py')},
        fit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        task_horizon_prior_amendment_sha256=hashlib.sha256((ROOT/'CALIBRATION_AMENDMENT_01.md').read_bytes()).hexdigest())
    models={};results={};deployment_candidates=[]
    for target in ['necessity','utility']:
        for variant in variants:
            allowed()
            train=select(rows,features,variant,target,{0});cal=select(rows,features,variant,target,{1})
            temperatures={r['temperature'] for r in train['rows']+cal['rows']};assert len(temperatures)==1
            prior=(train['y'].sum()+1)/(len(train['y'])+2)
            group={
                'raw':dict(kind='raw',temperature=float(next(iter(temperatures)))),
                'constant':dict(kind='constant',probability=float(prior)),
                'horizon_prior':dict(kind='horizon_prior',probabilities={str(h):float((train['y'][train['horizons']==h].sum()+1)/(np.sum(train['horizons']==h)+2)) for h in [50,100,200]}),
                'temperature':fit_temperature(train['z'],train['y']),
                'affine':fit_affine(train['z'],train['y'])}
            task_priors={}
            for task in range(10):
                for h in [50,100,200]:
                    mask=(train['tasks']==task)&(train['horizons']==h)
                    task_priors[f'{task}:{h}']=float((train['y'][mask].sum()+2*group['horizon_prior']['probabilities'][str(h)])/(mask.sum()+2))
            group['task_horizon_prior']=dict(kind='task_horizon_prior',probabilities=task_priors,
                                            pseudo_count=2,description='Task and horizon only, no current observation')
            readout_trials=[]
            if len(np.unique(train['y']))==2:
                for penalty in [.01,.1,1.,10.]:
                    model=fit_linear(train['features'],train['y'],penalty)
                    readout_trials.append((evaluate_model(model,cal),model))
                _,best=min(readout_trials,key=lambda pair:(pair[0]['log_loss'],pair[0]['brier'],-pair[1]['regularization']))
                group['linear']=best
                probabilities=predict(best,cal['z'],cal['features'],cal['horizons'])
                calibration=fit_temperature(logit(np.clip(probabilities,1e-12,1-1e-12)),cal['y'])
                group['linear_calibrated']={**best,'temperature':calibration['temperature'],
                    'temperature_fit_on_state1':True,'policy_argmax_identical_to_uncalibrated_linear':True}
            key=target+'__'+variant
            result=dict(target=target,variant=variant,train_coverage=train['coverage'],calibration_coverage=cal['coverage'],
                train_resolved=train['resolved'],calibration_resolved=cal['resolved'],train_total=train['total'],calibration_total=cal['total'],
                readout_trials=[dict(regularization=model['regularization'],calibration_metrics=measure) for measure,model in readout_trials],
                models={})
            for name,model in group.items():
                model_id=key+'__'+name
                models[model_id]=dict(target=target,variant=variant,model=model)
                result['models'][name]=dict(train=evaluate_model(model,train),calibration=evaluate_model(model,cal))
            # Eligibility prevents promoting an always-majority predictor as a useful decision gate.
            baselines=[result['models'][n]['calibration']['log_loss'] for n in ['constant','horizon_prior','raw','task_horizon_prior']]
            for name in ['affine','linear_calibrated']:
                if name not in result['models']:continue
                m=result['models'][name]['calibration']
                eligible=(target=='utility' and variant in ['binary_short','binary_compact','binary_window','binary_window3'] and m['auc'] is not None
                          and m['auc']>.55 and m['replan_recall']>0 and m['continue_recall']>0
                          and m['log_loss']<min(baselines))
                candidate=dict(model_id=key+'__'+name,eligible=bool(eligible),calibration_metrics=m,
                    screening_rule='utility; actual binary question; AUROC>0.55; both class recalls>0; calibration NLL better than raw and all three priors')
                deployment_candidates.append(candidate)
            results[key]=result
            print(json.dumps(dict(group=key,train_n=train['resolved'],cal_n=cal['resolved'],
                raw_cal_nll=result['models']['raw']['calibration']['log_loss'],
                best_prior_cal_nll=min(baselines[i] for i in [0,1,3]),
                best_readout_cal_nll=result['models'].get('linear_calibrated',{}).get('calibration',{}).get('log_loss'))))
    eligible=sorted([c for c in deployment_candidates if c['eligible']],key=lambda c:(c['calibration_metrics']['log_loss'],c['calibration_metrics']['brier'],c['model_id']))
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),model_count=len(models),
                    candidate_screen=deployment_candidates,proposed_deployment_models=[c['model_id'] for c in eligible[:2]],
                    interpretation='State1 is reused for model choice and temperature fitting; only state2 or later closed-loop checks can assess generalization')
    (out/'models.json').write_text(json.dumps(models,indent=2))
    (out/'development_scores.json').write_text(json.dumps(results,indent=2))
    manifest['models_sha256']=hashlib.sha256((out/'models.json').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(model_directory=str(out),proposed_deployment_models=manifest['proposed_deployment_models'],state2_seen=False)))


if __name__=='__main__':main()
