"""Read-only state2 diagnostics of already-frozen coefficients; no optimizer path."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
from calibration_data import ROOT,load_runs,select
from calibration_math import metrics,predict,paired_score_intervals
from local_policy import allowed


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--name',required=True)
    args=p.parse_args();assert args.name.isascii() and args.name.replace('_','').isalnum();allowed()
    folder=ROOT/'models'/args.name
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['state2_loaded'] is False
    models_path=folder/'models.json'
    assert hashlib.sha256(models_path.read_bytes()).hexdigest()==manifest['models_sha256']
    models=json.loads(models_path.read_text())
    rows,features,source=load_runs(args.runs.split(','),{2},manifest['variants'])
    assert source['laya_fingerprint']==manifest['source']['laya_fingerprint']
    output=dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),source=source,
        frozen_models_sha256=manifest['models_sha256'],fit_model_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
        fit_performed=False,results={},proposed_deployment_models=manifest['proposed_deployment_models'],
        interpretation='Resolved-label subset only; both-failure coverage is reported. This is a single fixed-noise intervention diagnostic, not a deployment-success estimate.')
    for model_id,definition in models.items():
        allowed();target=definition['target'];variant=definition['variant'];model=definition['model']
        data=select(rows,features,variant,target,{2})
        probs=predict(model,data['z'],data['features'],data['horizons'],data['tasks'])
        raw=models[target+'__'+variant+'__raw']['model']
        prior=models[target+'__'+variant+'__horizon_prior']['model']
        raw_p=predict(raw,data['z'],data['features'],data['horizons'])
        prior_p=predict(prior,data['z'],data['features'],data['horizons'])
        metadata_prior=models[target+'__'+variant+'__task_horizon_prior']['model']
        metadata_p=predict(metadata_prior,data['z'],data['features'],data['horizons'],data['tasks'])
        result=dict(target=target,variant=variant,model_kind=model['kind'],
            total_cases=data['total'],resolved_cases=data['resolved'],resolved_coverage=data['coverage'],
            metrics=metrics(data['y'],probs),versus_raw=paired_score_intervals(data['y'],probs,raw_p,data['tasks']),
            versus_horizon_prior=paired_score_intervals(data['y'],probs,prior_p,data['tasks']),
            versus_task_horizon_prior=paired_score_intervals(data['y'],probs,metadata_p,data['tasks']),by_horizon={})
        for h in [50,100,200]:
            mask=data['horizons']==h
            if mask.any():result['by_horizon'][str(h)]=metrics(data['y'][mask],probs[mask])
        output['results'][model_id]=result
        if target=='utility' and model['kind'] not in ['constant','horizon_prior','task_horizon_prior']:
            m=result['metrics']
            print(json.dumps(dict(model=model_id,n=m['n'],auc=m['auc'],nll=m['log_loss'],brier=m['brier'],
                replan_recall=m['replan_recall'],continue_recall=m['continue_recall'],coverage07=m['coverage_07'])))
    target=ROOT/'analysis'/f'{args.name}_state2_evaluation.json'
    assert not target.exists(),'State2 report already exists; do not silently overwrite a prior assessment'
    target.write_text(json.dumps(output,indent=2))
    print(json.dumps(dict(saved=str(target),fitting=False)))


if __name__=='__main__':main()
