"""Read-only per-horizon and coverage diagnostics of already frozen state0/1 fits."""
from pathlib import Path
import hashlib,json
import numpy as np
from calibration_data import ROOT,load_runs,select
from calibration_math import metrics,predict


def main():
    folder=ROOT/'models/calibrators_v1';manifest=json.loads((folder/'manifest.json').read_text())
    models_path=folder/'models.json';assert hashlib.sha256(models_path.read_bytes()).hexdigest()==manifest['models_sha256']
    models=json.loads(models_path.read_text())
    names=[r['run'] for r in manifest['source']['source_runs']]
    rows,features,source=load_runs(names,{0,1},manifest['variants']);results={}
    for name,definition in models.items():
        target,variant,model=definition['target'],definition['variant'],definition['model'];entry={}
        for sid in [0,1]:
            data=select(rows,features,variant,target,{sid})
            probs=predict(model,data['z'],data['features'],data['horizons'],data['tasks'])
            by_h={};auc_num=0.;auc_pairs=0
            for h in [50,100,200]:
                mask=data['horizons']==h
                m=metrics(data['y'][mask],probs[mask]);by_h[str(h)]=m
                pairs=m['replan_labels']*m['continue_labels']
                if pairs:auc_num+=pairs*m['auc'];auc_pairs+=pairs
            indices=[i for i,r in enumerate(rows) if r['variant']==variant and r['state']==sid]
            all_rows=[rows[i] for i in indices]
            all_p=predict(model,np.array([r['raw_logit_delta'] for r in all_rows]),features[indices],
                          np.array([r['horizon'] for r in all_rows]),np.array([r['task'] for r in all_rows]))
            entry[str(sid)]=dict(resolved=data['resolved'],total=data['total'],label_coverage=data['coverage'],
                by_horizon=by_h,within_horizon_pair_weighted_auc=auc_num/auc_pairs if auc_pairs else None,
                total_resolved_cross_class_pairs_within_horizon=auc_pairs,
                all_cases=dict(n=len(all_p),replan_predictions=int(np.sum(all_p>=.5)),above07=int(np.sum(np.maximum(all_p,1-all_p)>=.7))),
                unresolved_cases=dict(n=len(all_p)-data['resolved'],accuracy_unknown=True))
        results[name]=entry
        if target=='utility' and model['kind'] in ['raw','frozen_feature_linear']:
            print(json.dumps(dict(model=name,calibration_within_horizon_auc=entry['1']['within_horizon_pair_weighted_auc'],
                                  calibration_all_cases=entry['1']['all_cases'])))
    result=dict(source=source,models_sha256=manifest['models_sha256'],fit_performed=False,results=results,
        interpretation='Developer diagnostics only. Within-H AUC removes cross-horizon rank information. Unresolved cases have no correctness label; their probability coverage is not accuracy.')
    (ROOT/'analysis/calibration_development_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')


if __name__=='__main__':main()
