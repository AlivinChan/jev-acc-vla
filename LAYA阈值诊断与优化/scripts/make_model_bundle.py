"""Freeze state1-selected readouts and train-only inference fixtures for deployment."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from calibration_data import ROOT,load_runs
from calibration_math import predict
from local_policy import allowed


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);args=p.parse_args()
    assert args.name.isascii() and args.name.replace('_','').isalnum();allowed()
    folder=ROOT/'models'/args.name;manifest=json.loads((folder/'manifest.json').read_text())
    raw=(folder/'models.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest['models_sha256']
    assert manifest['state2_loaded'] is False
    models=json.loads(raw);selected=manifest['proposed_deployment_models']
    if not selected:
        print(json.dumps(dict(promoted=0,reason='No predeclared candidate passed state1 eligibility')));return
    training_runs=[r['run'] for r in manifest['source']['source_runs']
        if json.loads((ROOT/'raw'/r['run']/'data/manifest.json').read_text())['states']==[0]]
    rows,features,provenance=load_runs(training_runs,{0},manifest['variants'])
    assert provenance['laya_fingerprint']==manifest['source']['laya_fingerprint']
    sources={}
    for name in ['counterfactual_pilot_v1','counterfactual_train_v1']:
        for line in (ROOT/'raw'/name/'data/records.jsonl').read_text().splitlines():
            r=json.loads(line);assert r['id'] not in sources;sources[r['id']]=r
    entries={name:models[name] for name in selected};fixtures=[]
    for name,entry in entries.items():
        for h in [50,100,200]:
            index=next(i for i,r in enumerate(rows) if r['variant']==entry['variant'] and r['horizon']==h)
            row=rows[index];prob=float(predict(entry['model'],np.array([row['raw_logit_delta']]),features[index:index+1],np.array([h]))[0])
            state=row['model_state'] if entry['variant'] in ['binary_window','binary_window3'] else sources[row['id']]['state_text']
            fixtures.append(dict(id=row['id'],state=state,variant=entry['variant'],
                readout_model_id=name,expected_raw_logit_delta=row['raw_logit_delta'],expected_probability_replan=prob,
                expected_choice='replan' if prob>=.5 else 'continue',source_feature_row=row['feature_row'],
                source_runs=training_runs,split='state0 training; qualification only'))
    result=dict(entries=entries,qualification_fixtures=fixtures,laya_fingerprint=provenance['laya_fingerprint'],
        models_source_sha256=manifest['models_sha256'],fit_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
        state2_used_for_fitting=False,selection_source='predeclared state1 eligibility; no state2 selection')
    with (folder/'deployment_bundle.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(promoted=len(entries),fixtures=len(fixtures),bundle=str(folder/'deployment_bundle.json'))))


if __name__=='__main__':main()
