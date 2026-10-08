"""Strict split/row alignment for saved frozen-LAYA features."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['numbers3','binary_short','binary_compact']


def load_runs(names,states,variants=None):
    variants=VARIANTS if variants is None else variants
    expected={r['id'] for r in json.loads((ROOT/'fixtures/counterfactual_schedule.json').read_text()) if r['state'] in states}
    seen=set();rows=[];feature_rows=[];sources=[];fingerprints=set()
    for name in names:
        assert name.isascii() and name.replace('_','').isalnum()
        folder=ROOT/'raw'/name/'data'
        manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['frozen_parameters_unchanged'] and manifest['frozen_laya_before']==manifest['frozen_laya_after']
        fingerprints.add(manifest['frozen_laya_before'])
        assert json.loads((ROOT/'raw'/name/'supervisor.json').read_text())['status']=='completed'
        source=dict(run=name,scores_sha256=hashlib.sha256((folder/'scores.jsonl').read_bytes()).hexdigest(),
                    features_sha256=hashlib.sha256((folder/'features.npz').read_bytes()).hexdigest())
        sources.append(source)
        with np.load(folder/'features.npz',allow_pickle=False) as features:
            ids=features['row_ids'];matrix=features['features']
            for line in (folder/'scores.jsonl').read_text().splitlines():
                row=json.loads(line)
                assert row['state'] in states,'A forbidden split reached the fitter/evaluator'
                key=(row['id'],row['variant']);assert key not in seen
                assert row['id'] in expected and row['variant'] in variants
                assert str(ids[row['feature_row']])==row['id']+'|'+row['variant']
                assert hashlib.sha256(row['model_state'].encode()).hexdigest()==row['model_state_sha256']
                seen.add(key);rows.append(row);feature_rows.append(matrix[row['feature_row']].copy())
    assert seen=={(cid,v) for cid in expected for v in variants},'Incomplete or unexpected score coverage'
    assert len(fingerprints)==1
    return rows,np.asarray(feature_rows,dtype=float),dict(source_runs=sources,laya_fingerprint=next(iter(fingerprints)),
        states=sorted(states),unique_cases=len(expected),score_rows=len(rows))


def select(rows,features,variant,target,states):
    indices=[i for i,r in enumerate(rows) if r['variant']==variant and r['state'] in states and r['labels'][target] is not None]
    total=sum(r['variant']==variant and r['state'] in states for r in rows)
    selected=[rows[i] for i in indices]
    assert selected,'No resolved labels in this group'
    return dict(rows=selected,features=features[indices],
        y=np.array([r['labels'][target]=='replan' for r in selected],dtype=int),
        z=np.array([r['raw_logit_delta'] for r in selected]),
        horizons=np.array([r['horizon'] for r in selected]),tasks=np.array([r['task'] for r in selected]),
        total=total,resolved=len(selected),coverage=len(selected)/total)
