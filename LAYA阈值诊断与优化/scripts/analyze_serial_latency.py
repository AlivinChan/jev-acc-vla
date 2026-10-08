"""Describe already-measured VLA service receipts; no new timing benchmark."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=[];sources={}
    for name in ['representations_v1','cadence_v1','confirmation_v1','single_skip_v1']:
        p=ROOT/'raw'/name/'data/episodes.jsonl'
        assert json.loads((p.parents[1]/'supervisor.json').read_text(encoding='utf-8'))['status']=='completed'
        rr=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
        sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for r in rr:
            folder=p.parent/r['candidate_id']/f"t{r['task_id']:02d}_s{r['state_id']:02d}_h{r['horizon']}_{r['method']}"
            calls=json.loads((folder/'calls.json').read_text(encoding='utf-8'))
            assert len(calls)==r['vla_calls']
            rows.extend(dict(horizon=r['horizon'],seconds=c['predict_seconds'],run=name) for c in calls)
    result=dict(scope='Observed serial end-to-end VLA service receipt distribution; not an asynchronous latency measurement or independent timing samples',
                source_sha256=sources,by_horizon={})
    for h in [50,100,200]:
        a=np.array([r['seconds'] for r in rows if r['horizon']==h])
        assert len(a)>0 and np.isfinite(a).all() and (a>0).all()
        result['by_horizon'][str(h)]=dict(calls=len(a),seconds_median=float(np.median(a)),
            seconds_p95=float(np.quantile(a,.95)),equivalent_20Hz_steps_median=float(np.median(a)/.05),
            fraction_over_0_2_seconds=float(np.mean(a>.2)))
    (ROOT/'analysis/serial_vla_latency_development.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(result['by_horizon']))


if __name__=='__main__':main()
