"""Paired development comparisons, with success and full measured service costs separate."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'长动作块补充实验'

def rows_from(path):return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
def cost(r):return r['prediction_seconds']+r['gate_seconds']
def key(r):return (r['task_id'],r['state_id'],r['horizon'])
def summary(rows):
    return dict(n=len(rows),successes=sum(r['success'] for r in rows),success_rate=np.mean([r['success'] for r in rows]).item(),
        vla_calls=sum(r['vla_calls'] for r in rows),gate_calls=sum(r['gate_calls'] for r in rows),
        vla_seconds=sum(r['prediction_seconds'] for r in rows),gate_seconds=sum(r['gate_seconds'] for r in rows),
        total_service_seconds=sum(cost(r) for r in rows),
        steps=sum(r['steps'] for r in rows),generated_rows=sum(r['native_generated_rows'] for r in rows),
        used_rows=sum(r['steps'] for r in rows),gate_choices=dict(sum((Counter(r['gate_choices']) for r in rows),Counter())))
def paired(a,b):
    amap={key(r):r for r in a};bmap={key(r):r for r in b};keys=sorted(amap.keys()&bmap.keys())
    assert keys
    for k in keys:assert amap[k]['initial_state_sha256']==bmap[k]['initial_state_sha256']
    wins=sum(amap[k]['success'] and not bmap[k]['success'] for k in keys)
    losses=sum(bmap[k]['success'] and not amap[k]['success'] for k in keys)
    discord=wins+losses
    p_exact=min(1.0,2*sum(math.comb(discord,i) for i in range(min(wins,losses)+1))/(2**discord)) if discord else 1.
    tasks=sorted({k[0] for k in keys})
    by_task=np.array([[sum(cost(amap[k]) for k in keys if k[0]==t),sum(cost(bmap[k]) for k in keys if k[0]==t),
                      sum(int(amap[k]['success'])-int(bmap[k]['success']) for k in keys if k[0]==t),
                      sum(k[0]==t for k in keys)] for t in tasks])
    samples=by_task[np.random.default_rng(20261008).integers(0,len(tasks),(10000,len(tasks)))].sum(1)
    intervals=dict(cost_ratio_95_task_bootstrap=np.quantile(samples[:,0]/samples[:,1],[.025,.975]).tolist(),
                   success_delta_95_task_bootstrap=np.quantile(samples[:,2]/samples[:,3],[.025,.975]).tolist())
    return dict(n=len(keys),wins=wins,losses=losses,success_delta=(wins-losses)/len(keys),
        descriptive_mcnemar_exact_p=p_exact,multiple_search_comparisons_uncorrected=True,
        vla_call_ratio=sum(amap[k]['vla_calls'] for k in keys)/sum(bmap[k]['vla_calls'] for k in keys),
        total_service_cost_ratio=sum(cost(amap[k]) for k in keys)/sum(cost(bmap[k]) for k in keys),**intervals)

def main():
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
    folder=ROOT/'raw'/args.run/'data';rows=rows_from(folder/'episodes.jsonl')
    groups=defaultdict(list)
    for r in rows:groups[(r['candidate_id'],r['horizon'])].append(r)
    refmethods=['smol70','naive_k5','vlash_style_k5','fixed_r60'];ref={}
    for method in refmethods:
        ref[method]=[r for r in rows if r['method']==method]
        if ref[method]:continue
        for tid,sid,h in sorted({key(r) for r in rows}):
            path=OLD/f'raw/attempt_v1/smol/main/t{tid:02d}_s{sid:02d}_h{h}_{method}/result.json'
            if path.exists():ref[method].append(json.loads(path.read_text()))
    entries={}
    for (candidate,h),data in sorted(groups.items()):
        entry=summary(data);entry['comparisons']={}
        for method,refs in ref.items():
            if any(key(r) in {key(x) for x in refs} for r in data):entry['comparisons'][method]=paired(data,refs)
        if candidate!='queue_only' and ('queue_only',h) in groups:
            entry['comparisons']['queue_only']=paired(data,groups[('queue_only',h)])
        if candidate!='compact_skip1' and ('compact_skip1',h) in groups:
            entry['comparisons']['compact_skip1']=paired(data,groups[('compact_skip1',h)])
        paths=[folder/candidate/f"t{r['task_id']:02d}_s{r['state_id']:02d}_h{h}_{r['method']}"/'gate.json' for r in data]
        if all(path.exists() for path in paths):
            gate=[g for path in paths for g in json.loads(path.read_text())]
            if gate:
                entry['gate_tokens_mean']=float(np.mean([g['token_audit']['tokens'] for g in gate]))
                entry['gate_max_probability_range']=[min(max(g['probabilities'].values()) for g in gate),max(max(g['probabilities'].values()) for g in gate)]
                entry['gate_above_07']=sum(max(g['probabilities'].values())>=.7 for g in gate)
        entries[f'{candidate}_h{h}']=entry
    result=dict(run=args.run,completed=len(rows),status=json.loads((ROOT/'raw'/args.run/'supervisor.json').read_text())['status'],
        groups=entries,references={f'{m}_h{h}':summary([r for r in rr if r['horizon']==h]) for m,rr in ref.items()
            for h in sorted({r['horizon'] for r in rr})},
        cohort_role=json.loads((folder/'manifest.json').read_text())['config'].get('cohort_role','Developer selection'),
        interpretation='CIs descriptive, task-cluster resampling; measured serial service wall time, not physical robot speedup')
    (ROOT/'analysis'/f'{args.run}_summary.json').write_text(json.dumps(result,indent=2))
    for name,r in entries.items():
        smol=r['comparisons'].get('smol70',{})
        print(json.dumps(dict(group=name,n=r['n'],success=r['successes'],calls=r['vla_calls'],gates=r['gate_calls'],
            cost=round(r['total_service_seconds'],3),choices=r['gate_choices'],
            smol70_cost_ratio=smol.get('total_service_cost_ratio'),smol70_wins=smol.get('wins'),smol70_losses=smol.get('losses'))))


if __name__=='__main__':main()
