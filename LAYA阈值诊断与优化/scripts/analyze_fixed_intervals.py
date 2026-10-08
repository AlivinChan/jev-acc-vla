"""Assemble unique complete horizon/interval cells with explicit reuse provenance."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
from analyze_closed_loop import summary,paired
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main'
CELLS={(50,30),(100,30),(200,30),(50,45),(100,45),(200,45),(100,60),(200,60)}


def main():
    groups=defaultdict(list);sources={};seen=set()
    def add(row,path,interval):
        h=row['horizon'];key=(h,interval,row['task_id'],row['state_id'])
        if (h,interval) not in CELLS:return
        assert key not in seen,'Duplicate episode in a fixed-interval cell'
        calls=json.loads((path/'calls.json').read_text(encoding='utf-8'))
        assert [c['request_tick'] for c in calls]==list(range(0,row['steps'],interval))
        assert row['gate_calls']==0 and row['gate_seconds']==0
        seen.add(key);groups[(h,interval)].append(row)
        sources[':'.join(map(str,key))]=dict(result_path=str(path/'result.json'),
            result_sha256=hashlib.sha256((path/'result.json').read_bytes()).hexdigest(),
            calls_sha256=hashlib.sha256((path/'calls.json').read_bytes()).hexdigest())
    for h,interval in [(100,30),(200,60)]:
        for tid in range(10):
            for sid in range(3):
                path=OLD/f't{tid:02d}_s{sid:02d}_h{h}_smol70';add(json.loads((path/'result.json').read_text()),path,interval)
    for run in ['single_skip_v1','fixed_interval_v1']:
        base=ROOT/'raw'/run;assert json.loads((base/'supervisor.json').read_text())['status']=='completed'
        for line in (base/'data/episodes.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['method'] not in ['fixed_r60','fixed_interval']:continue
            interval=2*round(.3*row['horizon']) if row['method']=='fixed_r60' else row['gate_interval']
            path=base/'data'/row['candidate_id']/f"t{row['task_id']:02d}_s{row['state_id']:02d}_h{row['horizon']}_{row['method']}"
            add(row,path,interval)
    assert set(groups)==CELLS
    expected={(tid,sid) for tid in range(10) for sid in range(3)}
    for cell,rows in groups.items():assert len(rows)==30 and {(r['task_id'],r['state_id']) for r in rows}==expected
    comparisons={}
    references={};reference_sources={};within_horizon={}
    for h in [50,100,200]:
        references[h]=[]
        for tid,sid in sorted(expected):
            path=OLD/f't{tid:02d}_s{sid:02d}_h{h}_smol70/result.json'
            references[h].append(json.loads(path.read_text(encoding='utf-8')))
            reference_sources[f'H{h}_t{tid}_s{sid}']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for (h,interval),rows in sorted(groups.items()):
        within_horizon[f'H{h}_I{interval}']=paired(rows,references[h])
        assert within_horizon[f'H{h}_I{interval}']['n']==30
    for interval,reference_h in [(30,50),(45,50),(60,100)]:
        for h in [50,100,200]:
            if (h,interval) not in groups or h==reference_h:continue
            # Match task/state while deliberately varying H; no source row is modified.
            a=[dict(r,horizon=0) for r in groups[(h,interval)]]
            b=[dict(r,horizon=0) for r in groups[(reference_h,interval)]]
            comparisons[f'I{interval}_H{h}_vs_H{reference_h}']=paired(a,b)
    result=dict(groups={f'H{h}_I{i}':summary(rows) for (h,i),rows in sorted(groups.items())},
        same_interval_horizon_comparisons=comparisons,source_cells=sources,
        same_horizon_smol70_comparisons=within_horizon,
        smol70_references={f'H{h}':summary(rows) for h,rows in references.items()},
        reference_source_sha256=reference_sources,
        unique_horizon_interval_initial_cells=len(seen),
        interpretation='Developer attribution across 30 identical task/state starts per cell; reuse is explicit, historical timings are not contemporaneous. No gate cost or independent confirmation is implied.')
    (ROOT/'analysis/fixed_interval_matrix.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    for name,row in result['groups'].items():print(json.dumps(dict(cell=name,n=row['n'],successes=row['successes'],calls=row['vla_calls'],service=row['total_service_seconds'])))


if __name__=='__main__':main()
