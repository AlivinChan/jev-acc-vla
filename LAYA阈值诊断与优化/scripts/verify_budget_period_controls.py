"""Reproduce the already-frozen period choice from developer evidence only."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    definition=read(ROOT/'protocol/budget_period_controls_v1/definition.json')
    for name,digest in definition['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    latency=read(ROOT/'analysis/serial_vla_latency_development.json')
    summary=read(ROOT/'analysis/single_skip_development_pooled.json')
    rows=[]
    for name in ['single_skip_v1','single_skip_confirmation_v1']:
        for line in (ROOT/'raw'/name/'data/episodes.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row['candidate_id']=='window3_skip1':rows.append(row)
    result={}
    for h in [50,100,200]:
        rr=[r for r in rows if r['horizon']==h]
        assert len(rr)==30 and {(r['task_id'],r['state_id']) for r in rr}=={(t,s) for t in range(10) for s in range(3)}
        unit=latency['by_horizon'][str(h)]['seconds_median']
        target=summary['groups'][f'window3_skip1_h{h}']['total_service_seconds']
        grid=[]
        for i in range(1,h):
            calls=sum((r['steps']-1)//i+1 for r in rr);service=calls*unit
            grid.append(dict(interval=i,estimated_calls=calls,estimated_service_seconds=service,
                absolute_target_error_seconds=abs(service-target)))
        assert grid==definition['integer_interval_estimates'][str(h)]
        best=min(grid,key=lambda r:(r['absolute_target_error_seconds'],r['interval']))
        assert dict(target_service_seconds=target,unit_vla_service_seconds=unit,**best)==definition['candidates'][str(h)]
        result[h]=best['interval']
    output=dict(checks='All 347 integer-period estimates and three predeclared tie-broken selections reproduced from 90 developer episode lengths',
        intervals=result,errors=0,definition_sha256=hashlib.sha256((ROOT/'protocol/budget_period_controls_v1/definition.json').read_bytes()).hexdigest())
    (ROOT/'checks/budget_period_definition.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(output))


if __name__=='__main__':main()
