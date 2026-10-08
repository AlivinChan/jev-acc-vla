"""Reject a delay refactor unless d=1 reproduces every saved action and model input."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
    base=ROOT/'raw'/args.run;assert read(base/'supervisor.json')['status']=='completed'
    assert read(base/'data/manifest.json')['logical_delay_steps']==1
    rows=[json.loads(line) for line in (base/'data/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==9
    checks=0;sources={}
    for row in rows:
        h=row['horizon'];method=row['method'];cid=row['candidate_id']
        assert row['task_id']==0 and row['state_id']==0
        case_name=f't00_s00_h{h}_{method}'
        actual=base/'data'/cid/case_name
        source=ROOT/'raw/single_skip_v1/data/window3_skip1'/case_name if cid=='window3_skip1' else OLD/case_name
        sources[str(source/'result.json')]=hashlib.sha256((source/'result.json').read_bytes()).hexdigest()
        reference=read(source/'result.json')
        for field in ['success','steps','vla_calls','gate_calls','initial_state_sha256','executed_action_sha256','gate_choices']:
            assert row[field]==reference[field],(case_name,field);checks+=1
        with np.load(actual/'trace.npz') as a,np.load(source/'trace.npz') as b:
            for key in ['actions','physics']:assert np.array_equal(a[key],b[key]),(case_name,key);checks+=1
        calls=read(actual/'calls.json');old_calls=read(source/'calls.json')
        assert len(calls)==len(old_calls);checks+=1
        for index,(call,old_call) in enumerate(zip(calls,old_calls)):
            for key in ['request_tick','intended_delivery_tick','actual_delivery_tick','expired_prefix_rows','image_tick','state_tick','noise_seed','executed_rows','trigger']:
                assert call.get(key)==old_call.get(key),(case_name,index,key);checks+=1
            with np.load(actual/f'chunk_{index:03d}.npz') as a,np.load(source/f'chunk_{index:03d}.npz') as b:
                for key in ['actions','normalized']:assert np.array_equal(a[key],b[key]),(case_name,index,key);checks+=1
        gates=read(actual/'gate.json');old_gates=read(source/'gate.json')
        assert len(gates)==len(old_gates);checks+=1
        for gate,old_gate in zip(gates,old_gates):
            for key in ['tick','choice','model_state','probabilities','plan_age','remaining']:
                assert gate[key]==old_gate[key],(case_name,key);checks+=1
            assert gate['raw']['answers']['replanning']==old_gate['raw']['answers']['replanning'];checks+=1
    result=dict(run=args.run,episodes=len(rows),checks=checks,errors=0,source_result_sha256=sources,
        scope='d=1 refactor qualification: exact complete physics/actions, VLA outputs, request inputs and gate answers; timing excluded; not new efficacy samples')
    (ROOT/'checks'/f'{args.run}_exact_reproduction.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_result_sha256'}))


if __name__=='__main__':main()
