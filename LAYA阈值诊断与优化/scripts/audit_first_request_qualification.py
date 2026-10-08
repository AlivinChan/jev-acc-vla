"""Require live first-only gating to reproduce its precomputed causal branch exactly."""
from pathlib import Path
import argparse,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main'


def lines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
    folder=ROOT/'raw'/args.run/'data';rows=lines(folder/'episodes.jsonl');records={};scores={};checks=0
    assert json.loads((ROOT/'raw'/args.run/'supervisor.json').read_text())['status']=='completed'
    for name in ['counterfactual_pilot_v1','counterfactual_train_v1']:
        for r in lines(ROOT/'raw'/name/'data/records.jsonl'):
            if r['source_call_id']==1:records[(r['task'],r['state'],r['horizon'])]=(name,r)
    for name in ['scores_train_v1','scores_window_train_v1']:
        for r in lines(ROOT/'raw'/name/'data/scores.jsonl'):scores[(r['id'],r['variant'])]=r
    for row in rows:
        assert row['method']=='laya_first';tid,sid,h=row['task_id'],row['state_id'],row['horizon']
        name,record=records[(tid,sid,h)];source=OLD/record['case_name'];t=record['tick']
        case=folder/row['candidate_id']/f't{tid:02d}_s{sid:02d}_h{h}_laya_first'
        gates=json.loads((case/'gate.json').read_text());assert len(gates)==1 and gates[0]['tick']==t;checks+=1
        gate=gates[0];score=scores[(record['id'],row['candidate']['variant'])]
        assert gate['model_state']==score['model_state'];checks+=1
        assert gate['raw']['answers']['replanning']==score['raw']['answers']['replanning'];checks+=1
        choice=gate['choice'];branch=ROOT/'raw'/name/'data/cases'/record['id']
        with np.load(source/'trace.npz') as initial,np.load(branch/f'{choice}_trace.npz') as suffix,np.load(case/'trace.npz') as actual:
            for key in ['actions','physics']:
                assert np.array_equal(np.concatenate([initial[key][:t],suffix[key]]),actual[key]);checks+=1
        outcome=record['branches'][choice]
        assert row['success']==outcome['success'] and row['steps']==outcome['total_steps'];checks+=1
        calls=json.loads((case/'calls.json').read_text());old_calls=json.loads((branch/f'{choice}_calls.json').read_text())
        assert row['vla_calls']==len(calls)==1+len(old_calls);checks+=1
        for index,call in enumerate(calls[1:]):
            assert call['request_tick']==old_calls[index]['tick'] and call['noise_seed']==old_calls[index]['seed'];checks+=1
            with np.load(case/f'chunk_{index+1:03d}.npz') as actual,np.load(branch/f'{choice}_chunk_{index:03d}.npz') as old:
                for key in ['actions','normalized']:assert np.array_equal(actual[key],old[key]);checks+=1
    result=dict(run=args.run,episodes=len(rows),checks=checks,errors=0,
                scope='Live LAYA input/answer, exact source-prefix plus selected counterfactual branch, and all VLA chunks; qualification, not new efficacy data')
    (ROOT/'checks'/f'{args.run}_branch_qualification.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(result))


if __name__=='__main__':main()
