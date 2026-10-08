"""Enumerate every Smol70 call opportunity without outcome-based sample selection."""
from pathlib import Path
from collections import Counter
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main'
schedule=[]
for case in sorted(OLD.glob('t*_s*_h*_smol70')):
    result=json.loads((case/'result.json').read_text());calls=json.loads((case/'calls.json').read_text())
    timeline=json.loads((case/'timeline.json').read_text())
    tid,sid,h=result['task_id'],result['state_id'],result['horizon'];period=round(.3*h)
    for call in calls[1:]:
        tick=call['request_tick'];active_id=timeline[tick]['active_call'];active=calls[active_id]
        assert tick%period==0 and call['trigger']=='remaining_ratio_le_0.7'
        assert timeline[tick]['action_row']==tick-active['request_tick']==period
        schedule.append(dict(id=case.name+f'_t{tick:03d}',case_name=case.name,task=tid,state=sid,horizon=h,
            tick=tick,period=period,source_call_id=call['call_id'],active_call_id=active_id,
            active_origin=active['request_tick'],action_row=timeline[tick]['action_row'],
            split={0:'train',1:'calibration',2:'diagnostic_evaluation'}[sid],
            source_files={name:hashlib.sha256((case/name).read_bytes()).hexdigest()
                for name in ['result.json','calls.json','timeline.json','trace.npz','initial_state.npy',
                             f'chunk_{active_id:03d}.npz',f'chunk_{call["call_id"]:03d}.npz']}))
out=ROOT/'fixtures/counterfactual_schedule.json'
out.write_text(json.dumps(schedule,indent=2),encoding='utf-8',newline='\n')
meta=dict(cases=len(schedule),by_horizon=dict(Counter(r['horizon'] for r in schedule)),
          by_split=dict(Counter(r['split'] for r in schedule)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
          selection='all noninitial Smol70 requests, no outcome or LAYA score filtering')
(ROOT/'fixtures/counterfactual_manifest.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta))
