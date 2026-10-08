"""Verify queue provenance and that commands beyond the decision window cannot leak in."""
from pathlib import Path
import copy,json,re
import numpy as np
from gate_variants import variant_input
from window_state import prepare_window_state,from_counterfactual
ROOT=Path(__file__).resolve().parents[1]
source=ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main'
records=[json.loads(l) for name in ['counterfactual_pilot_v1','counterfactual_train_v1']
         for l in (ROOT/'raw'/name/'data/records.jsonl').read_text(encoding='utf-8').splitlines()]
checks=[]
for compact in [False,True]:
    variant='binary_window3' if compact else 'binary_window'
    for r in records:
        text=from_counterfactual(r,source,compact=compact)
        assert variant_input(variant,text)[0]==text
    checks.append(dict(name=variant+' 189 training records: original active chunk SHA and old command samples',passed=True))
    r=next(r for r in records if r['horizon']==200)
    with np.load(source/r['case_name']/f'chunk_{r["active_call_id"]:03d}.npz') as data:queue=data['actions'][r['action_row']:].copy()
    expected=prepare_window_state(r['state_text'],queue,compact=compact)
    altered=queue.copy();altered[r['period']+1:]+=777.
    assert prepare_window_state(r['state_text'],altered,compact=compact)==expected
    altered[0,0]+=3.
    assert prepare_window_state(r['state_text'],altered,compact=compact)!=expected
    limited={k:r[k] for k in ['active_call_id','action_row','case_name','source_files','state_text']}
    assert from_counterfactual(limited,source,compact=compact)==expected
    checks.append(dict(name=variant+' excludes post-window commands and all branch outcomes; responds to current queued action',passed=True))
(ROOT/'checks/window_input.json').write_text(json.dumps(checks,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(dict(checks=len(checks),failures=0)))
