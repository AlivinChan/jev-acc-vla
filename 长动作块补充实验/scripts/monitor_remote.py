"""Read-only compact remote progress snapshot; no model imports."""
from pathlib import Path
import json, subprocess

code = r'''
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json
r=Path('/mnt/4t/jev_vla_libero/experiments/long_chunk_laya')
result={'observed_utc':datetime.now(timezone.utc).isoformat()}
for attempt in ['attempt_v1','argmax_attempt_v1']:
 p=r/attempt
 if not p.exists():continue
 value={}
 for name in ['supervisor.json','smol/status.json']:
  if (p/name).exists():
   d=json.loads((p/name).read_text())
   value[name]={k:d[k] for k in ['status','phase','completed','planned','elapsed_seconds','error','lease_valid_until'] if k in d}
 rows=[]
 if (p/'smol/episodes.jsonl').exists():
  for line in (p/'smol/episodes.jsonl').read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
 value['episodes_by_phase']=dict(Counter(x['phase'] for x in rows))
 value['tasks_seen']=sorted(set(x['task_id'] for x in rows))
 value['vla_calls']=sum(x['vla_calls'] for x in rows)
 value['gate_choices']=dict(sum((Counter(x['gate_choices']) for x in rows),Counter()))
 lookup={(x['task_id'],x['state_id'],x['horizon'],x['method']):x for x in rows if x['phase']=='main'}
 checked=0; mismatches=[]
 for key,left in lookup.items():
  if key[3]!='laya' or set(left['gate_choices'])!={'uncertain'}:continue
  right=lookup.get((*key[:3],'smol70'))
  if right is None:continue
  checked+=1
  if left['executed_action_sha256']!=right['executed_action_sha256'] or left['success']!=right['success']:
   mismatches.append(list(key[:3]))
 value['completed_all_uncertain_pairs']=checked
 value['all_uncertain_trajectory_mismatches']=mismatches
 if (p/'official_reference/summary.json').exists():
  d=json.loads((p/'official_reference/summary.json').read_text())
  value['official_reference']={k:d[k] for k in ['episodes','successes','pc_success','freeze_verified_unchanged'] if k in d}
 elif (p/'official_reference/episodes.json').exists():
  value['official_episodes_so_far']=len(json.loads((p/'official_reference/episodes.json').read_text()))
 result[attempt]=value
print(json.dumps(result))
'''
run = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
                      'gpu4090-frp','python3 -'],input=code,text=True,encoding='utf-8',
                     capture_output=True,timeout=25)
if run.returncode:
    raise RuntimeError(run.stderr)
data=json.loads(run.stdout)
out=Path(__file__).resolve().parents[1]/'results'
out.mkdir(exist_ok=True)
(out/'current_remote.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
with (out/'remote_progress.jsonl').open('a',encoding='utf-8') as stream:
    stream.write(json.dumps(data)+'\n')
print(json.dumps(data))
