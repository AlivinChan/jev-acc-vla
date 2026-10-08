from pathlib import Path
from collections import Counter
import ast,hashlib,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main'
p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
folder=ROOT/'raw'/args.run/'data'
records=[json.loads(l) for l in (folder/'records.jsonl').read_text().splitlines()]
manifest=json.loads((folder/'manifest.json').read_text());checks=0;counts=Counter();failures=[]
def check(value,message):
    global checks
    checks+=1
    if not value:failures.append(message)
tree=ast.parse((ROOT/'scripts/collect_gate_counterfactuals.py').read_text(encoding='utf-8'))
scope={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='labels'],type_ignores=[]),'actual_labels','exec'),scope)
total_calls=0
for item in records:
    case=folder/'cases'/item['id'];source=OLD/item['case_name'];t=item['tick'];h=item['horizon'];period=item['period']
    with np.load(case/'continue_trace.npz') as c,np.load(case/'replan_trace.npz') as r,np.load(source/'trace.npz') as original:
        check(np.array_equal(c['start_physics'],r['start_physics']),item['id']+' start match')
        check(np.array_equal(c['start_physics'],original['physics'][t-1]),item['id']+' source prefix match')
        check(np.array_equal(c['actions'][0],r['actions'][0]) and np.array_equal(c['physics'][0],r['physics'][0]),item['id']+' d1 shared prefix')
        check(np.array_equal(r['actions'],original['actions'][t:]) and np.array_equal(r['physics'],original['physics'][t:]),item['id']+' exact R suffix')
        for choice,trace in [('continue',c),('replan',r)]:
            result=item['branches'][choice];calls=json.loads((case/f'{choice}_calls.json').read_text())
            check(result['post_steps']==len(trace['actions'])==len(trace['physics']),item['id']+' step accounting')
            check(result['vla_calls']==len(calls),item['id']+' call accounting')
            total_calls+=len(calls)
            for index,call in enumerate(calls):
                check(call['tick']>=t and call['delivery']==call['tick']+1 and call['nfe']==10,item['id']+' logical delivery and NFE')
                check((call['tick']-t)%period==0 and (choice=='replan' or call['tick']>t),item['id']+' intervention/common policy')
                check(call['seed']==20261007+item['task']*100000+item['state']*1000+call['tick'],item['id']+' paired fixed noise')
                with np.load(case/f'{choice}_chunk_{index:03d}.npz') as chunk:
                    check(chunk['actions'].shape==(h,7) and np.isfinite(chunk['actions']).all(),item['id']+' native output')
    check(hashlib.sha256(item['state_text'].encode()).hexdigest()==item['state_sha256'],item['id']+' input digest')
    check(item['labels']==scope['labels'](item['branches']['continue'],item['branches']['replan']),item['id']+' labels')
    counts[item['labels']['utility'] or 'both_failure']+=1
check(manifest['freeze_before']==manifest['freeze_after'],'VLA frozen')
check(manifest['freeze_before']['state_dict_sha256']=='b296dfca9e977fbe06d0f4a7971dbcfa368954e168c32f9c61295107f00eba82','same VLA')
check(total_calls==manifest['total_vla_calls'],'complete VLA accounting')
check(len(records)==manifest['planned'],'complete schedule')
result=dict(run=args.run,cases=len(records),checks=checks,errors=len(failures),failures=failures,
            utility_labels=dict(counts),total_vla_calls=total_calls,
            interpretation='Exactly paired intervention labels under the stated common recovery policy; not labels for arbitrary gating frequencies')
(ROOT/'checks'/f'{args.run}_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
assert not failures
