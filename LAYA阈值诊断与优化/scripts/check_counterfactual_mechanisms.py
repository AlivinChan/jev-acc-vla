"""Exercise actual branch code against a known baseline suffix without model weights."""
from pathlib import Path
from types import SimpleNamespace
import ast,collections,copy,hashlib,json,tempfile,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def definitions(path,names):
    return ast.Module(body=[n for n in ast.parse(path.read_text(encoding='utf-8')).body
        if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
OBS={'robot_state':{'eef':{'pos':np.zeros(3)},'gripper':{'qpos':np.zeros(2)}}}
class Env:
    def __init__(self):
        self.tick=0;self.position=0.
        self._env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(get_state=self.state)))
    def state(self):return np.array([self.tick,self.position],dtype=float)
    def step(self,action):
        self.tick+=1;self.position+=float(action[0])
        return OBS,0,self.tick==60,False,{'is_success':False}
    def close(self):pass
class Engine:
    suite=SimpleNamespace(get_task=lambda tid:SimpleNamespace(language='move block'))
    def restore(self,tid,sid,actions):
        env=Env()
        for a in actions:env.step(a)
        return env,OBS
class Predictor:
    def __init__(self):self.engine=Engine()
    def predict(self,obs,instruction,horizon,seed):
        a=np.repeat(np.arange(horizon,dtype=np.float32)[:,None],7,axis=1)
        return a,a[None],.01
scope=dict(Path=Path,np=np,collections=collections,copy=copy,hashlib=hashlib,json=json,time=time,
           object_positions=lambda env:{'block':np.zeros(3)})
exec(compile(definitions(ROOT/'scripts/controller_runtime.py',{'run_episode','state_text','digest_bytes','atomic'}),'actual_controller','exec'),scope)
runtime=SimpleNamespace(**{k:scope[k] for k in ['state_text','object_positions']})
def snap(env):return dict(physics=env.state(),controller={},objects={},rng_sha256='fixed')
branch_scope=dict(np=np,json=json,runtime=runtime,snapshot=snap,allowed=lambda:None,atomic=scope['atomic'])
exec(compile(definitions(ROOT/'scripts/collect_gate_counterfactuals.py',{'branch','labels'}),'actual_counterfactual','exec'),branch_scope)
checks=[]
temporary_root=(ROOT/'checks/temporary_fixtures').resolve();temporary_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='cf_',dir=temporary_root) as tmp:
    tmp=Path(tmp)
    assert tmp.resolve().is_relative_to(temporary_root)
    result=scope['run_episode'](Predictor(),None,tmp,'reference',0,0,50,'smol70')
    source=Path(result['case_path']);calls=json.loads((source/'calls.json').read_text())
    timeline=json.loads((source/'timeline.json').read_text())
    for call in calls[1:]:
        t=call['request_tick'];active=timeline[t]['active_call']
        item=dict(task=0,state=0,horizon=50,tick=t,period=15,active_origin=calls[active]['request_tick'],
                  active_call_id=active,source_call_id=call['call_id'],action_row=timeline[t]['action_row'])
        folder=tmp/f'branch{t}';folder.mkdir()
        outputs={}
        for choice in ['continue','replan']:
            outputs[choice]=branch_scope['branch'](Predictor(),item,source,folder,choice)
        r=np.load(folder/'replan_trace.npz');c=np.load(folder/'continue_trace.npz')
        assert np.array_equal(r['actions'][0],c['actions'][0]) and np.array_equal(r['physics'][0],c['physics'][0])
        assert r['actions'][1,0]==1 and c['actions'][1,0]==16
        rc=json.loads((folder/'replan_calls.json').read_text());cc=json.loads((folder/'continue_calls.json').read_text())
        assert [x['tick'] for x in rc]==list(range(t,60,15))
        assert [x['tick'] for x in cc]==list(range(t+15,60,15))
        assert outputs['replan'][0]['baseline_suffix_exact']
        r.close();c.close()
        checks.append(dict(name=f'exact R suffix and one skipped call at t={t}',passed=True))
label=branch_scope['labels']
def record(success,calls):return dict(success=success,vla_calls=calls)
assert label(record(False,2),record(True,3))['utility']=='replan'
assert label(record(True,4),record(False,3))['utility']=='continue'
assert label(record(True,4),record(True,3))==dict(necessity='continue',utility='replan',ambiguity='both_success')
assert label(record(True,3),record(True,3))['utility']=='continue'
assert label(record(False,1),record(False,9))['utility'] is None
checks.append(dict(name='success priority, call-count ties and unresolved failures',passed=True))
(ROOT/'checks/counterfactual_mechanisms.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(dict(checks=len(checks),failures=0)))
