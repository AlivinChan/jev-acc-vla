"""Exercise the actual rollout function with a deterministic, weight-free environment."""
from pathlib import Path
import ast,collections,copy,hashlib,json,tempfile,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'scripts/controller_runtime.py').read_text(encoding='utf-8')
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in
       {'run_episode','state_text','digest_bytes','atomic'}]
class Sim:
    def __init__(self):self.tick=0
    def get_state(self):return np.array([self.tick],dtype=float)
class Env:
    def __init__(self):
        from types import SimpleNamespace
        self.sim=Sim();self._env=SimpleNamespace(env=SimpleNamespace(sim=self.sim))
    def step(self,action):
        self.sim.tick+=1
        return observation(self.sim.tick),0,self.sim.tick==60,False,{'is_success':False}
    def close(self):pass
OBS={'robot_state':{'eef':{'pos':np.zeros(3)},'gripper':{'qpos':np.zeros(2)}}}
def observation(tick):
    return {'frame_tick':tick,'robot_state':{'eef':{'pos':np.array([tick,0.,0.])},'gripper':{'qpos':np.zeros(2)}}}
class Engine:
    def __init__(self):
        from types import SimpleNamespace
        self.suite=SimpleNamespace(get_task=lambda tid:SimpleNamespace(language='move block'))
    def restore(self,*args):return Env(),observation(0)
class Predictor:
    def __init__(self):self.engine=Engine();self.inputs=[]
    def predict(self,obs,instruction,horizon,seed):
        self.inputs.append(copy.deepcopy(obs))
        actions=np.repeat(np.arange(horizon,dtype=np.float32)[:,None],7,axis=1)
        return actions,actions[None],.01
class Gate:
    def __init__(self,choice):self.choice=choice
    def predict(self,state):return {'choice':self.choice}
scope=dict(Path=Path,np=np,collections=collections,copy=copy,hashlib=hashlib,json=json,time=time,
           object_positions=lambda env:{'block':np.zeros(3)})
exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_rollout_functions','exec'),scope)
checks=[]
with tempfile.TemporaryDirectory(prefix='laya_controller_') as temporary:
    root=Path(temporary)
    for interval in [5,10,20]:
        result=scope['run_episode'](Predictor(),Gate('replan'),root,f'every{interval}',0,0,200,'laya',interval)
        case=Path(result['case_path']);calls=json.loads((case/'calls.json').read_text())
        assert [c['request_tick'] for c in calls]==[0,*range(interval,60,interval)]
        assert all(c.get('expired_prefix_rows')==1 for c in calls[1:])
        assert all(c['actual_delivery_tick']==c['request_tick']+1 for c in calls[1:])
        assert result['steps']==60 and result['gate_calls']==len(calls)-1
        checks.append(dict(name=f'cadence {interval}; d=1 and expired prefix',passed=True))
    result=scope['run_episode'](Predictor(),Gate('continue'),root,'finite',0,0,50,'laya',20)
    calls=json.loads((Path(result['case_path'])/'calls.json').read_text())
    assert [c['request_tick'] for c in calls]==[0,49]
    assert calls[1]['trigger']=='mandatory_finite_queue' and calls[1]['executed_rows'][0]==1
    checks.append(dict(name='finite H50 queue guard despite continue',passed=True))
    result=scope['run_episode'](Predictor(),Gate('replan'),root,'minimum_age',0,0,200,'laya',5,25)
    calls=json.loads((Path(result['case_path'])/'calls.json').read_text())
    assert [c['request_tick'] for c in calls]==[0,25,50]
    checks.append(dict(name='minimum execution window',passed=True))
    for choice,method in [('replan','smol70'),('continue','fixed_r60')]:
        gated=scope['run_episode'](Predictor(),Gate(choice),root,f'skip_{choice}',0,0,50,'laya_skip1',15)
        control=scope['run_episode'](Predictor(),None,root,f'control_{choice}',0,0,50,method,15)
        gc=Path(gated['case_path']);cc=Path(control['case_path'])
        with np.load(gc/'trace.npz') as a,np.load(cc/'trace.npz') as b:
            assert np.array_equal(a['actions'],b['actions']) and np.array_equal(a['physics'],b['physics'])
        gates=json.loads((gc/'gate.json').read_text());calls=json.loads((gc/'calls.json').read_text())
        assert all(g['plan_age']==15 for g in gates)
        expected=[0,15,30,45] if choice=='replan' else [0,30]
        assert [c['request_tick'] for c in calls]==expected
        checks.append(dict(name=f'single-skip {choice} matches {method} exactly; age15 only',passed=True))
    for choice in ['replan','continue']:
        result=scope['run_episode'](Predictor(),Gate(choice),root,f'first_{choice}',0,0,50,'laya_first',15)
        case=Path(result['case_path']);calls=json.loads((case/'calls.json').read_text());gates=json.loads((case/'gate.json').read_text())
        assert len(gates)==1 and gates[0]['tick']==15
        assert [c['request_tick'] for c in calls]==([0,15,30,45] if choice=='replan' else [0,30,45])
        if choice=='replan':
            reference=root/'control_replan'/'t00_s00_h50_smol70'/'trace.npz'
            with np.load(case/'trace.npz') as a,np.load(reference) as b:
                assert np.array_equal(a['actions'],b['actions']) and np.array_equal(a['physics'],b['physics'])
        checks.append(dict(name=f'first-only gate {choice}: exactly one decision, then Smol70 recovery',passed=True))
    result=scope['run_episode'](Predictor(),None,root,'fixed30',0,0,50,'fixed_interval',30)
    with np.load(Path(result['case_path'])/'trace.npz') as a,np.load(root/'control_continue'/'t00_s00_h50_fixed_r60'/'trace.npz') as b:
        assert np.array_equal(a['actions'],b['actions']) and np.array_equal(a['physics'],b['physics'])
    checks.append(dict(name='fixed30 equals H50 fixed_r60; reference reuse is semantically exact',passed=True))
    values=np.array([.868,.304,.218,-.005,.046,-.001,-.994],dtype=np.float32)
    state=scope['state_text']('move block',OBS,Env(),{'block':np.zeros(3)},[values]*50,0,50,20)
    assert 'at most 20 steps' in state and '0.867999' not in state
    checks.append(dict(name='actual float32 serializer and prompt cadence',passed=True))
    for delay in [4,5]:
        for choice,method in [('replan','smol70'),('continue','fixed_r60')]:
            gated=scope['run_episode'](Predictor(),Gate(choice),root,f'd{delay}_skip_{choice}',0,0,50,'laya_skip1',15,logical_delay=delay)
            control=scope['run_episode'](Predictor(),None,root,f'd{delay}_control_{choice}',0,0,50,method,15,logical_delay=delay)
            gc=Path(gated['case_path']);cc=Path(control['case_path'])
            with np.load(gc/'trace.npz') as a,np.load(cc/'trace.npz') as b:
                assert np.array_equal(a['actions'],b['actions']) and np.array_equal(a['physics'],b['physics'])
            calls=json.loads((gc/'calls.json').read_text());gates=json.loads((gc/'gate.json').read_text())
            assert [c['request_tick'] for c in calls]==([0,15,30,45] if choice=='replan' else [0,30])
            assert all(c['actual_delivery_tick']==c['request_tick']+delay and c['expired_prefix_rows']==delay and c['executed_rows'][0]==delay for c in calls[1:])
            assert all(f'after {delay} additional logical steps' in g['state'] for g in gates)
            checks.append(dict(name=f'd={delay}: single-skip {choice} equals {method}; actual delay prefix and prompt',passed=True))
        finite=scope['run_episode'](Predictor(),Gate('continue'),root,f'd{delay}_finite',0,0,50,'laya',20,logical_delay=delay)
        calls=json.loads((Path(finite['case_path'])/'calls.json').read_text())
        assert [c['request_tick'] for c in calls]==[0,50-delay]
        assert calls[1]['actual_delivery_tick']==50 and calls[1]['executed_rows'][0]==delay
        checks.append(dict(name=f'd={delay}: finite queue triggers with exactly d commands left and delivers at tick50',passed=True))
        predictor=Predictor()
        vlash=scope['run_episode'](predictor,None,root,f'd{delay}_vlash',0,0,50,'vlash_style_k5',logical_delay=delay)
        calls=json.loads((Path(vlash['case_path'])/'calls.json').read_text())
        assert [c['request_tick'] for c in calls]==list(range(0,60,5))
        for c,obs in zip(calls[1:],predictor.inputs[1:]):
            boundary=c['request_tick']
            assert obs['frame_tick']==c['image_tick']==boundary-delay
            assert obs['robot_state']['eef']['pos'][0]==c['state_tick']==boundary
            assert c['actual_delivery_tick']==boundary and c['executed_rows'][0]==0
        checks.append(dict(name=f'd={delay}: all VLASH K5 boundaries use old image and current state, starting row0',passed=True))
        from window_state import prepare_window_state
        queue=np.arange(700,dtype=float).reshape(100,7)/1000
        state=scope['state_text']('move block',OBS,Env(),{'block':np.zeros(3)},queue,0,100,30,delay)
        changed=queue.copy();changed[30+delay:]+=10
        state2=scope['state_text']('move block',OBS,Env(),{'block':np.zeros(3)},changed,0,100,30,delay)
        for compact in [False,True]:
            a=prepare_window_state(state,queue,compact=compact)
            b=prepare_window_state(state2,changed,compact=compact)
            assert a==b
            if not compact:assert f'next {30+delay} existing commands' in a
        checks.append(dict(name=f'd={delay}: both window inputs exclude every command beyond interval+d',passed=True))
(ROOT/'checks/controller_mechanisms.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(dict(checks=len(checks),failures=0)))
