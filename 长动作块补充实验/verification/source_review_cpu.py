"""AST-extracted production scheduler check, CPU only, no model imports."""
from pathlib import Path
from types import SimpleNamespace
import ast, collections, copy, hashlib, json, tempfile, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'scripts/main_experiment.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
names = {'atomic', 'digest_bytes', 'object_positions', 'state_text', 'run_episode'}
module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
ns = dict(Path=Path, np=np, collections=collections, copy=copy, hashlib=hashlib, json=json, time=time)
exec(compile(module, str(source), 'exec'), ns)

class Env:
    def __init__(self, stop=230, success=False, trunc=False):
        self.tick=0; self.stop=stop; self.success=success; self.trunc=trunc; self.closed=False
        self._env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(get_state=lambda:np.array([self.tick]), data=SimpleNamespace(body_xpos=np.zeros((1,3)))), obj_body_id={'obj':0}))
    def obs(self):
        return {'images':np.array([self.tick]), 'robot_state':{'eef':{'pos':np.array([self.tick,0,0])},'gripper':{'qpos':np.zeros(2)}}}
    def step(self, action):
        self.tick+=1
        ended=self.tick==self.stop
        return self.obs(),0,ended and not self.trunc,ended and self.trunc,{'is_success':ended and self.success}
    def close(self): self.closed=True

class Predictor:
    def __init__(self, stop=230, success=False, trunc=False):
        self.env=Env(stop,success,trunc);self.inputs=[]
        self.engine=SimpleNamespace(restore=lambda *a:(self.env,self.env.obs()),suite=SimpleNamespace(get_task=lambda tid:SimpleNamespace(language='move object')))
    def predict(self,obs,instruction,horizon,seed):
        self.inputs.append((int(obs['images'][0]),int(obs['robot_state']['eef']['pos'][0])))
        arr=np.repeat((len(self.inputs)*10000+np.arange(horizon))[:,None],7,axis=1)
        return arr,arr[None].copy(),0.125

class Gate:
    def __init__(self,choice):self.choice=choice
    def predict(self,state):return {'choice':self.choice}

results=[]
with tempfile.TemporaryDirectory(prefix='long_chunk_cpu_') as tmp:
    out=Path(tmp)
    for h in [50,100,200]:
        cases={}
        for method in ['smol70','laya','naive_k5','vlash_style_k5']:
            predictor=Predictor(); result=ns['run_episode'](predictor,Gate('uncertain'),out,'cpu',0,0,h,method)
            path=Path(result['case_path']);calls=json.loads((path/'calls.json').read_text()); trace=np.load(path/'trace.npz')['actions']
            assert predictor.env.closed and result['steps']==230 and not result['success']
            assert all(c['native_output_shape']==[h,7] and c['nfe']==10 for c in calls)
            assert result['prediction_seconds']==len(calls)*.125
            if method in ['naive_k5','vlash_style_k5']:
                deliveries=[c['actual_delivery_tick'] for c in calls[1:] if 'actual_delivery_tick' in c]
                assert deliveries==list(range(5,230,5)), deliveries
                row=0 if method=='vlash_style_k5' else 1
                assert all(c['executed_rows'][0]==row for c in calls[1:] if c['executed_rows'])
                expected=[(t-1,t) if method=='vlash_style_k5' else (t-1,t-1) for t in deliveries]
                assert predictor.inputs[1:1+len(deliveries)]==expected
            cases[method]=(result,trace,calls)
            results.append({'horizon':h,'method':method,'calls':len(calls),'steps':result['steps'],'passed':True})
        assert np.array_equal(cases['smol70'][1],cases['laya'][1])
        assert [c['request_tick'] for c in cases['smol70'][2]]==[c['request_tick'] for c in cases['laya'][2]]
        expected=list(range(0,230,3*h//10))
        assert [c['request_tick'] for c in cases['smol70'][2]]==expected
        predictor=Predictor();result=ns['run_episode'](predictor,Gate('continue'),out,'continue',0,0,h,'laya')
        calls=json.loads((Path(result['case_path'])/'calls.json').read_text())
        assert all(c['trigger']=='mandatory_finite_queue' for c in calls[1:])
        assert result['steps']==230
    for method in ['smol70','laya','naive_k5','vlash_style_k5']:
        for success,trunc,stop in [(True,False,5),(False,True,17)]:
            predictor=Predictor(stop,success,trunc)
            result=ns['run_episode'](predictor,Gate('replan'),out,'stop_'+str(stop),0,0,50,method)
            assert result['success']==success and result['steps']==stop and predictor.env.closed
            if stop==5 and method in ['naive_k5','vlash_style_k5']:
                assert result['pending_at_termination']
                assert result['vla_calls']==(1 if method=='vlash_style_k5' else 2)
    audit={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cpu_only':True,'cases':results,
           'passed':['H50/100/200 actual action shapes','all-uncertain LAYA equals Smol70 executed actions and call ticks','remaining <= 0.7 triggers at 15/30/60','K5 deliveries 5,10,...225','naive row1 / VLASH row0','naive stale state / VLASH current state','all-continue finite queue guard','termination/truncation and pending call counts','environment closes','call and NFE metadata'],
           'limitations':'CPU mocks test scheduler only; real GPU model output/frozen weights require remote qualification.'}
    (ROOT/'verification/source_review_cpu.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
