"""Independent CPU contract check of actual primary rollout and argmax analyzer."""
from pathlib import Path
import ast,collections,copy,hashlib,importlib.util,json,sys,tempfile,time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'scripts/main_experiment.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
names={'atomic','digest_bytes','object_positions','state_text','run_episode'}
ns=dict(Path=Path,np=np,collections=collections,copy=copy,hashlib=hashlib,json=json,time=time,SimpleNamespace=SimpleNamespace)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(source),'exec'),ns)
fake_tree=ast.parse((ROOT/'verification/source_review_cpu.py').read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in fake_tree.body if isinstance(n,ast.ClassDef) and n.name in {'Env','Predictor'}],type_ignores=[]),'fake_classes','exec'),ns)
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_argmax as analyzer

class CheapPredictor(ns['Predictor']):
    def predict(self,*args):
        actions,norm,_=super().predict(*args)
        return actions,norm,0.000000001

class Gate:
    def __init__(self,folder,choice):
        self.folder=folder;self.choice=choice;self.index=0
        (folder/'laya_ipc/requests').mkdir(parents=True)
        (folder/'laya_ipc/responses').mkdir(parents=True)
    def predict(self,state):
        rid=f'{self.index:07d}';self.index+=1
        response=dict(status='ok',request_id=rid,warmup=False,choice=self.choice,raw_choice=self.choice,
                      probabilities={name:.8 if name==self.choice else .1 for name in ['continue','replan','uncertain']},
                      minimum_probability=0.0,usage={'truncated':False,'state_tokens_dropped':0},
                      token_audit={'max_len':1536,'head_max_len':192,'state':{'truncated':False},'options':{'options_distinct':3}},
                      cpu_fallback_count=0,state_sha256=hashlib.sha256(state.encode()).hexdigest(),
                      ipc_outer_seconds=0,sdk_synchronized_wall_ms=0,service_prewrite_wall_ms=0)
        ns['atomic'](self.folder/'laya_ipc/requests'/f'{rid}.json',{'state':state,'request_id':rid,'warmup':False})
        ns['atomic'](self.folder/'laya_ipc/responses'/f'{rid}.json',response)
        return response

checks=[]
with tempfile.TemporaryDirectory(prefix='argmax_review_') as temporary:
    for horizon in [50,100,200]:
        for choice in ['continue','replan','uncertain']:
            folder=Path(temporary)/f'h{horizon}_{choice}'
            row=ns['run_episode'](CheapPredictor(),Gate(folder,choice),folder,'argmax',0,0,horizon,'laya')
            audit=analyzer.core.Audit()
            case=analyzer.validate_case(folder,row,audit)
            assert case['valid'] and not audit.errors,audit.errors
            checks.append(dict(horizon=horizon,raw_choice=choice,valid=case['valid'],checks=audit.checked,vla_calls=row['vla_calls'],gate_calls=row['gate_calls']))
    empty=Path(temporary)/'empty';empty.mkdir()
    result=analyzer.analyze(empty,empty,Path(temporary)/'partial_result')
    assert result['completion_status']=='partial' and result['episodes_seen']==0
files=['laya_argmax_service.py','run_argmax_ablation.py','supervise_argmax.py','analyze_argmax.py']
for file in files:ast.parse((ROOT/'scripts'/file).read_text(encoding='utf-8'))
output=dict(cpu_only=True,status='passed',schema_validation=checks,empty_input_annotated_partial=True,
            source_sha256={f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in files},
            scope='Actual AST-extracted primary.run_episode with distinguishable action rows and real analyze_argmax.validate_case; no torch/model/CUDA imports.')
(ROOT/'verification/argmax_source_review_cpu.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
