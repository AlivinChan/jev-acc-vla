"""CPU only: audit common initial Gaussian prefixes across native action horizons."""
from pathlib import Path
import hashlib,json
from supervise import allowed
BASE=Path('/mnt/4t/jev_vla_libero');ROOT=BASE/'experiments/laya_gate_optimization'
allowed()
import torch
assert not torch.cuda.is_initialized()
sources=json.loads((BASE/'configs/baseline_sources.json').read_text())
config_path=BASE/'models'/('smolvla_libero_'+sources['checkpoint_revision'])/'config.json'
cfg=json.loads(config_path.read_text());dimension=cfg['max_action_dim'];assert dimension==32
records=[]
for tid in range(10):
    for sid in range(3):
        seed=20261007+tid*100000+sid*1000
        noise={h:torch.randn((1,h,dimension),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32,device='cpu') for h in [50,100,200]}
        matches={f'{a}:{b}':bool(torch.equal(noise[a],noise[b][:,:a])) for a,b in [(50,100),(50,200),(100,200)]}
        assert all(matches.values())
        records.append(dict(task=tid,state=sid,seed=seed,prefix_exact=matches,
            noise_sha256={str(h):hashlib.sha256(x.numpy().tobytes()).hexdigest() for h,x in noise.items()}))
assert not torch.cuda.is_initialized()
result=dict(cases=len(records),comparisons=3*len(records),all_prefixes_exact=True,cuda_initialized=False,
            torch=torch.__version__,model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            max_action_dim=dimension,records=records)
(ROOT/'checks/noise_prefix_cpu.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
