"""CPU-only snapshot of the installed SDK and fixed public configuration."""
from pathlib import Path
import json,subprocess,hashlib

ROOT=Path(__file__).resolve().parents[1]
code="""
from pathlib import Path
import json
r=Path('/mnt/4t/jev_vla_libero')
p=r/'envs/laya/lib/python3.12/site-packages/laya'
files={'sdk_'+n:(p/n).read_text() for n in ['agent.py','common.py','confidence.py','calibrate.py']}
s=json.loads((r/'configs/local_gate_sources.json').read_text())['laya']
files['rl_agent_config.json']=(r/s['path']/'rl_agent_config.json').read_text()
files['laya_source_config.json']=json.dumps(s,indent=2)
print(json.dumps(files))
"""
res=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','gpu4090-frp','python3 -'],input=code,
                   text=True,encoding='utf-8',capture_output=True,timeout=30,check=True)
folder=ROOT/'source_snapshot';folder.mkdir(exist_ok=True)
manifest={}
for name,contents in json.loads(res.stdout).items():
    p=folder/name;p.write_text(contents,encoding='utf-8',newline='\n')
    manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'files':len(manifest),'source':'pinned remote SDK0.3.22 and model config'}))
