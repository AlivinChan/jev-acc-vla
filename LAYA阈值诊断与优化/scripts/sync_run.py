"""Small read-only snapshots during execution; verified archive only after completion."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tarfile
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/mnt/4t/jev_vla_libero/experiments/laya_gate_optimization'

def remote_python(code):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','gpu4090-frp','python3 -'],
                     input=code,text=True,encoding='utf-8',capture_output=True,timeout=60,check=True)
    return json.loads(r.stdout)

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--metrics-only',action='store_true')
args=p.parse_args();assert args.run.isascii() and args.run.replace('_','').isalnum()
remote_run=f'{REMOTE}/runs/{args.run}'
if args.metrics_only:
    code=f"""
from pathlib import Path
import json
r=Path('{remote_run}')
names=['supervisor.json','data/status.json','data/manifest.json','data/freeze_before.json','data/freeze_after.json',
       'data/episodes.jsonl','data/records.jsonl','data/qualification.json','data/schedule.json','data/failure.json']
print(json.dumps({{n:(r/n).read_text() for n in names if (r/n).exists()}}))
"""
    files=remote_python(code)
    for name,value in files.items():
        target=ROOT/'raw'/args.run/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(value,encoding='utf-8',newline='\n')
    status=json.loads(files.get('data/status.json','{}'))
    print(json.dumps(dict(run=args.run,status=json.loads(files['supervisor.json'])['status'],
                         completed=status.get('completed'),planned=status.get('planned'))))
else:
    code=f"""
from pathlib import Path
import hashlib,json,tarfile
root=Path('{REMOTE}');r=Path('{remote_run}')
status=json.loads((r/'supervisor.json').read_text())
assert status['status']!='running','Do not copy a large archive while this GPU job is active'
(root/'artifacts').mkdir(exist_ok=True)
archive=root/'artifacts/{args.run}.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(r,arcname='{args.run}')
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,status=status['status'])))
"""
    meta=remote_python(code)
    artifacts=ROOT/'artifacts';artifacts.mkdir(exist_ok=True)
    archive=artifacts/f'{args.run}.tar.gz'
    # A large unthrottled SCP previously delayed SSH lease delivery; GPU stages are idle here.
    rate=1024 if meta['bytes']>8*1024*1024 else None
    command=['scp','-q']+(['-l',str(rate)] if rate else [])+[f"gpu4090-frp:{meta['path']}",str(archive)]
    timeout=max(180,int(meta['bytes']/(128*1024)*2+120)) if rate else 180
    subprocess.run(command,check=True,timeout=timeout)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==meta['sha256']
    raw=(ROOT/'raw').resolve();raw.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            assert (raw/member.name).resolve().is_relative_to(raw) and not member.issym() and not member.islnk()
        tar.extractall(raw,filter='data')
    meta['download_bandwidth_limit_kbit_per_second']=rate
    (artifacts/f'{args.run}_archive.json').write_text(json.dumps(meta,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(meta))
