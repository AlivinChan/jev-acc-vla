"""Sync this isolated experiment's small scripts/docs/fixtures to its own remote root."""
from pathlib import Path
import hashlib,json,subprocess,tarfile
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/mnt/4t/jev_vla_libero/experiments/laya_gate_optimization'
bundle=ROOT/'deployment.tar.gz'
with tarfile.open(bundle,'w:gz') as tar:
    for p in sorted(ROOT.rglob('*')):
        relative=p.relative_to(ROOT)
        if not p.is_file() or any(part in {'.git','__pycache__','raw','runs','runtime','tmp','.pdf_dependencies','output'} for part in relative.parts):continue
        if p.suffix in {'.gz','.pyc'}:continue
        tar.add(p,arcname=relative.as_posix(),recursive=False)
subprocess.run(['ssh','-o','BatchMode=yes','gpu4090-frp',f'mkdir -p {REMOTE}'],check=True,timeout=30)
subprocess.run(['scp','-q',str(bundle),f'gpu4090-frp:{REMOTE}/deployment.tar.gz'],check=True,timeout=60)
code=f"""
from pathlib import Path
import tarfile,hashlib,json
root=Path('{REMOTE}').resolve()
with tarfile.open(root/'deployment.tar.gz') as tar:
    for member in tar.getmembers():
        path=(root/member.name).resolve()
        assert path.is_relative_to(root) and not member.issym() and not member.islnk()
    tar.extractall(root,filter='data')
print(json.dumps({{'uploaded_sha256':hashlib.sha256((root/'deployment.tar.gz').read_bytes()).hexdigest()}}))
"""
result=subprocess.run(['ssh','-o','BatchMode=yes','gpu4090-frp','python3 -'],input=code,text=True,
                      capture_output=True,check=True,timeout=30)
print(result.stdout.strip())
print(json.dumps(dict(local_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),bytes=bundle.stat().st_size)))
