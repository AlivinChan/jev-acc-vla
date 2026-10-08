"""Upload only the new frozen random-control files; preserve active runtime bytes."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
from local_policy import allowed

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/mnt/4t/jev_vla_libero/experiments/laya_gate_optimization'
COHORTS = ['random_skip_pilot_d1_v1', 'random_skip_pilot_d4_v1', 'random_skip_validation_d4_v1']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def remote(code):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'gpu4090-frp', 'python3 -'],
                            input=code, text=True, encoding='utf-8', capture_output=True, check=True, timeout=45)
    return json.loads(result.stdout)


def main():
    allowed()
    files = {ROOT/'RANDOM_SKIP_PROTOCOL.md', ROOT/'checks/seeded_skip_cpu.json', ROOT/'checks/random_skip_available_initials.json'}
    locked = None
    for name in COHORTS:
        p = ROOT/'protocol'/name/'definition.json'
        d = read(p)
        assert sha(p) == read(p.parent/'freeze_receipt.json')['definition_sha256']
        files.update([p, p.parent/'freeze_receipt.json'])
        files.update(ROOT/'configs'/row['config'] for row in d['shards'])
        if locked is None:
            locked = d['locked_execution_source_sha256']
        assert locked == d['locked_execution_source_sha256']
    new_sources = ['seeded_skip_runtime.py', 'run_seeded_skip_closed_loop.py', 'audit_seeded_skip_closed_loop.py']
    files.update(ROOT/'scripts'/name for name in new_sources)
    for name, digest in locked.items():
        assert sha(ROOT/'scripts'/name) == digest
    old = {name: value for name, value in locked.items() if name not in new_sources}
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(files)}
    # The remote preflight checks existing core code and refuses conflicting new files.
    checked = remote(f"""
from pathlib import Path
import hashlib,json
r=Path({REMOTE!r})
old=json.loads({json.dumps(old)!r})
new=json.loads({json.dumps(hashes)!r})
assert all(hashlib.sha256((r/'scripts'/p).read_bytes()).hexdigest()==h for p,h in old.items())
assert all(not (r/p).exists() or hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in new.items())
print(json.dumps(dict(core_sources_verified=len(old),files=len(new))))
""")
    bundle = ROOT/'deployment_seeded_skip_v1.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        for path in sorted(files):
            tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    assert bundle.stat().st_size < 8*1024*1024
    subprocess.run(['scp', '-q', str(bundle), f'gpu4090-frp:{REMOTE}/{bundle.name}'], check=True, timeout=60)
    result = remote(f"""
from pathlib import Path
import hashlib,json,tarfile
r=Path({REMOTE!r}).resolve();archive=r/{bundle.name!r}
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha(bundle)!r}
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        assert (r/member.name).resolve().is_relative_to(r) and not member.issym() and not member.islnk()
    tar.extractall(r,filter='data')
hashes=json.loads({json.dumps(hashes)!r})
assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
print(json.dumps(dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(hashes))))
""")
    result.update(core_preflight=checked, files_sha256=hashes,
                  excluded='Existing core code, all models, all other configs, current runs and archives')
    (ROOT/'checks/deployment_seeded_skip_v1.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files_sha256'}))


if __name__ == '__main__':
    main()
