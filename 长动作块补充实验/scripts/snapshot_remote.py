"""Archive only a completed remote attempt, verify SHA256, safely mirror raw data."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/mnt/4t/jev_vla_libero/experiments/long_chunk_laya'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', choices=['attempt_v1', 'argmax_attempt_v1'], required=True)
    args = parser.parse_args()
    code = '''
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, tarfile
r = Path(REMOTE_VALUE)
attempt = ATTEMPT_VALUE
sup = json.loads((r / attempt / 'supervisor.json').read_text())
assert sup['status'] == 'completed', sup
archives = r / 'archives'
archives.mkdir(exist_ok=True)
archive = archives / (attempt + '_complete.tar.gz')
members = [attempt, 'environment', 'scripts', 'PLAN.md', 'ARGMAX_PLAN.md',
           'SOURCE_REVIEW.md', 'PILOT_RUNTIME_AUDIT.md', 'ARGMAX_SOURCE_REVIEW.md']
if not archive.exists():
    partial = archives / (attempt + '_complete.partial.tar.gz')
    assert not partial.exists(), 'Partial archive requires explicit inspection'
    def keep(info):
        return None if '__pycache__' in Path(info.name).parts or info.name.endswith('.pyc') else info
    with tarfile.open(partial, 'w:gz', compresslevel=3) as out:
        for member in members:
            if (r / member).exists():
                out.add(r / member, arcname=member, filter=keep)
    partial.rename(archive)
h = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
proof = dict(attempt=attempt, archive=str(archive), sha256=h.hexdigest(),
             bytes=archive.stat().st_size, supervisor_status=sup['status'],
             archived_utc=datetime.now(timezone.utc).isoformat())
(archives / (attempt + '_complete.proof.json')).write_text(json.dumps(proof, indent=2))
print(json.dumps(proof))
'''.replace('REMOTE_VALUE', repr(REMOTE)).replace('ATTEMPT_VALUE', repr(args.attempt))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'gpu4090-frp', 'python3 -'], input=code, text=True,
                            encoding='utf-8', capture_output=True, timeout=300, check=True)
    proof = json.loads(result.stdout)
    archives = ROOT / 'archives'
    archives.mkdir(exist_ok=True)
    local = archives / Path(proof['archive']).name
    if not local.exists():
        subprocess.run(['scp', '-q', 'gpu4090-frp:' + proof['archive'], str(local)],
                       timeout=300, check=True)
    assert digest(local) == proof['sha256'], 'Downloaded archive hash differs'
    target = ROOT / 'raw'
    target.mkdir(exist_ok=True)
    with tarfile.open(local) as source:
        for member in source.getmembers():
            resolved = (target / member.name).resolve()
            assert resolved.is_relative_to(target.resolve()), member.name
            assert member.isfile() or member.isdir(), 'Archive contains link/device: ' + member.name
        source.extractall(target, filter='data')
    proof.update(local_archive=str(local), local_sha256=digest(local), extracted_to=str(target))
    (archives / (args.attempt + '_complete.proof.json')).write_text(
        json.dumps(proof, indent=2), encoding='utf-8')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
