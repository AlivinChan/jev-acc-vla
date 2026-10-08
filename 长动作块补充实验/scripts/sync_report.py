"""Mirror the completed report bundle into this experiment's remote directory."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/mnt/4t/jev_vla_libero/experiments/long_chunk_laya'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = list(ROOT.glob('*.md'))
    for folder in ['scripts', 'verification', 'figures', 'references', 'source_snapshot', 'results']:
        files.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
                     and p.name not in ['report_delivery_manifest.json', 'delivery_receipt.json'])
    files.extend((ROOT / 'archives').glob('*.proof.json'))
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}
    manifest_path = ROOT / 'results/report_delivery_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    files.append(manifest_path)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    bundle = ROOT / 'archives' / ('report_bundle_' + stamp + '.tar.gz')
    assert not bundle.exists()
    with tarfile.open(bundle, 'w:gz') as archive:
        for path in sorted(set(files)):
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    expected = sha(bundle)
    remote_bundle = REMOTE + '/archives/' + bundle.name
    subprocess.run(['scp', '-q', str(bundle), 'gpu4090-frp:' + remote_bundle], check=True, timeout=180)
    code = '''
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, tarfile
r = Path(REMOTE_VALUE)
archive = Path(BUNDLE_VALUE)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(archive) == SHA_VALUE
for name in ['attempt_v1','argmax_attempt_v1']:
    assert json.loads((r/name/'supervisor.json').read_text())['status'] == 'completed'
with tarfile.open(archive) as source:
    for member in source.getmembers():
        assert member.isfile() or member.isdir(), member.name
        assert (r/member.name).resolve().is_relative_to(r.resolve()), member.name
    source.extractall(r)
manifest = json.loads((r/'results/report_delivery_manifest.json').read_text())
assert all(sha(r/name) == digest for name,digest in manifest.items())
(r/'raw').mkdir(exist_ok=True)
for name in ['attempt_v1','argmax_attempt_v1']:
    link = r/'raw'/name
    if not link.exists(): link.symlink_to(Path('..')/name, target_is_directory=True)
    assert link.resolve() == (r/name).resolve()
index = r.parents[1]/'INDEX.md'
text = index.read_text()
line = 'Completed: 360 main + 90 exploratory argmax + 30 official VLASH + 12 pilot = 492 episodes. All freeze and raw audits passed. Report: [long-chunk LAYA report](experiments/long_chunk_laya/REPORT.md). Both verified raw archives retained; no owned GPU process remains.'
old = next((item for item in text.splitlines() if item.startswith('Running: experiments/long_chunk_laya/attempt_v1;')), None)
if old: text = text.replace(old, line, 1)
elif line not in text: text += '\\n\\n## Completed long-chunk LAYA experiment\\n\\n' + line + '\\n'
index.write_text(text)
receipt = dict(utc=datetime.now(timezone.utc).isoformat(), files_verified=len(manifest),
               bundle=str(archive), bundle_sha256=sha(archive), report_sha256=sha(r/'REPORT.md'),
               report=str(r/'REPORT.md'), raw_links_verified=True, index_updated=True)
(r/'results/delivery_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''.replace('REMOTE_VALUE', repr(REMOTE)).replace('BUNDLE_VALUE', repr(remote_bundle)).replace('SHA_VALUE', repr(expected))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'gpu4090-frp', 'python3 -'], input=code, text=True, encoding='utf-8',
                            capture_output=True, timeout=120, check=True)
    receipt = json.loads(result.stdout)
    (ROOT / 'results/delivery_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
