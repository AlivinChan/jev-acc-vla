"""Recover exact source bytes for early runs; fail on any unmatched manifest hash."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--git-commit',required=True);args=p.parse_args()
manifest=json.loads((ROOT/'raw'/args.run/'data/manifest.json').read_text())
expected=manifest.get('scripts',{})
out=ROOT/'artifacts/run_sources'/args.run/'scripts';out.mkdir(parents=True,exist_ok=True)
audit={}
for name,digest in expected.items():
    path=ROOT/'scripts'/name
    value=path.read_bytes() if path.exists() else b''
    source='working bytes'
    if hashlib.sha256(value).hexdigest()!=digest:
        result=subprocess.run(['git','show',f'{args.git_commit}:scripts/{name}'],cwd=ROOT,capture_output=True,check=True)
        value=result.stdout;source='git '+args.git_commit
    assert hashlib.sha256(value).hexdigest()==digest,(name,digest,hashlib.sha256(value).hexdigest())
    (out/name).write_bytes(value);audit[name]=dict(sha256=digest,source=source)
(out.parent/'manifest.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(dict(run=args.run,matched_files=len(audit),errors=0)))
