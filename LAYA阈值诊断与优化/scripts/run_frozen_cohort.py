"""Sequential execution of an immutable cohort; archive/audit every shard before the next GPU stage."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from local_policy import allowed

ROOT = Path(__file__).resolve().parents[1]
BASE = '/mnt/4t/jev_vla_libero'
REMOTE = BASE + '/experiments/laya_gate_optimization'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remote_python(code):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'gpu4090-frp', 'python3 -'], input=code, text=True,
                            encoding='utf-8', capture_output=True, check=True, timeout=60)
    return json.loads(result.stdout)


def invoke(script, *args):
    result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts' / script), *args],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    print(result.stdout.strip(), flush=True)
    if result.returncode:
        print(result.stderr[-6000:], file=sys.stderr, flush=True)
        result.check_returncode()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    assert args.cohort.isascii() and args.cohort.replace('_', '').isalnum()
    folder = ROOT / 'protocol' / args.cohort
    definition = read(folder / 'definition.json')
    digest = sha(folder / 'definition.json')
    assert digest == read(folder / 'freeze_receipt.json')['definition_sha256']
    shards = definition['shards']
    entry = definition.get('entry_point', 'run_closed_loop.py')
    assert entry in ['run_closed_loop.py', 'run_verified_closed_loop.py', 'run_guarded_closed_loop.py', 'run_mechanism_closed_loop.py', 'run_seeded_skip_closed_loop.py']
    audit_entry = definition.get('audit_entry', 'audit_closed_loop.py')
    assert audit_entry in ['audit_closed_loop.py', 'audit_mechanism_closed_loop.py', 'audit_seeded_skip_closed_loop.py']
    assert 0 <= args.begin < len(shards)
    for name, expected in definition['locked_execution_source_sha256'].items():
        assert sha(ROOT / 'scripts' / name) == expected
    for shard in shards:
        assert sha(ROOT / 'configs' / shard['config']) == shard['config_sha256']
    for qualification in definition.get('qualification_runs', []):
        name = qualification['run']
        audit = read(ROOT / 'checks' / f'{name}_audit.json')
        manifest = read(ROOT / 'raw' / name / 'data/manifest.json')
        assert audit['errors'] == 0 and audit['episodes'] == qualification['episodes']
        assert read(ROOT / 'raw' / name / 'supervisor.json')['status'] == 'completed'
        assert manifest['logical_delay_steps'] == qualification['logical_delay_steps']
        assert manifest['freeze_before'] == manifest['freeze_after']
        if definition.get('initial_input_verification_required'):
            initial_audit = read(ROOT / 'checks' / f'{name}_initial_verification.json')
            assert initial_audit['errors'] == 0 and initial_audit['episodes'] == qualification['episodes']
        if definition.get('noise_alignment_audit_required'):
            assert read(ROOT/'checks'/f'{name}_noise_alignment.json')['errors'] == 0
        for script, expected in definition['locked_execution_source_sha256'].items():
            assert manifest['scripts'][script] == expected
    for shard in shards[:args.begin]:
        name = shard['run']
        audit = read(ROOT / 'checks' / f'{name}_audit.json')
        manifest = read(ROOT / 'raw' / name / 'data/manifest.json')
        assert audit['errors'] == 0 and audit['episodes'] == shard['episodes']
        assert read(ROOT / 'raw' / name / 'supervisor.json')['status'] == 'completed'
        assert manifest['config_sha256'] == shard['config_sha256']
        if definition.get('initial_input_verification_required'):
            initial_audit = read(ROOT / 'checks' / f'{name}_initial_verification.json')
            assert initial_audit['errors'] == 0 and initial_audit['episodes'] == shard['episodes']
        archive = read(ROOT / 'artifacts' / f'{name}_archive.json')
        assert sha(ROOT / 'artifacts' / f'{name}.tar.gz') == archive['sha256']
    for requirement in definition.get('exact_replay_qualifications', []):
        evidence = read(ROOT / requirement['file'])
        assert evidence['n'] == evidence['entire_trajectory_exact'] == requirement['n']
        for field in ['current', 'reference', 'candidates_filter']:
            assert evidence[field] == requirement[field]
    allowed()
    if args.check_only:
        print(json.dumps(dict(cohort=args.cohort, definition_sha256=digest,
                              remaining_shards=len(shards)-args.begin, check_only=True)))
        return
    runtime = ROOT / 'runtime'
    runtime.mkdir(exist_ok=True)
    status_path = runtime / f'{args.cohort}_chain.json'
    assert not status_path.exists(), 'Do not silently restart a recorded chain'
    state = dict(cohort=args.cohort, definition_sha256=digest, begin=args.begin,
                 started_utc=datetime.now(timezone.utc).isoformat(), completed_shards=[])

    def record(phase, **fields):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8', newline='\n')
        temp.replace(status_path)
        print(json.dumps(dict(phase=phase, **fields)), flush=True)

    try:
        for shard in shards[args.begin:]:
            allowed()
            name = shard['run']
            assert name.isascii() and name.replace('_', '').isalnum()
            assert shard['config'] == name + '.json'
            check = remote_python(f"from pathlib import Path\nimport hashlib,json\nr=Path('{REMOTE}')\nprint(json.dumps(dict(exists=(r/'runs/{name}').exists(),config_sha256=hashlib.sha256((r/'configs/{name}.json').read_bytes()).hexdigest())))")
            assert not check['exists'], 'Existing run requires explicit integrity review, never overwrite'
            assert check['config_sha256'] == shard['config_sha256']
            record('launching', run=name, completed_episodes=0, planned_episodes=shard['episodes'])
            command = (f'nohup {BASE}/scripts/run_in_workspace.sh python3 -u {REMOTE}/scripts/supervise.py '
                       f'--run {name} --entry {entry} --env smolvla --wall 5400 '
                       f'-- --config {name}.json > {REMOTE}/launch_{name}.log 2>&1 < /dev/null &')
            subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                            'gpu4090-frp', command], check=True, timeout=45)
            launch = time.monotonic()
            while True:
                allowed()
                snapshot = remote_python(f"""
from pathlib import Path
import json
r=Path('{REMOTE}/runs/{name}')
p=r/'supervisor.json';q=r/'data/status.json'
s=json.loads(p.read_text()) if p.exists() else {{}}
d=json.loads(q.read_text()) if q.exists() else {{}}
print(json.dumps(dict(status=s.get('status','starting'),completed=d.get('completed',0),
                     planned=d.get('planned',{shard['episodes']}),error=s.get('error'))))
""")
                record('running', remote_status=snapshot['status'], completed_episodes=snapshot['completed'],
                       planned_episodes=snapshot['planned'])
                if snapshot['status'] == 'completed':
                    assert snapshot['completed'] == snapshot['planned'] == shard['episodes']
                    break
                assert snapshot['status'] not in ['stopped', 'failed'], snapshot
                assert time.monotonic() - launch < 5700, 'Supervisor stage did not finish within bounded wall'
                time.sleep(30)
            record('archiving')
            invoke('sync_run.py', name)
            record('auditing')
            invoke(audit_entry, name)
            if definition.get('initial_input_verification_required'):
                invoke('audit_initial_verification.py', name)
            if definition.get('noise_alignment_audit_required'):
                invoke('audit_noise_alignment.py', name)
            audit = read(ROOT / 'checks' / f'{name}_audit.json')
            assert audit['errors'] == 0 and audit['episodes'] == shard['episodes']
            state['completed_shards'].append(dict(run=name, audit_sha256=sha(ROOT/'checks'/f'{name}_audit.json'),
                archive_sha256=read(ROOT/'artifacts'/f'{name}_archive.json')['sha256']))
            record('shard_complete')
        record('completed')
    except BaseException as error:
        record('halted', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
