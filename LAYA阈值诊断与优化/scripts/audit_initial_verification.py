"""Independent persisted-input, reference-repeat and physical call-ledger audit."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def receipt(values):
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                     sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in sorted(values.items())}


def npz_receipt(path):
    with np.load(path, allow_pickle=False) as data:
        return receipt({key: data[key] for key in data.files})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    args = parser.parse_args()
    base = ROOT/'raw'/args.run/'data'
    folder = base/'initial_verification'
    manifest = read(base/'manifest.json')
    rows = [json.loads(s) for s in (base/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    guards = [json.loads(s) for s in (folder/'guards.jsonl').read_text(encoding='utf-8').splitlines()]
    qualification = read(folder/'qualification_calls.json')
    summary = read(folder/'summary.json')
    checks = Counter()
    failures = []

    def check(value, kind, context=None):
        checks[kind] += 1
        if not value:
            failures.append(dict(kind=kind, context=context))

    cfg = manifest['config']
    expected_starts = {(t, s) for t in cfg['tasks'] for s in cfg['states']}
    expected_qual = {(t, s, h, rep) for t, s in expected_starts for h in cfg['horizons'] for rep in [0, 1]}
    qual_index = {(r['task'], r['state'], r['horizon'], r['repeat']): r for r in qualification}
    check(set(qual_index) == expected_qual and len(qualification) == len(expected_qual), 'complete unique reference repeats')
    check(len(guards) == len(rows) == summary['guarded_episodes'], 'one guard per episode')
    check(summary == manifest['initial_verification'], 'manifest verification matches evidence')
    check(summary['exact'] is True and summary['starts'] == len(expected_starts), 'all starts guarded')
    check(manifest['qualification_vla_calls'] == 2*len(cfg['horizons'])+len(qualification), 'warm plus reference call accounting')
    check(manifest['total_vla_calls'] == manifest['qualification_vla_calls']+sum(r['vla_calls'] for r in rows), 'all physical VLA calls accounted')
    check(summary['qualification_vla_calls'] == len(qualification), 'reference ledger count')
    check(abs(summary['qualification_prediction_seconds']-sum(r['seconds'] for r in qualification)) < 1e-9, 'reference ledger time')
    check(abs(summary['episode_input_capture_seconds']-sum(r['capture_seconds'] for r in guards)) < 1e-9, 'input capture overhead ledger')
    check(summary['input_capture_included_in_prediction_seconds'] is True, 'input capture included in service')
    env = read(folder/'environment.json')
    check(env == summary['environment'], 'execution backend settings unchanged')
    if cfg.get('deterministic_backend_forced', True):
        check(env['deterministic_algorithms'] and not env['deterministic_warn_only'] and env['cudnn_deterministic']
              and not env['cudnn_benchmark'] and env['cublas_workspace_config'] == ':4096:8', 'strict deterministic backend settings')
    else:
        check(not env['deterministic_algorithms'] and env['cublas_workspace_config'] is None,
              'original backend retained; no forced deterministic switch')
    refs = {}
    for t, s in sorted(expected_starts):
        start = folder/f't{t:02d}_s{s:02d}'
        ref = read(start/'reference.json')
        refs[(t, s)] = ref
        check(ref['restore'] == read(start/'restore.json'), 'saved reference restore', [t, s])
        check(npz_receipt(start/'raw_observation.npz') == ref['restore']['observation'], 'raw observation bytes', [t, s])
        for h in cfg['horizons']:
            target = ref['horizons'][str(h)]
            check(npz_receipt(start/f'h{h}_processed_input.npz') == target['processed_input'], 'canonical processed input bytes', [t, s, h])
            for rep in [0, 1]:
                entry = qual_index[(t, s, h, rep)]
                check(npz_receipt(start/f'h{h}_repeat{rep}_prediction.npz') == target['prediction'] == entry['prediction'],
                      'repeated prediction bytes', [t, s, h, rep])
                check(entry['processed_input'] == target['processed_input'] and entry['repeated_exact'], 'repeated input exact', [t, s, h, rep])
                check(0 < entry['capture_seconds'] < entry['seconds'], 'qualification capture contained in timing', [t, s, h, rep])
    for row, guard in zip(rows, guards):
        t, s, h = row['task_id'], row['state_id'], row['horizon']
        tag = [row['candidate_id'], t, s, h]
        ref = refs[(t, s)]
        case = base/row['candidate_id']/f't{t:02d}_s{s:02d}_h{h}_{row["method"]}'
        check([guard['candidate'], guard['task'], guard['state'], guard['horizon']] == tag, 'guard order matches episode', tag)
        check(guard['exact'] and guard['checked_before_first_action'], 'guard passed before action', tag)
        check(guard['restore'] == ref['restore'], 'full initial observation model physics paired', tag)
        check(receipt({'': np.load(case/'initial_state.npy')}) == ref['restore']['physics'], 'physical initial bytes independently checked', tag)
        check(guard['noise_seed'] == 20261007+t*100000+s*1000, 'reference seed fixed', tag)
        check(guard['processed_input'] == ref['horizons'][str(h)]['processed_input'], 'all initial model inputs paired', tag)
        check(npz_receipt(case/'chunk_000.npz') == guard['prediction'] == ref['horizons'][str(h)]['prediction'], 'all initial action and normalized bytes paired', tag)
        calls = read(case/'calls.json')
        check(0 < guard['capture_seconds'] < calls[0]['predict_seconds'], 'initial input copy timing contained', tag)
    result = dict(run=args.run, episodes=len(rows), reference_calls=len(qualification), checks=sum(checks.values()),
                  by_kind=dict(checks), errors=len(failures), failures=failures,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'checks'/f'{args.run}_initial_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['by_kind', 'failures']}, ensure_ascii=False))
    if failures:
        print(json.dumps(failures[:5]))
    assert not failures


if __name__ == '__main__':
    main()
