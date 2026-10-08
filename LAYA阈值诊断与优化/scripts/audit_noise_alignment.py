"""First K5 update contrasts, before any controller has changed the common trajectory."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import hashlib
import json
import numpy as np
from audit_initial_verification import receipt, npz_receipt

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['naive_k5', 'stale_row0_k5', 'vlash_noise_matched_k5', 'vlash_style_k5']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    args = parser.parse_args()
    base = ROOT/'raw'/args.run/'data'
    manifest = read(base/'manifest.json')
    delay = manifest['logical_delay_steps']
    rows = [json.loads(s) for s in (base/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = defaultdict(dict)
    checks = Counter()
    failures = []
    events = []

    def check(value, kind, key):
        checks[kind] += 1
        if not value:
            failures.append(dict(kind=kind, context=key))

    for row in rows:
        cid = row['candidate_id']
        if cid in NAMES:
            groups[(row['task_id'], row['state_id'], row['horizon'])][cid] = row
    cfg = manifest['config']
    expected = {(t, s, h) for t in cfg['tasks'] for s in cfg['states'] for h in cfg['horizons']}
    check(set(groups) == expected, 'complete common K5 groups', args.run)
    for key, group in sorted(groups.items()):
        check(set(group) == set(NAMES), 'all four controls present', key)
        folders, calls, traces = {}, {}, {}
        for cid, row in group.items():
            t, s, h = key
            case = base/cid/f't{t:02d}_s{s:02d}_h{h}_{row["method"]}'
            folders[cid] = case
            calls[cid] = read(case/'calls.json')
            traces[cid] = dict(np.load(case/'trace.npz', allow_pickle=False))
        reference = traces['naive_k5']
        check(all(np.array_equal(trace['actions'][:5], reference['actions'][:5])
                  and np.array_equal(trace['physics'][:5], reference['physics'][:5]) for trace in traces.values()),
              'identical five-action prefix before update', key)
        if not all(row['steps'] > 5 for row in group.values()):
            check(len({r['steps'] for r in group.values()}) == 1 and max(r['steps'] for r in group.values()) <= 5,
                  'common termination before update opportunity', key)
            events.append(dict(task=key[0], state=key[1], horizon=key[2], opportunity=False))
            continue
        inputs, metadata, predictions = {}, {}, {}
        for cid, row in group.items():
            case = folders[cid]
            inp = npz_receipt(case/'first_update_observation.npz')
            meta = read(case/'first_update_input.json')
            call = calls[cid][1]
            inputs[cid], metadata[cid] = inp, meta
            predictions[cid] = dict(np.load(case/'chunk_001.npz', allow_pickle=False))
            check(inp == meta['observation'], 'recorded raw update input bytes', [*key, cid])
            check(meta['noise_seed'] == call['noise_seed'] and meta['noise_seed_tick'] == call['noise_seed_tick'],
                  'saved input and prediction seed receipt', [*key, cid])
            check(call['actual_delivery_tick'] == 5 and call['executed_rows'][0] == (delay if cid == 'naive_k5' else 0),
                  'first K5 boundary and actual execution row', [*key, cid])
            expected_tick = 5-delay if cid == 'naive_k5' else 5
            check(meta['prediction_tick'] == call['request_tick'] == expected_tick, 'actual prediction time', [*key, cid])
        check(inputs['naive_k5'] == inputs['stale_row0_k5'], 'same complete stale input', key)
        check(metadata['naive_k5']['noise_seed'] == metadata['stale_row0_k5']['noise_seed']
              == metadata['vlash_noise_matched_k5']['noise_seed'], 'same image-tick noise seed', key)
        check(all(np.array_equal(predictions['naive_k5'][k], predictions['stale_row0_k5'][k]) for k in ['actions', 'normalized']),
              'same prediction block when stale input and noise match', key)
        stale = inputs['stale_row0_k5']
        current = inputs['vlash_noise_matched_k5']
        check(any(k.startswith('/robot_state/') for k in stale), 'robot-state fields identified', key)
        check({k:v for k,v in stale.items() if not k.startswith('/robot_state/')} ==
              {k:v for k,v in current.items() if not k.startswith('/robot_state/')}, 'current-state intervention preserves all other input', key)
        check(inputs['vlash_noise_matched_k5'] == inputs['vlash_style_k5'], 'original versus matched noise have same complete input', key)
        check(metadata['vlash_style_k5']['noise_seed']-metadata['vlash_noise_matched_k5']['noise_seed'] == delay,
              'only intended noise-clock offset', key)
        a = predictions['stale_row0_k5']['actions']
        b = predictions['vlash_noise_matched_k5']['actions']
        c = predictions['vlash_style_k5']['actions']
        events.append(dict(task=key[0], state=key[1], horizon=key[2], opportunity=True,
                           robot_state_changed={k:v for k,v in stale.items() if k.startswith('/robot_state/')} !=
                                               {k:v for k,v in current.items() if k.startswith('/robot_state/')},
                           state_alignment_action_max_abs=float(np.max(np.abs(a-b))),
                           noise_clock_action_max_abs=float(np.max(np.abs(b-c)))))
    result = dict(run=args.run, groups=len(groups), first_update_opportunities=sum(r['opportunity'] for r in events),
                  checks=sum(checks.values()), by_kind=dict(checks), errors=len(failures), failures=failures, events=events,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='First controlled update only; later policy trajectories may diverge')
    (ROOT/'checks'/f'{args.run}_noise_alignment.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['events','by_kind','failures']}))
    if failures:
        print(json.dumps(failures[:5]))
    assert not failures


if __name__ == '__main__':
    main()
