"""Post-hoc all-replan and pre-continue trace equality; no policy selection."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COHORT = 'validation_d1_v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_difference(a, b):
    common = min(len(a), len(b))
    changed = next((i for i in range(common) if not np.array_equal(a[i], b[i])), None)
    return changed if changed is not None else (common if len(a) != len(b) else None)


def blocks_equal(a, b):
    aa, bb = sorted(a.glob('chunk_*.npz')), sorted(b.glob('chunk_*.npz'))
    if len(aa) != len(bb):
        return False
    for x, y in zip(aa, bb):
        with np.load(x) as p, np.load(y) as q:
            if any(not np.array_equal(p[k], q[k]) for k in ['actions', 'normalized']):
                return False
    return True


def main():
    source = ROOT/'analysis'/f'{COHORT}_summary.json'
    data = read(source)
    assert read(ROOT/'runtime'/f'{COHORT}_chain.json')['phase'] == 'completed'
    assert data['total_episodes'] == 1050
    cases, violations = [], []
    for run, metadata in data['source_runs'].items():
        base = ROOT/'raw'/run/'data'
        assert sha(base/'episodes.jsonl') == metadata['episodes_sha256']
        rows = [json.loads(line) for line in (base/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        lookup = {(r['candidate_id'], r['task_id'], r['state_id'], r['horizon']):r for r in rows}
        for key, row in lookup.items():
            cid, task, state, h = key
            if cid != 'window3_skip1':
                continue
            ref = lookup[('smol70', task, state, h)]
            assert row['initial_state_sha256'] == ref['initial_state_sha256']
            def folder(r):
                return base/r['candidate_id']/f't{task:02d}_s{state:02d}_h{h}_{r["method"]}'
            here, there = folder(row), folder(ref)
            gate = read(here/'gate.json')
            assert Counter(g['choice'] for g in gate) == Counter(row['gate_choices'])
            ticks = [g['tick'] for g in gate if g['choice'] == 'continue']
            with np.load(here/'trace.npz') as a, np.load(there/'trace.npz') as b:
                action_first = first_difference(a['actions'], b['actions'])
                physics_first = first_difference(a['physics'], b['physics'])
                assert len(a['actions']) == row['steps'] and len(b['actions']) == ref['steps']
                prefix_end = min(ticks)+row['logical_delay_steps'] if ticks else len(a['actions'])
                prefix_end = min(prefix_end, len(a['actions']))
                prefix_exact = len(b['actions']) >= prefix_end and all(
                    np.array_equal(a[k][:prefix_end], b[k][:prefix_end]) for k in ['actions', 'physics'])
            calls, refcalls = read(here/'calls.json'), read(there/'calls.json')
            requests_equal = [c['request_tick'] for c in calls] == [c['request_tick'] for c in refcalls]
            no_continue_blocks_equal = blocks_equal(here, there) if not ticks else None
            full_trace_exact = action_first is None and physics_first is None
            record = dict(run=run, task=task, state=state, horizon=h, gate_calls=len(gate),
                          continue_choices=len(ticks), first_continue_tick=min(ticks) if ticks else None,
                          compared_pre_effect_prefix_steps=prefix_end, pre_effect_prefix_exact=prefix_exact,
                          first_action_difference_tick=action_first, first_physics_difference_tick=physics_first,
                          full_executed_trace_exact=full_trace_exact, request_ticks_exact=requests_equal,
                          no_continue_prediction_blocks_exact=no_continue_blocks_equal,
                          laya_success=row['success'], smol70_success=ref['success'],
                          outcome='win' if row['success'] and not ref['success'] else
                          'loss' if ref['success'] and not row['success'] else 'tie',
                          sources={p.relative_to(ROOT).as_posix():sha(p) for p in
                                   [here/'trace.npz', there/'trace.npz', here/'gate.json',
                                    here/'calls.json', there/'calls.json']})
            if not prefix_exact or (not ticks and not all([full_trace_exact, requests_equal, no_continue_blocks_equal])):
                violations.append({k:record[k] for k in record if k != 'sources'})
            cases.append(record)
    assert len(cases) == 150
    totals = {}
    for h in [50, 100, 200]:
        rows = [r for r in cases if r['horizon'] == h]
        no_c = [r for r in rows if not r['continue_choices']]
        totals[str(h)] = dict(n=len(rows), episodes_with_continue=sum(r['continue_choices'] > 0 for r in rows),
                             continue_choices=sum(r['continue_choices'] for r in rows),
                             no_continue_episodes=len(no_c),
                             no_continue_full_trace_exact=sum(r['full_executed_trace_exact'] for r in no_c),
                             pre_effect_prefix_exact=sum(r['pre_effect_prefix_exact'] for r in rows),
                             full_trace_exact=sum(r['full_executed_trace_exact'] for r in rows),
                             outcomes=dict(Counter(r['outcome'] for r in rows)),
                             no_continue_outcomes=dict(Counter(r['outcome'] for r in no_c)))
    output = dict(cohort=COHORT, posthoc=True, summary_sha256=sha(source),
                  protocol_sha256=sha(ROOT/'POSTHOC_GATE_TRACE_AUDIT.md'), script_sha256=sha(Path(__file__)),
                  totals=totals, violations=violations, cases=cases,
                  interpretation='All 150 primary pairs included. Endpoint changes after first CONTINUE are not labels for individual gates. This diagnostic neither refits a policy nor establishes universal numerical determinism.')
    path = ROOT/'checks/validation_d1_v2_gate_trace_equivalence.json'
    path.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(n=len(cases), violations=len(violations), totals=totals)))


if __name__ == '__main__':
    main()
