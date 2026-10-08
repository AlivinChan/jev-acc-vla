"""Exact technical replay comparison, without aggregating task-success outcomes."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def index(name):
    folder = ROOT/'raw'/name/'data'
    rows = [json.loads(line) for line in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    return {(r['candidate_id'], r['task_id'], r['state_id'], r['horizon']):
            folder/r['candidate_id']/f't{r["task_id"]:02d}_s{r["state_id"]:02d}_h{r["horizon"]}_{r["method"]}' for r in rows}


def gate_signature(path):
    return [{k: r.get(k) for k in ['tick', 'plan_age', 'state', 'model_state', 'choice', 'raw_choice', 'probabilities']}
            for r in read(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('current')
    parser.add_argument('reference')
    parser.add_argument('--candidates', help='Comma-separated subset for unchanged-path technical qualification')
    args = parser.parse_args()
    current, reference = index(args.current), index(args.reference)
    if args.candidates:
        selected = set(args.candidates.split(','))
        current = {k:v for k,v in current.items() if k[0] in selected}
        assert current and {k[0] for k in current} == selected
    assert set(current).issubset(reference)
    records = []
    for key, here in current.items():
        there = reference[key]
        a, b = np.load(here/'trace.npz'), np.load(there/'trace.npz')
        ac, bc = sorted(here.glob('chunk_*.npz')), sorted(there.glob('chunk_*.npz'))
        initial_exact = np.array_equal(np.load(here/'initial_state.npy'), np.load(there/'initial_state.npy'))
        blocks_exact = len(ac) == len(bc) and all(
            all(np.array_equal(np.load(x)[k], np.load(y)[k]) for k in ['actions', 'normalized'])
            for x, y in zip(ac, bc))
        actions_exact = np.array_equal(a['actions'], b['actions'])
        first_difference = next((i for i in range(min(len(a['actions']), len(b['actions'])))
                                 if not np.array_equal(a['actions'][i], b['actions'][i])), None)
        record = dict(candidate=key[0], task=key[1], state=key[2], horizon=key[3], initial_exact=initial_exact,
                      actions_exact=actions_exact, physics_exact=np.array_equal(a['physics'], b['physics']),
                      all_prediction_blocks_exact=blocks_exact,
                      all_gate_inputs_and_answers_exact=gate_signature(here/'gate.json') == gate_signature(there/'gate.json'),
                      first_action_difference_tick=first_difference)
        record['entire_trajectory_exact'] = all(record[k] for k in [
            'initial_exact', 'actions_exact', 'physics_exact', 'all_prediction_blocks_exact', 'all_gate_inputs_and_answers_exact'])
        records.append(record)
    result = dict(current=args.current, reference=args.reference, candidates_filter=args.candidates,
                  scope='Technical replay only; no outcome-based selection, no added efficacy sample',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cases=records, n=len(records), entire_trajectory_exact=sum(r['entire_trajectory_exact'] for r in records))
    path = ROOT/'checks/incidents'/f'{args.current}_vs_{args.reference}.json'
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
