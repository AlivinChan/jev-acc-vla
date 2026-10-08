"""Report the two already-identified replay discrepancies after full v2 completion."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    status = json.loads((ROOT/'runtime/validation_d1_v2_chain.json').read_text(encoding='utf-8'))
    assert status['phase'] == 'completed', 'Do not inspect efficacy while the main cohort is incomplete'
    pairs = [(12, 9, 50, 'vlash_style_k5'), (10, 0, 200, 'fixed45')]
    result = []
    for state, task, horizon, method in pairs:
        item = dict(state_id=state, task_id=task, horizon=horizon, candidate_id=method, runs={})
        for version in [1, 2]:
            name = f'validation_d1_s{state}_v{version}'
            path = ROOT/'raw'/name/'data/episodes.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            selected = [r for r in rows if r['state_id'] == state and r['task_id'] == task and
                        r['horizon'] == horizon and r['candidate_id'] == method]
            assert len(selected) == 1
            r = selected[0]
            item['runs'][name] = {key:r[key] for key in ['success', 'steps', 'vla_calls', 'initial_state_sha256']}
            item['runs'][name]['episodes_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        a, b = list(item['runs'].values())
        assert a['initial_state_sha256'] == b['initial_state_sha256']
        item['success_indicator_changed'] = a['success'] != b['success']
        result.append(item)
    output = dict(cases=result, interpretation='Exactly the two discrepancies identified before efficacy analysis. This describes their endpoints, does not locate their causes or establish determinism. All old v1 episodes remain excluded; no outcome-based replacement.')
    path = ROOT/'checks/incidents/known_replay_difference_outcomes.json'
    path.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
