"""Summarize the three registered K5 contrasts after the complete audited cohort."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = [
    ('execution_row', 'stale_row0_k5', 'naive_k5'),
    ('current_proprioception', 'vlash_noise_matched_k5', 'stale_row0_k5'),
    ('noise_clock', 'vlash_style_k5', 'vlash_noise_matched_k5'),
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) > 0 and np.isfinite(values).all()
    return dict(n=len(values), median=float(np.median(values)), mean=float(np.mean(values)),
                minimum=float(np.min(values)), maximum=float(np.max(values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', default='noise_alignment_validation_d4_v1')
    args = parser.parse_args()
    assert args.cohort.isascii() and args.cohort.replace('_', '').isalnum()
    source = ROOT/'analysis'/f'{args.cohort}_summary.json'
    data = read(source)
    definition_path = ROOT/'protocol'/args.cohort/'definition.json'
    definition = read(definition_path)
    assert definition['noise_alignment_audit_required']
    assert not definition.get('implementation_pilot', False)
    assert data['definition_sha256'] == sha(definition_path)
    assert data['total_episodes'] == definition['total_episodes']
    first_updates, seen, proofs = [], set(), {}
    for shard in definition['shards']:
        run = shard['run']
        folder = ROOT/'raw'/run/'data'
        proof_path = ROOT/'checks'/f'{run}_noise_alignment.json'
        proof = read(proof_path)
        assert proof['errors'] == 0 and sha(proof_path) == data['audit_sha256'][run+'_noise_alignment']
        assert read(folder.parent/'supervisor.json')['status'] == 'completed'
        assert sha(folder/'episodes.jsonl') == data['source_runs'][run]['episodes_sha256']
        proofs[run] = sha(proof_path)
        for event in proof['events']:
            t, s, h = event['task'], event['state'], event['horizon']
            key = (t, s, h)
            assert key not in seen
            seen.add(key)
            entry = dict(task=t, state=s, horizon=h, opportunity=event['opportunity'], contrasts={})
            if event['opportunity']:
                plans, actions = {}, {}
                for cid in ['naive_k5', 'stale_row0_k5', 'vlash_noise_matched_k5', 'vlash_style_k5']:
                    method = 'naive_k5' if cid == 'naive_k5' else 'vlash_style_k5'
                    case = folder/cid/f't{t:02d}_s{s:02d}_h{h}_{method}'
                    with np.load(case/'chunk_001.npz', allow_pickle=False) as archive:
                        plans[cid] = archive['actions'].copy()
                    call = read(case/'calls.json')[1]
                    row = call['executed_rows'][0]
                    assert call['actual_delivery_tick'] == 5
                    assert row == (definition['logical_delay_steps'] if cid == 'naive_k5' else 0)
                    actions[cid] = plans[cid][row]
                for name, a, b in CONTRASTS:
                    delta = actions[a]-actions[b]
                    entry['contrasts'][name] = dict(
                        translation_command_l2=float(np.linalg.norm(delta[:3])),
                        rotation_command_l2=float(np.linalg.norm(delta[3:6])),
                        gripper_command_absolute=float(abs(delta[6])),
                        same_full_prediction=bool(np.array_equal(plans[a], plans[b])))
                entry['robot_state_changed'] = event['robot_state_changed']
            first_updates.append(entry)
    expected = {(t, s, h) for t in definition['tasks'] for s in definition['states_in_execution_order']
                for h in definition['horizons']}
    assert seen == expected
    effects = {}
    for name, a, b in CONTRASTS:
        by_horizon = {}
        for h in definition['horizons']:
            row = data['groups'][f'{a}_h{h}']
            reference = data['groups'][f'{b}_h{h}']
            pair = row['comparisons'][b]
            first = [r for r in first_updates if r['horizon'] == h and r['opportunity']]
            assert pair['n'] == definition['episodes_per_method_horizon']
            by_horizon[str(h)] = dict(
                candidate_successes=row['successes'], reference_successes=reference['successes'],
                paired=pair, controlled_first_update_opportunities=len(first),
                first_executed_action_differences={field: describe([r['contrasts'][name][field] for r in first])
                    for field in ['translation_command_l2', 'rotation_command_l2', 'gripper_command_absolute']} if first else {},
                same_full_prediction=sum(r['contrasts'][name]['same_full_prediction'] for r in first))
        effects[name] = dict(candidate=a, reference=b, by_horizon=by_horizon)
    result = dict(cohort=args.cohort, summary_sha256=sha(source), definition_sha256=sha(definition_path),
                  noise_audit_sha256=proofs, effects=effects, first_updates=first_updates,
                  interpretation='Three predeclared one-mechanism policy contrasts. First update has a shared physical prefix and audited input/noise controls; later trajectories may diverge. Command differences are controller units, not metres or task quality. Whole-episode effects include downstream changes. No offset training and no claim to reproduce the official VLASH model.')
    output = ROOT/'analysis'/f'{args.cohort}_attribution.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(output=str(output), groups=len(seen), contrasts=list(effects), sha256=sha(output))))


if __name__ == '__main__':
    main()
