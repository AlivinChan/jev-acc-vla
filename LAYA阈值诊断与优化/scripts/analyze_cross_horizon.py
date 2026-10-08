"""Paired native-horizon effects at fixed policy definitions; never select a winner."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
from analyze_closed_loop import rows_from, summary, paired
from paired_resampling import uncertainty

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = [(100, 50), (200, 50), (200, 100)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_horizon_keys(rows, analysis_horizon=50):
    # H is the intervention here, not part of the pairing identity.
    return [dict(row, source_horizon=row['horizon'], horizon=analysis_horizon) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', required=True)
    args = parser.parse_args()
    assert args.cohort.isascii() and args.cohort.replace('_', '').isalnum()
    recipe_path = ROOT/'protocol/cross_horizon_analysis_v1/definition.json'
    recipe = read(recipe_path)
    assert sha(recipe_path) == read(recipe_path.parent/'freeze_receipt.json')['definition_sha256']
    for name, digest in recipe['source_sha256'].items():
        assert sha(ROOT/name) == digest
    source = ROOT/'analysis'/f'{args.cohort}_summary.json'
    data = read(source)
    definition_path = ROOT/'protocol'/args.cohort/'definition.json'
    definition = read(definition_path)
    assert sha(definition_path) == data['definition_sha256']
    assert data['total_episodes'] == definition['total_episodes']
    assert not definition.get('implementation_pilot', False)
    for name, digest in data['audit_sha256'].items():
        suffix = '.json' if name.endswith(('_initial_verification', '_noise_alignment')) else '_audit.json'
        path = ROOT/'checks'/(name+suffix)
        assert sha(path) == digest and read(path)['errors'] == 0
    groups = defaultdict(list)
    for run, metadata in data['source_runs'].items():
        path = ROOT/'raw'/run/'data/episodes.jsonl'
        assert sha(path) == metadata['episodes_sha256']
        assert read(path.parents[1]/'supervisor.json')['status'] == 'completed'
        for row in rows_from(path):
            groups[(row['candidate_id'], row['horizon'])].append(row)
    expected = {(t, s) for t in definition['tasks'] for s in definition['states_in_execution_order']}
    result = dict(cohort=args.cohort, summary_sha256=sha(source), definition_sha256=sha(definition_path),
                  recipe_sha256=sha(recipe_path), primary_length_control='fixed45', methods={})
    for cid, candidate in definition['candidates'].items():
        if set(candidate.get('horizons', definition['horizons'])) != {50, 100, 200}:
            continue
        fixed_cadence = candidate['method'] in ['fixed_interval', 'naive_k5', 'vlash_style_k5']
        entry = dict(request_rule=f'Fixed interval {candidate.get("interval", 5)} steps' if fixed_cadence else
                     'H also changes the request or gate schedule', comparisons={})
        for larger, smaller in CONTRASTS:
            a, b = groups[(cid, larger)], groups[(cid, smaller)]
            for rows in [a, b]:
                assert len(rows) == len(expected) and {(r['task_id'], r['state_id']) for r in rows} == expected
            amap = {(r['task_id'], r['state_id']): r for r in a}
            bmap = {(r['task_id'], r['state_id']): r for r in b}
            assert all(amap[k]['initial_state_sha256'] == bmap[k]['initial_state_sha256'] for k in expected)
            aa, bb = match_horizon_keys(a), match_horizon_keys(b)
            pair = paired(aa, bb)
            pair['paired_uncertainty'] = uncertainty(aa, bb)
            entry['comparisons'][f'h{larger}_vs_h{smaller}'] = dict(
                candidate_horizon=larger, reference_horizon=smaller,
                candidate=summary(a), reference=summary(b), paired=pair)
        result['methods'][cid] = entry
    assert 'fixed45' in result['methods']
    result['interpretation'] = 'Same task/initial pairs; fixed45 isolates native H at a fixed request interval. Other H-dependent schedules are joint interventions. Descriptive uncertainty only; no equivalence or model-selection claim.'
    output = ROOT/'analysis'/f'{args.cohort}_cross_horizon.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(cohort=args.cohort, methods=len(result['methods']), output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
