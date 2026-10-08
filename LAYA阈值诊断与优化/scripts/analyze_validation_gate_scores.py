"""Descriptive score distribution of the complete frozen primary validation policy."""
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


def main():
    source = ROOT/'analysis'/f'{COHORT}_summary.json'
    data = read(source)
    assert read(ROOT/'runtime'/f'{COHORT}_chain.json')['phase'] == 'completed'
    assert data['total_episodes'] == 1050 and data['primary_candidate'] == 'window3_skip1'
    groups, source_hashes = {h:[] for h in [50, 100, 200]}, {}
    for run, metadata in data['source_runs'].items():
        base = ROOT/'raw'/run/'data'
        assert sha(base/'episodes.jsonl') == metadata['episodes_sha256']
        for path in (base/'window3_skip1').glob('*/gate.json'):
            result = read(path.parent/'result.json')
            rows = read(path)
            assert all(set(r['probabilities']) == {'continue', 'replan'} for r in rows)
            groups[result['horizon']].extend(rows)
            source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    output_groups = {}
    for h, rows in groups.items():
        assert len(rows) == data['groups'][f'window3_skip1_h{h}']['gate_calls']
        values = np.asarray([max(r['probabilities'].values()) for r in rows])
        assert np.isfinite(values).all() and ((values >= .5) & (values <= 1)).all()
        output_groups[str(h)] = dict(judgments=len(rows), max_probability_min=float(values.min()),
            max_probability_median=float(np.median(values)), max_probability_max=float(values.max()),
            at_least_07=int((values >= .7).sum()), choices=dict(Counter(r['choice'] for r in rows)))
    result = dict(cohort=COHORT,
                  role='Descriptive frozen-policy score audit after cohort completion; correlated judgments are not independent task samples or correctness labels. No threshold changed.',
                  summary_sha256=sha(source), groups=output_groups, script_sha256=sha(Path(__file__)),
                  gate_file_sha256=source_hashes,
                  rounding_note='SDK-returned probabilities use four decimals; ties at 0.5000 can conceal smaller unrounded logit differences. Original choice/argmax checks remain in the execution audits.')
    path = ROOT/'analysis'/f'{COHORT}_score_distribution.json'
    if path.exists():
        assert read(path)['groups'] == output_groups, 'Never silently change existing score statistics'
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(cohort=COHORT, groups=output_groups)))


if __name__ == '__main__':
    main()
