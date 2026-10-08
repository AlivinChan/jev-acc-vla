"""Predeclared paired d5-versus-d4 sensitivity, only after both complete cohorts."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
from analyze_closed_loop import rows_from, summary, paired
from paired_resampling import uncertainty

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cohort(name, expected_delay):
    source = ROOT/'analysis'/f'{name}_summary.json'
    data = read(source)
    definition_path = ROOT/'protocol'/name/'definition.json'
    definition = read(definition_path)
    assert sha(definition_path) == data['definition_sha256']
    assert data['total_episodes'] == definition['total_episodes'] == 450
    assert definition['logical_delay_steps'] == expected_delay
    assert read(ROOT/'runtime'/f'{name}_chain.json')['phase'] == 'completed'
    for audit_name, digest in data['audit_sha256'].items():
        suffix = '.json' if audit_name.endswith('_initial_verification') else '_audit.json'
        path = ROOT/'checks'/(audit_name+suffix)
        assert sha(path) == digest and read(path)['errors'] == 0
    groups = defaultdict(list)
    for run, metadata in data['source_runs'].items():
        path = ROOT/'raw'/run/'data/episodes.jsonl'
        assert sha(path) == metadata['episodes_sha256']
        for row in rows_from(path):
            assert row['logical_delay_steps'] == expected_delay
            groups[(row['candidate_id'], row['horizon'])].append(row)
    assert sum(map(len, groups.values())) == 450
    return data, definition, groups, sha(source)


def main():
    recipe_path = ROOT/'protocol/cross_delay_analysis_v1/definition.json'
    recipe = read(recipe_path)
    assert sha(recipe_path) == read(recipe_path.parent/'freeze_receipt.json')['definition_sha256']
    for name, digest in recipe['source_sha256'].items():
        assert sha(ROOT/name) == digest
    a, adef, aa, asha = load_cohort(recipe['candidate_cohort'], 5)
    b, bdef, bb, bsha = load_cohort(recipe['reference_cohort'], 4)
    assert adef['candidates'] == bdef['candidates']
    assert adef['tasks'] == bdef['tasks'] and adef['states_in_execution_order'] == bdef['states_in_execution_order']
    assert set(aa) == set(bb)
    expected = {(t,s) for t in adef['tasks'] for s in adef['states_in_execution_order']}
    comparisons = {}
    for (cid,h), rows in sorted(aa.items()):
        refs = bb[(cid,h)]
        assert len(rows) == len(refs) == len(expected) == 30
        assert {(r['task_id'],r['state_id']) for r in rows} == expected
        assert {(r['task_id'],r['state_id']) for r in refs} == expected
        contrast = paired(rows,refs)
        contrast['paired_uncertainty'] = uncertainty(rows,refs)
        comparisons[f'{cid}_h{h}'] = dict(candidate_d5=summary(rows), reference_d4=summary(refs), paired=contrast)
    assert len(comparisons) == 15
    result = dict(recipe_sha256=sha(recipe_path), candidate_cohort=recipe['candidate_cohort'],
                  reference_cohort=recipe['reference_cohort'], candidate_summary_sha256=asha,
                  reference_summary_sha256=bsha, comparisons=comparisons,
                  interpretation=recipe['interpretation'])
    path = ROOT/'analysis/cross_delay_d5_vs_d4.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(comparisons=len(comparisons), output=str(path), sha256=sha(path))))


if __name__ == '__main__':
    main()
