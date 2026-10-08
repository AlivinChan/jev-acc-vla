"""Fixed order-invariant logit average on existing state0 branches; no learned parameters."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
from scipy.special import expit
from calibration_math import metrics
from local_policy import allowed

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    allowed()
    assert read(ROOT/'checks/choice_order_audit_v1_audit.json')['errors'] == 0
    score_path = ROOT/'raw/scores_window_train_v1/data/scores.jsonl'
    scores = {r['id']: r for r in lines(score_path) if r['variant'] == 'binary_window3'}
    assert len(scores) == 189 and all(r['state'] == 0 for r in scores.values())
    fixture_path = ROOT/'fixtures/choice_order_audit_v1.json'
    fixtures = {r['id']: r for r in read(fixture_path) if r['variant'] == 'binary_window3'}
    decisions_path = ROOT/'raw/choice_order_audit_v1/data/decisions.jsonl'
    pairs = defaultdict(dict)
    for r in lines(decisions_path):
        if r['variant'] == 'binary_window3':
            pairs[fixtures[r['id']]['source_record_id']][tuple(r['order'])] = r
    assert set(pairs) == set(scores)
    records = []
    methods = ['original', 'reversed', 'symmetric', 'always_replan', 'always_continue']
    for cid, pair in pairs.items():
        r = scores[cid]
        assert set(pair) == {(0, 1), (1, 0)}
        a, b = pair[(0, 1)], pair[(1, 0)]
        assert a['model_state_sha256'] == b['model_state_sha256'] == r['model_state_sha256']
        assert a['temperature'] == b['temperature'] == r['temperature']
        za = a['slot_logits'][0]-a['slot_logits'][1]
        zb = b['slot_logits'][1]-b['slot_logits'][0]
        p = dict(original=a['canonical_unrounded_probabilities']['replan'],
                 reversed=b['canonical_unrounded_probabilities']['replan'],
                 symmetric=float(expit((za+zb)/(2*a['temperature']))), always_replan=1., always_continue=0.)
        assert abs(float(expit(za/a['temperature']))-p['original']) < 1e-12
        assert abs(float(expit(zb/a['temperature']))-p['reversed']) < 1e-12
        choices = dict(original=a['choice'], reversed=b['choice'],
                       symmetric='replan' if p['symmetric'] >= .5 else 'continue',
                       always_replan='replan', always_continue='continue')
        records.append(dict(id=cid, task=r['task'], state=0, horizon=r['horizon'], tick=r['tick'],
                            label=r['labels']['necessity'], probability=p, choices=choices,
                            outcomes=r['outcomes'], symmetric_sdk_seconds=a['sdk_seconds']+b['sdk_seconds'],
                            order_invariant=bool(expit((za+zb)/(2*a['temperature'])) == expit((zb+za)/(2*a['temperature'])))))
    assert all(r['order_invariant'] for r in records)
    resolved = [r for r in records if r['label'] is not None]
    y = np.array([r['label'] == 'replan' for r in resolved], dtype=int)
    classification = {name: metrics(y, [r['probability'][name] for r in resolved]) for name in methods}
    opportunities = {}
    for name in methods:
        selected = [r['outcomes'][r['choices'][name]] for r in records]
        opportunities[name] = dict(opportunities=len(records), successful_branches=sum(s['success'] for s in selected),
                                    branch_vla_calls=sum(s['vla_calls'] for s in selected),
                                    continue_choices=sum(r['choices'][name] == 'continue' for r in records))
    first_path = ROOT/'analysis/first_intervention_development.json'
    first = [r for r in read(first_path)['cases'] if r['state'] == 0]
    assert len(first) == 30 and len({(r['task'], r['horizon']) for r in first}) == 30
    by_id = {r['id']: r for r in records}
    first_results = {}
    for name in methods:
        for h in [50, 100, 200]:
            selected = []
            for r in first:
                if r['horizon'] != h:
                    continue
                if r['opportunity']:
                    case = by_id[r['id']]
                    assert case['tick'] == round(.3*h)
                    original = r['choices']['binary_window3']
                    assert original['choice'] == case['choices']['original']
                    selected.append(r['outcomes'][case['choices'][name]])
                else:
                    selected.append(r['choices']['always_replan'])
            assert len(selected) == 10
            first_results[f'{name}_h{h}'] = dict(episodes=10, successes=sum(s['success'] for s in selected),
                                                vla_calls=sum(s['calls'] for s in selected))
    conditions = dict(
        success_not_lower_each_h=all(first_results[f'symmetric_h{h}']['successes'] >= first_results[f'original_h{h}']['successes'] for h in [50,100,200]),
        total_vla_calls_strictly_lower=sum(first_results[f'symmetric_h{h}']['vla_calls'] for h in [50,100,200]) < sum(first_results[f'original_h{h}']['vla_calls'] for h in [50,100,200]),
        replan_recall_not_lower=classification['symmetric']['replan_recall'] >= classification['original']['replan_recall'],
        balanced_accuracy_not_lower=classification['symmetric']['balanced_accuracy'] >= classification['original']['balanced_accuracy'])
    sources = [score_path, fixture_path, decisions_path, first_path, ROOT/'ORDER_ENSEMBLE_DEVELOPMENT.md']
    result = dict(states=[0], fitting=False, primary_validation_unchanged=True, inputs=189,
                  resolved=len(resolved), classification=classification, opportunities=opportunities,
                  first_request_results=first_results, eligibility_conditions=conditions,
                  eligible_for_next_development=all(conditions.values()),
                  symmetric_sdk_seconds_mean=float(np.mean([r['symmetric_sdk_seconds'] for r in records])),
                  records=records, source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},
                  interpretation='State0 exploratory branch selection, not independent validation or measured full-controller speedup. Paired branches use a fixed-noise realization. No deployment IPC fees are available for this ensemble.')
    path = ROOT/'analysis/order_ensemble_state0_development.json'
    assert not path.exists(), 'Keep the first recorded fixed-formula evaluation'
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records', 'source_sha256']}))


if __name__ == '__main__':
    main()
