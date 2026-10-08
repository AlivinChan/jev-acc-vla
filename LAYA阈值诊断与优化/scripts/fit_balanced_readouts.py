"""Four fixed class-balanced readouts: state0 fit, state1 selection, no state2 read."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from calibration_data import load_runs, select
from calibration_math import metrics, predict
from balanced_readout_math import fit
from local_policy import allowed

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    allowed()
    protocol = ROOT/'BALANCED_READOUT_PROTOCOL.md'
    check = json.loads((ROOT/'checks/balanced_readout_math.json').read_text(encoding='utf-8'))
    assert check['errors'] == 0 and check['module_sha256'] == sha(ROOT/'scripts/balanced_readout_math.py')
    out = ROOT/'models/balanced_readouts_v1'
    assert not out.exists(), 'Do not overwrite a fitted or selected model generation'
    names = ['scores_train_v1', 'scores_calibration_v1', 'scores_window_train_v1', 'scores_window_calibration_v1']
    variants = ['numbers3', 'binary_short', 'binary_compact', 'binary_window', 'binary_window3']
    rows, features, provenance = load_runs(names, {0,1}, variants)
    train = select(rows, features, 'binary_window3', 'utility', {0})
    cal = select(rows, features, 'binary_window3', 'utility', {1})
    assert len({r['temperature'] for r in train['rows']+cal['rows']}) == 1
    raw = dict(kind='raw', temperature=train['rows'][0]['temperature'])
    raw_metrics = metrics(cal['y'], predict(raw, cal['z']))
    first_path = ROOT/'analysis/first_intervention_development.json'
    first_source = json.loads(first_path.read_text(encoding='utf-8'))
    assert set(first_source['states']) == {0,1}
    first = [r for r in first_source['cases'] if r['state'] == 1]
    assert len(first) == 30
    indices = [i for i, r in enumerate(rows) if r['state'] == 1 and r['variant'] == 'binary_window3']
    assert len(indices) == 197
    selected_rows = [rows[i] for i in indices]

    def branches(probabilities=None):
        predictions = ({r['id']: ('replan' if p >= .5 else 'continue') for r,p in zip(selected_rows, probabilities)}
                       if probabilities is not None else None)
        groups = {}
        for h in [50,100,200]:
            outcomes = []
            for case in first:
                if case['horizon'] != h:
                    continue
                if not case['opportunity']:
                    outcomes.append(case['choices']['always_replan'])
                elif predictions is None:
                    outcomes.append(case['choices']['binary_window3'])
                else:
                    outcomes.append(case['outcomes'][predictions[case['id']]])
            assert len(outcomes) == 10
            groups[str(h)] = dict(episodes=10, successes=sum(r['success'] for r in outcomes),
                                  vla_calls=sum(r['calls'] for r in outcomes))
        return groups

    original_branches = branches()
    models, results, eligible = {}, {}, []
    for penalty in [.01,.1,1.,10.]:
        allowed()
        model = fit(train['features'], train['y'], penalty)
        model_id = 'utility__binary_window3__balanced_l2_'+str(penalty).replace('.', 'p')
        models[model_id] = dict(target='utility', variant='binary_window3', model=model)
        measure = metrics(cal['y'], predict(model, cal['z'], cal['features'], cal['horizons']))
        p = predict(model, np.array([r['raw_logit_delta'] for r in selected_rows]), features[indices],
                    np.array([r['horizon'] for r in selected_rows]))
        first_result = branches(p)
        total_calls = sum(r['vla_calls'] for r in first_result.values())
        conditions = dict(replan_recall_not_lower=measure['replan_recall'] >= raw_metrics['replan_recall'],
            balanced_accuracy_not_lower=measure['balanced_accuracy'] >= raw_metrics['balanced_accuracy'],
            both_recalls_nonzero=measure['replan_recall'] > 0 and measure['continue_recall'] > 0,
            first_success_not_lower_each_h=all(first_result[str(h)]['successes'] >= original_branches[str(h)]['successes'] for h in [50,100,200]),
            first_total_calls_strictly_lower=total_calls < sum(r['vla_calls'] for r in original_branches.values()))
        accepted = all(conditions.values())
        results[model_id] = dict(train=metrics(train['y'], predict(model, train['z'], train['features'], train['horizons'])),
            calibration=measure, first_intervention=first_result, eligibility_conditions=conditions, eligible=accepted)
        if accepted:
            eligible.append((total_calls, -measure['balanced_accuracy'], -penalty, model_id))
        print(json.dumps(dict(model=model_id, calibration=measure, first_intervention=first_result, conditions=conditions, eligible=accepted)))
    selected = [r[-1] for r in sorted(eligible)[:1]]
    out.mkdir(parents=True)
    model_path = out/'models.json'
    model_path.write_text(json.dumps(models, indent=2)+'\n', encoding='utf-8', newline='\n')
    results_path = out/'development_scores.json'
    results_path.write_text(json.dumps(dict(raw_calibration=raw_metrics, raw_first_intervention=original_branches,
                                           results=results), indent=2)+'\n', encoding='utf-8', newline='\n')
    manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(), source=provenance, variants=variants,
                    training_states=[0], selection_and_calibration_states=[1], diagnostic_evaluation_states=[2],
                    state2_loaded=False, VLA_training_steps=0, base_LAYA_training_steps=0,
                    training_target='utility', candidate_variants=['binary_window3'], readout_l2_grid=[.01,.1,1.,10.],
                    model_count=4, models_sha256=sha(model_path), results_sha256=sha(results_path),
                    proposed_deployment_models=selected,
                    selection_rule='All five predeclared conditions; then fewer first-intervention calls, greater balanced accuracy, stronger regularization',
                    class_balance_is_not_probability_calibration=True, protocol_sha256=sha(protocol),
                    source_first_intervention_sha256=sha(first_path), fit_script_sha256=sha(Path(__file__)),
                    fit_math_sha256=sha(ROOT/'scripts/balanced_readout_math.py'),
                    interpretation='State1 is a reused development selection set. State2 is not read by this fitter; future diagnosis cannot be called an untouched project test.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(proposed_deployment_models=selected, state2_seen=False, raw_calibration=raw_metrics,
                          raw_first_intervention=original_branches, models_sha256=sha(model_path))))


if __name__ == '__main__':
    main()
