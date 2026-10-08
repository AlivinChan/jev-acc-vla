"""Exact cost accounting and predeclared fixed-price sensitivity; no policy fitting."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
from analyze_closed_loop import rows_from, summary, paired
from paired_resampling import uncertainty

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factors(row):
    steps, calls = row['steps'], row['vla_calls']
    vla, gate = row['vla_seconds'], row['gate_seconds']
    assert steps > 0 and calls > 0 and vla > 0 and gate >= 0
    result = dict(environment_steps=steps, vla_calls_per_environment_step=calls/steps,
                  mean_vla_service_seconds=vla/calls, gate_multiplier=1+gate/vla)
    reconstructed = math.prod(result.values())
    assert math.isclose(reconstructed, row['total_service_seconds'], rel_tol=1e-12)
    return result


def decompose(row, reference):
    first, second = factors(row), factors(reference)
    ratios = {key: first[key]/second[key] for key in first}
    product = math.prod(ratios.values())
    observed = row['total_service_seconds']/reference['total_service_seconds']
    assert math.isclose(product, observed, rel_tol=1e-12)
    return dict(factor_ratios=ratios, reconstructed_cost_ratio=product,
                measured_cost_ratio=observed, absolute_reconstruction_error=abs(product-observed))


def fixed_rows(rows, price):
    result = copy.deepcopy(rows)
    for row in result:
        row['prediction_seconds'] = row['vla_calls']*price['vla_seconds_by_horizon'][str(row['horizon'])]
        row['gate_seconds'] = row['gate_calls']*price['window3_gate_seconds']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', required=True)
    args = parser.parse_args()
    assert args.cohort.isascii() and args.cohort.replace('_', '').isalnum()
    recipe_path = ROOT/'protocol/service_accounting_v2/definition.json'
    recipe = read(recipe_path)
    assert sha(recipe_path) == read(recipe_path.parent/'freeze_receipt.json')['definition_sha256']
    assert sha(Path(__file__)) == recipe['analyzer_sha256']
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
        if name.endswith('_initial_verification') or name.endswith('_noise_alignment'):
            audit_path = ROOT/'checks'/f'{name}.json'
        else:
            audit_path = ROOT/'checks'/f'{name}_audit.json'
        assert sha(audit_path) == digest and read(audit_path)['errors'] == 0
    groups = {}
    for run, metadata in data['source_runs'].items():
        path = ROOT/'raw'/run/'data/episodes.jsonl'
        assert sha(path) == metadata['episodes_sha256']
        assert read(path.parents[1]/'supervisor.json')['status'] == 'completed'
        for row in rows_from(path):
            key = f"{row['candidate_id']}_h{row['horizon']}"
            groups.setdefault(key, []).append(row)
    assert set(groups) == set(data['groups'])
    result = dict(cohort=args.cohort, recipe_sha256=sha(recipe_path), summary_sha256=sha(source),
                  prices=recipe['prices'], groups={}, interpretation=recipe['interpretation'])
    for key, row in data['groups'].items():
        actual = summary(groups[key])
        assert actual == {k: row[k] for k in actual}
        h = groups[key][0]['horizon']
        candidate = definition['candidates'][groups[key][0]['candidate_id']]
        fixed_eligible = not candidate['method'].startswith('laya') or groups[key][0]['candidate_id'] == 'window3_skip1'
        entry = dict(factors=factors(row), calls_per_100_environment_steps=100*row['vla_calls']/row['steps'],
                     service_seconds_per_simulated_second=20*row['total_service_seconds']/row['steps'],
                     measured_comparisons={}, fixed_price_comparisons={})
        for reference, comparison in row['comparisons'].items():
            refkey = f'{reference}_h{h}'
            ref = data['groups'][refkey]
            entry['measured_comparisons'][reference] = decompose(row, ref)
            assert math.isclose(comparison['total_service_cost_ratio'],
                                entry['measured_comparisons'][reference]['measured_cost_ratio'], rel_tol=1e-12)
            refcandidate = definition['candidates'][reference]
            reference_eligible = not refcandidate['method'].startswith('laya') or reference == 'window3_skip1'
            if fixed_eligible and reference_eligible:
                a, b = fixed_rows(groups[key], recipe['prices']), fixed_rows(groups[refkey], recipe['prices'])
                fixed = paired(a, b)
                fixed['paired_uncertainty'] = uncertainty(a, b)
                entry['fixed_price_comparisons'][reference] = fixed
        result['groups'][key] = entry
    output = ROOT/'analysis'/f'{args.cohort}_service_accounting.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(cohort=args.cohort, groups=len(groups), output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
