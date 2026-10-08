"""Freeze the predeclared developer-cost period comparison, without reading validation outcomes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from local_policy import allowed

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    allowed()
    cohort = 'budget_validation_d1_v1'
    folder = ROOT / 'protocol' / cohort
    assert not folder.exists(), 'Never overwrite an existing frozen cohort'
    periods_path = ROOT / 'protocol/budget_period_controls_v1/definition.json'
    assert sha(periods_path) == '91cabd60b255f58d834a6189a915266c7fa742843991eb8b6d1f69fb782e7fb4'
    periods = read(periods_path)
    check = read(ROOT / 'checks/budget_period_definition.json')
    assert check['errors'] == 0 and check['definition_sha256'] == sha(periods_path)
    prior_path = ROOT / 'protocol/validation_d1_v1/definition.json'
    assert sha(prior_path) == read(prior_path.parent / 'freeze_receipt.json')['definition_sha256']
    prior = read(prior_path)
    locked = prior['locked_execution_source_sha256']
    for name, digest in locked.items():
        assert sha(ROOT / 'scripts' / name) == digest
    candidates = {name: prior['candidates'][name] for name in ['smol70', 'fixed45', 'window3_skip1']}
    comparators = {}
    for h in [50, 100, 200]:
        cid = f'budget_period_h{h}'
        candidates[cid] = dict(method='fixed_interval', interval=periods['candidates'][str(h)]['interval'],
                               horizons=[h], role='developer-cost-selected fixed-period control')
        comparators[str(h)] = ['smol70', 'fixed45', cid]
    states = [22, 20, 24, 21, 23]
    configs = []
    for index, sid in enumerate(states):
        run = f'budget_d1_s{sid}_v1'
        path = ROOT / 'configs' / f'{run}.json'
        assert not path.exists()
        cfg = dict(tasks=list(range(10)), states=[sid], horizons=[50, 100, 200],
                   logical_delay_steps=1, schedule_seed=202610089, cohort_id=cohort,
                   cohort_role='Predeclared cost attribution cohort; no validation-dependent period selection',
                   candidates=candidates, locked_source_sha256=locked)
        count = sum(1 for t in cfg['tasks'] for h in cfg['horizons'] for c in candidates.values()
                    if h in c.get('horizons', cfg['horizons']))
        assert count == 120
        configs.append((path, cfg, dict(run=run, config=path.name, state=sid, episodes=count, order=index)))
    folder.mkdir(parents=True)
    shards = []
    for path, cfg, meta in configs:
        path.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8', newline='\n')
        shards.append(dict(**meta, config_sha256=sha(path)))
    evidence = ['BUDGET_PERIOD_PROTOCOL.md', 'protocol/budget_period_controls_v1/definition.json',
                'protocol/validation_d1_v1/definition.json', 'checks/budget_period_definition.json',
                'STATISTICS_AMENDMENT_01.md']
    definition = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), cohort_id=cohort,
        tasks=list(range(10)), states_in_execution_order=states, horizons=[50, 100, 200],
        episodes_per_method_horizon=50, total_episodes=600, candidates=candidates, shards=shards,
        primary_candidate='window3_skip1', primary_comparators=['same-H budget_period', 'smol70', 'fixed45'],
        comparators_by_horizon=comparators, locked_execution_source_sha256=locked,
        selection_evidence_sha256={n: sha(ROOT / n) for n in evidence}, vla_sha256=prior['vla_sha256'],
        minimum_probability=0, binary_argmax=True, readout_adaptation=False, logical_delay_steps=1,
        statistics=prior['statistics'], stop_policy=prior['stop_policy'],
        no_adaptation_during_cohort=True, no_early_stopping_for_favorable_results=True,
        interpretation='Periods selected only from state0-2 developer durations and service costs. Actual new-rollout costs need not match. Do not retune periods or claim exact budget equality. Earlier project exposure to public starts is not excluded.')
    path = folder / 'definition.json'
    path.write_text(json.dumps(definition, indent=2) + '\n', encoding='utf-8', newline='\n')
    receipt = dict(definition_sha256=sha(path), immutable=True)
    (folder / 'freeze_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(cohort=cohort, episodes=600, definition_sha256=sha(path), shards=shards)))


if __name__ == '__main__':
    main()
