"""Lock both delay sensitivities and their qualification stages before seeing their results."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from local_policy import allowed

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    assert not path.exists(), 'Frozen files are immutable'
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    allowed()
    prior_path = ROOT/'protocol/validation_d1_v1/definition.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    assert sha(prior_path) == json.loads((prior_path.parent/'freeze_receipt.json').read_text(encoding='utf-8'))['definition_sha256']
    locked = prior['locked_execution_source_sha256']
    for name, expected in locked.items():
        assert sha(ROOT/'scripts'/name) == expected
    qualification = json.loads((ROOT/'checks/delay_d1_qualification_v1_exact_reproduction.json').read_text(encoding='utf-8'))
    assert qualification['errors'] == 0 and qualification['episodes'] == 9
    mechanisms = json.loads((ROOT/'checks/controller_mechanisms.json').read_text(encoding='utf-8'))
    assert len(mechanisms) == 21 and all(r['passed'] for r in mechanisms)
    names = ['smol70', 'naive_k5', 'vlash_style_k5', 'fixed45', 'window3_skip1']
    candidates = {n:prior['candidates'][n] for n in names}
    evidence = ['DELAY_ROBUSTNESS_PROTOCOL.md', 'DELAY_COHORT_LOCK.md',
                'protocol/validation_d1_v1/definition.json',
                'checks/delay_d1_qualification_v1_exact_reproduction.json',
                'checks/controller_mechanisms.json', 'STATISTICS_AMENDMENT_01.md']
    proposals = []
    for delay in [4, 5]:
        for pilot in [True, False]:
            cohort = f'delay_{"pilot" if pilot else "validation"}_d{delay}_v1'
            folder = ROOT/'protocol'/cohort
            assert not folder.exists()
            tasks = [0] if pilot else list(range(10))
            states = [0] if pilot else [31, 30, 32]
            configs = []
            for index, sid in enumerate(states):
                run = f'delay_d{delay}_pilot_v1' if pilot else f'delay_d{delay}_s{sid}_v1'
                path = ROOT/'configs'/f'{run}.json'
                assert not path.exists()
                cfg = dict(tasks=tasks, states=[sid], horizons=[50, 100, 200], logical_delay_steps=delay,
                    schedule_seed=202610091, cohort_id=cohort,
                    cohort_role='Exposed-start implementation pilot; not new efficacy evidence' if pilot else 'Predeclared logical delay sensitivity; fixed primary policy, no selection on d4/d5 results',
                    candidates=candidates, locked_source_sha256=locked)
                count = len(tasks)*3*len(candidates)
                configs.append((path, cfg, dict(run=run, config=path.name, state=sid, episodes=count, order=index)))
            definition = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), cohort_id=cohort, tasks=tasks,
                states_in_execution_order=states, horizons=[50,100,200],
                episodes_per_method_horizon=len(tasks)*len(states), total_episodes=sum(m['episodes'] for _,_,m in configs),
                candidates=candidates, primary_candidate='window3_skip1', primary_comparators=['smol70','fixed45'],
                comparators_by_horizon={str(h):['smol70','fixed45','naive_k5','vlash_style_k5'] for h in [50,100,200]},
                locked_execution_source_sha256=locked, selection_evidence_sha256={n:sha(ROOT/n) for n in evidence},
                vla_sha256=prior['vla_sha256'], minimum_probability=0, binary_argmax=True, readout_adaptation=False,
                logical_delay_steps=delay, statistics=prior['statistics'], stop_policy=prior['stop_policy'],
                no_adaptation_during_cohort=True, no_early_stopping_for_favorable_results=True,
                implementation_pilot=pilot,
                interpretation='Paused-simulator logical delay, no actual concurrency speedup. Both d4 and d5 retained. K5 noise is seeded at actual prediction tick; ordinary d5 also makes the prescribed tick0 pending request in addition to initialization.')
            if not pilot:
                definition['qualification_runs'] = [dict(run=f'delay_d{delay}_pilot_v1', episodes=15, logical_delay_steps=delay)]
            proposals.append((folder, definition, configs))
    for folder, definition, configs in proposals:
        folder.mkdir(parents=True)
        shards = []
        for path, cfg, meta in configs:
            dump(path, cfg)
            shards.append(dict(**meta, config_sha256=sha(path)))
        definition['shards'] = shards
        dump(folder/'definition.json', definition)
        dump(folder/'freeze_receipt.json', dict(definition_sha256=sha(folder/'definition.json'), immutable=True))
        print(json.dumps(dict(cohort=definition['cohort_id'], episodes=definition['total_episodes'],
                              definition_sha256=sha(folder/'definition.json'))))


if __name__ == '__main__':
    main()
