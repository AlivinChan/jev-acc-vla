"""Register two state-blind Bernoulli controls using development-only frequencies."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
from local_policy import allowed

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert not path.exists(), f'Never overwrite a frozen definition: {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    allowed()
    prior_path = ROOT/'protocol/validation_d1_v2/definition.json'
    assert sha(prior_path) == read(prior_path.parent/'freeze_receipt.json')['definition_sha256']
    prior = read(prior_path)
    cpu = read(ROOT/'checks/seeded_skip_cpu.json')
    assert cpu['checks'] == 70 and cpu['errors'] == 0 and cpu['GPU_used'] is False
    initial_counts = read(ROOT/'checks/random_skip_available_initials.json')
    assert initial_counts['cuda_initialized'] is False and min(initial_counts['initial_states_per_task'].values()) > 47
    locked = copy.deepcopy(prior['locked_execution_source_sha256'])
    for name in ['seeded_skip_runtime.py', 'run_seeded_skip_closed_loop.py', 'audit_seeded_skip_closed_loop.py']:
        locked[name] = sha(ROOT/'scripts'/name)
    for name, digest in {**locked, **cpu['source_sha256']}.items():
        assert sha(ROOT/'scripts'/name) == digest
    development_path = ROOT/'analysis/gate_service_development.json'
    development = read(development_path)
    fractions = {h: [row['continues'], row['gates']] for h, row in development['by_horizon'].items()}
    assert fractions == {'50': [93,232], '100': [47,132], '200': [12,64]}
    candidates = {cid: copy.deepcopy(prior['candidates'][cid]) for cid in
                  ['smol70', 'fixed45', 'window3_skip1', 'naive_k5', 'vlash_style_k5']}
    for seed in [11131, 22261]:
        candidates[f'random_skip_s{seed}'] = dict(method='seeded_skip1', coin_seed=seed,
            continue_fractions=fractions, role='State-blind random single skip; fixed development frequency; no LAYA calls')
    main_candidates = {cid: candidates[cid] for cid in
                       ['smol70', 'fixed45', 'window3_skip1', 'random_skip_s11131', 'random_skip_s22261']}
    exact_d1 = dict(file='checks/incidents/random_d1_pilot_v1_vs_initial_repro_plain_v1.json', n=15,
                    current='random_d1_pilot_v1', reference='initial_repro_plain_v1',
                    candidates_filter='smol70,fixed45,window3_skip1,naive_k5,vlash_style_k5')
    exact_d4 = dict(file='checks/incidents/random_d4_pilot_v1_vs_delay_d4_pilot_v2.json', n=9,
                    current='random_d4_pilot_v1', reference='delay_d4_pilot_v2',
                    candidates_filter='smol70,fixed45,window3_skip1')
    setups = [
        dict(cohort='random_skip_pilot_d1_v1', delay=1, states=[10], tasks=[0], pilot=True,
             run_pattern='random_d1_pilot_v1', selected=candidates, qualifications=[], exact=[]),
        dict(cohort='random_skip_pilot_d4_v1', delay=4, states=[0], tasks=[0], pilot=True,
             run_pattern='random_d4_pilot_v1', selected=main_candidates,
             qualifications=[dict(run='random_d1_pilot_v1', episodes=21, logical_delay_steps=1)], exact=[exact_d1]),
        dict(cohort='random_skip_validation_d4_v1', delay=4, states=[47,45,46], tasks=list(range(10)), pilot=False,
             run_pattern='random_d4_s{state}_v1', selected=main_candidates,
             qualifications=[dict(run='random_d1_pilot_v1', episodes=21, logical_delay_steps=1),
                             dict(run='random_d4_pilot_v1', episodes=15, logical_delay_steps=4)], exact=[exact_d1, exact_d4]),
    ]
    for setup in setups:
        assert not (ROOT/'protocol'/setup['cohort']).exists()
        for state in setup['states']:
            assert not (ROOT/'configs'/(setup['run_pattern'].format(state=state)+'.json')).exists()
    for setup in setups:
        shards = []
        for order, state in enumerate(setup['states']):
            name = setup['run_pattern'].format(state=state)
            cfg = dict(tasks=setup['tasks'], states=[state], horizons=[50,100,200],
                       logical_delay_steps=setup['delay'], schedule_seed=2026100817,
                       cohort_id=setup['cohort'], cohort_role='Fixed-frequency random-skip controls; all sequences retained',
                       candidates=setup['selected'], locked_source_sha256=locked,
                       initial_input_verification_required=True, deterministic_backend_forced=False)
            cfg_path = ROOT/'configs'/f'{name}.json'
            write(cfg_path, cfg)
            count = len(setup['tasks'])*3*len(setup['selected'])
            shards.append(dict(run=name, config=cfg_path.name, state=state, episodes=count, order=order,
                               config_sha256=sha(cfg_path)))
        definition = dict(cohort_id=setup['cohort'], frozen_utc=datetime.now(timezone.utc).isoformat(),
            entry_point='run_seeded_skip_closed_loop.py', audit_entry='audit_seeded_skip_closed_loop.py',
            initial_input_verification_required=True, deterministic_backend_forced=False,
            tasks=setup['tasks'], states_in_execution_order=setup['states'], horizons=[50,100,200],
            episodes_per_method_horizon=len(setup['tasks'])*len(setup['states']),
            total_episodes=sum(s['episodes'] for s in shards), candidates=setup['selected'], shards=shards,
            primary_candidate='window3_skip1', primary_comparators=['random_skip_s11131','random_skip_s22261'],
            comparators_by_horizon={str(h): ['smol70','fixed45','random_skip_s11131','random_skip_s22261'] for h in [50,100,200]},
            logical_delay_steps=setup['delay'], locked_execution_source_sha256=locked,
            vla_sha256=prior['vla_sha256'], minimum_probability=0, laya_binary_argmax=True,
            random_baseline_not_a_laya_prediction=True, random_sequences_reported_separately=True,
            readout_adaptation=False, implementation_pilot=setup['pilot'],
            qualification_runs=setup['qualifications'], exact_replay_qualifications=setup['exact'],
            statistics=prior['statistics'], stop_policy=prior['stop_policy'],
            no_adaptation_during_cohort=True, no_early_stopping_for_favorable_results=True,
            selection_evidence_sha256={str(p.relative_to(ROOT)).replace('\\','/'): sha(p) for p in [
                prior_path, development_path, ROOT/'RANDOM_SKIP_PROTOCOL.md', ROOT/'checks/seeded_skip_cpu.json',
                ROOT/'checks/random_skip_available_initials.json']},
            interpretation='Two fixed state-blind random sequences at development-only continue frequencies. Actual new episode lengths, calls and costs may differ; two sequences are not independent extra starts. Primary LAYA remains original binary argmax.')
        path = ROOT/'protocol'/setup['cohort']/'definition.json'
        write(path, definition)
        write(path.parent/'freeze_receipt.json', dict(definition_sha256=sha(path), immutable=True))
        print(json.dumps(dict(cohort=setup['cohort'], episodes=definition['total_episodes'], definition_sha256=sha(path))))


if __name__ == '__main__':
    main()
