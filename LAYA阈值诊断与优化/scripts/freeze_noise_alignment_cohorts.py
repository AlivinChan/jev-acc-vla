"""Freeze the controlled-noise VLASH attribution queue before its GPU pilots."""
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
    assert not path.exists(), f'Do not overwrite a frozen record: {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    allowed()
    prior_path = ROOT/'protocol/validation_d1_v2/definition.json'
    assert sha(prior_path) == read(prior_path.parent/'freeze_receipt.json')['definition_sha256']
    prior = read(prior_path)
    cpu = read(ROOT/'checks/noise_alignment_cpu.json')
    assert cpu['checks'] == 36 and cpu['errors'] == 0
    locked = copy.deepcopy(prior['locked_execution_source_sha256'])
    for name in ['mechanism_runtime.py', 'run_mechanism_closed_loop.py',
                 'audit_mechanism_closed_loop.py', 'audit_noise_alignment.py']:
        locked[name] = sha(ROOT/'scripts'/name)
    for name, digest in locked.items():
        assert sha(ROOT/'scripts'/name) == digest
    for name, digest in cpu['source_sha256'].items():
        assert sha(ROOT/'scripts'/name) == digest
    candidates = {cid: copy.deepcopy(prior['candidates'][cid]) for cid in
                  ['smol70', 'fixed45', 'window3_skip1', 'naive_k5', 'vlash_style_k5']}
    candidates['vlash_noise_matched_k5'] = dict(method='vlash_style_k5', alignment_mode='current_state',
        noise_clock='image', role='Current proprio / stale image, same image-tick noise as naive')
    candidates['stale_row0_k5'] = dict(method='vlash_style_k5', alignment_mode='stale_state',
        noise_clock='image', role='Stale image and proprio with matched noise; execute row zero')
    k5 = ['naive_k5', 'vlash_style_k5', 'vlash_noise_matched_k5', 'stale_row0_k5']
    replay_requirement = dict(file='checks/incidents/noise_d1_pilot_v1_vs_initial_repro_plain_v1.json', n=6,
                              current='noise_d1_pilot_v1', reference='initial_repro_plain_v1',
                              candidates_filter='naive_k5,vlash_style_k5')
    setups = [
        dict(cohort='noise_alignment_pilot_d1_v1', delay=1, states=[10], tasks=[0], pilot=True,
             run_pattern='noise_d1_pilot_v1', selected={cid:candidates[cid] for cid in k5},
             qualifications=[], exact=[]),
        dict(cohort='noise_alignment_pilot_d4_v1', delay=4, states=[0], tasks=[0], pilot=True,
             run_pattern='noise_d4_pilot_v1', selected=candidates,
             qualifications=[dict(run='noise_d1_pilot_v1', episodes=12, logical_delay_steps=1)], exact=[replay_requirement]),
        dict(cohort='noise_alignment_validation_d4_v1', delay=4, states=[42,40,41], tasks=list(range(10)), pilot=False,
             run_pattern='noise_d4_s{state}_v1', selected=candidates,
             qualifications=[dict(run='noise_d1_pilot_v1', episodes=12, logical_delay_steps=1),
                             dict(run='noise_d4_pilot_v1', episodes=21, logical_delay_steps=4)], exact=[replay_requirement])]
    for setup in setups:
        assert not (ROOT/'protocol'/setup['cohort']).exists()
        for state in setup['states']:
            assert not (ROOT/'configs'/(setup['run_pattern'].format(state=state)+'.json')).exists()
    for setup in setups:
        shards = []
        for order, state in enumerate(setup['states']):
            name = setup['run_pattern'].format(state=state)
            cfg = dict(tasks=setup['tasks'], states=[state], horizons=[50,100,200],
                       logical_delay_steps=setup['delay'], schedule_seed=2026100813,
                       cohort_id=setup['cohort'], cohort_role='Controlled K5 noise/alignment attribution; no policy reselection',
                       candidates=setup['selected'], locked_source_sha256=locked,
                       initial_input_verification_required=True, deterministic_backend_forced=False)
            cfg_path = ROOT/'configs'/f'{name}.json'
            write(cfg_path, cfg)
            count = len(setup['tasks'])*3*len(setup['selected'])
            shards.append(dict(run=name, config=cfg_path.name, state=state, episodes=count, order=order,
                               config_sha256=sha(cfg_path)))
        definition = dict(cohort_id=setup['cohort'], frozen_utc=datetime.now(timezone.utc).isoformat(),
            entry_point='run_mechanism_closed_loop.py', audit_entry='audit_mechanism_closed_loop.py',
            initial_input_verification_required=True, noise_alignment_audit_required=True,
            deterministic_backend_forced=False, tasks=setup['tasks'], states_in_execution_order=setup['states'],
            horizons=[50,100,200], episodes_per_method_horizon=len(setup['tasks'])*len(setup['states']),
            total_episodes=sum(s['episodes'] for s in shards), candidates=setup['selected'], shards=shards,
            primary_candidate='window3_skip1' if not setup['pilot'] else 'vlash_noise_matched_k5',
            primary_comparators=['smol70','fixed45','vlash_noise_matched_k5'] if not setup['pilot'] else k5,
            mechanism_primary_contrasts=[['vlash_noise_matched_k5','stale_row0_k5'],
                                        ['stale_row0_k5','naive_k5'],['vlash_style_k5','vlash_noise_matched_k5']],
            comparators_by_horizon={str(h): (['smol70','fixed45']+k5 if not setup['pilot'] else k5) for h in [50,100,200]},
            logical_delay_steps=setup['delay'], locked_execution_source_sha256=locked,
            vla_sha256=prior['vla_sha256'], minimum_probability=0, binary_argmax=True,
            readout_adaptation=False, implementation_pilot=setup['pilot'],
            qualification_runs=setup['qualifications'], exact_replay_qualifications=setup['exact'],
            statistics=prior['statistics'], stop_policy=prior['stop_policy'],
            no_adaptation_during_cohort=True, no_early_stopping_for_favorable_results=True,
            selection_evidence_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [
                prior_path, ROOT/'VLASH_NOISE_ATTRIBUTION_PROTOCOL.md', ROOT/'checks/noise_alignment_cpu.json']},
            interpretation='First controlled update separates input state, action-row offset, and noise clock. Later trajectories differ. Frozen SmolVLA has no VLASH offset training; paused simulation is not real asynchronous execution.')
        path = ROOT/'protocol'/setup['cohort']/'definition.json'
        write(path, definition)
        write(path.parent/'freeze_receipt.json', dict(definition_sha256=sha(path), immutable=True))
        print(json.dumps(dict(cohort=setup['cohort'], episodes=definition['total_episodes'], definition_sha256=sha(path))))


if __name__ == '__main__':
    main()
