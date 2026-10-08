"""Apply only the predeclared technical repair, preserving all v1 scientific definitions."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
from local_policy import allowed

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ['validation_d1_v1', 'budget_validation_d1_v1', 'delay_pilot_d4_v1',
           'delay_validation_d4_v1', 'delay_pilot_d5_v1', 'delay_validation_d5_v1']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    assert not path.exists(), f'Immutable destination already exists: {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    allowed()
    qualification_name = 'guarded_runtime_qualification_v1'
    audit = read(ROOT/'checks'/f'{qualification_name}_audit.json')
    initial_audit = read(ROOT/'checks'/f'{qualification_name}_initial_verification.json')
    replay = read(ROOT/'checks/incidents'/f'{qualification_name}_vs_initial_repro_plain_v1.json')
    assert audit['errors'] == initial_audit['errors'] == 0
    assert audit['episodes'] == initial_audit['episodes'] == replay['entire_trajectory_exact'] == replay['n'] == 21
    assert read(ROOT/'raw'/qualification_name/'supervisor.json')['status'] == 'completed'
    manifest = read(ROOT/'raw'/qualification_name/'data/manifest.json')
    injected = read(ROOT/'checks/initial_guard_fault_injection.json')
    assert injected['errors'] == 0 and injected['checks'] == 4 and injected['cuda_initialized'] is False
    assert injected['verified_runtime_sha256'] == sha(ROOT/'scripts/verified_runtime.py')
    locked = dict(read(ROOT/'protocol/validation_d1_v1/definition.json')['locked_execution_source_sha256'])
    for name in ['run_guarded_closed_loop.py', 'verified_runtime.py']:
        locked[name] = sha(ROOT/'scripts'/name)
    for name, digest in locked.items():
        assert sha(ROOT/'scripts'/name) == manifest['scripts'][name] == digest
    staged = []
    for old_name in COHORTS:
        new_name = old_name[:-2]+'v2'
        prior_path = ROOT/'protocol'/old_name/'definition.json'
        assert sha(prior_path) == read(prior_path.parent/'freeze_receipt.json')['definition_sha256']
        assert not (ROOT/'protocol'/new_name).exists()
        prior = read(prior_path)
        definition = copy.deepcopy(prior)
        definition.update(cohort_id=new_name, frozen_utc=datetime.now(timezone.utc).isoformat(),
                          technical_revision_of=old_name, original_definition_sha256=sha(prior_path),
                          technical_amendment_sha256=sha(ROOT/'TECHNICAL_AMENDMENT_02.md'),
                          entry_point='run_guarded_closed_loop.py', initial_input_verification_required=True,
                          deterministic_backend_forced=False,
                          locked_execution_source_sha256=locked,
                          original_candidate_selection_unchanged=True,
                          v1_technical_attempts_excluded_from_efficacy_denominator=True)
        for shard in definition['shards']:
            old_cfg_path = ROOT/'configs'/shard['config']
            assert sha(old_cfg_path) == shard['config_sha256']
            cfg = read(old_cfg_path)
            assert cfg['candidates'] == prior['candidates']
            cfg.update(cohort_id=new_name, locked_source_sha256=locked,
                       initial_input_verification_required=True, deterministic_backend_forced=False,
                       technical_revision_of=old_name,
                       technical_amendment_sha256=sha(ROOT/'TECHNICAL_AMENDMENT_02.md'))
            shard['run'] = shard['run'][:-2]+'v2'
            shard['config'] = shard['run']+'.json'
            path = ROOT/'configs'/shard['config']
            assert not path.exists()
            staged.append((path, cfg, shard))
        qs = []
        for q in prior.get('qualification_runs', []):
            q = copy.deepcopy(q)
            q['run'] = q['run'][:-2]+'v2'
            qs.append(q)
        qs.append(dict(run=qualification_name, episodes=21, logical_delay_steps=1))
        definition['qualification_runs'] = qs
        definition['qualification_evidence_sha256'] = {
            str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [
                ROOT/'checks'/f'{qualification_name}_audit.json',
                ROOT/'checks'/f'{qualification_name}_initial_verification.json',
                ROOT/'checks/incidents'/f'{qualification_name}_vs_initial_repro_plain_v1.json',
                ROOT/'checks/initial_guard_fault_injection.json']}
        staged.append((ROOT/'protocol'/new_name/'definition.json', definition, None))
    # All destinations and inherited definitions have been validated before writing.
    for path, data, shard in staged:
        write(path, data)
        if shard is not None:
            shard['config_sha256'] = sha(path)
        else:
            write(path.parent/'freeze_receipt.json', dict(definition_sha256=sha(path), immutable=True))
            print(json.dumps(dict(cohort=data['cohort_id'], episodes=data['total_episodes'], definition_sha256=sha(path))))


if __name__ == '__main__':
    main()
