"""Wait for the existing registered queue, then run the frozen random-control extension."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time
import traceback
from local_policy import allowed

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ['random_skip_pilot_d1_v1', 'random_skip_pilot_d4_v1', 'random_skip_validation_d4_v1']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    allowed()
    status_path = ROOT/'runtime/seeded_skip_followup_v1.json'
    assert not status_path.exists(), 'Never silently restart a recorded extension'
    predecessor = ROOT/'runtime/registered_followups_v1.json'
    assert predecessor.exists() and (ROOT/'checks/deployment_seeded_skip_v1.json').exists()
    definitions = {}
    for name in COHORTS:
        p = ROOT/'protocol'/name/'definition.json'
        assert sha(p) == read(p.parent/'freeze_receipt.json')['definition_sha256']
        definitions[name] = sha(p)
    state = dict(started_utc=datetime.now(timezone.utc).isoformat(), predecessors='registered_followups_v1',
                 cohorts=COHORTS, definition_sha256=definitions, completed=[], no_outcome_dependent_selection=True)

    def record(phase, **fields):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8', newline='\n')
        temporary.replace(status_path)
        print(json.dumps(dict(phase=phase, **fields)), flush=True)

    def invoke(script, *args, log=None):
        allowed()
        command = [sys.executable, '-X', 'utf8', str(ROOT/'scripts'/script), *args]
        if log is None:
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            with (ROOT/'runtime'/log).open('w', encoding='utf-8') as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        record('waiting_for_registered_queue')
        while True:
            allowed()
            prior = read(predecessor)
            assert prior['phase'] != 'halted', 'Predecessor halted; extension must not launch'
            if prior['phase'] == 'completed':
                break
            time.sleep(20)
        for name in COHORTS:
            allowed()
            assert sha(ROOT/'protocol'/name/'definition.json') == definitions[name]
            record('executing_cohort', cohort=name)
            invoke('run_frozen_cohort.py', '--cohort', name)
            assert read(ROOT/'runtime'/f'{name}_chain.json')['phase'] == 'completed'
            if name == COHORTS[0]:
                current, reference, candidates, count = 'random_d1_pilot_v1', 'initial_repro_plain_v1', 'smol70,fixed45,window3_skip1,naive_k5,vlash_style_k5', 15
            elif name == COHORTS[1]:
                current, reference, candidates, count = 'random_d4_pilot_v1', 'delay_d4_pilot_v2', 'smol70,fixed45,window3_skip1', 9
            else:
                current = None
            if current:
                invoke('compare_forensic_runs.py', current, reference, '--candidates', candidates)
                proof = read(ROOT/'checks/incidents'/f'{current}_vs_{reference}.json')
                assert proof['n'] == proof['entire_trajectory_exact'] == count
            else:
                invoke('analyze_validation_cohort.py', '--cohort', name, log=name+'_analysis.log')
                invoke('build_validation_figure.py', '--cohort', name, log=name+'_figure.log')
                invoke('analyze_service_accounting_v2.py', '--cohort', name, log=name+'_service_accounting.log')
                invoke('analyze_cross_horizon.py', '--cohort', name, log=name+'_cross_horizon.log')
            state['completed'].append(name)
            record('cohort_complete', cohort=name)
        record('completed')
    except BaseException as error:
        record('halted', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
