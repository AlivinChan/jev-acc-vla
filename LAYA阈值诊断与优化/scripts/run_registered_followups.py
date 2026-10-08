"""Serial registered work after the active main chain; every stage fails closed."""
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
MAIN = 'validation_d1_v2'
FOLLOWUPS = ['budget_validation_d1_v2', 'delay_pilot_d4_v2', 'delay_validation_d4_v2',
             'delay_pilot_d5_v2', 'delay_validation_d5_v2', 'noise_alignment_pilot_d1_v1',
             'noise_alignment_pilot_d4_v1', 'noise_alignment_validation_d4_v1']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    allowed()
    status_path = ROOT/'runtime/registered_followups_v1.json'
    assert not status_path.exists(), 'Never silently restart a recorded queue'
    assert (ROOT/'runtime'/f'{MAIN}_chain.json').exists(), 'Main chain must already be running'
    definitions = {}
    for name in [MAIN, *FOLLOWUPS]:
        p = ROOT/'protocol'/name/'definition.json'
        assert sha(p) == read(p.parent/'freeze_receipt.json')['definition_sha256']
        definitions[name] = sha(p)
    state = dict(started_utc=datetime.now(timezone.utc).isoformat(), main=MAIN, followups=FOLLOWUPS,
                 definition_sha256=definitions, completed=[], no_outcome_dependent_selection=True)

    def record(phase, **fields):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8', newline='\n')
        temp.replace(status_path)
        print(json.dumps(dict(phase=phase, **fields)), flush=True)

    def invoke(script, *args, log=None):
        allowed()
        command = [sys.executable, '-X', 'utf8', str(ROOT/'scripts'/script), *args]
        if log is None:
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            with (ROOT/'runtime'/log).open('w', encoding='utf-8') as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def analyze(name):
        definition = read(ROOT/'protocol'/name/'definition.json')
        if not definition.get('implementation_pilot', False):
            record('analyzing', cohort=name)
            invoke('analyze_validation_cohort.py', '--cohort', name, log=name+'_analysis.log')
            if not name.startswith('noise_alignment_'):
                invoke('build_validation_figure.py', '--cohort', name, log=name+'_figure.log')

    try:
        record('waiting_for_main')
        while True:
            allowed()
            main_status = read(ROOT/'runtime'/f'{MAIN}_chain.json')
            if main_status['phase'] == 'completed':
                break
            assert main_status['phase'] != 'halted', 'Main chain halted; registered follow-ups must not start'
            time.sleep(20)
        analyze(MAIN)
        for name in FOLLOWUPS:
            allowed()
            assert sha(ROOT/'protocol'/name/'definition.json') == definitions[name]
            record('executing_cohort', cohort=name)
            invoke('run_frozen_cohort.py', '--cohort', name)
            assert read(ROOT/'runtime'/f'{name}_chain.json')['phase'] == 'completed'
            if name == 'noise_alignment_pilot_d1_v1':
                invoke('compare_forensic_runs.py', 'noise_d1_pilot_v1', 'initial_repro_plain_v1',
                       '--candidates', 'naive_k5,vlash_style_k5')
                proof = read(ROOT/'checks/incidents/noise_d1_pilot_v1_vs_initial_repro_plain_v1.json')
                assert proof['n'] == proof['entire_trajectory_exact'] == 6
            analyze(name)
            state['completed'].append(name)
            record('cohort_complete', cohort=name)
        record('completed')
    except BaseException as error:
        record('halted', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
