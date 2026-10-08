"""Separate one-hour exploratory ablation, after primary and official completion."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback

from supervise import BASE, ROOT, allowed, cleanup, atomic


def safe_attempt(value):
    return value.isascii() and value.replace('_', '').isalnum()


def primary_complete(attempt):
    assert safe_attempt(attempt)
    primary_root = ROOT / attempt
    supervisor = json.loads((primary_root / 'supervisor.json').read_text())
    smol = json.loads((primary_root / 'smol/status.json').read_text())
    official = json.loads((primary_root / 'official_reference/summary.json').read_text())
    if not (supervisor['status'] == 'completed' and smol['phase'] == 'completed'
            and smol['completed'] == 372 and official['episodes'] == 30
            and official['freeze_verified_unchanged'] is True):
        raise RuntimeError('Primary 360 + pilot 12 and official 30 must finish before argmax ablation')
    return dict(attempt=attempt, primary_completed=smol['completed'], official_episodes=official['episodes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', default='argmax_attempt_v1')
    parser.add_argument('--primary-attempt', default='attempt_v1')
    args = parser.parse_args()
    assert safe_attempt(args.attempt) and args.attempt.startswith('argmax_attempt_')
    assert safe_attempt(args.primary_attempt)
    # Same lock as the primary supervisor: never overlap either GPU campaign.
    lock = (ROOT / 'experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = None
    root = ROOT / args.attempt
    owns_output = False
    started = time.monotonic()
    records = []
    stop = {'signal': None}
    try:
        allowed()
        reference = primary_complete(args.primary_attempt)
        root.mkdir(exist_ok=False)
        owns_output = True
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, lambda number, frame: stop.update(signal=number))
        command = [str(BASE / 'envs/smolvla/bin/python'), '-u',
                   str(ROOT / 'scripts/run_argmax_ablation.py'), '--out', str(root / 'smol')]
        atomic(root / 'supervisor_pid.json', dict(pid=os.getpid(), argv=sys.argv,
                                                 started_utc=datetime.now(timezone.utc).isoformat(),
                                                 exploratory=True, minimum_probability=0.0,
                                                 wall_limit_seconds=3600, primary_reference=reference))
        with (root / 'argmax.log').open('w') as log:
            process = subprocess.Popen(command, cwd=BASE, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            rec = dict(phase='argmax', argv=command, pid=process.pid,
                       started_utc=datetime.now(timezone.utc).isoformat())
            records.append(rec)
            last_resource_sample = -60.0
            while process.poll() is None:
                if stop['signal'] is not None:
                    raise RuntimeError('Argmax supervisor received termination signal')
                if time.monotonic() - started > 3600:
                    raise TimeoutError('Exploratory argmax 3600 second wall limit')
                lease = allowed()
                if time.monotonic() - last_resource_sample >= 60:
                    sample = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name,memory.used,memory.free,utilization.gpu',
                         '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
                    with (root / 'gpu_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                                     output=sample.stdout.strip(), returncode=sample.returncode)) + '\n')
                    last_resource_sample = time.monotonic()
                atomic(root / 'supervisor.json', dict(status='running', records=records,
                       elapsed_seconds=time.monotonic() - started, exploratory=True,
                       minimum_probability=0.0, primary_reference=reference,
                       lease_valid_until=lease['valid_until_utc']))
                time.sleep(2)
            rec.update(exit_code=process.returncode, ended_utc=datetime.now(timezone.utc).isoformat())
            if process.returncode != 0:
                raise RuntimeError(f'Argmax ablation exited with {process.returncode}')
            cleanup(process)
            process = None
        atomic(root / 'supervisor.json', dict(status='completed', records=records,
                                             elapsed_seconds=time.monotonic() - started,
                                             exploratory=True, minimum_probability=0.0,
                                             primary_reference=reference))
    except BaseException as error:
        cleanup(process)
        if owns_output:
            atomic(root / 'supervisor.json', dict(status='failed', records=records,
                   elapsed_seconds=time.monotonic() - started, error=repr(error),
                   traceback=traceback.format_exc(), exploratory=True, minimum_probability=0.0))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
