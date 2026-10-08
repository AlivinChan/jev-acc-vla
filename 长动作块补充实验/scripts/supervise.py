"""Supervise sequential experiment stages and clean up owned processes."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, fcntl, json, os, signal, subprocess, sys, time, traceback

BASE = Path('/mnt/4t/jev_vla_libero')
ROOT = BASE/'experiments/long_chunk_laya'


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def allowed():
    if (ROOT/'STOP_REQUESTED.json').exists():
        raise RuntimeError('This experiment is stopped')
    return {}


def cleanup(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic()+8
    while time.monotonic()<deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', default='attempt_v1')
    p.add_argument('--pilot-only', action='store_true')
    args = p.parse_args()
    assert args.attempt.isascii() and args.attempt.replace('_','').isalnum()
    root = ROOT/args.attempt
    root.mkdir(exist_ok=False)
    lock = (ROOT/'experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    allowed()
    started = time.monotonic()
    commands = [[str(BASE/'envs/smolvla/bin/python'), '-u', str(ROOT/'scripts/main_experiment.py'),
                 '--out', str(root/'smol')]]
    if args.pilot_only:
        commands[0].append('--pilot-only')
    else:
        commands.append([str(BASE/'envs/vlash/bin/python'), '-u',
                         str(ROOT/'scripts/official_reference.py'), '--states','3',
                         '--suite','libero_spatial','--delay','1', '--out',str(root/'official_reference')])
    process = None
    records = []
    last_resource_sample = -60.0
    stop = {'signal': None}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda number, frame: stop.update(signal=number))
    try:
        for i, command in enumerate(commands):
            allowed()
            with (root/f'phase_{i}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=BASE, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                rec = dict(phase=i, argv=command, pid=process.pid,
                           started_utc=datetime.now(timezone.utc).isoformat())
                records.append(rec)
                while process.poll() is None:
                    if stop['signal'] is not None:
                        raise RuntimeError('Supervisor received termination signal')
                    if time.monotonic()-started>7200:
                        raise TimeoutError('Whole campaign 7200 second wall limit')
                    lease = allowed()
                    if time.monotonic()-last_resource_sample >= 60:
                        sample = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.free,utilization.gpu',
                                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
                        with (root/'gpu_samples.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                                       output=sample.stdout.strip(), returncode=sample.returncode))+'\n')
                        last_resource_sample = time.monotonic()
                    atomic(root/'supervisor.json', dict(status='running', records=records,
                           elapsed_seconds=time.monotonic()-started))
                    time.sleep(2)
                rec.update(exit_code=process.returncode, ended_utc=datetime.now(timezone.utc).isoformat())
                if process.returncode!=0:
                    raise RuntimeError(f'Phase {i} failed with exit {process.returncode}')
                cleanup(process)
                process = None
        atomic(root/'supervisor.json', dict(status='completed', records=records,
                                            elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        cleanup(process)
        atomic(root/'supervisor.json', dict(status='failed', records=records,
               elapsed_seconds=time.monotonic()-started, error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        lock.close()


if __name__=='__main__':
    main()
