"""Supervise one experiment process group and preserve partial output."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
import traceback

BASE = Path('/mnt/4t/jev_vla_libero')
ROOT = BASE / 'experiments/laya_gate_optimization'


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)





def allowed():
    if (ROOT/'STOP_REQUESTED.json').exists():
        raise RuntimeError('This experiment is stopped')
    return {}


def cleanup(child):
    if child is None:return
    try:os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:return
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:os.killpg(child.pid, 0)
        except ProcessLookupError:break
        time.sleep(.1)
    try:os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait()


def main():
    import fcntl
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--entry',required=True)
    p.add_argument('--env',choices=['laya','smolvla'],required=True)
    p.add_argument('--wall',type=int,default=3600)
    p.add_argument('extra',nargs=argparse.REMAINDER)
    args=p.parse_args()
    assert args.run.isascii() and args.run.replace('_','').isalnum()
    entry=(ROOT/'scripts'/args.entry).resolve()
    assert entry.is_file() and entry.parent == (ROOT/'scripts').resolve() and entry.suffix=='.py'
    assert 30 <= args.wall <= 7200
    ROOT.joinpath('runs').mkdir(exist_ok=True)
    lock=(ROOT/'experiment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'runs'/args.run
    child=None
    owns_output=False
    started=time.monotonic()
    rec={}
    stopped={'signal':None}
    try:
        allowed()
        out.mkdir(exist_ok=False)
        owns_output=True
        # Freeze executable bytes, so later stages cannot change a running worker's service imports.
        source=out/'source'
        source.mkdir()
        source_hashes={}
        for original in sorted((ROOT/'scripts').glob('*.py')):
            value=original.read_bytes()
            (source/original.name).write_bytes(value)
            source_hashes[original.name]=hashlib.sha256(value).hexdigest()
        atomic(out/'source_manifest.json',source_hashes)
        for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:
            signal.signal(sig,lambda number,frame:stopped.update(signal=number))
        extra=args.extra[1:] if args.extra[:1]==['--'] else args.extra
        command=[str(BASE/'envs'/args.env/'bin/python'),'-u',str(source/entry.name),'--out',str(out/'data'),*extra]
        with (out/'worker.log').open('w') as log:
            child=subprocess.Popen(command,cwd=BASE,stdin=subprocess.DEVNULL,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
            rec=dict(pid=child.pid,supervisor_pid=os.getpid(),argv=command,
                     started_utc=datetime.now(timezone.utc).isoformat())
            while child.poll() is None:
                if stopped['signal'] is not None:raise RuntimeError('Termination signal')
                if time.monotonic()-started > args.wall:raise TimeoutError('Bounded stage wall limit')
                lease=allowed()
                atomic(out/'supervisor.json',dict(status='running',record=rec,
                       elapsed_seconds=time.monotonic()-started))
                time.sleep(2)
            rec.update(exit_code=child.returncode,ended_utc=datetime.now(timezone.utc).isoformat())
            if child.returncode:raise RuntimeError('Worker exited '+str(child.returncode))
            cleanup(child);child=None
        atomic(out/'supervisor.json',dict(status='completed',record=rec,elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        cleanup(child)
        if owns_output:atomic(out/'supervisor.json',dict(status='failed',
            record=rec,error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))
        raise
    finally:lock.close()


if __name__=='__main__':main()
