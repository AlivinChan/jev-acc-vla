"""Frozen Laya CUDA service for the separately registered long-chunk experiment.

Run only with the remote envs/laya interpreter. The controller atomically writes
requests/<id>.json containing request_id and state (English text), then reads
responses/<id>.json. Optional warmup=true labels a separately accounted request.
The service does not decide scheduling: its uncertain output is interpreted by
the controller's fixed remaining/H <= 0.7 fallback. A stop or stop.json file,
or 15 minutes without a request, ends the service. No model loads on import.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback


ROOT = Path('/mnt/4t/jev_vla_libero')
EXPERIMENT_ROOT = ROOT / 'experiments/long_chunk_laya'
MAX_LEN = 1536
HEAD_MAX_LEN = 192
MINIMUM_PROBABILITY = 0.7
IDLE_SECONDS = 900
LABELS = {'replan', 'continue', 'uncertain'}
QUESTIONS = {
    'replanning': {
        'type': 'choice',
        'instructions': (
            'Should the robot request a new action chunk now, or continue the finite '
            'queued actions until its next scheduled check? Use the task, observed '
            'robot and object states, plan age, remaining queue, sampled future '
            'commands, next-check interval, and replacement delay stated in the input. '
            'Object coordinates are privileged simulator observations, not visual '
            'estimates. Movement since planning can be ordinary task progress. '
            'The input contains no future outcome or predicted future state. '
            'Choose uncertain if these observations do not justify either decision.'
        ),
        'criteria': {
            'replan': 'Current task-relevant evidence indicates the queued actions need replacement; request a new plan now.',
            'continue': 'The queued actions remain suitable for the task until the next check and stated replacement delay; continue for this interval.',
            'uncertain': 'There is insufficient evidence to choose reliably between replanning and continuing.',
        },
    }
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def checked_answer(raw):
    usage = raw.get('usage', {})
    if usage.get('truncated') or usage.get('state_tokens_dropped', 0) or usage.get('options'):
        raise ValueError('SDK reports truncated state or collapsed choices')
    answer = raw['answers']['replanning']
    if answer.get('type') != 'choice':
        raise ValueError('Wrong answer type')
    probabilities = answer['probabilities']
    if set(probabilities) != LABELS:
        raise ValueError('Wrong answer label set')
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
           for p in probabilities.values()):
        raise ValueError('Non-finite or invalid probability')
    if abs(sum(probabilities.values()) - 1) > 0.002:
        raise ValueError('Probabilities do not sum to one')
    raw_choice = answer['choice']
    if raw_choice not in LABELS or probabilities[raw_choice] < max(probabilities.values()) - 0.0002:
        raise ValueError('Choice disagrees with probabilities')
    choice = raw_choice if probabilities[raw_choice] >= MINIMUM_PROBABILITY else 'uncertain'
    return choice, raw_choice, probabilities


def evaluate(agent, torch, build_sequence, request, received_ns):
    state = request.get('state')
    if not isinstance(state, str) or not state.strip():
        raise ValueError('state must be nonempty English text')
    question = QUESTIONS['replanning']
    packed_question = {'t': question['type'], 'ins': question['instructions'], 'crit': question['criteria']}
    audit_started = time.monotonic_ns()
    sequence = build_sequence(agent.tok, state, packed_question, max_len=MAX_LEN,
                              head_max_len=HEAD_MAX_LEN, return_stats=True,
                              return_truncation_stats=True)
    option_stats, truncation_stats = sequence[-2:]
    if truncation_stats['truncated'] or truncation_stats.get('state_tokens_dropped', 0):
        raise ValueError('Preflight token audit rejects a truncated state')
    if option_stats['options_distinct'] != 3:
        raise ValueError('Preflight token audit rejects collapsed choices')
    audit_ended = time.monotonic_ns()
    before_fallback = getattr(agent, 'cpu_fallback_count', 0)
    if agent.device.type != 'cuda':
        raise RuntimeError('CUDA required; CPU fallback is prohibited')
    torch.cuda.synchronize()
    prediction_started = time.monotonic_ns()
    with torch.no_grad():
        raw = agent.system_one(state=state, questions=QUESTIONS,
                               max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
    torch.cuda.synchronize()
    prediction_ended = time.monotonic_ns()
    if agent.device.type != 'cuda' or getattr(agent, 'cpu_fallback_count', 0) != before_fallback:
        raise RuntimeError('SDK CPU fallback invalidates this request')
    choice, raw_choice, probabilities = checked_answer(raw)
    response_prewrite_ns = time.monotonic_ns()
    return {
        'status': 'ok', 'request_id': request['request_id'],
        'warmup': bool(request.get('warmup', False)),
        'choice': choice, 'raw_choice': raw_choice, 'probabilities': probabilities,
        'minimum_probability': MINIMUM_PROBABILITY,
        'usage': raw.get('usage', {}), 'raw': raw,
        'token_audit': {'tokens': len(sequence[0]), 'options': option_stats,
                        'state': truncation_stats, 'max_len': MAX_LEN, 'head_max_len': HEAD_MAX_LEN},
        'state_sha256': digest_bytes(state.encode('utf-8')),
        'prompt_sha256': digest_bytes(canonical_bytes(QUESTIONS)),
        'clock': {'received_monotonic_ns': received_ns,
                  'prediction_start_monotonic_ns': prediction_started,
                  'prediction_end_monotonic_ns': prediction_ended,
                  'response_prewrite_monotonic_ns': response_prewrite_ns,
                  'response_prewrite_utc': utc_now()},
        'token_audit_wall_ms': (audit_ended - audit_started) / 1e6,
        'sdk_synchronized_wall_ms': (prediction_ended - prediction_started) / 1e6,
        'service_prewrite_wall_ms': (response_prewrite_ns - received_ns) / 1e6,
        'timing_scope': 'Token audit and SDK wall time include host work; IPC publication and controller polling are separately timed by caller.',
        'cpu_fallback_count': getattr(agent, 'cpu_fallback_count', 0),
    }


def run(folder):
    if not folder.is_relative_to(EXPERIMENT_ROOT.resolve()) or folder == EXPERIMENT_ROOT.resolve():
        raise ValueError('IPC folder must be a child of the registered remote experiment root')
    folder.mkdir(parents=True, exist_ok=True)
    # A unique folder per service prevents racing workers and stale response reuse.
    with (folder / 'service_claim.json').open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'utc': utc_now()}, stream)
    count = 0
    try:
        if not Path(sys.executable).absolute().is_relative_to(ROOT / 'envs/laya'):
            raise RuntimeError('Use the pinned remote envs/laya interpreter')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        for name in ['HF_HOME', 'TORCH_HOME', 'PYTHONPYCACHEPREFIX']:
            if name not in os.environ or not Path(os.environ[name]).resolve().is_relative_to(ROOT):
                raise RuntimeError(name + ' must point inside the approved remote workspace')
        import torch
        import laya
        from laya.common import build_sequence

        source_path = ROOT / 'configs/local_gate_sources.json'
        source = json.loads(source_path.read_text())['laya']
        if laya.__version__ != source['sdk_version']:
            raise RuntimeError('Pinned Laya SDK version mismatch')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(4)
        model_path = (ROOT / source['path']).resolve()
        if not model_path.is_relative_to(ROOT / 'models'):
            raise RuntimeError('Checkpoint is outside the approved model directory')
        load_started = time.monotonic_ns()
        agent = laya.Agent(str(model_path), device='cuda', compile=False, fast=False,
                           expected_sha256={item['path']: item['sha256'] for item in source['assets']})
        agent.model.eval()
        for parameter in agent.model.parameters():
            parameter.requires_grad_(False)
        torch.cuda.synchronize()
        load_ended = time.monotonic_ns()
        if agent.device.type != 'cuda' or any(p.requires_grad for p in agent.model.parameters()):
            raise RuntimeError('Frozen CUDA model qualification failed')
        dtype_counts = Counter()
        for parameter in agent.model.parameters():
            dtype_counts[str(parameter.dtype)] += parameter.numel()
        requests = folder / 'requests'
        responses = folder / 'responses'
        requests.mkdir(exist_ok=True)
        responses.mkdir(exist_ok=True)
        atomic_json(folder / 'prompt.json', QUESTIONS)
        atomic_json(folder / 'ready.json', {
            'status': 'ready', 'pid': os.getpid(), 'ready_utc': utc_now(),
            'ready_monotonic_ns': time.monotonic_ns(), 'source': source,
            'model_path': str(model_path), 'torch': torch.__version__, 'laya': laya.__version__,
            'source_config_sha256': digest_bytes(source_path.read_bytes()),
            'service_source_sha256': digest_bytes(Path(__file__).read_bytes()),
            'prompt_sha256': digest_bytes(canonical_bytes(QUESTIONS)),
            'frozen': True, 'training': agent.model.training,
            'parameters_by_dtype': dict(dtype_counts),
            'device': str(agent.device), 'gpu': torch.cuda.get_device_name(),
            'load_and_integrity_wall_ms': (load_ended - load_started) / 1e6,
            'max_len': MAX_LEN, 'head_max_len': HEAD_MAX_LEN,
            'threshold': MINIMUM_PROBABILITY, 'warmup_performed': False,
            'idle_timeout_seconds': IDLE_SECONDS,
            'compile': False, 'fast': False,
        })
        handled = set()
        last_activity = time.monotonic()
        while True:
            if (folder / 'stop').exists() or (folder / 'stop.json').exists():
                reason = 'stop_requested'
                break
            if time.monotonic() - last_activity >= IDLE_SECONDS:
                reason = 'idle_timeout'
                break
            for path in sorted(requests.glob('*.json')):
                if path.name in handled:
                    continue
                received_ns = time.monotonic_ns()
                request = json.loads(path.read_text(encoding='utf-8'))
                request_id = request.get('request_id')
                if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id):
                    raise ValueError('request_id must be a safe, nonempty identifier')
                if path.stem != request_id:
                    raise ValueError('request_id must match its request filename')
                handled.add(path.name)
                try:
                    response = evaluate(agent, torch, build_sequence, request, received_ns)
                except Exception as exc:
                    response = {'status': 'error', 'request_id': request_id, 'choice': 'uncertain',
                                'error': type(exc).__name__ + ': ' + str(exc),
                                'clock': {'received_monotonic_ns': received_ns,
                                          'failed_monotonic_ns': time.monotonic_ns(), 'failed_utc': utc_now()}}
                    atomic_json(responses / path.name, response)
                    raise
                atomic_json(responses / path.name, response)
                count += 1
                last_activity = time.monotonic()
            time.sleep(0.002)
        atomic_json(folder / 'stopped.json', {'status': 'stopped', 'reason': reason,
                                             'requests_completed': count, 'utc': utc_now(),
                                             'monotonic_ns': time.monotonic_ns()})
        return 0
    except BaseException as exc:
        atomic_json(folder / 'failure.json', {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc),
                                             'requests_completed': count, 'utc': utc_now(),
                                             'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', required=True)
    return run(Path(parser.parse_args().folder).resolve())


if __name__ == '__main__':
    raise SystemExit(main())
