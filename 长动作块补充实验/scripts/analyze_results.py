"""CPU-only audit of main_experiment.py artifacts; incomplete data remain partial."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METHODS = ['smol70', 'laya', 'naive_k5', 'vlash_style_k5']
HORIZONS = [50, 100, 200]
SEED = 20261007
BOOTSTRAP_REPLICATES = 10000
Z95 = 1.959963984540054


def sha(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b):
    return finite_number(a) and finite_number(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)


def ratio(a, b):
    return a / b if b > 0 else None


def wilson(successes, total):
    if not total:
        return [None, None]
    p = successes / total
    denominator = 1 + Z95 * Z95 / total
    center = (p + Z95 * Z95 / (2 * total)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / total + Z95 * Z95 / (4 * total * total)) / denominator
    return [0.0 if successes == 0 else max(0.0, center - half),
            1.0 if successes == total else min(1.0, center + half)]


def distribution(values):
    if not values:
        return {'n': 0}
    a = np.asarray(values, dtype=np.float64)
    return {'n': len(values), 'mean': float(a.mean()), 'min': float(a.min()),
            'p05': float(np.quantile(a, .05)), 'p50': float(np.quantile(a, .5)),
            'p95': float(np.quantile(a, .95)), 'max': float(a.max())}


class Audit:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checked = 0

    def check(self, condition, message, case=None):
        self.checked += 1
        if not condition:
            self.errors.append({'case': case, 'message': message})
        return bool(condition)


def episode_key(row):
    return row['phase'], row['task_id'], row['state_id'], row['horizon'], row['method']


def case_directory(run, key):
    phase, tid, sid, horizon, method = key
    return run / phase / f't{tid:02d}_s{sid:02d}_h{horizon}_{method}'


def validate_gate(gate, audit, name):
    audit.check(gate.get('status') == 'ok', 'Gate response status is not ok', name)
    audit.check(gate.get('choice') in {'replan', 'continue', 'uncertain'}, 'Unknown gate choice', name)
    audit.check(not gate.get('warmup', False), 'Warmup appears in episode gate costs', name)
    probabilities = gate.get('probabilities', {})
    valid = set(probabilities) == {'replan', 'continue', 'uncertain'} and all(
        finite_number(x) and 0 <= x <= 1 for x in probabilities.values())
    audit.check(valid, 'Gate probabilities malformed', name)
    if valid:
        audit.check(abs(sum(probabilities.values()) - 1) <= .002, 'Gate probability sum invalid', name)
        raw_choice = gate.get('raw_choice')
        raw_valid = raw_choice in probabilities and probabilities.get(raw_choice, -1) >= max(probabilities.values()) - .0002
        audit.check(raw_valid, 'Gate raw choice is not an argmax', name)
        if raw_valid:
            expected = raw_choice if probabilities[raw_choice] >= .7 else 'uncertain'
            audit.check(gate.get('choice') == expected and gate.get('minimum_probability') == .7,
                        'Gate confidence threshold differs from frozen 0.7 rule', name)
    usage = gate.get('usage', {})
    token = gate.get('token_audit', {})
    audit.check(not usage.get('truncated') and not usage.get('state_tokens_dropped', 0) and not usage.get('options'),
                'SDK usage reports truncation or collapsed choices', name)
    audit.check(token.get('max_len') == 1536 and token.get('head_max_len') == 192,
                'Gate token budgets differ from protocol', name)
    audit.check(not token.get('state', {}).get('truncated') and token.get('options', {}).get('options_distinct') == 3,
                'Preflight audit reports truncation or indistinct choices', name)
    audit.check(gate.get('cpu_fallback_count') == 0, 'Gate used CPU fallback', name)
    audit.check(gate.get('tick', -1) > 0 and gate.get('tick', -1) % 5 == 0,
                'Gate request is outside registered five-step cadence', name)
    state = gate.get('state', '')
    audit.check(isinstance(state, str) and hashlib.sha256(state.encode()).hexdigest() == gate.get('state_sha256'),
                'Gate state hash mismatch', name)
    for field in ['full_gate_seconds', 'ipc_outer_seconds', 'sdk_synchronized_wall_ms', 'service_prewrite_wall_ms']:
        audit.check(finite_number(gate.get(field)) and gate[field] >= 0, 'Invalid gate timing: ' + field, name)
    if finite_number(gate.get('full_gate_seconds')) and finite_number(gate.get('ipc_outer_seconds')):
        audit.check(gate['full_gate_seconds'] + 1e-6 >= gate['ipc_outer_seconds'], 'Full gate cost omits IPC time', name)


def validate_episode(run, row, audit):
    key = episode_key(row)
    case = case_directory(run, key)
    name = str(case.relative_to(run))
    errors_before = len(audit.errors)
    try:
        horizon = row['horizon']
        result = read_json(case / 'result.json')
        audit.check(result == row, 'episodes.jsonl and case result differ', name)
        audit.check(not (case / 'failure.json').exists(), 'Completed case also has failure.json', name)
        initial = np.load(case / 'initial_state.npy', allow_pickle=False)
        audit.check(np.isfinite(initial).all(), 'Non-finite initial state', name)
        audit.check(sha(initial) == row['initial_state_sha256'], 'Initial state hash mismatch', name)
        with np.load(case / 'trace.npz', allow_pickle=False) as archive:
            actions = archive['actions']
            physics = archive['physics']
        calls = read_json(case / 'calls.json')
        gates = read_json(case / 'gate.json')
        timeline = read_json(case / 'timeline.json')
        steps = row['steps']
        audit.check(isinstance(steps, int) and 0 < steps <= 230, 'Invalid control step count', name)
        audit.check(actions.shape == (steps, 7) and np.isfinite(actions).all(), 'Trace actions shape/finite violation', name)
        audit.check(physics.ndim == 2 and len(physics) == steps and np.isfinite(physics).all(),
                    'Trace physics shape/finite violation', name)
        audit.check(sha(actions) == row['executed_action_sha256'], 'Executed action hash mismatch', name)
        audit.check(len(timeline) == steps, 'Timeline length differs from executed steps', name)
        audit.check(len(calls) == row['vla_calls'] and len(calls) > 0, 'VLA count mismatch', name)
        audit.check(len(gates) == row['gate_calls'], 'Gate count mismatch', name)
        audit.check(row['nfe'] == 10 * len(calls), 'Episode NFE mismatch', name)
        audit.check(row['real_time'] is False and row['logical_delay_steps'] == 1, 'Incorrect timing claim or delay', name)
        audit.check(close(row['simulated_seconds'], steps / 20), 'Simulated duration mismatch', name)
        audit.check(isinstance(row['success'], bool), 'Success is not Boolean', name)
        audit.check(bool(timeline[-1]['success']) == row['success'], 'Terminal success mismatch', name)
        audit.check(not any(t['terminated'] or t['truncated'] for t in timeline[:-1]),
                    'Execution continued after a recorded terminal step', name)
        audit.check(steps == 230 or timeline[-1]['terminated'] or timeline[-1]['truncated'],
                    'Short trace has no terminal/truncation reason', name)
        for field in ['prediction_seconds', 'gate_seconds', 'episode_wall_seconds']:
            audit.check(finite_number(row.get(field)) and row[field] >= 0, 'Invalid episode cost: ' + field, name)
        audit.check(close(sum(c['predict_seconds'] for c in calls), row['prediction_seconds']), 'VLA cost sum mismatch', name)
        audit.check(close(sum(g['full_gate_seconds'] for g in gates), row['gate_seconds']), 'Gate cost sum mismatch', name)
        audit.check(row['episode_wall_seconds'] + 1e-6 >= row['prediction_seconds'] + row['gate_seconds'],
                    'Episode wall time is smaller than serial model+gate cost', name)
        audit.check(Counter(g['choice'] for g in gates) == Counter(row['gate_choices']), 'Gate choice counts mismatch', name)
        audit.check(row['method'] == 'laya' or not gates, 'Non-Laya method used a gate', name)
        expected_rows = defaultdict(list)
        for tick, item in enumerate(timeline):
            audit.check(item['tick'] == tick, 'Timeline tick order mismatch', name)
            expected_rows[item['active_call']].append(item['action_row'])
            audit.check(item['queue_remaining'] == horizon - item['action_row'] - 1,
                        'Finite queue length disagrees with action row', name)
            audit.check(0 <= item['active_call'] < len(calls), 'Invalid active call index', name)
        native_rows = 0
        unused_rows = 0
        for call_id, call in enumerate(calls):
            audit.check(call['call_id'] == call_id, 'Call ID/order mismatch', name)
            audit.check(call['native_output_shape'] == [horizon, 7] and call['nfe'] == 10, 'Call H/NFE mismatch', name)
            audit.check(call['noise_seed'] == SEED + row['task_id'] * 100000 + row['state_id'] * 1000 + call['request_tick'],
                        'Call RNG seed disagrees with registered tick-keyed schedule', name)
            audit.check(finite_number(call['predict_seconds']) and call['predict_seconds'] >= 0,
                        'Invalid call cost', name)
            used = call['executed_rows']
            audit.check(used == expected_rows[call_id], 'Call consumed rows differ from timeline', name)
            audit.check(all(isinstance(i, int) and 0 <= i < horizon for i in used), 'Consumed row outside finite chunk', name)
            audit.check(not used or used == list(range(used[0], used[0] + len(used))), 'Consumed rows are not contiguous', name)
            audit.check(call['unused_rows'] == horizon - len(used), 'Call unused-row count mismatch', name)
            with np.load(case / f'chunk_{call_id:03d}.npz', allow_pickle=False) as archive:
                chunk, normalized = archive['actions'], archive['normalized']
            audit.check(chunk.shape == (horizon, 7) and normalized.shape == (1, horizon, 7), 'Saved chunk shape mismatch', name)
            audit.check(np.isfinite(chunk).all() and np.isfinite(normalized).all(), 'Saved chunk contains non-finite values', name)
            ticks = [i for i, t in enumerate(timeline) if t['active_call'] == call_id]
            if ticks and all(0 <= index < horizon for index in used):
                audit.check(np.array_equal(actions[ticks], chunk[used].astype(actions.dtype)),
                            'Executed actions differ from corresponding generated chunk rows', name)
                if call_id == 0:
                    audit.check(ticks[0] == 0 and used[0] == 0, 'Initial chunk does not begin at row zero/tick zero', name)
                elif row['method'] == 'vlash_style_k5':
                    audit.check(used[0] == 0 and call.get('image_tick') == ticks[0] - 1 and call.get('state_tick') == ticks[0]
                                and call.get('current_state_alignment') is True, 'VLASH-style input alignment mismatch', name)
                else:
                    audit.check(used[0] == 1 and ticks[0] == call['request_tick'] + 1
                                and call.get('expired_prefix_rows') == 1, 'Naive logical-delay chunk installation mismatch', name)
                if call_id > 0:
                    audit.check(call.get('actual_delivery_tick') == ticks[0] == call['intended_delivery_tick'],
                                'Actual/intended delivery mismatch', name)
            native_rows += horizon
            unused_rows += horizon - len(used)
        audit.check(sum(len(c['executed_rows']) for c in calls) == steps, 'Executed row total mismatch', name)
        audit.check(row['native_generated_rows'] == native_rows and row['unused_generated_rows'] == unused_rows
                    and native_rows - unused_rows == steps, 'Native generation/utilization accounting mismatch', name)
        for gate in gates:
            validate_gate(gate, audit, name)
            response = run / 'laya_ipc' / 'responses' / (gate['request_id'] + '.json')
            request = run / 'laya_ipc' / 'requests' / (gate['request_id'] + '.json')
            if response.exists() and request.exists():
                original_response = read_json(response)
                original_request = read_json(request)
                audit.check(original_request['state'] == gate['state'], 'IPC input differs from saved gate state', name)
                audit.check(all(gate.get(k) == v for k, v in original_response.items()), 'IPC response differs from episode gate', name)
            else:
                audit.check(False, 'Missing original gate IPC request or response', name)
        return {'row': row, 'calls': calls, 'gates': gates, 'valid': len(audit.errors) == errors_before}
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        audit.check(False, type(exc).__name__ + ': ' + str(exc), name)
        return {'row': row, 'calls': [], 'gates': [], 'valid': False}


def aggregate(rows):
    total = len(rows)
    successes = sum(r['success'] for r in rows)
    steps = sum(r['steps'] for r in rows)
    calls = sum(r['vla_calls'] for r in rows)
    vla = sum(r['prediction_seconds'] for r in rows)
    gate = sum(r['gate_seconds'] for r in rows)
    generated = sum(r['native_generated_rows'] for r in rows)
    unused = sum(r['unused_generated_rows'] for r in rows)
    return {'episodes': total, 'successes': successes, 'success_rate': ratio(successes, total),
            'success_wilson95': wilson(successes, total), 'steps': steps,
            'vla_calls': calls, 'nfe': sum(r['nfe'] for r in rows), 'calls_per_100_steps': ratio(calls * 100, steps),
            'prediction_seconds': vla, 'gate_calls': sum(r['gate_calls'] for r in rows),
            'gate_seconds': gate, 'vla_plus_gate_seconds': vla + gate,
            'episode_wall_seconds': sum(r['episode_wall_seconds'] for r in rows),
            'simulated_seconds': sum(r['simulated_seconds'] for r in rows),
            'native_generated_rows': generated, 'unused_generated_rows': unused,
            'executed_rows_per_vla_call': ratio(steps, calls), 'generated_row_utilization': ratio(steps, generated),
            'unused_fraction': ratio(unused, generated)}


def clustered_cost_ratios(pairs, horizon):
    clusters = defaultdict(lambda: np.zeros(6, dtype=np.float64))
    for laya, base in pairs:
        clusters[laya['task_id']] += [laya['prediction_seconds'] + laya['gate_seconds'],
                                     base['prediction_seconds'] + base['gate_seconds'],
                                     laya['vla_calls'], base['vla_calls'],
                                     laya['episode_wall_seconds'], base['episode_wall_seconds']]
    if not clusters:
        return {'clusters': 0, 'replicates': 0, 'ratios': {}}
    matrix = np.stack([clusters[tid] for tid in sorted(clusters)])
    point = matrix.sum(axis=0)
    rng = np.random.default_rng(SEED + horizon)
    samples = rng.integers(0, len(matrix), size=(BOOTSTRAP_REPLICATES, len(matrix)))
    draws = matrix[samples].sum(axis=1)
    values = {}
    for name, numerator, denominator in [('vla_plus_gate', 0, 1), ('vla_calls', 2, 3), ('episode_wall', 4, 5)]:
        valid = draws[:, denominator] > 0
        resampled = draws[valid, numerator] / draws[valid, denominator]
        values[name] = {'point': ratio(float(point[numerator]), float(point[denominator])),
                        'bootstrap95_percentile': np.quantile(resampled, [.025, .975]).tolist() if len(resampled) else [None, None],
                        'valid_replicates': int(valid.sum())}
    return {'clusters': len(matrix), 'task_ids': sorted(clusters), 'pairs': len(pairs),
            'seed': SEED + horizon, 'replicates': BOOTSTRAP_REPLICATES,
            'resampling': 'Paired task-cluster bootstrap: sample tasks with replacement, retain all available paired states within each task.',
            'ratios': values, 'few_clusters_caution': len(matrix) < 20}


def paired_metrics(cases):
    lookup = {(r['row']['task_id'], r['row']['state_id'], r['row']['horizon'], r['row']['method']): r
              for r in cases if r['valid'] and r['row']['phase'] == 'main'}
    result = []
    for horizon in HORIZONS:
        pairs = []
        all_uncertain_checks = []
        cells = Counter()
        for tid in range(10):
            for sid in range(3):
                left = lookup.get((tid, sid, horizon, 'laya'))
                right = lookup.get((tid, sid, horizon, 'smol70'))
                if left is None or right is None:
                    continue
                laya, base = left['row'], right['row']
                pairs.append((laya, base))
                cells['both_success' if laya['success'] and base['success'] else
                      'laya_only' if laya['success'] else 'smol70_only' if base['success'] else 'both_failure'] += 1
                if left['gates'] and all(g['choice'] == 'uncertain' for g in left['gates']):
                    all_uncertain_checks.append({'task_id': tid, 'state_id': sid,
                        'request_ticks_equal': [c['request_tick'] for c in left['calls']] == [c['request_tick'] for c in right['calls']],
                        'actions_equal': laya['executed_action_sha256'] == base['executed_action_sha256'],
                        'success_equal': laya['success'] == base['success'],
                        'steps_equal': laya['steps'] == base['steps']})
        net = cells['laya_only'] - cells['smol70_only']
        result.append({'horizon': horizon, 'paired_episodes': len(pairs),
                       **{key: cells[key] for key in ['both_success', 'laya_only', 'smol70_only', 'both_failure']},
                       'discordant_pairs': cells['laya_only'] + cells['smol70_only'],
                       'success_net_count_laya_minus_smol70': net,
                       'success_rate_difference': ratio(net, len(pairs)),
                       'cost_ratio_laya_over_smol70': clustered_cost_ratios(pairs, horizon),
                       'all_uncertain_matched_trajectory_audit': all_uncertain_checks})
    return result


def gate_metrics(cases):
    grouped = defaultdict(list)
    for case in cases:
        if case['valid']:
            for gate in case['gates']:
                grouped[(case['row']['phase'], case['row']['horizon'], gate['prompt_sha256'])].append(gate)
    result = []
    for (phase, horizon, prompt), gates in sorted(grouped.items()):
        result.append({'phase': phase, 'horizon': horizon, 'prompt_sha256': prompt, 'requests': len(gates),
                       'choices': dict(Counter(g['choice'] for g in gates)),
                       'raw_choices': dict(Counter(g['raw_choice'] for g in gates)),
                       'probabilities': {label: distribution([g['probabilities'][label] for g in gates])
                                         for label in ['replan', 'continue', 'uncertain']},
                       'maximum_probability': distribution([max(g['probabilities'].values()) for g in gates]),
                       'full_gate_seconds': distribution([g['full_gate_seconds'] for g in gates]),
                       'sdk_synchronized_wall_ms': distribution([g['sdk_synchronized_wall_ms'] for g in gates]),
                       'tokens': distribution([g['token_audit']['tokens'] for g in gates])})
    return result


def analyze(run, out):
    out.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    manifest = read_json(run / 'run_manifest.json') if (run / 'run_manifest.json').exists() else {}
    status = read_json(run / 'status.json') if (run / 'status.json').exists() else {}
    rows = []
    if (run / 'episodes.jsonl').exists():
        for line_index, line in enumerate((run / 'episodes.jsonl').read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                audit.check(False, f'episodes.jsonl line {line_index}: {exc}')
    expected = {('pilot', 0, 0, h, m) for h in HORIZONS for m in METHODS}
    expected |= {('main', tid, sid, h, m) for tid in range(10) for sid in range(3) for h in HORIZONS for m in METHODS}
    keys = []
    cases = []
    for row in rows:
        try:
            key = episode_key(row)
            if audit.check(key in expected and key not in keys, 'Unexpected or duplicate episode key', str(key)):
                cases.append(validate_episode(run, row, audit))
            keys.append(key)
        except (KeyError, TypeError) as exc:
            audit.check(False, 'Malformed episode summary: ' + str(exc))
    available = set(keys)
    missing = sorted(expected - available)
    init_hashes = defaultdict(set)
    for case in cases:
        row = case['row']
        init_hashes[(row['task_id'], row['state_id'])].add(row['initial_state_sha256'])
    audit.check(all(len(hashes) == 1 for hashes in init_hashes.values()), 'Paired initial state hashes differ across arms/horizons/phases')
    audit.check(manifest.get('horizons') == HORIZONS and manifest.get('methods') == METHODS,
                'Manifest horizons/methods differ from frozen design')
    audit.check(manifest.get('tasks') == list(range(10)) and manifest.get('state_ids') == [0, 1, 2],
                'Manifest task/state partition differs from frozen design')
    audit.check(manifest.get('real_time') is False and manifest.get('inference_overlap') is False,
                'Manifest incorrectly labels execution as real time/overlapped')
    audit.check(manifest.get('logical_delay_steps') == 1 and manifest.get('gate_interval') == 5,
                'Manifest delay/gate interval mismatch')
    audit.check(manifest.get('checkpoint_training_horizon') == 50 and manifest.get('native_length_extrapolation') is True,
                'Manifest does not disclose native length extrapolation')
    audit.check(manifest.get('no_optimizer') is True and manifest.get('privileged_gate_state') is True,
                'Manifest freeze/privileged input declaration mismatch')
    ready = manifest.get('laya_ready', {})
    if ready:
        audit.check(ready.get('frozen') is True and ready.get('training') is False and ready.get('device') == 'cuda',
                    'Laya ready metadata does not show frozen CUDA inference')
        prompt_hashes = {g['prompt_sha256'] for case in cases for g in case['gates']}
        audit.check(not prompt_hashes or prompt_hashes == {ready['prompt_sha256']}, 'More than one prompt or mismatch with ready record')
    else:
        audit.warnings.append('Laya ready metadata is not yet available.')
    qualification = read_json(run / 'qualification.json') if (run / 'qualification.json').exists() else []
    qual_calls = 0
    qual_seconds = 0.0
    for item in qualification:
        horizon = item['horizon']
        audit.check(horizon in HORIZONS and item['native_shape'] == [horizon, 7] and item.get('repeated_exact') is True
                    and item.get('finite') is True, 'Invalid native-horizon qualification summary')
        try:
            with np.load(run / f'qualification_h{horizon}.npz', allow_pickle=False) as archive:
                qa, qn = archive['actions'], archive['normalized']
            audit.check(qa.shape == (horizon, 7) and qn.shape == (1, horizon, 7)
                        and np.isfinite(qa).all() and np.isfinite(qn).all(), 'Qualification saved shape/finite failure')
        except (OSError, ValueError, KeyError) as exc:
            audit.check(False, 'Qualification artifact missing/invalid: ' + str(exc))
        audit.check(finite_number(item.get('warm_seconds')) and finite_number(item.get('hot_seconds'))
                    and item['warm_seconds'] >= 0 and item['hot_seconds'] >= 0, 'Invalid qualification cost')
        qual_calls += 2
        qual_seconds += item['warm_seconds'] + item['hot_seconds']
    if len(qualification) != 3:
        audit.warnings.append('Qualification is incomplete: expected H=50/100/200, two calls each.')
    all_rows = [case['row'] for case in cases]
    valid_rows = [case['row'] for case in cases if case['valid']]
    totals = {phase: aggregate([row for row in valid_rows if row['phase'] == phase]) for phase in ['pilot', 'main']}
    manifest_finished = status.get('phase') == 'completed'
    if manifest_finished:
        audit.check(bool(ready), 'Finished run lacks frozen Laya ready metadata')
        audit.check(manifest.get('frozen_parameters_unchanged') is True
                    and manifest.get('parameter_sha256_before') == manifest.get('parameter_sha256_after')
                    and bool(manifest.get('parameter_sha256_before')), 'SmolVLA parameter freeze hash mismatch/missing')
        audit.check(len(qualification) == 3 and {q['horizon'] for q in qualification} == set(HORIZONS),
                    'Finished run lacks all three qualifications')
        audit.check(manifest.get('total_vla_calls_including_qualification') == qual_calls + sum(r['vla_calls'] for r in all_rows),
                    'Run VLA count does not reconcile qualification+pilot+main')
        audit.check(close(manifest.get('total_prediction_seconds_including_qualification'),
                          qual_seconds + sum(r['prediction_seconds'] for r in all_rows)), 'Run prediction cost does not reconcile')
    else:
        audit.warnings.append('Run lacks completed status; frozen-parameter terminal comparison and total cost reconciliation remain pending.')
    failures = sorted(str(path.relative_to(run)) for path in run.rglob('failure.json'))
    if failures:
        audit.check(False, 'Failure artifacts present: ' + ', '.join(failures))
    complete = (not missing and len(rows) == 372 and len(valid_rows) == 372 and manifest_finished and not audit.errors)
    completion = 'complete' if complete else 'partial'
    if not complete:
        audit.warnings.append('Partial or invalid evidence: do not present this run as the completed 360-main+12-pilot experiment.')
    cells = []
    for horizon in HORIZONS:
        for method in METHODS:
            chosen = [row for row in valid_rows if row['phase'] == 'main' and row['horizon'] == horizon and row['method'] == method]
            call_lengths = [len(call['executed_rows']) for case in cases if case['valid'] and case['row'] in chosen for call in case['calls']]
            cells.append({'horizon': horizon, 'method': method, **aggregate(chosen),
                          'expected_episodes': 30, 'arm_complete': len(chosen) == 30,
                          'gate_choices': dict(sum((Counter(row['gate_choices']) for row in chosen), Counter())),
                          'executed_rows_per_call_distribution': distribution(call_lengths)})
    pairs = paired_metrics(cases)
    gates = gate_metrics(cases)
    warmup = read_json(run / 'laya_warmup.json') if (run / 'laya_warmup.json').exists() else None
    metrics = {
        'schema_version': 1, 'run_path': str(run), 'completion_status': completion,
        'groups': [{**cell, 'n': cell['episodes'],
                    'wilson_low': cell['success_wilson95'][0], 'wilson_high': cell['success_wilson95'][1],
                    'calls_per100steps': cell['calls_per_100_steps']} for cell in cells],
        'main_cells': cells, 'phase_totals': totals, 'paired_laya_smol70': pairs, 'gate_by_prompt_phase_horizon': gates,
        'startup_and_qualification': {'smolvla_load_seconds': manifest.get('smolvla_load_seconds'),
            'laya_load_and_integrity_seconds': ready.get('load_and_integrity_wall_ms', 0) / 1000 if ready else None,
            'vla_qualification_calls': qual_calls, 'vla_qualification_nfe': 10 * qual_calls,
            'vla_qualification_seconds': qual_seconds,
            'laya_warmup_requests': 1 if warmup else 0,
            'laya_warmup_ipc_seconds': warmup.get('ipc_outer_seconds') if warmup else None},
        'cost_scope': 'Prediction includes preprocessing, transfers, model, postprocessing and synchronization. Gate includes state construction and IPC. Their sum is measured serial service wall time, not GPU kernel active time. Episode wall is the closed-loop body: it includes initial prediction and in-loop chunk saves, but excludes reset/settle, final artifact serialization and env.close. It already includes VLA and gate time; do not add them again. Cold loads, qualification and pilot are listed separately. Full campaign wall is available separately from supervisor.json.',
        'uncertainty_scope': 'Wilson intervals describe observed episode success under a binomial approximation. Episodes share tasks and are exposed development states. Cost ratio intervals resample ten paired task clusters; only ten clusters limits precision. Neither proves 2 percentage point noninferiority or generalization.',
        'method_scope': 'vlash_style_k5 is a delayed-image/current-state alignment mechanism on frozen SmolVLA. It is not the official VLASH checkpoint or its retrained asynchronous policy.',
        'timing_scope': 'Unpaced serial simulator, logical delay d=1, no actual inference/control overlap. No real-time speedup claim.',
    }
    audit_result = {'schema_version': 1, 'audited_utc': datetime.now(timezone.utc).isoformat(),
                    'run_path': str(run), 'completion_status': completion,
                    'audit_status': 'passed_complete' if complete else 'failed' if audit.errors else 'passed_partial',
                    'checks': audit.checked, 'errors': audit.errors, 'warnings': audit.warnings,
                    'episodes_seen': len(rows), 'episodes_valid': len(valid_rows),
                    'main_seen': sum(r.get('phase') == 'main' for r in rows),
                    'pilot_seen': sum(r.get('phase') == 'pilot' for r in rows),
                    'main_expected': 360, 'pilot_expected': 12,
                    'missing_episode_keys': [list(key) for key in missing],
                    'paired_initial_state_groups': len(init_hashes),
                    'main_arm_counts': [{'horizon': cell['horizon'], 'method': cell['method'], 'valid': cell['episodes'], 'expected': 30} for cell in cells],
                    'qualification_repeated_exact_note': 'The runner reports exact repeated outputs; only one array per H is retained, so the second array cannot be independently compared from this mirror.',
                    'failure_artifacts': failures,
                    'source_manifest_sha256': hashlib.sha256((run / 'run_manifest.json').read_bytes()).hexdigest() if manifest else None}
    write_json(out / 'audit.json', audit_result)
    write_json(out / 'metrics.json', metrics)
    columns = ['horizon', 'method', 'episodes', 'expected_episodes', 'arm_complete', 'successes', 'success_rate',
               'success_wilson95_low', 'success_wilson95_high', 'steps', 'vla_calls', 'nfe', 'calls_per_100_steps',
               'prediction_seconds', 'gate_calls', 'gate_seconds', 'vla_plus_gate_seconds', 'episode_wall_seconds',
               'simulated_seconds', 'native_generated_rows', 'unused_generated_rows', 'executed_rows_per_vla_call',
               'generated_row_utilization', 'unused_fraction']
    with (out / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for cell in cells:
            flat = {key: cell.get(key) for key in columns}
            flat['success_wilson95_low'], flat['success_wilson95_high'] = cell['success_wilson95']
            writer.writerow(flat)
    write_claim_validation(out, audit_result, metrics)
    return audit_result


def write_claim_validation(out, audit, metrics):
    fmt = lambda value: 'NA' if value is None else f'{value:.4f}'
    lines = ['# 长动作块补充实验：声明与证据检查', '',
             f"状态：**{audit['completion_status']}**；审计：{audit['audit_status']}。",
             f"已见主实验 {audit['main_seen']}/360、pilot {audit['pilot_seen']}/12；有效完整轨迹 {audit['episodes_valid']}/372。",
             '尚未完成或有审计错误时，下列表格仅为当前有效原件的描述性汇总，禁止称为完整实验结论。' if audit['completion_status'] != 'complete' else
             '372 个登记回合的文件、形状、执行动作绑定、成本与固定参数身份检查已通过。', '',
             '| H | 方法 | 成功/回合 | Wilson 95% | 调用/100步 | VLA+门控秒 | 回合主体墙钟秒 |',
             '|---|---|---:|---|---:|---:|---:|']
    for cell in metrics['main_cells']:
        interval = ', '.join(fmt(v) for v in cell['success_wilson95'])
        lines.append(f"| {cell['horizon']} | {cell['method']} | {cell['successes']}/{cell['episodes']} | [{interval}] | {fmt(cell['calls_per_100_steps'])} | {fmt(cell['vla_plus_gate_seconds'])} | {fmt(cell['episode_wall_seconds'])} |")
    lines += ['', '## Laya 与 Smol70 的配对结果', '',
              '费用比为 Laya / Smol70；小于 1 只说明此样本中的对应费用较低，必须同时查看成功损益与审计状态。区间按任务整簇配对重采样 10,000 次，固定种子；十个任务簇的区间精度有限。', '']
    for pair in metrics['paired_laya_smol70']:
        lines.append(f"- H={pair['horizon']}：{pair['paired_episodes']} 对；仅 Laya 成功 {pair['laya_only']}，仅 Smol70 成功 {pair['smol70_only']}，成功净差 {pair['success_net_count_laya_minus_smol70']:+d}。")
        for label, value in pair['cost_ratio_laya_over_smol70']['ratios'].items():
            lines.append(f"  {label} 比 {fmt(value['point'])}，配对任务簇 bootstrap 95% [{', '.join(fmt(v) for v in value['bootstrap95_percentile'])}]。")
        checks = pair['all_uncertain_matched_trajectory_audit']
        mismatches = sum(not all(c[key] for key in ['request_ticks_equal', 'actions_equal', 'success_equal', 'steps_equal']) for c in checks)
        lines.append(f'  全部弃权的 Laya 配对 {len(checks)} 个；实际请求时刻/动作/结果不一致 {mismatches} 个。')
    main_gate_choices = Counter()
    for group in metrics['gate_by_prompt_phase_horizon']:
        if group['phase'] == 'main':
            main_gate_choices.update(group['choices'])
    lines += ['', '## 可支持与不可支持的解释', '',
              f'- 主实验门控实际选择：{dict(main_gate_choices)}。概率分布和原始最大概率选择见 metrics.json。',
              '- H=100/200 使用在 H=50 上训练的冻结权重，属于推断长度外推；未进行长时域训练。',
              '- Laya 接收仿真特权物体坐标；本实验不验证从相机提取这些坐标的能力或成本，也不能声称直接视觉部署已成立。',
              '- Laya 的0.7接受阈值和1536 token输入预算未在本机器人分布重新校准；SDK的熵式confidence不当作正确率。',
              '- Smol70 指剩余动作比例 ≤70% 时请求补充，不是执行70%后请求；Laya uncertain 按该规则回退。',
              '- vlash_style_k5 仅在相同 SmolVLA 上实现旧图像/当前状态对齐机制，须与官方 VLASH 检查点结果分列。',
              '- 仿真没有按20Hz墙钟推进，推断与控制串行。d=1是逻辑延迟；不能将回合墙钟比或模型费用比称为实际异步机器人加速。',
              '- VLA+门控费用包含主机预处理、传输、同步、状态构造和IPC，不是纯GPU活跃秒。回合墙钟已包含这些费用，不能重复相加。加载、qualification、pilot 单列；前期工程与所有准备成本未被完整测量。',
              '- 调用量下降本身不证明Laya判断有效。应结合配对质量变化、含门控费用比和预算匹配对照；结果不支持时仍保留负结果。',
              '- 每格30回合来自10个任务、3个已暴露标准起点，不构成未见确认集；Wilson区间不是任务聚类校正区间。该规模不证明2个百分点非劣或跨任务泛化。',
              '- 只保存了每个H的一份qualification动作数组；重复完全一致由运行器断言记录，无法用本镜像独立比较未保留的第二数组。', '']
    if not main_gate_choices.get('continue', 0):
        lines.append('当前完整记录中没有阈值后 continue；不能宣称“Laya 自适应跳过已得到验证”。若只有 uncertain，应优先检查它是否复现Smol70并增加门控开销。')
    if audit['errors']:
        lines += ['', '## 未通过的审计项', '']
        lines.extend('- ' + (str(error['case']) + ': ' if error['case'] else '') + error['message'] for error in audit['errors'])
    if audit['warnings']:
        lines += ['', '## 保留事项', '']
        lines.extend('- ' + warning for warning in audit['warnings'])
    (out / 'claim_validation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run.resolve(), args.out.resolve())
    print(json.dumps({key: result[key] for key in ['completion_status', 'audit_status', 'checks', 'episodes_seen', 'episodes_valid']}, ensure_ascii=False))
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
