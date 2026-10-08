"""CPU audit of the post-hoc tau=0 Laya ablation; original tau=0.7 stays frozen.

--baseline accepts the original SmolVLA run directory, or its metrics.json. A
metrics file must resolve to its accessible run_path for raw paired auditing;
otherwise only descriptive aggregate comparisons are possible and status is
partial. No model packages are imported.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_results as core


HORIZONS = [50, 100, 200]
N_BOOT = 10000
SEED = 20261008


def gate_zero(gate, audit, label):
    audit.check(gate.get('status') == 'ok' and not gate.get('warmup', False), 'Invalid ablation gate status/warmup', label)
    probabilities = gate.get('probabilities', {})
    valid = set(probabilities) == {'continue', 'replan', 'uncertain'} and all(
        core.finite_number(x) and 0 <= x <= 1 for x in probabilities.values())
    audit.check(valid, 'Invalid three-way probabilities', label)
    audit.check(gate.get('minimum_probability') == 0.0, 'Ablation gate is not tau=0', label)
    audit.check(gate.get('choice') == gate.get('raw_choice') and gate.get('choice') in probabilities,
                'Tau=0 choice must equal raw choice, including semantic uncertain', label)
    if valid:
        audit.check(abs(sum(probabilities.values()) - 1) <= .002, 'Probability sum mismatch', label)
        audit.check(probabilities.get(gate.get('raw_choice'), -1) >= max(probabilities.values()) - .0002,
                    'Raw choice is not probability argmax', label)
    usage = gate.get('usage', {})
    token = gate.get('token_audit', {})
    audit.check(not usage.get('truncated') and not usage.get('state_tokens_dropped', 0) and not usage.get('options'),
                'SDK reports truncated/collapsed input', label)
    audit.check(token.get('max_len') == 1536 and token.get('head_max_len') == 192
                and not token.get('state', {}).get('truncated')
                and token.get('options', {}).get('options_distinct') == 3,
                'Gate packing audit failed', label)
    audit.check(gate.get('cpu_fallback_count') == 0, 'Gate CPU fallback occurred', label)
    audit.check(gate.get('tick', -1) > 0 and gate.get('tick', -1) % 5 == 0, 'Wrong gate cadence', label)
    state = gate.get('state', '')
    audit.check(isinstance(state, str) and hashlib.sha256(state.encode()).hexdigest() == gate.get('state_sha256'),
                'Gate state hash mismatch', label)
    for field in ['full_gate_seconds', 'ipc_outer_seconds', 'sdk_synchronized_wall_ms', 'service_prewrite_wall_ms']:
        audit.check(core.finite_number(gate.get(field)) and gate[field] >= 0, 'Invalid gate timing: ' + field, label)
    if core.finite_number(gate.get('full_gate_seconds')) and core.finite_number(gate.get('ipc_outer_seconds')):
        audit.check(gate['full_gate_seconds'] + 1e-6 >= gate['ipc_outer_seconds'], 'Gate cost omits IPC', label)


def validate_case(run, row, audit):
    key = core.episode_key(row)
    case = core.case_directory(run, key)
    label = str(case.relative_to(run))
    before = len(audit.errors)
    calls, gates = [], []
    try:
        audit.check(core.read_json(case / 'result.json') == row, 'JSONL/case result mismatch', label)
        initial = np.load(case / 'initial_state.npy', allow_pickle=False)
        audit.check(np.isfinite(initial).all() and core.sha(initial) == row['initial_state_sha256'],
                    'Initial-state finite/hash mismatch', label)
        with np.load(case / 'trace.npz', allow_pickle=False) as archive:
            actions, physics = archive['actions'], archive['physics']
        calls = core.read_json(case / 'calls.json')
        gates = core.read_json(case / 'gate.json')
        timeline = core.read_json(case / 'timeline.json')
        horizon, steps = row['horizon'], row['steps']
        audit.check(isinstance(steps, int) and 0 < steps <= 230, 'Invalid step count', label)
        audit.check(actions.shape == (steps, 7) and np.isfinite(actions).all(), 'Invalid executed actions', label)
        audit.check(physics.ndim == 2 and len(physics) == steps and np.isfinite(physics).all(), 'Invalid physics trace', label)
        audit.check(core.sha(actions) == row['executed_action_sha256'], 'Executed-action hash mismatch', label)
        audit.check(len(timeline) == steps, 'Timeline length mismatch', label)
        audit.check(len(calls) == row['vla_calls'] and len(calls) > 0 and row['nfe'] == len(calls) * 10,
                    'Call/NFE count mismatch', label)
        audit.check(len(gates) == row['gate_calls'], 'Gate count mismatch', label)
        audit.check(row['real_time'] is False and row['logical_delay_steps'] == 1, 'Invalid timing declaration', label)
        audit.check(core.close(row['simulated_seconds'], steps / 20), 'Simulated duration mismatch', label)
        audit.check(isinstance(row['success'], bool) and row['success'] == bool(timeline[-1]['success']),
                    'Terminal success mismatch', label)
        audit.check(not any(t['terminated'] or t['truncated'] for t in timeline[:-1]), 'Steps executed after termination', label)
        audit.check(steps == 230 or timeline[-1]['terminated'] or timeline[-1]['truncated'], 'Unexplained early termination', label)
        for field in ['prediction_seconds', 'gate_seconds', 'episode_wall_seconds']:
            audit.check(core.finite_number(row.get(field)) and row[field] >= 0, 'Invalid cost: ' + field, label)
        audit.check(core.close(sum(c['predict_seconds'] for c in calls), row['prediction_seconds']), 'VLA cost sum mismatch', label)
        audit.check(core.close(sum(g['full_gate_seconds'] for g in gates), row['gate_seconds']), 'Gate cost sum mismatch', label)
        audit.check(row['episode_wall_seconds'] + 1e-6 >= row['prediction_seconds'] + row['gate_seconds'],
                    'Serial costs exceed episode wall time', label)
        audit.check(Counter(g['choice'] for g in gates) == Counter(row['gate_choices']), 'Gate choice count mismatch', label)
        used_rows = defaultdict(list)
        used_ticks = defaultdict(list)
        for tick, item in enumerate(timeline):
            audit.check(item['tick'] == tick and 0 <= item['active_call'] < len(calls), 'Invalid timeline tick/call', label)
            audit.check(item['queue_remaining'] == horizon - item['action_row'] - 1, 'Finite queue/row mismatch', label)
            used_rows[item['active_call']].append(item['action_row'])
            used_ticks[item['active_call']].append(tick)
        for cid, call in enumerate(calls):
            used, ticks = call['executed_rows'], used_ticks[cid]
            audit.check(call['call_id'] == cid and call['native_output_shape'] == [horizon, 7] and call['nfe'] == 10,
                        'Call ID/native shape/NFE mismatch', label)
            audit.check(call['noise_seed'] == 20261007 + row['task_id'] * 100000 + row['state_id'] * 1000 + call['request_tick'],
                        'Tick-keyed seed mismatch', label)
            audit.check(core.finite_number(call['predict_seconds']) and call['predict_seconds'] >= 0, 'Invalid per-call cost', label)
            audit.check(used == used_rows[cid] and all(isinstance(i, int) and 0 <= i < horizon for i in used),
                        'Consumed rows mismatch or out of bounds', label)
            audit.check(not used or used == list(range(used[0], used[0] + len(used))), 'Non-contiguous row consumption', label)
            audit.check(call['unused_rows'] == horizon - len(used), 'Unused-row accounting mismatch', label)
            with np.load(case / f'chunk_{cid:03d}.npz', allow_pickle=False) as archive:
                chunk, normalized = archive['actions'], archive['normalized']
            audit.check(chunk.shape == (horizon, 7) and normalized.shape == (1, horizon, 7)
                        and np.isfinite(chunk).all() and np.isfinite(normalized).all(), 'Chunk finite/shape failure', label)
            if ticks and all(0 <= i < horizon for i in used):
                audit.check(np.array_equal(actions[ticks], chunk[used].astype(actions.dtype)), 'Executed action differs from chunk row', label)
                if cid == 0:
                    audit.check(used[0] == 0 and ticks[0] == 0 and call['request_tick'] == 0, 'Initial chunk alignment failure', label)
                else:
                    audit.check(used[0] == 1 and ticks[0] == call['request_tick'] + 1
                                and call.get('expired_prefix_rows') == 1
                                and call.get('actual_delivery_tick') == ticks[0] == call['intended_delivery_tick'],
                                'Logical d=1 action installation failure', label)
        audit.check(sum(len(call['executed_rows']) for call in calls) == steps, 'Total consumed rows mismatch', label)
        audit.check(row['native_generated_rows'] == len(calls) * horizon
                    and row['unused_generated_rows'] == len(calls) * horizon - steps,
                    'Generated/unused totals mismatch', label)
        gate_ticks = set()
        for gate in gates:
            gate_zero(gate, audit, label)
            audit.check(gate['tick'] not in gate_ticks, 'Duplicate gate tick', label)
            gate_ticks.add(gate['tick'])
            original_response = core.read_json(run / 'laya_ipc/responses' / (gate['request_id'] + '.json'))
            original_request = core.read_json(run / 'laya_ipc/requests' / (gate['request_id'] + '.json'))
            audit.check(original_request['state'] == gate['state'] and all(gate.get(k) == v for k, v in original_response.items()),
                        'Original IPC evidence differs from episode gate', label)
            matching = [call for call in calls if call['request_tick'] == gate['tick']]
            if gate['choice'] == 'replan':
                audit.check(len(matching) == 1 and matching[0]['trigger'] == 'laya_replan', 'Replan choice did not trigger VLA', label)
            elif gate['choice'] == 'continue':
                audit.check(not matching, 'Continue choice unexpectedly triggered VLA', label)
            else:
                expected_request = gate['remaining'] <= .7 * horizon
                audit.check(bool(matching) == expected_request, 'Semantic uncertain does not follow Smol70 fallback', label)
        return {'row': row, 'calls': calls, 'gates': gates, 'valid': len(audit.errors) == before}
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        audit.check(False, type(exc).__name__ + ': ' + str(exc), label)
        return {'row': row, 'calls': calls, 'gates': gates, 'valid': False}


def jsonl_rows(path, audit):
    rows = []
    if not path.exists():
        audit.warnings.append('Not yet present: ' + str(path))
        return rows
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                audit.check(False, f'{path.name} line {line_number}: {exc}')
    return rows


def cluster_success(rows, seed):
    clusters = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    for row in rows:
        clusters[row['task_id']] += [int(row['success']), 1]
    if not clusters:
        return {'clusters': 0, 'interval95': [None, None]}
    matrix = np.stack([clusters[key] for key in sorted(clusters)])
    rng = np.random.default_rng(seed)
    totals = matrix[rng.integers(0, len(matrix), size=(N_BOOT, len(matrix)))].sum(axis=1)
    rates = totals[:, 0] / totals[:, 1]
    return {'clusters': len(matrix), 'seed': seed, 'replicates': N_BOOT,
            'interval95': np.quantile(rates, [.025, .975]).tolist(), 'kind': 'task-cluster percentile bootstrap'}


def compare_pairs(left_rows, right_rows, horizon):
    lookup = {(row['task_id'], row['state_id'], row['horizon']): row for row in right_rows}
    pairs = [(row, lookup[(row['task_id'], row['state_id'], horizon)]) for row in left_rows
             if row['horizon'] == horizon and (row['task_id'], row['state_id'], horizon) in lookup]
    counts = Counter()
    clusters = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    for left, right in pairs:
        counts['both_success' if left['success'] and right['success'] else
               'argmax_only' if left['success'] else 'baseline_only' if right['success'] else 'both_failure'] += 1
        clusters[left['task_id']] += [int(left['success']) - int(right['success']), 1]
    difference_interval = [None, None]
    if clusters:
        matrix = np.stack([clusters[key] for key in sorted(clusters)])
        rng = np.random.default_rng(SEED + horizon)
        totals = matrix[rng.integers(0, len(matrix), size=(N_BOOT, len(matrix)))].sum(axis=1)
        difference_interval = np.quantile(totals[:, 0] / totals[:, 1], [.025, .975]).tolist()
    net = counts['argmax_only'] - counts['baseline_only']
    return {'horizon': horizon, 'pairs': len(pairs),
            **{key: counts[key] for key in ['both_success', 'argmax_only', 'baseline_only', 'both_failure']},
            'discordant_pairs': counts['argmax_only'] + counts['baseline_only'],
            'success_net_count_argmax_minus_baseline': net,
            'success_rate_difference': core.ratio(net, len(pairs)),
            'success_difference_task_cluster_bootstrap95': difference_interval,
            'success_difference_seed': SEED + horizon,
            'cost_ratios_argmax_over_baseline': core.clustered_cost_ratios(pairs, horizon)}


def load_baseline(path, audit):
    metrics = {}
    if path.is_file():
        metrics = core.read_json(path)
        candidate = Path(metrics.get('run_path', ''))
        run = candidate if candidate.is_dir() and (candidate / 'episodes.jsonl').exists() else None
    else:
        run = path if path.is_dir() else None
    if run is None:
        audit.warnings.append('Baseline raw run is unavailable; aggregate metrics cannot validate paired initial states or paired uncertainty.')
        return [], {}, metrics, None
    manifest = core.read_json(run / 'run_manifest.json') if (run / 'run_manifest.json').exists() else {}
    rows = [row for row in jsonl_rows(run / 'episodes.jsonl', audit) if row.get('phase') == 'main']
    seen = set()
    good = []
    for row in rows:
        key = core.episode_key(row)
        valid = audit.check(key not in seen, 'Duplicate baseline main episode', str(key))
        seen.add(key)
        case = core.case_directory(run, key)
        try:
            initial = np.load(case / 'initial_state.npy', allow_pickle=False)
            valid &= audit.check(core.sha(initial) == row['initial_state_sha256'] and np.isfinite(initial).all(),
                                 'Baseline initial-state artifact hash mismatch', str(key))
            valid &= audit.check(core.read_json(case / 'result.json') == row, 'Baseline summary/original result mismatch', str(key))
            if valid:
                good.append(row)
        except (OSError, ValueError, KeyError) as exc:
            audit.check(False, 'Baseline original evidence unavailable: ' + str(exc), str(key))
    audit.check(manifest.get('frozen_parameters_unchanged') is True and
                manifest.get('parameter_sha256_before') == manifest.get('parameter_sha256_after'),
                'Original baseline lacks matched terminal freeze hashes')
    return good, manifest, metrics, run


def analyze(run, baseline, out):
    out.mkdir(parents=True, exist_ok=True)
    audit = core.Audit()
    manifest = core.read_json(run / 'run_manifest.json') if (run / 'run_manifest.json').exists() else {}
    status = core.read_json(run / 'status.json') if (run / 'status.json').exists() else {}
    baseline_rows, base_manifest, baseline_metrics, base_run = load_baseline(baseline, audit)
    rows = jsonl_rows(run / 'episodes.jsonl', audit)
    expected = {('argmax', tid, sid, horizon, 'laya') for tid in range(10) for sid in range(3) for horizon in HORIZONS}
    keys, cases = [], []
    for row in rows:
        try:
            key = core.episode_key(row)
            if audit.check(key in expected and key not in keys, 'Unexpected or duplicate ablation episode', str(key)):
                cases.append(validate_case(run, row, audit))
            keys.append(key)
        except (KeyError, TypeError) as exc:
            audit.check(False, 'Malformed episode: ' + str(exc))
    missing = sorted(expected - set(keys))
    audit.check(manifest.get('minimum_probability') == 0.0 and manifest.get('exploratory') is True,
                'Manifest must disclose tau=0 and exploratory status')
    audit.check(manifest.get('horizons') == HORIZONS and manifest.get('tasks') == list(range(10))
                and manifest.get('state_ids') == [0, 1, 2], 'Ablation partition/H mismatch')
    audit.check(manifest.get('real_time') is False and manifest.get('inference_overlap') is False
                and manifest.get('logical_delay_steps') == 1 and manifest.get('gate_interval') == 5,
                'Logical timing protocol mismatch')
    audit.check(manifest.get('no_optimizer') is True and manifest.get('privileged_gate_state') is True
                and manifest.get('checkpoint_training_horizon') == 50,
                'Freeze/privileged-input/training-horizon declaration mismatch')
    ready = manifest.get('laya_ready', {})
    if ready:
        audit.check(ready.get('frozen') is True and ready.get('training') is False and ready.get('device') == 'cuda'
                    and ready.get('threshold') == 0.0, 'Laya readiness does not qualify frozen tau=0 CUDA inference')
        hashes = {gate['prompt_sha256'] for case in cases for gate in case['gates']}
        audit.check(not hashes or hashes == {ready['prompt_sha256']}, 'Gate prompt differs within ablation')
        if base_manifest.get('laya_ready'):
            audit.check(ready['prompt_sha256'] == base_manifest['laya_ready']['prompt_sha256'],
                        'Prompt changed as well as threshold; not a threshold-only ablation')
    elif rows:
        audit.check(False, 'Ablation Laya ready metadata missing')
    base_smol = {(row['task_id'], row['state_id'], row['horizon']): row for row in baseline_rows if row['method'] == 'smol70'}
    matched = 0
    for case in cases:
        row = case['row']
        key = row['task_id'], row['state_id'], row['horizon']
        counterpart = base_smol.get(key)
        if counterpart:
            paired = audit.check(row['initial_state_sha256'] == counterpart['initial_state_sha256'],
                                 'Argmax/Smol70 initial state hash differs', str(key))
            case['valid'] &= paired
            matched += int(paired)
        else:
            audit.warnings.append('No original Smol70 paired state for ' + str(key))
    freeze_records = {}
    for name in ['freeze_before.json', 'freeze_after.json']:
        if (run / name).exists():
            freeze_records[name] = core.read_json(run / name)
            freeze = freeze_records[name]
            audit.check(all(freeze.get(flag) is True for flag in
                            ['all_parameters_frozen', 'all_parameter_grads_none', 'all_modules_eval'])
                        and freeze.get('training_steps') == freeze.get('optimizer_steps') == 0,
                        'Freeze record does not show unchanged inference-only parameters: ' + name)
    if 'freeze_before.json' in freeze_records:
        audit.check(freeze_records['freeze_before.json'].get('state_dict_sha256') == manifest.get('parameter_sha256_before'),
                    'Before-freeze evidence differs from manifest')
    if base_manifest.get('parameter_sha256_before'):
        audit.check(manifest.get('parameter_sha256_before') == base_manifest['parameter_sha256_before'],
                    'Ablation and original baseline checkpoint parameter identities differ')
    qualification = core.read_json(run / 'qualification.json') if (run / 'qualification.json').exists() else []
    qual_seconds, qual_calls = 0.0, 0
    for item in qualification:
        horizon = item['horizon']
        audit.check(horizon in HORIZONS and item['native_shape'] == [horizon, 7]
                    and item.get('repeated_exact') is True and item.get('finite') is True, 'Qualification record malformed')
        try:
            with np.load(run / f'qualification_h{horizon}.npz', allow_pickle=False) as archive:
                actions, norm = archive['actions'], archive['normalized']
            audit.check(actions.shape == (horizon, 7) and norm.shape == (1, horizon, 7)
                        and np.isfinite(actions).all() and np.isfinite(norm).all(), 'Qualification shape/finite failure')
        except (OSError, ValueError, KeyError) as exc:
            audit.check(False, 'Qualification artifact error: ' + str(exc))
        qual_seconds += item['warm_seconds'] + item['hot_seconds']
        qual_calls += 2
    terminal = status.get('phase') == 'completed'
    warmup = core.read_json(run / 'laya_warmup.json') if (run / 'laya_warmup.json').exists() else None
    if warmup:
        audit.check(warmup.get('warmup') is True and warmup.get('minimum_probability') == 0.0
                    and warmup.get('choice') == warmup.get('raw_choice'), 'Warmup is not separately marked tau=0')
    if terminal:
        audit.check(manifest.get('frozen_parameters_unchanged') is True and bool(manifest.get('parameter_sha256_before'))
                    and manifest.get('parameter_sha256_before') == manifest.get('parameter_sha256_after'),
                    'Terminal parameter identity is not frozen')
        audit.check(set(freeze_records) == {'freeze_before.json', 'freeze_after.json'}, 'Both explicit freeze records required')
        if 'freeze_after.json' in freeze_records:
            audit.check(freeze_records['freeze_after.json'].get('state_dict_sha256') == manifest.get('parameter_sha256_after'),
                        'After-freeze evidence differs from manifest')
        if len(freeze_records) == 2:
            audit.check(freeze_records['freeze_before.json'] == freeze_records['freeze_after.json'],
                        'Before/after freeze records differ')
        audit.check(len(qualification) == 3 and {q['horizon'] for q in qualification} == set(HORIZONS),
                    'Finished ablation lacks three paired qualifications')
        audit.check(manifest.get('total_vla_calls_including_qualification') == qual_calls + sum(row['vla_calls'] for row in rows),
                    'Total VLA count reconciliation failed')
        audit.check(core.close(manifest.get('total_prediction_seconds_including_qualification'),
                              qual_seconds + sum(row['prediction_seconds'] for row in rows)), 'Total VLA cost reconciliation failed')
        audit.check(bool(warmup) and manifest.get('total_gate_calls_including_warmup') == 1 + sum(row['gate_calls'] for row in rows),
                    'Total gate count does not reconcile explicit warmup and episode calls')
    failures = sorted(str(path.relative_to(run)) for path in run.rglob('failure.json'))
    audit.check(not failures, 'Failure artifacts present: ' + ', '.join(failures))
    valid_rows = [case['row'] for case in cases if case['valid']]
    complete = not missing and len(rows) == len(valid_rows) == matched == 90 and terminal and not audit.errors
    completion = 'complete' if complete else 'partial'
    if not complete:
        audit.warnings.append('Incomplete/invalid ablation; no complete 90-episode result claim is permitted.')
    groups = []
    for horizon in HORIZONS:
        selected = [row for row in valid_rows if row['horizon'] == horizon]
        aggregate = core.aggregate(selected)
        groups.append({'horizon': horizon, 'method': 'laya_argmax_tau0', **aggregate,
                       'n': len(selected), 'expected_n': 30,
                       'wilson_low': aggregate['success_wilson95'][0], 'wilson_high': aggregate['success_wilson95'][1],
                       'calls_per100steps': aggregate['calls_per_100_steps'],
                       'gate_choices': dict(sum((Counter(row['gate_choices']) for row in selected), Counter())),
                       'success_task_cluster_bootstrap95': cluster_success(selected, SEED + horizon)})
    comparisons = []
    for method in ['smol70', 'laya', 'naive_k5', 'vlash_style_k5']:
        method_rows = [row for row in baseline_rows if row['method'] == method]
        for horizon in HORIZONS:
            pairs = compare_pairs(valid_rows, method_rows, horizon)
            selected = [row for row in method_rows if row['horizon'] == horizon]
            aggregate = core.aggregate(selected)
            if not selected and baseline_metrics:
                aggregate = next((g for g in baseline_metrics.get('groups', baseline_metrics.get('main_cells', []))
                                  if g.get('horizon') == horizon and g.get('method') == method), aggregate)
            comparisons.append({'baseline_method': method, 'baseline_threshold': .7 if method == 'laya' else None,
                                **pairs, 'baseline_descriptive': aggregate,
                                'comparison_is_posthoc_and_cross_run': True})
    metrics = {'schema_version': 1, 'run_path': str(run), 'baseline_path': str(baseline),
               'completion_status': completion, 'exploratory': True, 'minimum_probability': 0.0,
               'original_threshold_contract_unchanged': .7, 'groups': groups,
               'paired_comparisons': comparisons, 'gate_by_prompt_phase_horizon': core.gate_metrics(cases),
               'phase_totals': {'argmax': core.aggregate(valid_rows)},
               'startup_and_qualification': {'qualification_vla_calls': qual_calls, 'qualification_nfe': qual_calls * 10,
                   'qualification_prediction_seconds': qual_seconds, 'smolvla_load_seconds': manifest.get('smolvla_load_seconds'),
                   'laya_load_and_integrity_seconds': ready.get('load_and_integrity_wall_ms', 0) / 1000 if ready else None,
                   'laya_warmup_requests': 1 if warmup else 0,
                   'laya_warmup_ipc_seconds': warmup.get('ipc_outer_seconds') if warmup else None},
               'parameter_identity': {'before': manifest.get('parameter_sha256_before'), 'after': manifest.get('parameter_sha256_after'),
                   'baseline_before': base_manifest.get('parameter_sha256_before'), 'freeze_records': freeze_records},
               'limits': ['Post-hoc exploration motivated by tau=0.7 abstentions; not a new independent confirmation set.',
                   'Raw semantic uncertain remains a possible argmax and still falls back to Smol70; tau=0 removes confidence rejection only.',
                   'Measured serial VLA+gate service wall cost includes host work; it is not GPU kernel time or real-time robot speedup.',
                   'Only 10 task clusters; bootstrap intervals do not establish 2 percentage point noninferiority or generalization.',
                   'Cross-run GPU interference and time order can influence latency. Paired origins do not randomize run order.',
                   'Same frozen H=50-trained checkpoint extrapolated to H100/200; privileged simulator object coordinates.',
                   'VLASH-style is a same-SmolVLA input-alignment control, not the official VLASH trained checkpoint.']}
    result = {'schema_version': 1, 'audited_utc': datetime.now(timezone.utc).isoformat(), 'completion_status': completion,
              'audit_status': 'passed_complete' if complete else 'failed' if audit.errors else 'passed_partial',
              'checks': audit.checked, 'errors': audit.errors, 'warnings': audit.warnings,
              'expected_episodes': 90, 'episodes_seen': len(rows), 'episodes_valid': len(valid_rows),
              'initial_states_matched_to_smol70': matched, 'missing_episode_keys': [list(key) for key in missing],
              'horizon_counts': {str(h): sum(row['horizon'] == h for row in valid_rows) for h in HORIZONS},
              'parameter_identity': metrics['parameter_identity'], 'failure_artifacts': failures,
              'baseline_raw_path': str(base_run) if base_run else None,
              'baseline_validation_scope': 'Baseline JSONL/original result/initial state identities and freeze manifest verified here; full baseline trace audit remains the original analyze_results.py report.',
              'run_manifest_sha256': hashlib.sha256((run / 'run_manifest.json').read_bytes()).hexdigest() if manifest else None}
    core.write_json(out / 'argmax_metrics.json', metrics)
    core.write_json(out / 'argmax_audit.json', result)
    summary(out, result, metrics)
    return result


def summary(out, audit, metrics):
    fmt = lambda value: 'NA' if value is None else f'{value:.4f}'
    lines = ['# Laya 原始 argmax 探索性消融', '',
             f"状态：**{audit['completion_status']}**；审计：{audit['audit_status']}。有效回合 {audit['episodes_valid']}/90；与原Smol70初始状态匹配 {audit['initial_states_matched_to_smol70']}/90。", '',
             '该消融是在观察0.7阈值弃权后追加的开发分析，原360回合与0.7阈值结果保持原样。tau=0仅取消置信度拒绝；语义选项uncertain仍可被原始argmax选中，并按Smol70回退。', '',
             '| H | 成功/回合 | Wilson95% | 调用/100步 | VLA+门控秒 | 墙钟秒 | 门控选择 |',
             '|---|---:|---|---:|---:|---:|---|']
    for group in metrics['groups']:
        lines.append(f"| {group['horizon']} | {group['successes']}/{group['n']} | [{fmt(group['wilson_low'])}, {fmt(group['wilson_high'])}] | {fmt(group['calls_per100steps'])} | {fmt(group['vla_plus_gate_seconds'])} | {fmt(group['episode_wall_seconds'])} | {group['gate_choices']} |")
    lines += ['', '## 同起点配对比较', '',
              '费用比为 argmax / 对照；置信区间按任务簇整簇配对重采样10,000次。成功净差为argmax独自成功数减去对照独自成功数。', '',
              '| H | 原对照 | 配对数 | 仅argmax成功 | 仅对照成功 | 成功净差 | VLA+门控费用比及95% |',
              '|---|---|---:|---:|---:|---:|---|']
    for comparison in metrics['paired_comparisons']:
        cost = comparison['cost_ratios_argmax_over_baseline']['ratios'].get('vla_plus_gate', {})
        interval = cost.get('bootstrap95_percentile', [None, None])
        lines.append(f"| {comparison['horizon']} | {comparison['baseline_method']} | {comparison['pairs']} | {comparison['argmax_only']} | {comparison['baseline_only']} | {comparison['success_net_count_argmax_minus_baseline']:+d} | {fmt(cost.get('point'))} [{fmt(interval[0])}, {fmt(interval[1])}] |")
    lines += ['', '## 解释限制', '',
              '- 数据未满90回合或审计失败时，上表仅描述当前有效原件，不能声称消融已完成。',
              '- 原始分数、选择和逐状态改变见argmax_metrics.json。降低阈值本身不是训练、校准或语义判断有效性的证明。',
              '- 必须联合查看成功损益与包含门控的费用比。调用减少不能单独证明收益；失败回合可能更长，费用与成功不能拆开解释。',
              '- 无真实20Hz墙钟推进、无模型与控制并发；d=1为逻辑延迟。模型+门控秒包含主机处理、同步与IPC，不能作为纯GPU秒或实际异步机器人加速。',
              '- 此消融在原实验后顺序运行，共享GPU的负载与时间顺序可能影响时延；配对起点不能消除这一因素。',
              '- 三十回合共享十个任务；任务簇bootstrap区间精度有限，Wilson区间又采用二项近似。未见泛化、2个百分点非劣均未被此规模证明。',
              '- H100/200仍为H50训练权重的推断长度外推；Laya输入含仿真特权坐标。VLASH-style结果不是官方VLASH检查点复现。',
              '- qualification仅保存每H一份动作数组，重复相等由运行器断言记录；第二份数组无法独立复算。']
    if audit['errors']:
        lines += ['', '## 审计未通过项', '']
        lines.extend('- ' + (str(item['case']) + ': ' if item['case'] else '') + item['message'] for item in audit['errors'])
    if audit['warnings']:
        lines += ['', '## 保留事项', '']
        lines.extend('- ' + warning for warning in audit['warnings'])
    (out / 'argmax_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run.resolve(), args.baseline.resolve(), args.out.resolve())
    print(json.dumps({key: result[key] for key in ['completion_status', 'audit_status', 'episodes_seen', 'episodes_valid']}, ensure_ascii=False))
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
