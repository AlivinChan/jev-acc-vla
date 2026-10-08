"""Independently audit the fresh 30-episode official VLASH reference on CPU.

Read only the archived raw artifacts; write derived audit.json, episodes.csv,
and summary.md to a separate --out directory. No policy or simulator imports.
"""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import traceback

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [100 * (center - radius), 100 * (center + radius)]


class Audit:
    def __init__(self):
        self.checks = 0
        self.errors = []
        self.files = {}

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def record(self, path):
        self.files[str(path)] = sha256(path)

    def read(self, path):
        self.record(path)
        return json.loads(path.read_text(encoding='utf-8'))


def audit_run(run, script, audit):
    check = audit.check
    contract = audit.read(run / 'contract.json')
    episodes = audit.read(run / 'episodes.json')
    summary = audit.read(run / 'summary.json')
    official = audit.read(run / 'official_eval_info.json')
    aliases = audit.read(run / 'missing_alias_audit.json')
    loading = audit.read(run / 'load_audit.json')
    before = audit.read(run / 'freeze_before.json')
    after = audit.read(run / 'freeze_after.json')
    audit.record(run / 'predict_calls.jsonl')
    predictions = [json.loads(line) for line in (run / 'predict_calls.jsonl').read_text().splitlines() if line.strip()]
    for key, expected in dict(suite='libero_spatial', delay_steps=1, prediction_horizon=50,
                              execution_horizon=5, seed=42, initial_states_per_task=3,
                              scheduled_episodes=30, batch_size=10, action_quant=1,
                              max_steps=230, smoke_only=False, training_steps=0, optimizer_steps=0).items():
        check(contract.get(key) == expected, f'Contract mismatch: {key}')
    check(contract['task_ids'] == list(range(10)), 'Contract task IDs differ from 0..9')
    check(contract['checkpoint_repo'] == 'mit-han-lab/vlash-pi05-libero-async5', 'Checkpoint repository mismatch')
    check(contract['checkpoint_revision'] == 'c85d5962a7b823f9f391f97d60227d6ca8bd7bc7', 'Checkpoint revision mismatch')
    check(contract['source_commit'] == 'd2da223e2251a4a5020b667e77cf8ad57e594144', 'VLASH source commit mismatch')
    if script is not None:
        audit.record(script)
        check(contract['reference_script_sha256'] == sha256(script), 'Runtime wrapper hash differs from reviewed script')
    check(not aliases['unresolved'], 'Unresolved missing model weights')
    check(loading == contract['load_audit'], 'Loading records differ from contract')
    check(all(not row['unexpected'] for row in loading), 'Unexpected model weights at load')
    missing = {key for row in loading for key in row['missing']}
    check(set(aliases['proven_aliases']) == missing, 'Missing weights do not match proved aliases')
    check(before == after, 'Before/after full freeze attestations differ')
    for label, snapshot in [('before', before), ('after', after)]:
        for flag in ['all_parameters_frozen', 'all_parameter_grads_none', 'all_modules_eval']:
            check(snapshot[flag] is True, f'{label}: {flag} failed')
        check(snapshot['training_steps'] == snapshot['optimizer_steps'] == 0, f'{label}: nonzero training counters')
        check(bool(re.fullmatch(r'[0-9a-f]{64}', snapshot['state_dict_sha256'])), f'{label}: invalid state digest')
        check(bool(snapshot['state_tensor_sha256']), f'{label}: empty tensor hash inventory')
        check(all(re.fullmatch(r'[0-9a-f]{64}', value) for value in snapshot['state_tensor_sha256'].values()),
              f'{label}: invalid per-tensor digest')
        counts = Counter()
        for parameter in snapshot['parameters']:
            check(parameter['requires_grad'] is False and parameter['grad_is_none'] is True,
                  f"{label}: unfrozen parameter {parameter['name']}")
            counts[parameter['dtype']] += math.prod(parameter['shape'])
        check(dict(counts) == snapshot['parameter_numel_by_dtype'], f'{label}: dtype inventory mismatch')
        check(sum(counts.values()) == contract['parameter_count'], f'{label}: parameter count mismatch')
    for omitted, loaded in aliases['proven_aliases'].items():
        check(loaded not in missing, f'Alias points to another missing key: {omitted}')
        check(before['state_tensor_sha256'][omitted] == before['state_tensor_sha256'][loaded],
              f'Alias tensor digest mismatch: {omitted}')
    check(summary['state_dict_sha256'] == before['state_dict_sha256'], 'Summary freeze digest mismatch')
    check(summary['freeze_verified_unchanged'] is True, 'Summary freeze flag missing')
    check(summary['parameter_numel_by_dtype'] == before['parameter_numel_by_dtype'], 'Summary dtype counts mismatch')
    check(summary['training_steps'] == summary['optimizer_steps'] == 0, 'Summary reports training')

    expected_pairs = {(tid, sid) for tid in range(10) for sid in range(3)}
    actual_pairs = [(row['task_id'], row['init_state_id']) for row in episodes]
    check(len(actual_pairs) == len(set(actual_pairs)) == 30, 'Episode pairs missing or duplicated')
    check(set(actual_pairs) == expected_pairs, 'Actual initial states differ from each task IDs 0/1/2')
    check(len(episodes) == summary['episodes'] == contract['scheduled_episodes'] == 30, 'Episode count mismatch')
    grouped = defaultdict(list)
    trace_parents = set()
    for row in episodes:
        declared = PurePosixPath(row['trace'].replace('\\', '/'))
        name = f"rollout_{row['batch']:03d}.npz"
        check(declared.name == name, 'Episode trace filename does not match batch')
        trace_parents.add(str(declared.parent))
        grouped[name].append(row)
        check(row['env_slot'] == row['task_id'], 'Environment slot/task ordering mismatch')
        check(row['batch'] == row['init_state_id'] == row['reset_count_before'], 'Initial state/reset rotation mismatch')
        check(row['seed'] == 42 + row['env_slot'], 'Official rollout seed mismatch')
    check(len(trace_parents) == 1, 'Episodes reference multiple remote output directories')
    check(set(grouped) == {f'rollout_{i:03d}.npz' for i in range(3)}, 'Expected exactly three rollout batches')
    check({p.name for p in run.glob('rollout_*.npz')} == set(grouped), 'Raw trace set differs from episode references')
    episode_details = []
    trace_reports = []
    for name, rows in sorted(grouped.items()):
        path = run / name
        audit.record(path)
        check(len(rows) == 10, f'{name}: batch must have ten episodes')
        with np.load(path, allow_pickle=False) as data:
            action, reward, done, success = data['action'], data['reward'], data['done'], data['success']
            check(action.ndim == 3 and action.shape[0] == 10 and action.shape[2] == 7, f'{name}: action shape mismatch')
            check(0 < action.shape[1] <= 230, f'{name}: horizon outside cap')
            check(done.shape == success.shape == reward.shape == action.shape[:2], f'{name}: trace shapes disagree')
            check(np.isfinite(action).all() and np.isfinite(reward).all(), f'{name}: nonfinite action/reward')
            check(np.isin(done, [0, 1]).all() and np.isin(success, [0, 1]).all(), f'{name}: invalid flags')
            check(done.any(axis=1).all(), f'{name}: unterminated stored episode')
            check(not np.any(done[:, :-1] & ~done[:, 1:]), f'{name}: done is not monotonic')
            for row in rows:
                slot = row['env_slot']
                first = int(np.flatnonzero(done[slot])[0])
                strict = bool(success[slot, :first + 1].any())
                published = bool(success[slot, :min(first + 2, success.shape[1])].any())
                check(row['trace_sha256'] == audit.files[str(path)], f'{name}: stored trace hash mismatch')
                check(row['first_done_index'] == first, f'{name}/{slot}: first termination mismatch')
                check(row['success_strict'] == strict, f'{name}/{slot}: strict success mismatch')
                check(row['success_official_mask'] == published, f'{name}/{slot}: official mask success mismatch')
                episode_details.append(dict(task_id=row['task_id'], init_state_id=row['init_state_id'],
                                            batch=row['batch'], env_slot=slot, seed=row['seed'],
                                            steps=first + 1, success_strict=strict,
                                            success_official_mask=published, trace=name,
                                            trace_sha256=audit.files[str(path)]))
            trace_reports.append(dict(file=name, sha256=audit.files[str(path)],
                                      batch=rows[0]['batch'], batch_size=action.shape[0],
                                      batch_control_steps=action.shape[1]))

    successes = sum(row['success_strict'] for row in episode_details)
    official_successes = sum(row['success_official_mask'] for row in episode_details)
    disagreements = sum(row['success_strict'] != row['success_official_mask'] for row in episode_details)
    strict_steps = sum(row['steps'] for row in episode_details)
    check(summary['successes'] == successes, 'Summary success count mismatch')
    check(abs(summary['pc_success'] - 100 * successes / 30) < 1e-8, 'Summary strict success rate mismatch')
    check(abs(summary['official_pc_success'] - 100 * official_successes / 30) < 1e-8, 'Summary official success rate mismatch')
    check(abs(official['overall']['pc_successes'] - 100 * official_successes / 30) < 1e-8,
          'Upstream evaluator success rate mismatch')
    check(disagreements == summary['mask_disagreements'], 'Success mask disagreement count mismatch')

    by_batch = defaultdict(list)
    for index, call in enumerate(predictions):
        check(call['call_index'] == index, 'Predict global call index mismatch')
        check(call['status'] == 'completed', 'Failed predict call in completed official run')
        check(call['batch_size'] == 10 and call['output_shape'] == [10, 5, 7], 'Predict batch/output shape mismatch')
        check(call['native_prediction_horizon'] == 50 and call['returned_execution_horizon'] == 5,
              'Predict native/execution horizon mismatch')
        check(call['all_model_parameters_frozen'] is True, 'Predict freeze flag false')
        check(math.isfinite(call['wall_seconds']) and call['wall_seconds'] >= 0, 'Invalid predict timing')
        by_batch[call['rollout_batch']].append(call)
    check(set(by_batch) == {0, 1, 2}, 'Predict batch set mismatch')
    active_elements = 0
    for trace in trace_reports:
        calls = by_batch[trace['batch']]
        expected_calls = (trace['batch_control_steps'] + 4) // 5
        check(len(calls) == expected_calls, f"Batch {trace['batch']}: K5 call count mismatch")
        for index, call in enumerate(calls):
            check(call['call_in_rollout'] == index and call['control_step'] == index * 5,
                  f"Batch {trace['batch']}: prediction cadence mismatch")
            active_elements += sum(row['steps'] > call['control_step'] for row in episode_details
                                   if row['batch'] == trace['batch'])
    predict_seconds = sum(row['wall_seconds'] for row in predictions)
    batch_elements = sum(row['batch_size'] for row in predictions)
    check(len(predictions) == summary['predict_calls'] == summary['predict_call_attempts'], 'Predict count summary mismatch')
    check(batch_elements == summary['predict_batch_elements'], 'Predict batch-element total mismatch')
    check(math.isclose(predict_seconds, summary['predict_wall_seconds'], rel_tol=1e-12, abs_tol=1e-8),
          'Predict wall total mismatch')
    check(0 <= predict_seconds <= summary['wall_seconds'] + 1e-6, 'Prediction wall exceeds evaluation wall')
    per_task = []
    for tid in range(10):
        rows = [row for row in episode_details if row['task_id'] == tid]
        per_task.append(dict(task_id=tid, episodes=len(rows), successes=sum(row['success_strict'] for row in rows),
                             initial_state_ids=sorted(row['init_state_id'] for row in rows),
                             strict_control_steps=sum(row['steps'] for row in rows)))
    return dict(episodes=30, successes=successes, pc_success=100 * successes / 30,
                wilson95_percent=wilson(successes, 30), mask_disagreements=disagreements,
                unique_initial_state_pairs=len(set(actual_pairs)), nominal_initial_states_match_primary=True,
                exact_cross_wrapper_physics_match_verified=False, strict_control_steps=strict_steps,
                actual_batched_environment_steps=sum(t['batch_size'] * t['batch_control_steps'] for t in trace_reports),
                predict_calls=len(predictions), predict_batch_elements=batch_elements,
                active_episode_prediction_elements=active_elements,
                already_done_prediction_elements=batch_elements - active_elements,
                predict_wall_seconds=predict_seconds, evaluation_wall_seconds=summary['wall_seconds'],
                freeze_attestations_equal=before == after, parameter_numel_by_dtype=before['parameter_numel_by_dtype'],
                parameter_count=contract['parameter_count'], state_dict_sha256=before['state_dict_sha256'],
                alias_count=len(aliases['proven_aliases']), per_task=per_task,
                episode_details=sorted(episode_details, key=lambda row: (row['task_id'], row['init_state_id'])),
                traces=trace_reports,
                limitations=['Official and Smol runs share nominal task/state IDs; official artifacts contain no initial physics snapshots, so bitwise cross-wrapper pairing is unverified.',
                             'Freeze hashes attest the runtime tensor state before and after. Archived hashes agree; the CPU audit does not reload checkpoint tensors.',
                             'Batch timing includes already-done environment slots and is not comparable to single-environment inference latency.',
                             'Wilson interval is descriptive for 30 fixed states across ten tasks; it does not establish independent generalization or paper-table equivalence.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--script', type=Path, default=Path(__file__).with_name('official_reference.py'))
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    if out == run or out.is_relative_to(run):
        raise ValueError('Derived audit output must be outside the raw run directory')
    out.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    result = {}
    try:
        result = audit_run(run, args.script.resolve(), audit)
    except Exception as exc:
        audit.errors.append(repr(exc))
        result['traceback'] = traceback.format_exc()
    result.update(verified=not audit.errors, checks=audit.checks, errors=audit.errors,
                  run=str(run), audited_utc=datetime.now(timezone.utc).isoformat(),
                  auditor_sha256=sha256(Path(__file__)), source_file_sha256=audit.files)
    (out / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# 官方 VLASH 新30回合：独立原件审计', '',
             f"审计状态：{'通过' if result['verified'] else '未通过'}；检查 {audit.checks} 项，错误 {len(audit.errors)} 项。", '']
    if 'episode_details' in result:
        with (out / 'episodes.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['episode_details'][0]))
            writer.writeheader()
            writer.writerows(result['episode_details'])
        lines += [f"严格成功 {result['successes']}/30 = {result['pc_success']:.1f}%；严格控制步合计 {result['strict_control_steps']}。",
                  f"每任务初态ID均为0/1/2；30组合唯一。官方mask与严格成功统计分歧 {result['mask_disagreements']} 个。", '',
                  f"批量预测 {result['predict_calls']} 次，实际计算 {result['predict_batch_elements']} 个样本，其中 {result['already_done_prediction_elements']} 个来自已结束槽位。",
                  f"同步预测wall合计 {result['predict_wall_seconds']:.3f} 秒，评估wall {result['evaluation_wall_seconds']:.3f} 秒。", '',
                  '| 任务ID | 初态ID | 成功/3 | 严格控制步 |', '|---:|---|---:|---:|']
        for row in result['per_task']:
            lines.append(f"| {row['task_id']} | {row['initial_state_ids']} | {row['successes']}/3 | {row['strict_control_steps']} |")
        lines += ['', '冻结前后完整哈希清单、dtype/参数数目、加载别名、原始动作/奖励/终止标记与预测日志已交叉核对。',
                  '主实验与官方参考仅名义任务/初态ID相同；官方未存起始physics快照，不能声称跨封装逐bit配对。此CPU审计不重新加载模型张量，也不测量真实异步控制。', '',
                  '[逐回合数据](episodes.csv) · [完整审计](audit.json)']
    if audit.errors:
        lines += ['', '错误：'] + ['- ' + message for message in audit.errors]
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({key: result.get(key) for key in ['verified', 'checks', 'errors', 'episodes', 'successes', 'strict_control_steps']}))
    return 0 if result['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
