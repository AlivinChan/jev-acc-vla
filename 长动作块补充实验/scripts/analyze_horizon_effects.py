"""CPU-only paired effects of native H100/H200 versus H50, with fixed task/state.

Reuses the primary analyzer's task-cluster cost bootstrap; no new experiment,
threshold fitting, source modification, or model imports are performed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import analyze_results as core


MAIN_METHODS = ['smol70', 'laya', 'naive_k5', 'vlash_style_k5']
DISPLAY = {'smol70': 'Smol70', 'laya': 'Laya τ=0.7', 'naive_k5': 'Naive K5',
           'vlash_style_k5': 'VLASH-style K5', 'laya_argmax_tau0': 'Laya argmax τ=0'}
HORIZONS = [50, 100, 200]
N_BOOTSTRAP = 10000
SUCCESS_STEP_SEED = 20261008


def load_rows(run, phase, errors):
    status = core.read_json(run / 'status.json') if (run / 'status.json').exists() else {}
    manifest = core.read_json(run / 'run_manifest.json') if (run / 'run_manifest.json').exists() else {}
    path = run / 'episodes.jsonl'
    rows = []
    keys = set()
    if not path.exists():
        errors.append({'source': str(path), 'error': 'missing episodes.jsonl'})
        return rows, status, manifest
    for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get('phase') != phase:
                continue
            key = core.episode_key(row)
            _, tid, sid, horizon, method = key
            if key in keys or tid not in range(10) or sid not in range(3) or horizon not in HORIZONS:
                raise ValueError('duplicate or out-of-scope episode key')
            if method not in (MAIN_METHODS if phase == 'main' else ['laya']):
                raise ValueError('out-of-scope method')
            keys.add(key)
            if core.read_json(core.case_directory(run, key) / 'result.json') != row:
                raise ValueError('episodes.jsonl and original result.json disagree')
            if not isinstance(row['success'], bool) or row['vla_calls'] < 1 or not 1 <= row['steps'] <= 230:
                raise ValueError('invalid episode outcome/count')
            if any(not core.finite_number(row[field]) or row[field] < 0
                   for field in ['prediction_seconds', 'gate_seconds', 'episode_wall_seconds']):
                raise ValueError('invalid timing values')
            rows.append(row)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append({'source': str(path), 'line': lineno, 'error': type(exc).__name__ + ': ' + str(exc)})
    return rows, status, manifest


def success_steps_bootstrap(pairs, longer_horizon):
    clusters = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    for longer, reference in pairs:
        clusters[longer['task_id']] += [int(longer['success']) - int(reference['success']),
                                       longer['steps'] - reference['steps'], 1]
    if not clusters:
        return {'task_clusters': 0, 'success_rate_difference_95': [None, None],
                'mean_step_difference_95': [None, None]}
    matrix = np.stack([clusters[key] for key in sorted(clusters)])
    seed = SUCCESS_STEP_SEED + longer_horizon
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(matrix), size=(N_BOOTSTRAP, len(matrix)))
    totals = matrix[indices].sum(axis=1)
    return {'task_clusters': len(matrix), 'seed': seed, 'replicates': N_BOOTSTRAP,
            'success_rate_difference_95': np.quantile(totals[:, 0] / totals[:, 2], [.025, .975]).tolist(),
            'mean_step_difference_95': np.quantile(totals[:, 1] / totals[:, 2], [.025, .975]).tolist(),
            'method': 'Paired task-cluster percentile bootstrap; keep all available paired states in each resampled task.'}


def compare(rows, source_phase, source_method, longer_horizon, errors):
    selected = [row for row in rows if row['method'] == source_method]
    lookup = {(row['task_id'], row['state_id'], row['horizon']): row for row in selected}
    pairs = []
    missing = []
    outcomes = Counter()
    pair_records = []
    for tid in range(10):
        for sid in range(3):
            longer = lookup.get((tid, sid, longer_horizon))
            reference = lookup.get((tid, sid, 50))
            if longer is None or reference is None:
                missing.append([tid, sid])
                continue
            if longer['initial_state_sha256'] != reference['initial_state_sha256']:
                errors.append({'source': source_phase, 'method': source_method, 'horizon': longer_horizon,
                               'task_id': tid, 'state_id': sid, 'error': 'H-long and H50 initial state hashes differ'})
                continue
            pairs.append((longer, reference))
            outcomes['both_success' if longer['success'] and reference['success'] else
                     'longer_only' if longer['success'] else 'h50_only' if reference['success'] else 'both_failure'] += 1
            pair_records.append({'task_id': tid, 'state_id': sid, 'initial_state_sha256': longer['initial_state_sha256'],
                                 'longer_success': longer['success'], 'h50_success': reference['success'],
                                 'steps_longer_minus_h50': longer['steps'] - reference['steps'],
                                 'vla_calls_longer_minus_h50': longer['vla_calls'] - reference['vla_calls'],
                                 'vla_plus_gate_seconds_longer_minus_h50':
                                     longer['prediction_seconds'] + longer['gate_seconds']
                                     - reference['prediction_seconds'] - reference['gate_seconds']})
    n = len(pairs)
    net = outcomes['longer_only'] - outcomes['h50_only']
    long_rows = [pair[0] for pair in pairs]
    reference_rows = [pair[1] for pair in pairs]
    long_metrics, reference_metrics = core.aggregate(long_rows), core.aggregate(reference_rows)
    steps_delta = long_metrics['steps'] - reference_metrics['steps']
    both_success = [(longer, reference) for longer, reference in pairs if longer['success'] and reference['success']]
    cost = core.clustered_cost_ratios(pairs, longer_horizon)
    return {'phase': source_phase, 'method': source_method if source_phase == 'main' else 'laya_argmax_tau0',
            'source_method': source_method, 'longer_horizon': longer_horizon, 'reference_horizon': 50,
            'paired_episodes': n, 'expected_pairs': 30, 'pairing_complete': n == 30,
            'exploratory_tau0': source_phase == 'argmax',
            **{key: outcomes[key] for key in ['both_success', 'longer_only', 'h50_only', 'both_failure']},
            'discordant_pairs': outcomes['longer_only'] + outcomes['h50_only'],
            'success_net_count_longer_minus_h50': net, 'success_rate_difference': core.ratio(net, n),
            'longer': long_metrics, 'h50': reference_metrics,
            'cost_ratios_longer_over_h50': cost,
            'steps_total_longer_minus_h50': steps_delta,
            'steps_mean_paired_difference': core.ratio(steps_delta, n),
            'steps_total_ratio_longer_over_h50': core.ratio(long_metrics['steps'], reference_metrics['steps']),
            'paired_success_steps_uncertainty': success_steps_bootstrap(pairs, longer_horizon),
            'both_success_steps_descriptive': {'pairs': len(both_success),
                'mean_steps_longer_minus_h50': core.ratio(sum(left['steps'] - right['steps'] for left, right in both_success), len(both_success)),
                'scope': 'Post-outcome subset description; not a causal speed estimate or replacement for all-pair results.'},
            'missing_task_state_pairs': missing, 'paired_details': pair_records}


def analyze(main_run, argmax_run, out):
    errors = []
    rows, status, manifest = load_rows(main_run, 'main', errors)
    comparisons = [compare(rows, 'main', method, horizon, errors) for method in MAIN_METHODS for horizon in [100, 200]]
    argmax_rows, argmax_status, argmax_manifest = [], {}, {}
    if argmax_run is not None:
        argmax_rows, argmax_status, argmax_manifest = load_rows(argmax_run, 'argmax', errors)
        comparisons.extend(compare(argmax_rows, 'argmax', 'laya', horizon, errors) for horizon in [100, 200])
    main_complete = len(rows) == 360 and status.get('phase') == 'completed'
    argmax_complete = argmax_run is None or (len(argmax_rows) == 90 and argmax_status.get('phase') == 'completed')
    complete = main_complete and argmax_complete and all(row['pairing_complete'] for row in comparisons) and not errors
    result = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'completion_status': 'complete' if complete else 'partial',
              'requested_input_scope': 'main_and_exploratory_argmax' if argmax_run else 'main_only',
              'main_run': str(main_run), 'argmax_run': str(argmax_run) if argmax_run else None,
              'main_episodes_seen': len(rows), 'main_expected': 360,
              'argmax_episodes_seen': len(argmax_rows), 'argmax_expected': 90 if argmax_run else 0,
              'main_parameter_hash': manifest.get('parameter_sha256_before'),
              'argmax_parameter_hash': argmax_manifest.get('parameter_sha256_before'),
              'comparisons': comparisons, 'errors': errors,
              'cost_bootstrap_definition': 'Reuse analyze_results.clustered_cost_ratios unchanged: 10,000 paired task-cluster draws, seed 20261007+longer_H, ratio of resampled summed costs, complete states within each task.',
              'validation_scope': 'Original result.json/JSONL consistency and same task/state/initial physical hash across H. Full trace/chunk finite checks and terminal model freeze are delegated to audit.json and argmax_audit.json.',
              'interpretation_limits': [
                  'Smol70 changes native length and the remaining-70% request threshold together. Its nominal first request is after 15/30/60 consumed steps at H50/100/200, so cross-H effects cannot be attributed to output length alone.',
                  'Laya tau=0.7 uncertain fallback inherits the H-dependent Smol70 threshold; its cross-H contrast can also change request frequency.',
                  'Naive K5 and VLASH-style K5 retain nominal five-step block cadence, but H still changes generated trajectories, unused tails and per-call cost.',
                  'Same H50-trained frozen checkpoint at all horizons: H100/200 are native inference-length extrapolation, not long-horizon retraining.',
                  'Actual rows consumed, especially rows51+/101+, must be read from task_breakdown.json; native generation of 200 rows does not establish use or quality of all 200.',
                  'Argmax tau=0 is a post-hoc exploratory stage after primary tau=0.7 observations. Keep stages separate and do not promote it to an independent confirmation set.',
                  'Cost ratios use serial measured service wall time including host work and gate IPC. No paced control, physical inference delay, inference/control overlap or real-time robot speedup is measured.',
                  'Only ten task clusters and three exposed states per task; bootstrap intervals do not establish generalization or 2 percentage point noninferiority.',
                  'Step counts include failures reaching the cap; all-pair step reductions can reflect changed success rather than faster successful control.']}
    out.mkdir(parents=True, exist_ok=True)
    core.write_json(out / 'horizon_effects.json', result)
    write_markdown(out, result)
    return result


def write_markdown(out, result):
    fmt = lambda x: 'NA' if x is None else f'{x:.3f}'
    lines = ['# 加长输出相对H=50的配对结果', '',
             f"状态：**{result['completion_status']}**；本文件范围：{result['requested_input_scope']}。主回合 {result['main_episodes_seen']}/360，纳入的argmax回合 {result['argmax_episodes_seen']}/{result['argmax_expected']}。", '',
             '每个比较匹配同一任务、同一初始状态，并核对初始物理状态哈希。费用比统一为长块/H50；成功净差为“仅长块成功”减“仅H50成功”。pilot不参与。', '',
             '| 方法 | 长块/H50 | 配对数 | 仅长块成功 | 仅H50成功 | 成功净差 | 调用量比及95% | VLA+门控费用比及95% | 平均步数变化 |',
             '|---|---|---:|---:|---:|---:|---|---|---:|']
    for row in result['comparisons']:
        ratio_cells = []
        for metric in ['vla_calls', 'vla_plus_gate']:
            cost = row['cost_ratios_longer_over_h50']['ratios'].get(metric, {})
            interval = cost.get('bootstrap95_percentile', [None, None])
            ratio_cells.append(f"{fmt(cost.get('point'))} [{fmt(interval[0])}, {fmt(interval[1])}]")
        lines.append(f"| {DISPLAY[row['method']]} | {row['longer_horizon']}/50 | {row['paired_episodes']} | {row['longer_only']} | {row['h50_only']} | {row['success_net_count_longer_minus_h50']:+d} | {ratio_cells[0]} | {ratio_cells[1]} | {fmt(row['steps_mean_paired_difference'])} |")
    lines += ['', '费用区间沿用主分析定义：按任务整簇配对重采样10,000次，固定种子；同任务的三个状态一起保留。成功率差与平均步数差的任务簇区间、总步数、两边原始费用以及逐配对结果均保存在horizon_effects.json。十个任务簇不足以支持宽泛的稳健性结论。', '',
              '## 解释边界', '',
              '- **Smol70同时改变输出长度和刷新周期。** H50/100/200的首次名义触发点分别为执行15/30/60步后；剩余量降到70%时请求。因此其跨H差异不能纯归因于输出变长。',
              '- Laya τ=0.7的uncertain回退继承该H相关触发条件；它的跨H差异也可能伴随调用频率变化。Naive K5和VLASH-style K5保持名义5步周期，但H仍会改变预测轨迹、未用尾部和单次推断成本。',
              '- 权重在全部H保持冻结，原检查点训练长度为50。H100/200是推断长度外推，不是重新训练得到的长规划策略。',
              '- 结合task_breakdown中的实际最大消费行及第51/101行以后执行量解读；输出200行不等于控制执行满200行。',
              '- τ=0消融在观察原阈值结果后追加，属于独立阶段的探索性分析，不能并入主实验或当作未见确认集。',
              '- 推断、门控与仿真串行执行；VLA+门控秒包含主机工作与IPC。本表不测实际机器人异步墙钟加速。费用比下降必须同时检查成功损益。',
              '- 平均步数变化包含失败跑满上限的回合；步数减少可能来自任务成败变化，不能直接称为成功执行更快。JSON中的“双成功”子集也仅供事后描述。',
              '- 每方法每H仅30个暴露开发起点，分属十个任务；不据此宣称泛化或2个百分点非劣。完整原件审计仍以audit.json/argmax_audit.json为准。']
    if result['requested_input_scope'] == 'main_only':
        lines += ['', '当前仅分析主实验。尚未将追加argmax阶段纳入，此文件的complete只表示所请求的主实验分析输入完整。']
    if result['completion_status'] != 'complete':
        lines += ['', '当前输入或配对未完整：所有数字仅为已取得配对的描述，禁止称为预定实验的完整结论。']
    if result['errors']:
        lines += ['', '## 输入不一致', '']
        lines.extend('- ' + json.dumps(error, ensure_ascii=False) for error in result['errors'])
    (out / 'horizon_effects.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--argmax', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.main.resolve(), args.argmax.resolve() if args.argmax else None, args.out.resolve())
    print(json.dumps({key: result[key] for key in ['completion_status', 'requested_input_scope', 'main_episodes_seen', 'argmax_episodes_seen']}, ensure_ascii=False))
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
