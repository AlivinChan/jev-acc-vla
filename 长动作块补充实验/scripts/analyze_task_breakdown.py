"""Lightweight read-only JSON diagnostics, complementary to the full trace audits.

Only main phase rows and the separately labelled exploratory argmax phase are
included. No model imports, threshold selection, fitting, or causal inference.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


HORIZONS = [50, 100, 200]
MAIN_METHODS = ['smol70', 'laya', 'naive_k5', 'vlash_style_k5']
DISPLAY = {'smol70': 'Smol70', 'laya': 'Laya τ=0.7', 'naive_k5': 'Naive K5',
           'vlash_style_k5': 'VLASH-style K5', 'laya_argmax_tau0': 'Laya argmax τ=0'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def probability_range(values):
    if not values:
        return {'n': 0, 'min': None, 'max': None, 'mean': None, 'median': None}
    return {'n': len(values), 'min': min(values), 'max': max(values),
            'mean': statistics.fmean(values), 'median': statistics.median(values)}


def load_cases(run, phase, errors):
    cases = []
    if run is None:
        return cases, {}
    status = read_json(run / 'status.json') if (run / 'status.json').exists() else {}
    path = run / 'episodes.jsonl'
    if not path.exists():
        errors.append({'source': str(run), 'error': 'episodes.jsonl missing'})
        return cases, status
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get('phase') != phase:
                continue
            tid, sid, horizon, method = row['task_id'], row['state_id'], row['horizon'], row['method']
            key = tid, sid, horizon, method
            if key in seen:
                raise ValueError('duplicate episode')
            seen.add(key)
            if tid not in range(10) or sid not in range(3) or horizon not in HORIZONS:
                raise ValueError('unexpected task/state/horizon')
            if method not in (MAIN_METHODS if phase == 'main' else ['laya']):
                raise ValueError('unexpected method')
            case = run / phase / f't{tid:02d}_s{sid:02d}_h{horizon}_{method}'
            result = read_json(case / 'result.json')
            calls = read_json(case / 'calls.json')
            gates = read_json(case / 'gate.json')
            timeline = read_json(case / 'timeline.json')
            if result != row or len(calls) != row['vla_calls'] or len(gates) != row['gate_calls'] or len(timeline) != row['steps']:
                raise ValueError('summary and original JSON evidence disagree')
            cases.append({'row': row, 'method': method if phase == 'main' else 'laya_argmax_tau0',
                          'calls': calls, 'gates': gates, 'timeline': timeline,
                          'case_path': str(case), 'source_phase': phase})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append({'source': str(path), 'line': line_number, 'error': type(exc).__name__ + ': ' + str(exc)})
    return cases, status


def gate_breakdown(gates):
    valid = [gate for gate in gates if set(gate.get('probabilities', {})) == {'replan', 'continue', 'uncertain'}
             and all(isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p)
                     for p in gate['probabilities'].values())]
    raw = Counter(gate.get('raw_choice', 'MISSING') for gate in gates)
    effective = Counter(gate.get('choice', 'MISSING') for gate in gates)
    return {'requests': len(gates), 'valid_probability_records': len(valid),
            'raw_choices': {key: raw[key] for key in ['replan', 'continue', 'uncertain']},
            'effective_choices': {key: effective[key] for key in ['replan', 'continue', 'uncertain']},
            'probability_range': {label: probability_range([gate['probabilities'][label] for gate in valid])
                                  for label in ['replan', 'continue', 'uncertain']},
            'maximum_probability_range': probability_range([max(gate['probabilities'].values()) for gate in valid]),
            'confidence_rejections': sum(gate.get('choice') == 'uncertain' and gate.get('raw_choice') != 'uncertain'
                                         for gate in gates),
            'semantic_uncertain': sum(gate.get('raw_choice') == 'uncertain' for gate in gates),
            'prompt_hashes': sorted({gate.get('prompt_sha256', 'MISSING') for gate in gates})}


def aggregate(cases):
    rows = [case['row'] for case in cases]
    steps = sum(row['steps'] for row in rows)
    calls = sum(row['vla_calls'] for row in rows)
    gate_seconds = sum(row['gate_seconds'] for row in rows)
    prediction_seconds = sum(row['prediction_seconds'] for row in rows)
    gates = [gate for case in cases for gate in case['gates']]
    return {'n': len(rows), 'successes': sum(row['success'] for row in rows),
            'failures': sum(not row['success'] for row in rows), 'state_ids': sorted(row['state_id'] for row in rows),
            'steps': steps, 'vla_calls': calls, 'calls_per100steps': calls * 100 / steps if steps else None,
            'prediction_seconds': prediction_seconds, 'gate_calls': len(gates), 'gate_seconds': gate_seconds,
            'gate_seconds_per_call': gate_seconds / len(gates) if gates else None,
            'vla_plus_gate_seconds': prediction_seconds + gate_seconds,
            'episode_wall_seconds': sum(row['episode_wall_seconds'] for row in rows),
            'executed_rows_per_vla_call': steps / calls if calls else None,
            'unused_generated_rows': sum(row['unused_generated_rows'] for row in rows),
            'native_generated_rows': sum(row['native_generated_rows'] for row in rows),
            'request_triggers': dict(Counter(call['trigger'] for case in cases for call in case['calls'])),
            'gate': gate_breakdown(gates)}


def execution_coverage(cases):
    calls = [call for case in cases for call in case['calls']]
    used = [index for call in calls for index in call['executed_rows']]
    generated = sum(case['row']['native_generated_rows'] for case in cases)
    return {'episodes': len(cases), 'vla_calls': len(calls), 'native_generated_rows': generated,
            'actually_executed_rows': len(used),
            'max_executed_generated_row_1based': max(used) + 1 if used else None,
            'executed_rows_51_and_later': sum(index >= 50 for index in used),
            'executed_rows_101_and_later': sum(index >= 100 for index in used),
            'calls_with_any_row_51_or_later': sum(any(index >= 50 for index in call['executed_rows']) for call in calls),
            'calls_with_any_row_101_or_later': sum(any(index >= 100 for index in call['executed_rows']) for call in calls),
            'calls_with_no_executed_row': sum(not call['executed_rows'] for call in calls),
            'generated_row_utilization': len(used) / generated if generated else None,
            'mean_executed_rows_per_generated_chunk': len(used) / len(calls) if calls else None,
            'unused_generated_rows': generated - len(used),
            'counting_scope': 'Counts actually consumed rows from each saved call; row numbering is one-based. Rows 51+ and 101+ are inclusive and the latter is a subset of the former. Uninstalled terminal calls remain in generated-row/call denominators.'}


def failed_case_observations(cases, baseline_lookup):
    result = []
    for case in cases:
        row = case['row']
        if row['success']:
            continue
        final = case['timeline'][-1]
        gates = case['gates']
        base = baseline_lookup.get((row['task_id'], row['state_id'], row['horizon']))
        observation = {
            'task_id': row['task_id'], 'state_id': row['state_id'], 'horizon': row['horizon'], 'method': case['method'],
            'steps': row['steps'], 'reached_230_step_cap': row['steps'] == 230,
            'terminal_flag': final['terminated'], 'truncated_flag': final['truncated'],
            'recorded_final_success': final['success'], 'vla_calls': row['vla_calls'],
            'gate_calls': row['gate_calls'], 'gate_seconds': row['gate_seconds'],
            'prediction_seconds': row['prediction_seconds'], 'episode_wall_seconds': row['episode_wall_seconds'],
            'pending_at_termination': row.get('pending_at_termination'),
            'last_active_call': final['active_call'], 'last_action_row': final['action_row'],
            'last_queue_remaining': final['queue_remaining'],
            'request_ticks': [call['request_tick'] for call in case['calls']],
            'request_triggers': dict(Counter(call['trigger'] for call in case['calls'])),
            'effective_gate_choices': dict(Counter(gate['choice'] for gate in gates)),
            'raw_gate_choices': dict(Counter(gate['raw_choice'] for gate in gates)),
            'all_gate_choices_uncertain': bool(gates) and all(gate['choice'] == 'uncertain' for gate in gates),
            'evidence': {'result': str(Path(case['case_path']) / 'result.json'),
                         'timeline': str(Path(case['case_path']) / 'timeline.json'),
                         'calls': str(Path(case['case_path']) / 'calls.json'),
                         'gate': str(Path(case['case_path']) / 'gate.json')},
        }
        if base and base is not case:
            base_row = base['row']
            observation['same_origin_smol70'] = {
                'initial_state_hash_equal': row['initial_state_sha256'] == base_row['initial_state_sha256'],
                'success': base_row['success'], 'steps': base_row['steps'], 'vla_calls': base_row['vla_calls'],
                'executed_action_hash_equal': row['executed_action_sha256'] == base_row['executed_action_sha256'],
                'request_ticks_equal': observation['request_ticks'] == [call['request_tick'] for call in base['calls']],
            }
        result.append(observation)
    return result


def build(main_run, argmax_run, out):
    errors = []
    main_cases, main_status = load_cases(main_run, 'main', errors)
    argmax_cases, argmax_status = load_cases(argmax_run, 'argmax', errors)
    cases = main_cases + argmax_cases
    methods = MAIN_METHODS + (['laya_argmax_tau0'] if argmax_run is not None else [])
    expected_main = 360
    expected_argmax = 90 if argmax_run is not None else 0
    complete = (len(main_cases) == expected_main and main_status.get('phase') == 'completed'
                and len(argmax_cases) == expected_argmax
                and (argmax_run is None or argmax_status.get('phase') == 'completed') and not errors)
    grouped = defaultdict(list)
    by_task_method = defaultdict(list)
    for case in cases:
        row = case['row']
        grouped[(row['task_id'], row['horizon'], case['method'])].append(case)
        by_task_method[(row['task_id'], case['method'])].append(case)
    task_horizon_rows = [{'task_id': tid, 'horizon': horizon, 'method': method, 'expected_n': 3,
                          **aggregate(grouped[(tid, horizon, method)])}
                         for tid in range(10) for horizon in HORIZONS for method in methods]
    task_rows = [{'task_id': tid, 'method': method, 'expected_n': 9,
                  **aggregate(by_task_method[(tid, method)])} for tid in range(10) for method in methods]
    coverage_rows = [{'horizon': horizon, 'method': method,
                      **execution_coverage([case for case in cases if case['row']['horizon'] == horizon and case['method'] == method])}
                     for horizon in HORIZONS for method in methods]
    baseline_lookup = {(case['row']['task_id'], case['row']['state_id'], case['row']['horizon']): case
                       for case in main_cases if case['method'] == 'smol70'}
    failures = failed_case_observations(cases, baseline_lookup)
    gate_cases = [case for case in cases if case['method'] in ['laya', 'laya_argmax_tau0']]
    method_gates = {method: gate_breakdown([gate for case in gate_cases if case['method'] == method for gate in case['gates']])
                    for method in ['laya', 'laya_argmax_tau0'] if method in methods}
    failure_counts = {method: {
        'failed_episodes': sum(f['method'] == method for f in failures),
        'failed_episodes_at_230_steps': sum(f['method'] == method and f['reached_230_step_cap'] for f in failures),
        'failed_episodes_with_pending_request': sum(f['method'] == method and bool(f['pending_at_termination']) for f in failures),
        'all_uncertain_failures_action_identical_to_smol70': sum(
            f['method'] == method and f['all_gate_choices_uncertain']
            and f.get('same_origin_smol70', {}).get('initial_state_hash_equal', False)
            and f.get('same_origin_smol70', {}).get('executed_action_hash_equal', False) for f in failures),
        'failures_whose_paired_smol70_succeeded': sum(f['method'] == method
            and f.get('same_origin_smol70', {}).get('initial_state_hash_equal', False)
            and f.get('same_origin_smol70', {}).get('success', False) for f in failures),
    } for method in methods}
    result = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'main_run': str(main_run), 'argmax_run': str(argmax_run) if argmax_run else None,
        'completion_status': 'complete' if complete else 'partial',
        'main_episodes': len(main_cases), 'main_expected': expected_main,
        'argmax_episodes': len(argmax_cases), 'argmax_expected': expected_argmax,
        'validation_scope': 'Lightweight JSON consistency only. Full action-to-chunk, NPZ finiteness, model freeze and cost reconciliation belong to audit.json and argmax_audit.json. This script does not replace them.',
        'task_horizon_method': task_horizon_rows,
        'task_method_all_horizons': task_rows,
        'execution_coverage_by_horizon_method': coverage_rows,
        'gate_totals_by_method': method_gates,
        'failure_counts_by_method': failure_counts,
        'failed_episode_observations': failures,
        'errors': errors,
        'limits': ['No threshold selection, fitted model, or result-dependent subgroup promotion.',
                   'Three states per task/H/method; task differences are descriptive, not generalization estimates.',
                   'Raw argmax ablation was added after observing primary abstentions and remains exploratory.',
                   'Failure records identify observable events, not causes; failure does not label individual gate choices as wrong.',
                   'Task-completion failure and all-uncertain identity checks are based on recorded simulator success flags and action hashes, not new physical replay.',
                   'No paced robot control or actual inference/control overlap; measured seconds are serial service/episode wall costs.']}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'task_breakdown.json', result)
    write_markdown(out, result, methods)
    return result


def write_markdown(out, result, methods):
    fmt = lambda x: 'NA' if x is None else f'{x:.3f}'
    ranges = lambda stats: 'NA' if not stats['n'] else f"{stats['min']:.3f}–{stats['max']:.3f}"
    table = {(row['task_id'], row['horizon'], row['method']): row for row in result['task_horizon_method']}
    lines = ['# 长动作块实验：任务层诊断', '',
             f"状态：**{result['completion_status']}**。主实验 {result['main_episodes']}/{result['main_expected']} 回合；argmax探索消融 {result['argmax_episodes']}/{result['argmax_expected']} 回合。", '',
             '本文件补充任务细分，完整动作绑定、模型冻结和成本审计仍以audit.json及argmax_audit.json为准。pilot不计入下表。未满时仅作部分记录，禁止写成最终完成结果。', '',
             '## 各任务、动作长度、方法的成功数', '',
             '| 任务ID | H | ' + ' | '.join(DISPLAY[m] for m in methods) + ' |',
             '|---|---|' + '|'.join('---:' for _ in methods) + '|']
    for tid in range(10):
        for horizon in HORIZONS:
            cells = [f"{table[(tid,horizon,m)]['successes']}/{table[(tid,horizon,m)]['n']}" for m in methods]
            lines.append(f'| {tid} | {horizon} | ' + ' | '.join(cells) + ' |')
    lines += ['', '每格预定3个相同官方初始状态。三个H沿用这些起点，不能把跨H记录当成新增独立场景。', '',
              '## 生成长动作块的实际执行覆盖', '',
              '生成行索引从1开始。第51行以后含第51行，第101行以后含第101行，后者包含在前者计数中。统计实际消费的动作行；结束前启动但未接入的块仍计入生成总量和块数。', '',
              '| H | 方法 | 已执行最大生成行索引 | 执行第51行及以后 | 执行第101行及以后 | 生成行利用率 | 每块平均执行行数 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in result['execution_coverage_by_horizon_method']:
        lines.append(f"| {row['horizon']} | {DISPLAY[row['method']]} | {row['max_executed_generated_row_1based'] if row['max_executed_generated_row_1based'] is not None else 'NA'} | {row['executed_rows_51_and_later']} | {row['executed_rows_101_and_later']} | {fmt(row['generated_row_utilization'])} | {fmt(row['mean_executed_rows_per_generated_chunk'])} |")
    lines += ['', '原生输出H=200只证明模型生成200行；实际控制是否使用超过50/100行由本表判断。仅生成未执行的远期行不能当作长时域控制能力的证据。', '',
              '## 每任务的调用与门控费用', '',
              '此表合并三个H，每任务每方法预定9个回合；H细分费用完整保存在JSON的task_horizon_method。', '',
              '| 任务ID | 方法 | 回合 | VLA调用 | 调用/100步 | VLA秒 | 门控次数 | 门控秒 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in result['task_method_all_horizons']:
        lines.append(f"| {row['task_id']} | {DISPLAY[row['method']]} | {row['n']} | {row['vla_calls']} | {fmt(row['calls_per100steps'])} | {fmt(row['prediction_seconds'])} | {row['gate_calls']} | {fmt(row['gate_seconds'])} |")
    lines += ['', 'VLA秒含预处理、推断、传输与同步；门控秒含状态构造、IPC和推断。它们是串行墙钟服务成本，不能作为纯GPU活跃秒或实际异步加速。', '',
              '## Laya原始选择、实际选择与概率范围', '',
              '计数顺序均为 replan / continue / uncertain。原始选择与置信度阈值后的实际选择分别列出；τ=0仍保留语义uncertain。概率范围覆盖该任务三个H的全部门控输入，H细分见JSON。', '',
              '| 任务ID | 方法 | 原始选择计数 | 实际选择计数 | P(replan)范围 | P(continue)范围 | P(uncertain)范围 |',
              '|---|---|---|---|---|---|---|']
    for row in result['task_method_all_horizons']:
        if row['method'] not in ['laya', 'laya_argmax_tau0']:
            continue
        gate = row['gate']
        raw = '/'.join(str(gate['raw_choices'][k]) for k in ['replan', 'continue', 'uncertain'])
        effective = '/'.join(str(gate['effective_choices'][k]) for k in ['replan', 'continue', 'uncertain'])
        values = [ranges(gate['probability_range'][k]) for k in ['replan', 'continue', 'uncertain']]
        lines.append(f"| {row['task_id']} | {DISPLAY[row['method']]} | {raw} | {effective} | " + ' | '.join(values) + ' |')
    lines += ['', '## 失败回合可核实的观察', '',
              '| 方法 | 失败回合 | 失败且达230步 | 结束时有待接入请求 | 全弃权失败且动作同Smol70 | 本方法失败但配对Smol70成功 |',
              '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        row = result['failure_counts_by_method'][method]
        lines.append(f"| {DISPLAY[method]} | {row['failed_episodes']} | {row['failed_episodes_at_230_steps']} | {row['failed_episodes_with_pending_request']} | {row['all_uncertain_failures_action_identical_to_smol70']} | {row['failures_whose_paired_smol70_succeeded']} |")
    lines += ['', '这些计数只能证明日志中的事件关系：到达步数上限不是“模型不理解任务”的证据，待接入请求不是失败原因，整体任务失败也不能把某次continue或replan标成错误决策。', '',
              '全弃权且执行动作哈希与Smol70一致时，可确认两条已保存动作序列相同；不能由此推断其他输入、提示或训练方案都无效。逐失败回合的步数、请求时刻、队列末状态、门控选择及证据路径均在JSON的failed_episode_observations。', '',
              '任务层成功数每格只有3个起点，仅用于描述本次暴露开发状态；不选择新阈值，不将事后表现较好的任务提升为确认性结论。']
    if result['errors']:
        lines += ['', '## 不完整或不一致的原件', '']
        lines.extend('- ' + json.dumps(error, ensure_ascii=False) for error in result['errors'])
    (out / 'task_breakdown.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--argmax', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.main.resolve(), args.argmax.resolve() if args.argmax else None, args.out.resolve())
    print(json.dumps({key: result[key] for key in ['completion_status', 'main_episodes', 'argmax_episodes']}, ensure_ascii=False))
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
