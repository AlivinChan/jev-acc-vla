"""Render already-complete audited analyses; no new statistics or policy selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    'validation_d1_v2': '主验证：d=1，每方法/H 50起点',
    'budget_validation_d1_v2': '费用接近的固定周期：d=1，每方法/H 50起点',
    'delay_validation_d4_v2': '延迟敏感性：d=4，每方法/H 30起点',
    'delay_validation_d5_v2': '延迟敏感性：d=5，每方法/H 30起点',
    'noise_alignment_validation_d4_v1': 'VLASH噪声与状态归因：d=4，每方法/H 30起点',
    'random_skip_validation_d4_v1': '随机跳过对照：d=4，每方法/H 30起点',
}
LABELS = {
    'smol70': 'Smol70', 'fixed45': '固定I45', 'naive_k5': '普通K5',
    'vlash_style_k5': 'VLASH式K5', 'window3_skip1': 'LAYA三样本窗口',
    'compact_skip1': 'LAYA紧凑输入', 'window3_first': 'LAYA仅首次',
    'compact_minr30_i5': '紧凑输入＋最短0.3H＋每5步',
    'vlash_noise_matched_k5': 'VLASH式K5＋图像时刻噪声',
    'stale_row0_k5': '旧状态K5＋从row0执行',
    'random_skip_s11131': '随机跳过种子11131',
    'random_skip_s22261': '随机跳过种子22261',
    'budget_period_h50': '费用接近I19', 'budget_period_h100': '费用接近I38',
    'budget_period_h200': '费用接近I71',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci_text(point, interval, percent=False):
    if percent:
        return f'{point*100:+.1f} [{interval[0]*100:+.1f}, {interval[1]*100:+.1f}]'
    return f'{point:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]'


def main():
    lines = ['# 完整配对验证读数', '',
             '本文件从完整、已审计分析生成；不读取未完成分片，不重新选择策略。区间为已登记的任务＋初态分层配对bootstrap 95%描述性区间。总费用为同步模型服务秒，不能解释为实际机器人加速。随机对照不调用LAYA，其门控秒为单列记录的CPU选择开销，LAYA调用数仍为0。', '']
    sources, completed = {}, []
    for cohort, title in COHORTS.items():
        source = ROOT/'analysis'/f'{cohort}_summary.json'
        if not source.exists():
            continue
        definition_path = ROOT/'protocol'/cohort/'definition.json'
        definition, data = read(definition_path), read(source)
        assert sha(definition_path) == data['definition_sha256']
        assert data['total_episodes'] == definition['total_episodes']
        assert read(ROOT/'runtime'/f'{cohort}_chain.json')['phase'] == 'completed'
        for name, digest in data['audit_sha256'].items():
            suffix = '.json' if name.endswith(('_initial_verification', '_noise_alignment')) else '_audit.json'
            audit = ROOT/'checks'/(name+suffix)
            assert sha(audit) == digest and read(audit)['errors'] == 0
        sources[source.relative_to(ROOT).as_posix()] = sha(source)
        completed.append(cohort)
        lines += [f'## {title}', '',
                  f'完整回合数{data["total_episodes"]}；[分析原件]({source.name})。每一行使用相同task/state起点；方法/H之间的重复不增加独立起点数。', '',
                  '| 方法 | H | 成功 | VLA调用 | LAYA调用 | VLA秒 | 门控秒 | 总服务秒 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        order = [c for c in LABELS if c in definition['candidates']]
        assert set(order) == set(definition['candidates']), 'Label every candidate explicitly'
        for cid in order:
            for h in definition['candidates'][cid].get('horizons', definition['horizons']):
                r = data['groups'][f'{cid}_h{h}']
                assert r['n'] == definition['episodes_per_method_horizon']
                lines.append(f'| {LABELS[cid]} | {h} | {r["successes"]}/{r["n"]} | {r["vla_calls"]} | {r["gate_calls"]} | {r["vla_seconds"]:.3f} | {r["gate_seconds"]:.3f} | {r["total_service_seconds"]:.3f} |')
        lines += ['', '### 主LAYA相对各参照', '',
                  '| H | 对照 | 配对胜/负 | 成功差pp [区间] | 总费用比 [区间] |',
                  '|---:|---|---:|---:|---:|']
        for h in definition['horizons']:
            row = data['groups'][f'{data["primary_candidate"]}_h{h}']
            for cid, c in row['comparisons'].items():
                if cid == data['primary_candidate']:
                    continue
                u = c['paired_uncertainty']['intervals']['hierarchical_task_and_initial']
                lines.append(f'| {h} | {LABELS[cid]} | {c["wins"]}/{c["losses"]} | {ci_text(c["success_delta"],u["success_delta_95"],True)} | {ci_text(c["total_service_cost_ratio"],u["cost_ratio_95"])} |')
        accounting = source.with_name(f'{cohort}_service_accounting.json')
        if accounting.exists():
            prices = read(accounting)
            assert prices['summary_sha256'] == sha(source)
            sources[accounting.relative_to(ROOT).as_posix()] = sha(accounting)
            lines += ['', '### 固定开发单价敏感性（仅主LAYA相对Smol70）', '',
                      '| H | 固定单价费用比 [区间] | 轨迹步数比 | 每步调用率比 | 单次VLA时长比 | 门控乘数比 |',
                      '|---:|---:|---:|---:|---:|---:|']
            for h in definition['horizons']:
                entry = prices['groups'][f'{data["primary_candidate"]}_h{h}']
                c = entry['fixed_price_comparisons']['smol70']
                u = c['paired_uncertainty']['intervals']['hierarchical_task_and_initial']
                f = entry['measured_comparisons']['smol70']['factor_ratios']
                lines.append(f'| {h} | {ci_text(c["total_service_cost_ratio"],u["cost_ratio_95"])} | {f["environment_steps"]:.3f} | {f["vla_calls_per_environment_step"]:.3f} | {f["mean_vla_service_seconds"]:.3f} | {f["gate_multiplier"]:.3f} |')
            lines += ['', '后四列相乘精确还原实测费用比；这是算术拆分，不是因果中介分析。', '']
        cross = source.with_name(f'{cohort}_cross_horizon.json')
        if cross.exists():
            native = read(cross)
            assert native['summary_sha256'] == sha(source)
            sources[cross.relative_to(ROOT).as_posix()] = sha(cross)
            lines += ['', '### 固定I45的原生长度对照', '',
                      '| H比较 | 成功数（较长/较短） | 配对胜/负 | 成功差pp [区间] | 总费用比 [区间] |',
                      '|---|---:|---:|---:|---:|']
            for pair in native['methods']['fixed45']['comparisons'].values():
                c = pair['paired']; u = c['paired_uncertainty']['intervals']['hierarchical_task_and_initial']
                lines.append(f'| {pair["candidate_horizon"]} vs {pair["reference_horizon"]} | {pair["candidate"]["successes"]}/{pair["reference"]["successes"]} | {c["wins"]}/{c["losses"]} | {ci_text(c["success_delta"],u["success_delta_95"],True)} | {ci_text(c["total_service_cost_ratio"],u["cost_ratio_95"])} |')
        lines.append('')
    cross_delay = ROOT/'analysis/cross_delay_d5_vs_d4.json'
    if cross_delay.exists():
        data = read(cross_delay)
        assert sha(ROOT/'analysis'/f'{data["candidate_cohort"]}_summary.json') == data['candidate_summary_sha256']
        assert sha(ROOT/'analysis'/f'{data["reference_cohort"]}_summary.json') == data['reference_summary_sha256']
        sources[cross_delay.relative_to(ROOT).as_posix()] = sha(cross_delay)
        lines += ['## 同起点d5相对d4的全部配对比较', '',
                  '同一task/state/H，d5为比较对象、d4为参照。含整个冻结策略对延迟的响应，不是单独图像/状态变量归因，也不是真实异步速度。', '',
                  '| 方法 | H | 成功（d5/d4，各30） | 配对胜/负 | 成功差pp [区间] | 总费用比 [区间] |',
                  '|---|---:|---:|---:|---:|---:|']
        for key, row in data['comparisons'].items():
            cid, h = key.rsplit('_h',1)
            c = row['paired']; u = c['paired_uncertainty']['intervals']['hierarchical_task_and_initial']
            lines.append(f'| {LABELS[cid]} | {h} | {row["candidate_d5"]["successes"]}/{row["reference_d4"]["successes"]} | {c["wins"]}/{c["losses"]} | {ci_text(c["success_delta"],u["success_delta_95"],True)} | {ci_text(c["total_service_cost_ratio"],u["cost_ratio_95"])} |')
        lines.append('')
    assert completed, 'No complete cohort exists yet'
    output = ROOT/'analysis/VALIDATION_READOUT.md'
    output.write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), completed=completed,
                   sources=sources, output_sha256=sha(output), script_sha256=sha(Path(__file__)))
    (ROOT/'checks/validation_readout.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(completed=completed, output=str(output))))


if __name__ == '__main__':
    main()
