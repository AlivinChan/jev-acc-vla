"""Format already-authorized partial descriptions; never read active outcomes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    'noise_alignment_validation_d4_v1': '输入对齐对照',
    'random_skip_validation_d4_v1': '随机跳过对照',
}
LABELS = {
    'smol70': 'Smol70', 'fixed45': '固定I45', 'window3_skip1': 'LAYA三样本窗口',
    'naive_k5': '普通K5', 'vlash_style_k5': '原VLASH式K5',
    'vlash_noise_matched_k5': 'VLASH式匹配噪声', 'stale_row0_k5': '旧状态row0',
    'random_skip_s11131': '随机跳过11131', 'random_skip_s22261': '随机跳过22261',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    stop_path = ROOT/'checks/final_stop.json'
    stop = read(stop_path)
    assert stop['remote_owned_gpu_workers'] == stop['remote_owned_processes'] == 0
    lines = ['# 未完成组的部分样本描述', '',
        '本表仅格式化冻结部分样本分析器的输出。部分组不是完整确认；没有新增区间、显著性检验、候选选择或机制归因。', '']
    sources = {}
    for cohort, title in COHORTS.items():
        path = ROOT/'analysis'/f'{cohort}_partial_descriptive.json'
        if not path.exists():
            continue
        data = read(path)
        assert data['final_stop_sha256'] == sha(stop_path)
        assert data['status'] == 'partial_descriptive_descriptive_only'
        assert data['intervals_computed'] is data['formal_confirmation_completed'] is False
        definition_path = ROOT/'protocol'/cohort/'definition.json'
        assert sha(definition_path) == data['definition_sha256']
        definition = read(definition_path)
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        states = [s['state'] for s in data['included_complete_shards']]
        lines.extend([f'## {title}', '',
            f"原计划{data['planned_episodes']}回合；连续完整并纳入描述{data['included_episodes']}回合；初态{states}。遗漏分片：{', '.join(data['omitted_shards'])}。", ''])
        if not data['groups']:
            lines.extend(['没有可纳入描述的完整分片，不计算成功率或费用比。', ''])
            continue
        lines.extend(['| 方法 | H | 成功/n | VLA调用 | 门控调用 | 总服务秒 |',
                      '|---|---:|---:|---:|---:|---:|'])
        for cid in definition['candidates']:
            for h in definition['horizons']:
                group = data['groups'][f'{cid}_h{h}']
                lines.append(f"| {LABELS[cid]} | {h} | {group['successes']}/{group['n']} | {group['vla_calls']} | {group['gate_calls']} | {group['total_service_seconds']:.3f} |")
        lines.extend(['', '| LAYA对参照 | H | 配对数 | 胜/负 | 成功差pp | 总费用比 |',
                      '|---|---:|---:|---:|---:|---:|'])
        for h in definition['horizons']:
            for cid in definition['candidates']:
                if cid == data['primary_candidate']:
                    continue
                comparison = data['primary_point_comparisons'][f"{data['primary_candidate']}_h{h}_vs_{cid}"]
                lines.append(f"| {LABELS[cid]} | {h} | {comparison['n']} | {comparison['wins']}/{comparison['losses']} | {100*comparison['success_delta']:+.1f} | {comparison['total_service_cost_ratio']:.3f} |")
        lines.extend(['', '以上是未完成组的点值描述。固定执行顺序和运行耗时可能带来完成选择效应；完整分片内的配对不能消除这一限制，不能替代原计划完整确认。', ''])
    assert sources, 'No frozen partial descriptions are available; do not read active outcomes'
    output = ROOT/'analysis/PARTIAL_SAMPLE_READOUT.md'
    output.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    receipt = dict(final_stop_sha256=sha(stop_path), source_sha256=sources,
                   output_sha256=sha(output), script_sha256=sha(Path(__file__)),
                   new_statistics_computed=False)
    (ROOT/'checks/partial_readout.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
