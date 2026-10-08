"""Check report tables against complete analyses and verify local evidence links.

This is a report-integrity check, not a new statistical analysis. Narrative claims
and interpretation still require human review.
"""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import unquote
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'Smol70': 'smol70', '固定I45': 'fixed45', '普通K5': 'naive_k5',
    '同模型VLASH式K5': 'vlash_style_k5', 'LAYA三样本窗口': 'window3_skip1',
    'LAYA紧凑输入': 'compact_skip1',
}
SECTIONS = {
    '完整配对主验证：1,050回合': 'validation_d1_v2',
    '4步逻辑延迟对照：450回合': 'delay_validation_d4_v2',
    '5步逻辑延迟对照：450回合': 'delay_validation_d5_v2',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sections(text):
    return {match.group(1): match.group(2) for match in re.finditer(
        r'^## ([^\n]+)\n(.*?)(?=^## |\Z)', text, re.M | re.S)}


def first_table(section):
    match = re.search(r'(?m)^\|.*(?:\n\|.*)+', section)
    assert match, 'Expected a report table'
    rows = [[cell.strip() for cell in line.strip().strip('|').split('|')]
            for line in match.group(0).splitlines()]
    assert len(rows) >= 3 and all(re.fullmatch(r':?-+:?', c) for c in rows[1])
    return rows[0], rows[2:]


def main():
    report = ROOT/'REPORT.md'
    text = report.read_text(encoding='utf-8')
    parts = sections(text)
    sources, entries, local_links, figures = {}, [], [], []
    catalogue_path = ROOT/'figures/figure_catalog.json'
    catalogue = read(catalogue_path)

    def source(cohort):
        path = ROOT/'analysis'/f'{cohort}_summary.json'
        data = read(path)
        definition = ROOT/'protocol'/cohort/'definition.json'
        assert sha(definition) == data['definition_sha256']
        assert data['total_episodes'] == read(definition)['total_episodes']
        assert read(ROOT/'runtime'/f'{cohort}_chain.json')['phase'] == 'completed'
        for name, digest in data['audit_sha256'].items():
            suffix = '.json' if name.endswith(('_initial_verification', '_noise_alignment')) else '_audit.json'
            audit = ROOT/'checks'/(name+suffix)
            assert sha(audit) == digest and read(audit)['errors'] == 0
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return data

    for title, cohort in SECTIONS.items():
        if title not in parts:
            continue
        data = source(cohort)
        header, rows = first_table(parts[title])
        expected_methods = set(LABELS) if cohort == 'validation_d1_v2' else set(LABELS)-{'LAYA紧凑输入'}
        assert {r[0] for r in rows} == expected_methods
        n = read(ROOT/'protocol'/cohort/'definition.json')['episodes_per_method_horizon']
        for column, h in enumerate((50, 100, 200), 1):
            assert f'H{h}：成功/{n}' in header[column]
        for row in rows:
            cid = LABELS[row[0]]
            assert len(row) == 4
            for column, h in enumerate((50, 100, 200), 1):
                group = data['groups'][f'{cid}_h{h}']
                actual = row[column].split('；')
                expected = [str(group['successes']), f"{group['total_service_seconds']:.3f}"]
                assert actual == expected, (title, cid, h, actual, expected)
                entries.append(dict(section=title, method=cid, horizon=h, values=expected))

    budget_title = '费用接近对照：600回合'
    if budget_title in parts:
        data = source('budget_validation_d1_v2')
        _, rows = first_table(parts[budget_title])
        assert len(rows) == 3
        for row, h, period in zip(rows, (50, 100, 200), (19, 38, 71)):
            candidate = data['groups'][f'window3_skip1_h{h}']
            cid = f'budget_period_h{h}'
            reference = data['groups'][f'{cid}_h{h}']
            comparison = candidate['comparisons'][cid]
            expected = [f'{h}；I{period}',
                        f"{candidate['successes']}；{reference['successes']}",
                        f"{candidate['total_service_seconds']:.3f}；{reference['total_service_seconds']:.3f}",
                        f"{comparison['total_service_cost_ratio']:.3f}",
                        f"{comparison['wins']}/{comparison['losses']}"]
            assert row == expected, (budget_title, h, row, expected)
            entries.append(dict(section=budget_title, horizon=h, values=expected))

    delay_title = '同起点：延迟从4步变为5步'
    if delay_title in parts:
        path = ROOT/'analysis/cross_delay_d5_vs_d4.json'
        data = read(path)
        for role in ('candidate', 'reference'):
            cohort = data[f'{role}_cohort']
            source(cohort)
            assert sha(ROOT/'analysis'/f'{cohort}_summary.json') == data[f'{role}_summary_sha256']
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        _, rows = first_table(parts[delay_title])
        displays = {value: name for name, value in LABELS.items()}
        displays['vlash_style_k5'] = 'VLASH式K5'
        expected_rows = []
        for key, group in data['comparisons'].items():
            cid, h = key.rsplit('_h', 1)
            c = group['paired']
            u = c['paired_uncertainty']['intervals']['hierarchical_task_and_initial']
            delta, ratio = u['success_delta_95'], u['cost_ratio_95']
            expected_rows.append([displays[cid], h,
                f"{group['candidate_d5']['successes']}/{group['reference_d4']['successes']}",
                f"{c['wins']}/{c['losses']}",
                f"{100*c['success_delta']:+.1f} [{100*delta[0]:+.1f}, {100*delta[1]:+.1f}]",
                f"{c['total_service_cost_ratio']:.3f} [{ratio[0]:.3f}, {ratio[1]:.3f}]"])
        assert len(rows) == len(expected_rows) == 15 and rows == expected_rows
        entries.extend(dict(section=delay_title, values=row) for row in rows)

    partial_title = '8. 未完成组的输入对齐部分样本'
    if partial_title in parts:
        path = ROOT/'analysis/noise_alignment_validation_d4_v1_partial_descriptive.json'
        data = read(path)
        stop_path = ROOT/'checks/final_stop.json'
        stop = read(stop_path)
        assert data['final_stop_sha256'] == sha(stop_path)
        assert stop['remote_owned_gpu_workers'] == stop['remote_owned_processes'] == 0
        assert data['status'] == 'partial_descriptive_descriptive_only'
        assert data['intervals_computed'] is data['formal_confirmation_completed'] is False
        definition_path = ROOT/'protocol'/data['cohort']/'definition.json'
        assert data['definition_sha256'] == sha(definition_path)
        definition = read(definition_path)
        for origin, digest in data['source_sha256'].items():
            assert sha(ROOT/origin) == digest, origin
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        sources[stop_path.relative_to(ROOT).as_posix()] = sha(stop_path)
        table_texts = [m.group(0) for m in re.finditer(r'(?m)^\|.*(?:\n\|.*)+', parts[partial_title])]
        assert len(table_texts) == 2
        header, rows = first_table(table_texts[0])
        assert header == ['方法', 'H', '成功/n', 'VLA调用', '门控调用', '总服务秒']
        displays = {value: name for name, value in LABELS.items()}
        displays.update(vlash_style_k5='原VLASH式K5', vlash_noise_matched_k5='VLASH式匹配噪声', stale_row0_k5='旧状态row0')
        expected_rows = []
        for cid in definition['candidates']:
            for h in definition['horizons']:
                group = data['groups'][f'{cid}_h{h}']
                expected_rows.append([displays[cid], str(h), f"{group['successes']}/{group['n']}",
                    str(group['vla_calls']), str(group['gate_calls']), f"{group['total_service_seconds']:.3f}"])
        assert rows == expected_rows and len(rows) == len(data['groups'])
        entries.extend(dict(section=partial_title, values=row) for row in rows)
        header, rows = first_table(table_texts[1])
        assert header == ['LAYA对参照', 'H', '配对数', '胜/负', '成功差pp', '总费用比']
        expected_rows = []
        for h in definition['horizons']:
            for cid in definition['candidates']:
                if cid == data['primary_candidate']:
                    continue
                c = data['primary_point_comparisons'][f"{data['primary_candidate']}_h{h}_vs_{cid}"]
                expected_rows.append([displays[cid], str(h), str(c['n']), f"{c['wins']}/{c['losses']}",
                    f"{100*c['success_delta']:+.1f}", f"{c['total_service_cost_ratio']:.3f}"])
        assert rows == expected_rows and len(rows) == len(data['primary_point_comparisons'])
        entries.extend(dict(section=partial_title, values=row) for row in rows)

    for match in re.finditer(r'(!?)\[([^\]]*)\]\(([^)]+)\)', text):
        image, label, target = match.groups()
        if target.startswith(('https://', 'http://', '#')):
            continue
        path = (ROOT/unquote(target.split('#', 1)[0])).resolve()
        assert path.is_file(), ('Missing report evidence link', target)
        local_links.append(target)
        if image:
            assert path.is_relative_to(ROOT)
            relative = path.relative_to(ROOT).as_posix()
            records = [entry for entry in catalogue if relative in entry['exports']]
            assert len(records) == 1 and records[0]['review_status'] in {'visually_inspected_accepted', 'accepted'}, relative
            for origin, digest in records[0]['source_data'].items():
                assert sha(ROOT/origin) == digest, ('Stale figure source', relative, origin)
            figures.append(dict(path=relative, sha256=sha(path)))

    assert entries and figures
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha(report),
                   script_sha256=sha(Path(__file__)), source_sha256=sources,
                   figure_catalog_sha256=sha(catalogue_path), verified_rows=entries,
                   local_links=local_links, figures=figures, errors=0,
                   scope='Complete main/budget/delay tables, all 15 cross-delay contrasts, all available resource-stop group and primary point-comparison rows, denominators, evidence links and accepted figure sources. Does not automatically validate narrative numbers or causal interpretation.')
    output = ROOT/'checks/report_integrity.json'
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(verified_rows=len(entries), local_links=len(local_links), figures=len(figures), errors=0)))


if __name__ == '__main__':
    main()
