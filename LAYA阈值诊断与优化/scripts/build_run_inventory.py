"""Inventory archived evidence by its actual unit; never aggregate policy outcomes."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = {'representations_v1', 'cadence_v1', 'confirmation_v1', 'single_skip_v1',
               'fixed_interval_v1', 'single_skip_confirmation_v1'}
EXCLUDED = {'validation_d1_s10_v1', 'validation_d1_s12_v1'}
FORENSIC = {'initial_repro_probe_v1', 'initial_repro_plain_v1', 'verified_runtime_qualification_v1'}
QUALIFICATION = {'first_request_qualification_v1', 'delay_d1_qualification_v1', 'guarded_runtime_qualification_v1'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def count_rows(path):
    return len([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()])


def main():
    records = []
    totals = defaultdict(Counter)
    validation_status = Counter()
    for receipt_path in sorted((ROOT/'artifacts').glob('*_archive.json')):
        name = receipt_path.name.removesuffix('_archive.json')
        archive = read(receipt_path)
        tar = receipt_path.with_name(name+'.tar.gz')
        assert tar.is_file() and tar.stat().st_size == archive['bytes'] and sha(tar) == archive['sha256']
        folder = ROOT/'raw'/name/'data'
        status = read(folder.parent/'supervisor.json')['status']
        manifest_path = folder/'manifest.json'
        manifest = read(manifest_path)
        cohort = manifest.get('config', {}).get('cohort_id')
        definition_path = ROOT/'protocol'/cohort/'definition.json' if cohort else None
        definition = read(definition_path) if definition_path and definition_path.exists() else None
        units = {}
        if (folder/'episodes.jsonl').exists():
            units['closed_loop_episodes'] = count_rows(folder/'episodes.jsonl')
            if name in DEVELOPMENT:
                role = 'development'
            elif name in EXCLUDED:
                role = 'excluded_old_validation_attempt'
            elif name in FORENSIC:
                role = 'forensic_or_rejected_backend_qualification'
            elif name in QUALIFICATION or definition and definition.get('implementation_pilot', False):
                role = 'implementation_qualification'
            elif definition:
                role = 'registered_validation'
            else:
                raise AssertionError('Unclassified episode run: '+name)
        elif (folder/'records.jsonl').exists():
            rows = [json.loads(s) for s in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
            units['counterfactual_opportunities'] = len(rows)
            units['counterfactual_suffix_branches'] = sum(len(row['branches']) for row in rows)
            assert units['counterfactual_suffix_branches'] == 2*len(rows)
            role = 'counterfactual_label_collection'
        elif (folder/'scores.jsonl').exists():
            units['offline_laya_scores'] = count_rows(folder/'scores.jsonl')
            role = 'offline_score_diagnosis'
        elif (folder/'decisions.jsonl').exists():
            units['offline_laya_decisions'] = count_rows(folder/'decisions.jsonl')
            role = 'offline_prompt_diagnosis'
        else:
            raise AssertionError('Unknown evidence unit: '+name)
        if (folder/'warmup.jsonl').exists():
            units['warmup_decisions_excluded'] = count_rows(folder/'warmup.jsonl')
        elif 'warmup_gate_calls' in manifest:
            units['warmup_decisions_excluded'] = manifest['warmup_gate_calls']
        if 'qualification_vla_calls' in manifest:
            units['qualification_vla_calls_excluded'] = manifest['qualification_vla_calls']
        if 'initial_verification' in manifest:
            units['initial_reference_calls_within_qualification'] = manifest['initial_verification']['qualification_vla_calls']
        audit_path = ROOT/'checks'/f'{name}_audit.json'
        audit = read(audit_path) if audit_path.exists() else None
        errors = audit.get('errors') if audit else None
        primary_ready = False
        if cohort and role == 'registered_validation':
            summary_path = ROOT/'analysis'/f'{cohort}_summary.json'
            if summary_path.exists():
                summary = read(summary_path)
                assert summary['definition_sha256'] == sha(ROOT/'protocol'/cohort/'definition.json')
                assert summary['total_episodes'] == definition['total_episodes']
                primary_ready = name in summary['source_runs'] and errors == 0 and status == 'completed'
        record = dict(run=name, role=role, supervisor_status=status, units=units, cohort=cohort,
                      end_weight_hash_available=('freeze_after' in manifest) if role == 'registered_validation' else None,
                      complete_cohort_summary_available=primary_ready,
                      audit_checks=audit.get('checks') if audit else None, audit_errors=errors,
                      archive_bytes=archive['bytes'], archive_sha256=archive['sha256'],
                      manifest_sha256=sha(manifest_path), archive_receipt_sha256=sha(receipt_path))
        records.append(record)
        totals[role].update(units)
        if role == 'registered_validation':
            category = ('complete_analyzed_cohorts' if primary_ready else
                        'complete_audited_shards_in_incomplete_cohorts' if status == 'completed' and errors == 0 else
                        'incomplete_or_unaudited_run_records')
            validation_status[category] += units['closed_loop_episodes']
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(), archived_runs=len(records),
                  totals_by_role={role: dict(values) for role, values in totals.items()}, runs=records,
                  registered_validation_records_by_completion=dict(validation_status),
                  scope='This optimization folder only; prior SmolVLA and official pi0.5 references remain in the predecessor experiment. No task-success outcomes are read into this inventory.',
                  unit_warning='Do not add branches, model judgments, repeated methods, qualifications, or reused trajectories to independent initial-state denominators. Reference calls are a subset of qualification calls, not another additive count. Ancillary call totals reflect available manifest fields only; an interrupted run can lack its closing manifest and counters. Episode-record counts parse every nonempty JSON line.')
    output = ROOT/'analysis/run_inventory.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    lines = ['# 已归档证据清单', '', '仅统计本优化目录；前轮SmolVLA与官方π0.5参考另存。每次生成均核验压缩原件SHA，不读取成功率作选择。', '',
             '分支、离线判断、资格复跑均不能充当独立测试初态；初始参考调用包含在资格调用总数内，不再相加。', '',
             '| 注册验证记录的完成状态 | 方法/H回合记录数 |', '|---|---:|',
             f'| 整批完整、已分析 | {validation_status["complete_analyzed_cohorts"]} |',
             f'| 分片完成并已审计，但整批未完成 | {validation_status["complete_audited_shards_in_incomplete_cohorts"]} |',
             f'| 作业未完成或尚未通过审计 | {validation_status["incomplete_or_unaudited_run_records"]} |', '',
             '后两类不纳入完整批次的主结果；记录数不等于独立初态数。中止片可能缺少结束manifest及附加调用计数，不能将缺失计数理解为零；回合计数逐行验证JSON可解析。', '',
             '| 运行 | 证据用途 | 单位与数量 | 作业状态 | 审计错误 | 整批验证汇总就绪 |', '|---|---|---|---|---:|---|']
    for row in records:
        primary_units = {key: value for key, value in row['units'].items() if not key.endswith('_excluded')
                         and key != 'initial_reference_calls_within_qualification'}
        units = '; '.join(f'{key}={value}' for key, value in primary_units.items())
        lines.append(f'| {row["run"]} | {row["role"]} | {units} | {row["supervisor_status"]} | {row["audit_errors"]} | {row["complete_cohort_summary_available"]} |')
    lines.extend(['', f'生成时间UTC：{result["generated_utc"]}。完整单位、原件指纹及资格计数见[JSON清单](run_inventory.json)。', ''])
    (ROOT/'analysis/run_inventory.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps(dict(archived_runs=len(records), totals_by_role=result['totals_by_role'],
                          registered_validation_records_by_completion=dict(validation_status))))


if __name__ == '__main__':
    main()
