"""Predeclared descriptive reporting of complete shards for an incomplete cohort."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from analyze_closed_loop import rows_from, summary

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', required=True)
    args = parser.parse_args()
    recipe_path = ROOT/'protocol/partial_sample_reporting_v1/definition.json'
    recipe = read(recipe_path)
    assert sha(recipe_path) == read(recipe_path.parent/'freeze_receipt.json')['definition_sha256']
    assert args.cohort in recipe['cohort_definition_sha256']
    for name, digest in recipe['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    stop_path = ROOT/'checks/final_stop.json'
    stop = read(stop_path)
    assert stop['remote_owned_gpu_workers'] == 0
    assert stop.get('remote_owned_processes', 0) == 0
    definition_path = ROOT/'protocol'/args.cohort/'definition.json'
    assert sha(definition_path) == recipe['cohort_definition_sha256'][args.cohort]
    definition = read(definition_path)
    assert not (ROOT/'analysis'/f'{args.cohort}_summary.json').exists(), 'Use the full-cohort analysis when available'
    chain_path = ROOT/'runtime'/f'{args.cohort}_chain.json'
    if chain_path.exists():
        assert read(chain_path)['phase'] == 'halted'

    chosen, omitted, all_rows, sources = [], [], [], {}
    gap = False
    for shard in definition['shards']:
        run = shard['run']
        archive_path = ROOT/'artifacts'/f'{run}_archive.json'
        base = ROOT/'raw'/run
        audits = [ROOT/'checks'/f'{run}_audit.json', ROOT/'checks'/f'{run}_initial_verification.json']
        if definition.get('noise_alignment_audit_required'):
            audits.append(ROOT/'checks'/f'{run}_noise_alignment.json')
        eligible = archive_path.exists() and all(p.exists() for p in audits)
        if eligible:
            eligible = read(base/'supervisor.json')['status'] == 'completed' and all(read(p)['errors'] == 0 for p in audits)
        if not eligible:
            gap = True
            omitted.append(run)
            continue
        assert not gap, 'Do not select a later shard after omitting an earlier one'
        archive = read(archive_path)
        assert sha(ROOT/'artifacts'/f'{run}.tar.gz') == archive['sha256']
        manifest_path = base/'data/manifest.json'
        manifest = read(manifest_path)
        assert manifest['freeze_before'] == manifest['freeze_after']
        assert manifest['config_sha256'] == shard['config_sha256']
        assert manifest['config']['cohort_id'] == args.cohort
        for name, digest in definition['locked_execution_source_sha256'].items():
            assert manifest['scripts'][name] == digest
        for p in audits[:2]:
            assert read(p)['episodes'] == shard['episodes']
        path = base/'data/episodes.jsonl'
        rows = rows_from(path)
        expected = {(cid, t, shard['state'], h) for cid, cfg in definition['candidates'].items()
                    for t in definition['tasks'] for h in cfg.get('horizons', definition['horizons'])}
        actual = [(r['candidate_id'], r['task_id'], r['state_id'], r['horizon']) for r in rows]
        assert len(actual) == len(set(actual)) == shard['episodes'] and set(actual) == expected
        chosen.append(dict(run=run, state=shard['state'], episodes=len(rows), archive_sha256=archive['sha256']))
        all_rows.extend(rows)
        for p in [archive_path, manifest_path, path, *audits]:
            sources[p.relative_to(ROOT).as_posix()] = sha(p)
    assert len(all_rows) < definition['total_episodes'], 'A complete cohort requires its primary analysis'

    groups = defaultdict(list)
    for row in all_rows:
        groups[(row['candidate_id'], row['horizon'])].append(row)
    summaries = {f'{cid}_h{h}': summary(rows) for (cid, h), rows in sorted(groups.items())}
    primary = definition['primary_candidate']
    comparisons = {}
    for h in definition['horizons'] if all_rows else []:
        aa = {(r['task_id'], r['state_id']): r for r in groups[(primary, h)]}
        for cid in definition['candidates']:
            if cid == primary:
                continue
            bb = {(r['task_id'], r['state_id']): r for r in groups[(cid, h)]}
            assert set(aa) == set(bb) and len(aa) == len(chosen)*len(definition['tasks'])
            assert all(aa[k]['initial_state_sha256'] == bb[k]['initial_state_sha256'] for k in aa)
            wins = sum(bool(aa[k]['success']) and not bool(bb[k]['success']) for k in aa)
            losses = sum(bool(bb[k]['success']) and not bool(aa[k]['success']) for k in aa)
            comparison = dict(n=len(aa), wins=wins, losses=losses, success_delta=(wins-losses)/len(aa),
                total_service_cost_ratio=summaries[f'{primary}_h{h}']['total_service_seconds']/summaries[f'{cid}_h{h}']['total_service_seconds'])
            comparisons[f'{primary}_h{h}_vs_{cid}'] = comparison
    result = dict(cohort=args.cohort, status='partial_descriptive_descriptive_only',
                  recipe_sha256=sha(recipe_path), definition_sha256=sha(definition_path),
                  final_stop_sha256=sha(stop_path), source_sha256=sources,
                  planned_episodes=definition['total_episodes'], included_episodes=len(all_rows),
                  included_complete_shards=chosen, omitted_shards=omitted,
                  primary_candidate=primary, groups=summaries, primary_point_comparisons=comparisons,
                  intervals_computed=False, formal_confirmation_completed=False,
                  interpretation=recipe['interpretation'])
    output = ROOT/'analysis'/f'{args.cohort}_partial_descriptive.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(cohort=args.cohort, included_episodes=len(all_rows), planned_episodes=definition['total_episodes'], output_sha256=sha(output))))


if __name__ == '__main__':
    main()
