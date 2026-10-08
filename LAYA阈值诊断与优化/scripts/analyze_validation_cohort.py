"""Analyze only a complete, frozen paired-validation cohort; never select its winner."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json
from analyze_closed_loop import rows_from,summary,paired
from paired_resampling import uncertainty

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',default='validation_d1_v1');args=p.parse_args()
    assert args.cohort.isascii() and args.cohort.replace('_','').isalnum()
    folder=ROOT/'protocol'/args.cohort;definition=read(folder/'definition.json')
    assert sha(folder/'definition.json')==read(folder/'freeze_receipt.json')['definition_sha256']
    groups=defaultdict(list);seen=set();sources={};audits={}
    for shard in definition['shards']:
        name=shard['run'];base=ROOT/'raw'/name
        assert read(base/'supervisor.json')['status']=='completed',name+' is incomplete'
        audit_path=ROOT/'checks'/f'{name}_audit.json';audit=read(audit_path)
        assert audit['errors']==0 and audit['episodes']==shard['episodes'];audits[name]=sha(audit_path)
        if definition.get('initial_input_verification_required'):
            initial_audit_path=ROOT/'checks'/f'{name}_initial_verification.json'
            initial_audit=read(initial_audit_path)
            assert initial_audit['errors']==0 and initial_audit['episodes']==shard['episodes']
            audits[name+'_initial_verification']=sha(initial_audit_path)
        if definition.get('noise_alignment_audit_required'):
            noise_audit_path=ROOT/'checks'/f'{name}_noise_alignment.json'
            assert read(noise_audit_path)['errors']==0
            audits[name+'_noise_alignment']=sha(noise_audit_path)
        manifest=read(base/'data/manifest.json')
        assert manifest['config_sha256']==shard['config_sha256']
        assert manifest['config']['cohort_id']==args.cohort
        assert manifest['config']['candidates']==definition['candidates']
        for script,digest in definition['locked_execution_source_sha256'].items():
            assert manifest['scripts'][script]==digest
        assert manifest['freeze_before']==manifest['freeze_after']
        assert manifest['freeze_after']['state_dict_sha256']==definition['vla_sha256']
        path=base/'data/episodes.jsonl';rows=rows_from(path);assert len(rows)==shard['episodes']
        sources[name]=dict(episodes_sha256=sha(path),manifest_sha256=sha(base/'data/manifest.json'),
            archive=read(ROOT/'artifacts'/f'{name}_archive.json'))
        for row in rows:
            cid=row['candidate_id'];h=row['horizon']
            assert row['state_id']==shard['state'] and row['logical_delay_steps']==definition['logical_delay_steps']
            assert row['candidate']==definition['candidates'][cid]
            key=(cid,row['task_id'],row['state_id'],h);assert key not in seen
            seen.add(key);groups[(cid,h)].append(row)
    expected={(t,s) for t in definition['tasks'] for s in definition['states_in_execution_order']}
    cells={(cid,h) for cid,c in definition['candidates'].items() for h in c.get('horizons',definition['horizons'])}
    assert set(groups)==cells and len(seen)==definition['total_episodes']
    for rows in groups.values():
        assert len(rows)==definition['episodes_per_method_horizon']
        assert {(r['task_id'],r['state_id']) for r in rows}==expected
    entries={}
    for (cid,h),rows in sorted(groups.items()):
        entry=summary(rows);entry['role']=definition['candidates'][cid]['role'];entry['comparisons']={}
        comparators=definition.get('comparators_by_horizon',{}).get(str(h),
            ['smol70','fixed45','naive_k5','vlash_style_k5','compact_skip1'])
        for ref in comparators:
            refs=groups[(ref,h)];comparison=paired(rows,refs)
            assert comparison['n']==definition['episodes_per_method_horizon']
            comparison['paired_uncertainty']=uncertainty(rows,refs)
            entry['comparisons'][ref]=comparison
        entry['by_task']={str(t):summary([r for r in rows if r['task_id']==t]) for t in definition['tasks']}
        entries[f'{cid}_h{h}']=entry
    output=ROOT/'analysis'/f'{args.cohort}_summary.json'
    result=dict(cohort=args.cohort,definition_sha256=sha(folder/'definition.json'),
        total_episodes=len(seen),primary_candidate=definition['primary_candidate'],
        primary_comparators=definition['primary_comparators'],groups=entries,
        source_runs=sources,audit_sha256=audits,
        interpretation='Candidates and bytes frozen before these starts. Earlier project exposure not excluded. Complete serial-service, logical-delay comparison; no real-time speedup or equivalence claim. No candidate reselection performed.')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    for name,r in entries.items():
        c=r['comparisons']['smol70'];u=c['paired_uncertainty']['intervals']['hierarchical_task_and_initial']
        print(json.dumps(dict(group=name,n=r['n'],successes=r['successes'],calls=r['vla_calls'],gates=r['gate_calls'],
            service_seconds=r['total_service_seconds'],vs_smol70_cost_ratio=c['total_service_cost_ratio'],
            vs_smol70_wins=c['wins'],vs_smol70_losses=c['losses'],hierarchical_intervals=u)))


if __name__=='__main__':main()
