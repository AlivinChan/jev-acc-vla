"""Combine declared developer confirmation with exact fixed-period reference cells."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
from analyze_closed_loop import summary,paired,rows_from

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main'


def main():
    groups=defaultdict(list);sources={};seen=set()
    for run in ['single_skip_v1','fixed_interval_v1','single_skip_confirmation_v1']:
        base=ROOT/'raw'/run
        assert json.loads((base/'supervisor.json').read_text(encoding='utf-8'))['status']=='completed'
        assert json.loads((ROOT/'checks'/f'{run}_audit.json').read_text(encoding='utf-8'))['errors']==0
        path=base/'data/episodes.jsonl'
        sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        for row in rows_from(path):
            h=row['horizon'];cid=row['candidate_id']
            if cid not in ['compact_skip1','window3_skip1','fixed_r60']:
                if row['method']=='fixed_interval' and row['gate_interval']==2*round(.3*h):cid='fixed_r60'
                else:continue
            key=(cid,row['task_id'],row['state_id'],h)
            assert key not in seen,'Do not double-count a reused cell'
            seen.add(key);groups[(cid,h)].append(row)
    expected={(tid,sid) for tid in range(10) for sid in range(3)}
    assert set(groups)=={(cid,h) for cid in ['compact_skip1','window3_skip1','fixed_r60'] for h in [50,100,200]}
    for rows in groups.values():assert len(rows)==30 and {(r['task_id'],r['state_id']) for r in rows}==expected
    entries={}
    for (cid,h),rows in sorted(groups.items()):
        row=summary(rows);row['comparisons']={}
        references={name:[json.loads((OLD/f't{tid:02d}_s{sid:02d}_h{h}_{name}/result.json').read_text(encoding='utf-8'))
                          for tid,sid in sorted(expected)] for name in ['smol70','vlash_style_k5']}
        references.update(fixed_r60=groups[('fixed_r60',h)],compact_skip1=groups[('compact_skip1',h)])
        for name,ref in references.items():
            comparison=paired(rows,ref);assert comparison['n']==30
            row['comparisons'][name]=comparison
        row['by_task']={str(t):summary([r for r in rows if r['task_id']==t]) for t in range(10)}
        entries[f'{cid}_h{h}']=row
    result=dict(groups=entries,source_sha256=sources,unique_new_and_reused_policy_cells=len(seen),
        interpretation='30 exposed developer starts per policy/H, including selection pilot. Fixed-period cells reused exactly once. Historical baseline costs; no independent or real-time improvement claim.')
    (ROOT/'analysis/single_skip_development_pooled.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    for name,row in entries.items():
        c=row['comparisons']['smol70']
        print(json.dumps(dict(policy=name,n=row['n'],successes=row['successes'],calls=row['vla_calls'],
              gates=row['gate_calls'],cost_ratio=c['total_service_cost_ratio'],wins=c['wins'],losses=c['losses'])))


if __name__=='__main__':main()
