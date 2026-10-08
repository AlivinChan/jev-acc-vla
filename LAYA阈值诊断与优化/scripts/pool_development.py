from pathlib import Path
from collections import defaultdict
import json
from analyze_closed_loop import rows_from,summary,paired,key
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main'
allowed={'compact_r30':[50,100,200],'compact_minr30_i5':[50,100,200],'short5':[50],'compact5':[200]}
groups=defaultdict(list);origins={}
for run in ['representations_v1','cadence_v1','confirmation_v1']:
    for row in rows_from(ROOT/'raw'/run/'data/episodes.jsonl'):
        candidate=row['candidate_id'];h=row['horizon']
        if candidate not in allowed or h not in allowed[candidate]:continue
        unique=(candidate,*key(row));assert unique not in origins
        origins[unique]=run;groups[(candidate,h)].append(row)
results={}
for (candidate,h),rows in sorted(groups.items()):
    assert len(rows)==30 and {r['state_id'] for r in rows}=={0,1,2}
    item=summary(rows);item['comparisons']={}
    for method in ['smol70','naive_k5','vlash_style_k5']:
        reference=[json.loads((OLD/f't{r["task_id"]:02d}_s{r["state_id"]:02d}_h{h}_{method}/result.json').read_text()) for r in rows]
        item['comparisons'][method]=paired(rows,reference)
    results[f'{candidate}_h{h}']=item
    baseline=item['comparisons']['smol70']
    print(json.dumps(dict(candidate=candidate,horizon=h,n=30,success=item['successes'],calls=item['vla_calls'],
        gates=item['gate_calls'],cost=round(item['total_service_seconds'],3),
        cost_ratio=baseline['total_service_cost_ratio'],cost_CI=baseline['cost_ratio_95_task_bootstrap'],
        wins=baseline['wins'],losses=baseline['losses'])))
out=dict(groups=results,source_runs=['representations_v1','cadence_v1','confirmation_v1'],
         interpretation='Pooled exposed developer states0-2; includes selection and confirmation, not a new test set',
         unique_case_count=sum(len(rows) for rows in groups.values()))
(ROOT/'analysis/pooled_development.json').write_text(json.dumps(out,indent=2))
