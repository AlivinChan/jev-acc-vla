from pathlib import Path
from collections import Counter
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
rows=[json.loads(line) for line in (ROOT/'raw/diagnosis_v1/data/decisions.jsonl').read_text().splitlines()]
summary={}
def describe(values):
    values=sorted(values)
    return dict(min=min(values),median=statistics.median(values),mean=statistics.mean(values),max=max(values))
original={r['id']:r for r in rows if r['variant']=='original3'}
for variant in dict.fromkeys(r['variant'] for r in rows):
    cases=[r for r in rows if r['variant']==variant and r['kind']=='developer_replay']
    controls=[r for r in rows if r['variant']==variant and r['kind']!='developer_replay']
    entry=dict(n=len(cases),max_probability=describe([r['max_probability'] for r in cases]),
        cross_07=sum(r['max_probability']>=.7 for r in cases),
        raw_choices=dict(Counter(r['raw_choice'] for r in cases)),
        changed_from_original=sum(r['raw_choice']!=original[r['id']]['raw_choice'] for r in cases),
        tokens=describe([r['token_audit']['tokens'] for r in cases]),
        head_dropped=describe([r['token_audit']['head_dropped'] for r in cases]),
        fits_512=sum(r['token_audit']['fits_512'] for r in cases),
        sdk_wall_ms=describe([r['sdk_wall_seconds']*1000 for r in cases]),
        semantic_controls=dict(n=len(controls),correct=sum(r['raw_choice']==r['semantic_label'] for r in controls),
            cross_07=sum(r['max_probability']>=.7 for r in controls),
            probabilities=[dict(id=r['id'],label=r['semantic_label'],choice=r['raw_choice'],p=r['max_probability']) for r in controls]))
    if variant=='original3':
        entry['old_probability_max_error']=max(r['old_probability_max_error'] for r in cases)
        entry['conditional_binary_probability']=describe([max(r['probabilities'][k] for k in ['continue','replan'])/
            sum(r['probabilities'][k] for k in ['continue','replan']) for r in cases])
        entry['temperature_one_max_probability']=describe([max(r['temperature_one_diagnostic_only'].values()) for r in cases])
    summary[variant]=entry
out=ROOT/'analysis';out.mkdir(exist_ok=True)
(out/'diagnosis_summary.json').write_text(json.dumps(summary,indent=2))
for name,s in summary.items():
    print(json.dumps(dict(variant=name,n=s['n'],prob=s['max_probability'],cross_07=s['cross_07'],choices=s['raw_choices'],
       changed=s['changed_from_original'],tokens=s['tokens']['mean'],head_dropped=s['head_dropped'],
       sdk_ms=s['sdk_wall_ms']['median'],controls=s['semantic_controls']['correct'],
       controls_07=s['semantic_controls']['cross_07'],old_error=s.get('old_probability_max_error'))))
