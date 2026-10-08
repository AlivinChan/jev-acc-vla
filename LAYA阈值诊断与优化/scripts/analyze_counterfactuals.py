"""Descriptive branch opportunity counts; do not confuse these with independent episodes."""
from pathlib import Path
from collections import Counter
import argparse,json
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--states',required=True)
p.add_argument('--name',required=True);args=p.parse_args()
states={int(s) for s in args.states.split(',')};records={};preparation=[]
for name in args.runs.split(','):
    folder=ROOT/'raw'/name/'data';manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['frozen_parameters_unchanged']
    preparation.append(dict(run=name,cases=manifest['planned'],VLA_calls=manifest['total_vla_calls'],
        VLA_seconds=manifest['total_prediction_seconds'],
        stage_wall_seconds=json.loads((ROOT/'raw'/name/'supervisor.json').read_text())['elapsed_seconds']))
    for line in (folder/'records.jsonl').read_text().splitlines():
        record=json.loads(line)
        if record['state'] in states:
            assert record['id'] not in records
            records[record['id']]=record
expected={r['id'] for r in json.loads((ROOT/'fixtures/counterfactual_schedule.json').read_text()) if r['state'] in states}
assert set(records)==expected
groups={}
for horizon in [50,100,200]:
    rows=[r for r in records.values() if r['horizon']==horizon]
    counts=Counter();utility=Counter();necessity=Counter()
    for r in rows:
        c=r['branches']['continue']['success'];s=r['branches']['replan']['success']
        counts['both_success' if c and s else 'both_failure' if not c and not s else 'continue_only' if c else 'replan_only']+=1
        utility[r['labels']['utility'] or 'unresolved']+=1
        necessity[r['labels']['necessity'] or 'unresolved']+=1
    groups[str(horizon)]=dict(opportunities=len(rows),paired_outcomes=dict(counts),utility_labels=dict(utility),
        necessity_labels=dict(necessity),resolved_coverage=sum(r['labels']['utility'] is not None for r in rows)/len(rows),
        source_initial_states=len({(r['task'],r['state']) for r in rows}),
        continue_minus_replan_call_difference=dict(Counter(r['branches']['continue']['vla_calls']-r['branches']['replan']['vla_calls'] for r in rows)))
result=dict(states=sorted(states),unique_opportunities=len(records),groups=groups,preparation=preparation,
    interpretation='All registered opportunities from the stated initial states; repeated positions within a trajectory are correlated. These are single fixed-noise intervention outcomes under a common Smol70 tail.')
(ROOT/'analysis'/f'{args.name}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(states=sorted(states),unique_opportunities=len(records),groups=groups)))
