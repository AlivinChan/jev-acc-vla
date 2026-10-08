"""One first-request intervention per episode, including both-failure and early-ending cases."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main'
VARIANTS=['binary_short','binary_compact','binary_window','binary_window3']


def lines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--counterfactual-runs',required=True)
    p.add_argument('--score-runs',required=True);p.add_argument('--states',required=True);p.add_argument('--name',required=True)
    args=p.parse_args();states={int(s) for s in args.states.split(',')};records={};scores={};sources={}
    for name in args.counterfactual_runs.split(','):
        base=ROOT/'raw'/name;assert json.loads((base/'supervisor.json').read_text())['status']=='completed'
        path=base/'data/records.jsonl';sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for r in lines(path):
            if r['state'] in states and r['source_call_id']==1:
                key=(r['task'],r['state'],r['horizon']);assert key not in records;records[key]=r
    for name in args.score_runs.split(','):
        base=ROOT/'raw'/name;assert json.loads((base/'supervisor.json').read_text())['status']=='completed'
        path=base/'data/scores.jsonl';sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for r in lines(path):
            if r['state'] in states and r['variant'] in VARIANTS:
                key=(r['id'],r['variant']);assert key not in scores;scores[key]=r
    cases=[]
    for h in [50,100,200]:
        for tid in range(10):
            for sid in sorted(states):
                baseline=json.loads((OLD/f't{tid:02d}_s{sid:02d}_h{h}_smol70/result.json').read_text())
                key=(tid,sid,h);r=records.get(key)
                original=dict(success=baseline['success'],calls=baseline['vla_calls'],steps=baseline['steps'])
                if r is None:
                    assert baseline['vla_calls']==1,'Missing a first-request intervention'
                    choices={name:dict(**original,choice='no_opportunity') for name in ['always_replan','always_skip','oracle',*VARIANTS]}
                    cases.append(dict(task=tid,state=sid,horizon=h,opportunity=False,choices=choices))
                    continue
                assert r['tick']==round(.3*h) and r['branches']['replan']['baseline_suffix_exact']
                outcomes={choice:dict(success=b['success'],calls=1+b['vla_calls'],steps=b['total_steps']) for choice,b in r['branches'].items()}
                assert outcomes['replan']==original
                oracle=max(['continue','replan'],key=lambda choice:(outcomes[choice]['success'],-outcomes[choice]['calls'],choice=='continue'))
                selections={'always_replan':'replan','always_skip':'continue','oracle':oracle}
                for variant in VARIANTS:selections[variant]=scores[(r['id'],variant)]['projected_action_choice']
                choices={name:dict(**outcomes[choice],choice=choice) for name,choice in selections.items()}
                cases.append(dict(task=tid,state=sid,horizon=h,opportunity=True,id=r['id'],outcomes=outcomes,choices=choices))
    groups={}
    for h in [50,100,200]:
        group=[r for r in cases if r['horizon']==h];active=[r for r in group if r['opportunity']]
        reference_success=sum(r['choices']['always_replan']['success'] for r in group)
        reference_calls=sum(r['choices']['always_replan']['calls'] for r in group)
        for method in ['always_replan','always_skip','oracle',*VARIANTS]:
            chosen=[r['choices'][method] for r in group];skips=sum(c['choice']=='continue' for c in chosen)
            fraction=skips/len(active) if active else 0.
            random_success=reference_success+fraction*sum(int(r['outcomes']['continue']['success'])-int(r['outcomes']['replan']['success']) for r in active)
            random_calls=reference_calls+fraction*sum(r['outcomes']['continue']['calls']-r['outcomes']['replan']['calls'] for r in active)
            wins=sum(c['success'] and not r['choices']['always_replan']['success'] for r,c in zip(group,chosen))
            losses=sum(not c['success'] and r['choices']['always_replan']['success'] for r,c in zip(group,chosen))
            groups[f'{method}_h{h}']=dict(episodes=len(group),opportunities=len(active),skips=skips,
                successes=sum(c['success'] for c in chosen),vla_calls=sum(c['calls'] for c in chosen),wins=wins,losses=losses,
                calls_saved=reference_calls-sum(c['calls'] for c in chosen),
                matched_skip_count_random_policy_expected_successes=random_success,
                matched_skip_count_random_policy_expected_calls=random_calls)
    result=dict(states=sorted(states),groups=groups,cases=cases,source_sha256=sources,
        intervention_rule='Only the first Smol70 request can be skipped; afterwards both branches follow Smol70. No addition of effects from multiple opportunities.',
        oracle_note='Uses observed future branch outcomes; unattainable deployment upper reference, not a deployable LAYA result.',
        missing_label_note='Both-failure cases and episodes ending before the first request are included in episode success and call totals.',
        cost_note='VLA call counts only. No gate cost included, so this is not a measured total-service speedup.',
        randomness_note='Expected outcome of a uniformly random subset with the same skip count per H, not a confidence interval or significance test.')
    (ROOT/'analysis'/f'{args.name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    for name,r in groups.items():print(json.dumps(dict(method=name,**r)))


if __name__=='__main__':main()
