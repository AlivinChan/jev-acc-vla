"""Audit the full predeclared option-order probe; report sensitivity, not robot correctness."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', nargs='?', default='choice_order_audit_v1')
    args = parser.parse_args()
    assert args.run.isascii() and args.run.replace('_', '').isalnum()
    folder = ROOT/'raw'/args.run
    assert read(folder/'supervisor.json')['status'] == 'completed'
    manifest = read(folder/'data/manifest.json')
    definition_path = ROOT/'protocol/choice_order_audit_v1/definition.json'
    definition = read(definition_path)
    assert sha(definition_path) == read(definition_path.parent/'freeze_receipt.json')['definition_sha256']
    assert manifest['definition_sha256'] == sha(definition_path) and manifest['definition'] == definition
    assert manifest['frozen_laya_before'] == manifest['frozen_laya_after'] == definition['frozen_laya_sha256']
    assert manifest['frozen_parameters_unchanged'] and manifest['no_optimizer']
    fixture_path = ROOT/'fixtures/choice_order_audit_v1.json'
    assert sha(fixture_path) == definition['fixture_sha256']
    fixture = read(fixture_path)
    expected = {r['id']:r for r in fixture}
    rows_path = folder/'data/decisions.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines()]
    warmups = [json.loads(line) for line in (folder/'data/warmup.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == definition['decisions'] and len(warmups) == definition['warmups']
    grouped = defaultdict(dict)
    checks = Counter()
    for row in rows:
        original = expected[row['id']]
        order = tuple(row['order'])
        assert order in [(0,1),(1,0)] and order not in grouped[row['id']]
        assert row['variant'] == original['variant'] and row['kind'] == original['kind']
        assert row['model_state_sha256'] == original['model_state_sha256']
        audit = row['token_audit']
        assert audit['head_dropped'] == 0 and not audit['state']['truncated'] and audit['options']['options_distinct'] == 2
        z = np.array(row['slot_logits'], dtype=float)/row['temperature']
        assert z.shape == (2,) and np.isfinite(z).all() and row['temperature'] > 0
        p = np.exp(z-z.max());p /= p.sum()
        canonical = {label:float(p[order.index(i)]) for i,label in enumerate(['replan','continue'])}
        assert max(abs(canonical[k]-row['canonical_unrounded_probabilities'][k]) for k in canonical) < 1e-12
        assert max(abs(canonical[k]-row['probabilities'][k]) for k in canonical) < .00006
        assert row['choice'] in canonical and row['probabilities'] == row['raw']['answers']['replanning']['probabilities']
        assert row['choice'] == row['raw']['answers']['replanning']['choice']
        assert abs(sum(row['probabilities'].values())-1) < .0002 and row['sdk_seconds'] > 0
        assert abs(canonical['replan']-canonical['continue']) < 1e-6 or row['choice'] == max(canonical, key=canonical.get)
        if order == (0,1):
            assert row['probabilities'] == original['expected_probabilities'] and row['choice'] == original['expected_choice']
            checks['historical_original_order_exact'] += 1
        if 'semantic_label' in original:
            assert row['semantic_label'] == original['semantic_label']
        grouped[row['id']][order] = row
        checks['input_schema_and_untruncated'] += 1
        checks['logits_to_semantic_probability'] += 1
        checks['raw_choice_and_timing'] += 1
    assert set(grouped) == set(expected) and all(set(r) == {(0,1),(1,0)} for r in grouped.values())
    groups = defaultdict(list)
    for cid, pair in grouped.items():
        record = expected[cid]
        groups[(record['variant'],record['kind'])].append(pair)
    result = {}
    for (variant,kind), pairs in sorted(groups.items()):
        a = [pair[(0,1)] for pair in pairs];b = [pair[(1,0)] for pair in pairs]
        pa = np.array([r['canonical_unrounded_probabilities']['replan'] for r in a])
        pb = np.array([r['canonical_unrounded_probabilities']['replan'] for r in b])
        delta = pb-pa
        entry = dict(n=len(pairs), choice_agreement=sum(x['choice']==y['choice'] for x,y in zip(a,b))/len(pairs),
            original_replan=sum(x['choice']=='replan' for x in a), reverse_replan=sum(x['choice']=='replan' for x in b),
            original_R_to_reverse_C=sum(x['choice']=='replan' and y['choice']=='continue' for x,y in zip(a,b)),
            original_C_to_reverse_R=sum(x['choice']=='continue' and y['choice']=='replan' for x,y in zip(a,b)),
            reverse_minus_original_p_replan_mean=float(delta.mean()),
            absolute_probability_difference_mean=float(np.abs(delta).mean()),
            absolute_probability_difference_max=float(np.abs(delta).max()),
            absolute_probability_difference_p95=float(np.quantile(np.abs(delta),.95)),
            p_replan_correlation=float(np.corrcoef(pa,pb)[0,1]) if np.std(pa)>0 and np.std(pb)>0 else None,
            order_metrics={})
        for label, rr, probabilities in [('original',a,pa),('reversed',b,pb)]:
            times=np.array([r['sdk_seconds'] for r in rr]);confidence=np.maximum(probabilities,1-probabilities)
            metric=dict(above_07=int(np.sum(confidence>=.7)), max_probability_range=[float(confidence.min()),float(confidence.max())],
                sdk_seconds_mean=float(times.mean()),sdk_seconds_median=float(np.median(times)))
            if all('semantic_label' in r for r in rr):
                metric['explicit_semantic_correct']=sum(r['choice']==r['semantic_label'] for r in rr)
            entry['order_metrics'][label]=metric
        result[variant+'__'+kind]=entry
    output=dict(run=args.run,inputs=len(grouped),decisions=len(rows),groups=result,
        definition_sha256=sha(definition_path),decisions_sha256=sha(rows_path),
        archive=read(ROOT/'artifacts'/f'{args.run}_archive.json'),
        interpretation='Only option order varied. Same-state probability and choice sensitivity does not establish robot correctness. Explicit semantic controls are a separate small diagnostic, not LIBERO successes.')
    (ROOT/'analysis'/f'{args.run}_summary.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8',newline='\n')
    audit=dict(run=args.run,inputs=len(grouped),decisions=len(rows),checks=dict(checks),errors=0,
        audit_script_sha256=sha(Path(__file__)),manifest_sha256=sha(folder/'data/manifest.json'))
    (ROOT/'checks'/f'{args.run}_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(audit))
    for name,entry in result.items():
        print(json.dumps(dict(group=name,**entry)))


if __name__ == '__main__':
    main()
