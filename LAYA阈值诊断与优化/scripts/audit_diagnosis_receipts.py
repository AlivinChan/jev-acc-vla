"""Read-only probability/fixture audit of the original completed threshold diagnosis."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT/'raw/diagnosis_v1/data'
    manifest = read(base/'manifest.json')
    fixture_path = ROOT/'fixtures/diagnostic_replay.json'
    fixture = {r['id']: r for r in read(fixture_path)}
    summary = read(ROOT/'analysis/diagnosis_summary.json')
    rows = [json.loads(s) for s in (base/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert read(base.parent/'supervisor.json')['status'] == 'completed'
    counts, failures = Counter(), []

    def check(ok, kind, context):
        counts[kind] += 1
        if not ok:
            failures.append(dict(kind=kind, context=context))

    check(sha(fixture_path) == manifest['fixture_sha256'], 'immutable input fixture', 'manifest')
    variants = manifest['variants']
    expected = {(variant, identity) for variant in variants for identity in fixture}
    keys = [(r['variant'], r['id']) for r in rows]
    check(len(keys) == len(set(keys)) and set(keys) == expected, 'complete unique diagnostic matrix', len(rows))
    maximum_error = 0.
    for row in rows:
        key = [row['variant'], row['id']]
        source = fixture[row['id']]
        answer = row['raw']['answers']['replanning']
        check(row['kind'] == source['kind'] and row['warmup'] is False, 'source kind and warmup exclusion', key)
        check(hashlib.sha256(row['state'].encode('utf-8')).hexdigest() == row['state_sha256'], 'state bytes', key)
        labels = list(row['logits'])
        scaled = [row['logits'][label]/row['temperature'] for label in labels]
        exps = [math.exp(value-max(scaled)) for value in scaled]
        exact = {label: value/sum(exps) for label, value in zip(labels, exps)}
        error = max(abs(exact[label]-row['probabilities'][label]) for label in labels)
        maximum_error = max(maximum_error, error)
        check(error <= .00006 and set(exact) == set(answer['probabilities']), 'SDK softmax mapping within output rounding', key)
        check(row['probabilities'] == answer['probabilities'] and row['raw_choice'] == answer['choice'], 'unaltered SDK decision receipt', key)
        check(row['max_probability'] == max(row['probabilities'].values()) == answer['answer_confidence'], 'maximum-probability field semantics', key)
        check(row['probabilities'][row['raw_choice']] >= max(row['probabilities'].values())-.0001, 'choice is a probability maximizer within reporting precision', key)
        expected_temperature = manifest['temperature_by_options']['choice:2' if len(labels) == 2 else 'choice:3-5']
        check(row['temperature'] == expected_temperature, 'correct option-count temperature', key)
        check(row['token_audit']['head_dropped'] == 0 and row['token_audit']['state']['state_tokens_dropped'] == 0,
              'no head or state truncation in tested inputs', key)
        if row['variant'] == 'original3' and row['kind'] == 'developer_replay':
            check(row['state'] == source['state'] and row['state_sha256'] == source['old_state_sha256'], 'original replay source bytes', key)
            check(row['probabilities'] == source['old_probabilities'] and row['old_probability_max_error'] == 0,
                  'original historical probabilities reproduced', key)
    for variant in variants:
        chosen = [r for r in rows if r['variant'] == variant and r['kind'] == 'developer_replay']
        controls = [r for r in rows if r['variant'] == variant and r['kind'] != 'developer_replay']
        old = summary[variant]
        check(len(chosen) == old['n'] == 270 and len(controls) == old['semantic_controls']['n'] == 10,
              'replay and semantic controls remain separate', variant)
        check(sum(r['max_probability'] >= .7 for r in chosen) == old['cross_07'] and
              dict(Counter(r['raw_choice'] for r in chosen)) == old['raw_choices'], 'reported threshold counts and decisions', variant)
        check(statistics.mean(r['token_audit']['tokens'] for r in chosen) == old['tokens']['mean'], 'reported mean token count', variant)
    result = dict(run='diagnosis_v1', decisions=len(rows), robot_replay_decisions=1620,
                  semantic_control_decisions=60, checks=sum(counts.values()), by_kind=dict(counts),
                  errors=len(failures), failures=failures, maximum_softmax_rounding_error=maximum_error,
                  inputs_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
                                 for p in [fixture_path, base/'manifest.json', base/'decisions.jsonl',
                                           ROOT/'analysis/diagnosis_summary.json']}, auditor_sha256=sha(Path(__file__)),
                  scope='Receipt consistency only; semantic controls and confident scores do not establish robot-gating accuracy')
    output = ROOT/'checks/diagnosis_v1_audit.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ['decisions', 'checks', 'errors', 'maximum_softmax_rounding_error']}))
    if failures:
        print(json.dumps(failures[:5]))
    assert not failures


if __name__ == '__main__':
    main()
