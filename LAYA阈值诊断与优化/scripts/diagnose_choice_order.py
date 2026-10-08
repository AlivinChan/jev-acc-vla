"""Only permute the two option slots; preserve canonical semantic labels and input bytes."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
from gate_variants import questions_for, VARIANTS
from score_counterfactuals import parameter_digest
from supervise import allowed

BASE = Path('/mnt/4t/jev_vla_libero')
ROOT = BASE / 'experiments/laya_gate_optimization'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT/'runs')
    folder = ROOT/'protocol/choice_order_audit_v1'
    definition = json.loads((folder/'definition.json').read_text())
    assert sha(folder/'definition.json') == json.loads((folder/'freeze_receipt.json').read_text())['definition_sha256']
    assert sha(ROOT/'fixtures/choice_order_audit_v1.json') == definition['fixture_sha256']
    for name, digest in definition['script_sha256'].items():
        assert sha(Path(__file__).parent/name) == digest
    allowed()
    out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    import laya
    from laya.common import build_sequence, _encode_question_text, QTYPES, temp_bucket
    torch.set_num_threads(4)
    source = json.loads((BASE/'configs/local_gate_sources.json').read_text())['laya']
    assert laya.__version__ == source['sdk_version']
    agent = laya.Agent(str(BASE/source['path']), device='cuda', compile=False, fast=False,
                       expected_sha256={a['path']:a['sha256'] for a in source['assets']})
    agent.model.eval()
    for parameter in agent.model.parameters():
        parameter.requires_grad_(False)
    assert agent.device.type == 'cuda'
    before = parameter_digest(agent.model, torch)
    assert before == definition['frozen_laya_sha256']
    records = json.loads((ROOT/'fixtures/choice_order_audit_v1.json').read_text())
    captured = []
    original = agent._forward

    def observe(batch):
        logits, action = original(batch)
        captured.append(logits.copy())
        return logits, action

    agent._forward = observe
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), definition=definition,
        definition_sha256=sha(folder/'definition.json'), source=source, no_optimizer=True,
        frozen_laya_before=before, timing_scope='Synchronized full SDK call; separate token audit; no VLA or robot rollouts')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))

    def evaluate(record, order):
        state = record['model_state']
        assert hashlib.sha256(state.encode()).hexdigest() == record['model_state_sha256']
        spec = VARIANTS[record['variant']]
        questions = questions_for(spec['schema'])
        q = questions['replanning']
        q['option_order'] = order
        assert list(q['criteria']) == ['replan', 'continue']
        internal = dict(t=q['type'], ins=q['instructions'], crit=q['criteria'])
        ids, markers, options, truncation = build_sequence(agent.tok, state, internal,
            max_len=spec['max_len'], head_max_len=spec['head'], option_order=order,
            return_stats=True, return_truncation_stats=True)
        full_head = len(_encode_question_text(agent.tok, 'choice question: '+q['instructions'], add_special_tokens=False))
        assert markers[0]-2 == full_head and not truncation['truncated'] and options['options_distinct'] == 2
        captured.clear()
        fallback = getattr(agent, 'cpu_fallback_count', 0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            raw = agent.system_one(state=state, questions=questions, max_len=spec['max_len'], head_max_len=spec['head'])
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        assert agent.device.type == 'cuda' and getattr(agent, 'cpu_fallback_count', 0) == fallback
        assert len(captured) == 1
        logits = np.asarray(captured[0][0][:2], dtype=np.float64)
        temperature = float(agent.temperature_by_options.get(temp_bucket(QTYPES['choice'], 2), agent.temperature[QTYPES['choice']]))
        p = np.exp(logits/temperature - np.max(logits/temperature))
        p /= p.sum()
        canonical = {label:float(p[order.index(i)]) for i,label in enumerate(q['criteria'])}
        answer = raw['answers']['replanning']
        error = max(abs(canonical[label]-answer['probabilities'][label]) for label in canonical)
        assert error < .00006
        if order == [0, 1]:
            assert answer['choice'] == record['expected_choice']
            assert answer['probabilities'] == record['expected_probabilities'], 'Original-order replay changed'
        result = dict(id=record['id'], variant=record['variant'], kind=record['kind'], order=order,
            model_state_sha256=record['model_state_sha256'], slot_logits=logits.tolist(),
            temperature=temperature, canonical_unrounded_probabilities=canonical,
            probabilities=answer['probabilities'], choice=answer['choice'], raw=raw,
            semantic_mapping_error=error, sdk_seconds=elapsed,
            token_audit=dict(tokens=len(ids), head_dropped=0, state=truncation, options=options))
        if 'semantic_label' in record:
            result['semantic_label'] = record['semantic_label']
        return result

    with (out/'warmup.jsonl').open('w') as stream:
        for variant in definition['counts']:
            example = next(r for r in records if r['variant'] == variant)
            for order in definition['option_orders']:
                stream.write(json.dumps(evaluate(example, order))+'\n')
    with (out/'decisions.jsonl').open('w') as stream:
        for index, record in enumerate(records):
            allowed()
            orders = definition['option_orders'][::(-1 if index % 2 else 1)]
            for order in orders:
                stream.write(json.dumps(evaluate(record, order))+'\n')
                stream.flush()
            if (index+1) % 50 == 0:
                print(json.dumps(dict(completed=index+1, planned=len(records))), flush=True)
    after = parameter_digest(agent.model, torch)
    assert before == after and not agent.model.training and all(not p.requires_grad and p.grad is None for p in agent.model.parameters())
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(), frozen_laya_after=after,
                    frozen_parameters_unchanged=True, decisions=2*len(records), warmups=8)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out/'status.json').write_text(json.dumps(dict(phase='completed', completed=len(records), planned=len(records), decisions=2*len(records))))
    print(json.dumps(dict(completed=len(records), decisions=2*len(records), frozen_parameters_unchanged=True)), flush=True)


if __name__ == '__main__':
    main()
