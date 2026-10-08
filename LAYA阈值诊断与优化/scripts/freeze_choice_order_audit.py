"""Freeze final model inputs and historical predictions for a pure option-order probe."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
from gate_variants import questions_for, VARIANTS
from local_policy import allowed

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    allowed()
    fixture = ROOT / 'fixtures/choice_order_audit_v1.json'
    folder = ROOT / 'protocol/choice_order_audit_v1'
    assert not fixture.exists() and not folder.exists()
    records = []
    source_paths = ['raw/diagnosis_v1/data/decisions.jsonl', 'raw/scores_window_train_v1/data/scores.jsonl']
    for index, source in enumerate(source_paths):
        for line in (ROOT / source).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            variant = row['variant']
            if index == 0 and variant not in ['binary_full', 'binary_short', 'binary_compact']:
                continue
            if index == 1 and variant != 'binary_window3':
                continue
            text = row['state'] if index == 0 else row['model_state']
            digest = hashlib.sha256(text.encode()).hexdigest()
            assert digest == row['state_sha256' if index == 0 else 'model_state_sha256']
            if index == 0:
                q = questions_for(VARIANTS[variant]['schema'])
                assert hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest() == row['prompt_sha256']
            else:
                assert row['state'] == 0
            answer = row['raw']['answers']['replanning']
            item = dict(id=f'{index}:{variant}:{row["id"]}', source=source, source_record_id=row['id'],
                variant=variant, model_state=text, model_state_sha256=digest,
                kind=row.get('kind', 'developer_counterfactual_state0'),
                expected_probabilities=answer['probabilities'], expected_choice=answer['choice'])
            if 'semantic_label' in row:
                item['semantic_label'] = row['semantic_label']
            records.append(item)
    counts = dict(Counter(r['variant'] for r in records))
    assert counts == dict(binary_full=280, binary_short=280, binary_compact=280, binary_window3=189)
    assert len({r['id'] for r in records}) == len(records) == 1029
    fixture.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8', newline='\n')
    folder.mkdir(parents=True)
    definition = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), inputs=1029, decisions=2058,
        warmups=8, counts=counts, fixture_sha256=sha(fixture), option_orders=[[0, 1], [1, 0]],
        source_sha256={p:sha(ROOT/p) for p in source_paths},
        script_sha256={p:sha(ROOT/'scripts'/p) for p in ['gate_variants.py', 'diagnose_choice_order.py']},
        protocol_sha256=sha(ROOT/'CHOICE_ORDER_AUDIT_PROTOCOL.md'),
        frozen_laya_sha256='ae9a05c819a0ab99493067336fd9f5c186dce48265e75ef027978f2597dfac71',
        no_outcomes_in_model_input=True, no_current_cohort_policy_change=True)
    path = folder / 'definition.json'
    path.write_text(json.dumps(definition, indent=2) + '\n', encoding='utf-8', newline='\n')
    (folder/'freeze_receipt.json').write_text(json.dumps(dict(definition_sha256=sha(path), immutable=True),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(inputs=len(records), decisions=len(records)*2, fixture_sha256=sha(fixture), definition_sha256=sha(path))))


if __name__ == '__main__':
    main()
