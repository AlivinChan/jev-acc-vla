"""Distinguish generated length from actually consumed action positions on developer trajectories."""
from pathlib import Path
import hashlib
import json
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    groups = defaultdict(list)
    sources = {}
    for name in ['single_skip_v1', 'single_skip_confirmation_v1']:
        folder = ROOT / 'raw' / name / 'data'
        path = folder / 'episodes.jsonl'
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['candidate_id'] != 'window3_skip1':
                continue
            case = folder / row['candidate_id'] / f"t{row['task_id']:02d}_s{row['state_id']:02d}_h{row['horizon']}_{row['method']}"
            calls = read(case / 'calls.json')
            positions = [index for call in calls for index in call['executed_rows']]
            assert len(positions) == row['steps']
            groups[row['horizon']].append(dict(task=row['task_id'], state=row['state_id'],
                executed=len(positions), generated=row['native_generated_rows'], calls=len(calls),
                beyond_training_h50=sum(i >= 50 for i in positions),
                highest_row_executed=max(positions),
                row_usage=np.bincount(positions, minlength=row['horizon']).tolist()))
    summary = {}
    for h, rows in sorted(groups.items()):
        assert len(rows) == 30 and {(r['task'], r['state']) for r in rows} == {(t, s) for t in range(10) for s in range(3)}
        used = sum(r['executed'] for r in rows)
        generated = sum(r['generated'] for r in rows)
        beyond = sum(r['beyond_training_h50'] for r in rows)
        histogram = np.sum([r['row_usage'] for r in rows], axis=0)
        assert int(histogram.sum()) == used and int(histogram[50:].sum()) == beyond
        summary[str(h)] = dict(episodes=len(rows), generated_rows=generated, executed_rows=used,
            consumed_fraction=used/generated, executed_rows_at_positions_50_or_later=beyond,
            beyond_training_h50_fraction_of_executed=beyond/used,
            highest_row_executed=max(r['highest_row_executed'] for r in rows),
            episodes_executing_beyond_training_h50=sum(r['beyond_training_h50'] > 0 for r in rows),
            row_usage=histogram.tolist())
    result = dict(candidate='window3_skip1', source_sha256=sources, by_horizon=summary,
        records={str(h): rows for h, rows in groups.items()},
        interpretation='Developer trajectories only. Zero-based action position >=50 is outside the original trained output length. Position coverage and reuse are accounting measures, not estimates of long-tail correctness or causes of task failure.')
    (ROOT / 'analysis/executed_row_coverage_development.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({h:{k:v for k,v in s.items() if k != 'row_usage'} for h,s in summary.items()}))


if __name__ == '__main__':
    main()
