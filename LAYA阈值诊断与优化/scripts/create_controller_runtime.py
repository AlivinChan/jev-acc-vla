"""Derive an isolated controller with a small, asserted diff from the read-only baseline."""
from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parents[1]
original=root.parent/'长动作块补充实验/scripts/main_experiment.py'
text=original.read_text(encoding='utf-8')
before=hashlib.sha256(original.read_bytes()).hexdigest()
# Keep the already verified predictor, physical stepping, finite queue and d=1 semantics.
text=text.split('\ndef main():',1)[0]+'\n'
changes=[
 ("ROOT = BASE / 'experiments/long_chunk_laya'", "ROOT = BASE / 'experiments/laya_gate_optimization'"),
 ("str(ROOT / 'source_snapshot')", "str(BASE / 'experiments/long_chunk_laya/source_snapshot')"),
 ('def state_text(instruction, obs, env, anchor, queue, age, horizon):',
  'def state_text(instruction, obs, env, anchor, queue, age, horizon, gate_interval=5):'),
 ('np.round(np.asarray(a), 3).tolist()', 'np.round(np.asarray(a, dtype=np.float64), 3).tolist()'),
 ("'One action = one control step. Next gate check in at most 5 steps; '",
  "f'One action = one control step. Next gate check in at most {gate_interval} steps; '"),
 ('def run_episode(predictor, gate, out, phase, tid, sid, horizon, method):',
  'def run_episode(predictor, gate, out, phase, tid, sid, horizon, method, gate_interval=5, minimum_age=0):'),
 ("elif method == 'laya' and tick % 5 == 0:",
  "elif method == 'laya' and tick % gate_interval == 0 and tick-origin >= minimum_age:"),
 ('queue, tick-origin, horizon)', 'queue, tick-origin, horizon, gate_interval)'),
 ('result = dict(phase=phase, task_id=tid, state_id=sid, horizon=horizon, method=method,',
  'result = dict(phase=phase, task_id=tid, state_id=sid, horizon=horizon, method=method,\n                      gate_interval=gate_interval, minimum_age=minimum_age,')]
for old,new in changes:
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
ast.parse(text)
target=root/'scripts/controller_runtime.py'
target.write_text(text,encoding='utf-8',newline='\n')
meta=dict(original_sha256=before,generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),changes=changes,
          unused_original_main_removed=True,baseline_readonly=True)
(root/'checks/controller_diff.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(dict(changes=len(changes),original_sha256=before)))
