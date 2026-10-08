"""First/middle/last decisions from each existing developer trajectory."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parents[1]
old=root.parent/'长动作块补充实验'
records=[]
for path in sorted((old/'raw/attempt_v1/smol/main').glob('t*_s*_h*_laya/gate.json')):
    rows=json.loads(path.read_text(encoding='utf-8'))
    for i in sorted({0,len(rows)//2,len(rows)-1}):
        row=rows[i]
        records.append(dict(id=path.parent.name+f'_g{i:03d}',kind='developer_replay',
            case=path.parent.name,tick=row['tick'],state=row['state'],
            old_probabilities=row['probabilities'],old_state_sha256=row['state_sha256']))
controls=[
 ('replan','The operator changed the task from moving the red block to moving the blue block. The queued actions still grasp the red block. The robot has not started the grasp. A new plan is needed to follow the new instruction.'),
 ('replan','The next queued command closes the gripper at an empty location. The target object was just moved far away by a person. The remaining commands target only the old empty location.'),
 ('replan','The operator says stop moving left and move right instead. Every remaining queued command moves left. The robot can request a replacement plan now.'),
 ('replan','The queued actions are for opening a drawer, but the new task is to close the drawer. All remaining commands pull the drawer open.'),
 ('replan','Only one action remains in the finite queue. The task is unfinished and there are five steps until the next check. Without a new plan, the queue will be exhausted.'),
 ('continue','A new plan was just generated for the unchanged task. The next five queued actions correctly approach the target. The target has not moved, the robot follows the plan, and plenty of actions remain. Continue until the next check.'),
 ('continue','The robot is lifting the correct object as planned. The grasp is secure. The next five commands continue this required lift, and the target and surroundings have not changed. The finite queue has 40 actions left.'),
 ('continue','The task is to move right. The queued plan moves right toward the correct target. No relevant observation or instruction has changed. There are 80 valid queued actions and the next check is in five steps.'),
 ('continue','The drawer is opening exactly as requested. The next queued actions maintain the intended pull. No slip or task change is observed, and 30 actions remain. The next check is in five steps.'),
 ('continue','The current plan was generated one step ago. The task, object position, and robot state agree with that plan. Its next five actions are still appropriate. There are 99 actions left and no reason to replace it now.')]
for i,(label,state) in enumerate(controls):
    records.append(dict(id=f'synthetic_{i:02d}',kind='synthetic_explicit_semantic_control',
                        state=state,semantic_label=label))
out=root/'fixtures';out.mkdir(exist_ok=True)
p=out/'diagnostic_replay.json'
p.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
meta=dict(records=len(records),replay=sum(r['kind']=='developer_replay' for r in records),
    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    selection='first/middle/last, deduplicated indices, old developer trajectories only',
    controls='manually constructed explicit semantic cases; not robotic success/calibration labels')
(out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta))
