"""Paired initial predictions: length changes the shared prefix, not only appended actions."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main'
records=[]
for tid in range(10):
    for sid in range(3):
        plans={};initials=[];source={}
        for h in [50,100,200]:
            folder=OLD/f't{tid:02d}_s{sid:02d}_h{h}_smol70'
            with np.load(folder/'chunk_000.npz') as data:plans[h]=data['actions'].astype(np.float64)
            initials.append(np.load(folder/'initial_state.npy'))
            source[str(h)]=hashlib.sha256((folder/'chunk_000.npz').read_bytes()).hexdigest()
        assert all(np.array_equal(initials[0],x) for x in initials[1:])
        diffs={}
        for h in [100,200]:
            delta=plans[h][:50]-plans[50]
            diffs[str(h)]=dict(first50_exact=bool(np.array_equal(plans[h][:50],plans[50])),
                translation_rmse=float(np.sqrt(np.mean(delta[:,:3]**2))),rotation_rmse=float(np.sqrt(np.mean(delta[:,3:6]**2))),
                gripper_rmse=float(np.sqrt(np.mean(delta[:,-1]**2))),
                by_window={str(end):float(np.sqrt(np.mean(delta[:end,:3]**2))) for end in [1,5,15,30,50]})
        records.append(dict(task=tid,state=sid,differences_from_h50=diffs,source_sha256=source))
summary={}
for h in [100,200]:
    rows=[r['differences_from_h50'][str(h)] for r in records]
    summary[str(h)]=dict(cases=len(rows),exact_prefix_cases=sum(r['first50_exact'] for r in rows),
        median_translation_rmse=float(np.median([r['translation_rmse'] for r in rows])),
        median_rotation_rmse=float(np.median([r['rotation_rmse'] for r in rows])),
        median_gripper_rmse=float(np.median([r['gripper_rmse'] for r in rows])),
        median_translation_rmse_by_window={str(end):float(np.median([r['by_window'][str(end)] for r in rows])) for end in [1,5,15,30,50]})
result=dict(summary=summary,records=records,
    scope='Same initial simulator state and fixed seed; CPU noise-prefix equality audited separately. Controller-unit differences, not physical displacement or proof of lower policy quality.')
(ROOT/'analysis/native_initial_chunk_differences.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(summary))
