"""Describe only existing commands used before the next check and declared delivery."""
import hashlib,json,re
import numpy as np
from gate_variants import compact_state


def prepare_window_state(state,queue,compact=False):
    horizon,age,remaining=map(int,re.search(r'Native plan length (\d+); age (\d+); remaining (\d+) actions',state).groups())
    interval=int(re.search(r'Next gate check in at most (\d+) steps',state).group(1))
    delay=int(re.search(r'replacement is used after (\d+) additional logical step',state).group(1))
    assert 1<=delay<=5
    queue=np.asarray(queue,dtype=np.float64)
    assert queue.shape==(remaining,7) and np.isfinite(queue).all()
    length=min(remaining,interval+delay);window=queue[:length]
    assert length>=1
    offsets=sorted({0,min(4,length-1),(length-1)//2,max(0,length-2),length-1})
    lines=[line for line in compact_state(state).splitlines() if not line.startswith('Queued [offset,')]
    round3=lambda x:np.round(np.asarray(x,dtype=np.float64),3).tolist()
    if compact:
        offsets=sorted({0,(length-1)//2,length-1})
        lines.append('Next-check command window only, including replacement delay.')
        lines.append('Queued [offset,dx,dy,dz,gripper], controller units; rotations omitted: '+
            json.dumps([[i,*round3(window[i,:3]),round(float(window[i,-1]),3)] for i in offsets],separators=(',',':')))
        return '\n'.join(lines)
    lines.append(f'Decision window: next {length} existing commands, including the possible {delay}-step replacement delay. No later queue commands are shown.')
    lines.append('Window [offset,dx,dy,dz,rotation3,gripper], controller units not metres: '+
                 json.dumps([[i,*round3(window[i])] for i in offsets],separators=(',',':')))
    lines.append('Window mean command [dx,dy,dz,rotation3,gripper]: '+json.dumps(round3(window.mean(axis=0)),separators=(',',':')))
    jump=float(np.linalg.norm(np.diff(window[:,:3],axis=0),axis=1).max()) if length>1 else 0.
    switches=int(np.sum(np.sign(window[1:,-1])!=np.sign(window[:-1,-1])))
    lines.append(f'Maximum adjacent translation-command change norm: {round(jump,3)} controller units; gripper-command sign switches: {switches}.')
    return '\n'.join(lines)


def from_counterfactual(record,source_root,compact=False):
    name=f'chunk_{record["active_call_id"]:03d}.npz';path=source_root/record['case_name']/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['source_files'][name]
    with np.load(path,allow_pickle=False) as archive:queue=archive['actions'][record['action_row']:].copy()
    text=record['state_text']
    line=next(line for line in text.splitlines() if line.startswith('Remaining planned commands at offsets '))
    offsets=json.loads(line.split('offsets ',1)[1].split(' [dx',1)[0])
    commands=json.loads(line.split('metres: ',1)[1])
    assert np.array_equal(np.round(queue[np.array(offsets)].astype(np.float64),3),commands)
    return prepare_window_state(text,queue,compact=compact)
