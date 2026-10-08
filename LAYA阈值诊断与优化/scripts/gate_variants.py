"""Predeclared diagnosis inputs; no observed future or outcome enters a gate."""
import copy
import json
import re

FULL_INSTRUCTION=(
    'Should the robot request a new action chunk now, or continue the finite '
    'queued actions until its next scheduled check? Use the task, observed '
    'robot and object states, plan age, remaining queue, sampled future '
    'commands, next-check interval, and replacement delay stated in the input. '
    'Object coordinates are privileged simulator observations, not visual '
    'estimates. Movement since planning can be ordinary task progress. '
    'The input contains no future outcome or predicted future state. '
    'Choose uncertain if these observations do not justify either decision.')
CRITERIA={
    'replan':'Current task-relevant evidence indicates the queued actions need replacement; request a new plan now.',
    'continue':'The queued actions remain suitable for the task until the next check and stated replacement delay; continue for this interval.',
    'uncertain':'There is insufficient evidence to choose reliably between replanning and continuing.'}
SHORT_INSTRUCTION=('Choose whether to request a new robot action plan now or execute the queued plan until the next check. '
                   'Use current observations and queued commands; ordinary motion can be task progress. '
                   'The input contains no future outcome. Choose the better supported option.')


def clean_numbers(state):
    # The old float32 round(...,3).tolist() exposed binary tails. Preserve its intended precision.
    return re.sub(r'(?<![\w.])-?\d+\.\d+(?:[eE][+-]?\d+)?',
                  lambda m: str(round(float(m.group()),3)),state)


def compact_state(state):
    state=clean_numbers(state)
    lines=state.splitlines()
    if not any(l.startswith('Native plan length ') for l in lines):
        return state  # Synthetic semantic controls have no simulator schema.
    result=[lines[0],lines[1],lines[2],
            'Observed object xyz metres; delta since planning. These are simulator measurements.']
    for line in lines:
        if ': [[' in line and not line.startswith('Remaining '):
            name,value=line.split(': ',1)
            old,current=json.loads(value)
            delta=[round(b-a,3) for a,b in zip(old,current)]
            result.append(f'{name}: '+json.dumps(current,separators=(',',':'))+
                          (' delta=0' if all(x==0 for x in delta) else ' delta='+json.dumps(delta,separators=(',',':'))))
        elif line.startswith('Current EEF xyz:'):
            result.append(line)
        elif line.startswith('Remaining planned commands at offsets '):
            offsets_text= line.split('offsets ',1)[1].split(' [dx',1)[0]
            offsets=json.loads(offsets_text)
            commands=json.loads(line.split('metres: ',1)[1])
            chosen=sorted(set([0,min(2,len(commands)-1),len(commands)-1]))
            result.append('Queued [offset,dx,dy,dz,gripper], controller units; rotations omitted: '+
                json.dumps([[offsets[i],*[round(float(v),3) for v in commands[i][:3]],
                             round(float(commands[i][-1]),3)] for i in chosen],separators=(',',':')))
    return '\n'.join(result)


VARIANTS={
    'original3':dict(format='original',schema='full3',head=192,max_len=1536),
    'numbers3':dict(format='numbers',schema='full3',head=192,max_len=1536),
    'fullhead3':dict(format='numbers',schema='full3',head=256,max_len=1536),
    'binary_full':dict(format='numbers',schema='full2',head=256,max_len=1536),
    'binary_short':dict(format='numbers',schema='short2',head=192,max_len=1536),
    'binary_compact':dict(format='compact',schema='short2',head=192,max_len=768),
    'binary_window':dict(format='window',schema='short2',head=192,max_len=1024),
    'binary_window3':dict(format='window3',schema='short2',head=192,max_len=768),
}


def questions_for(schema):
    instruction=FULL_INSTRUCTION
    criteria=copy.deepcopy(CRITERIA)
    if schema.endswith('2'):
        criteria.pop('uncertain')
        instruction=instruction.rsplit('Choose uncertain',1)[0].strip()+' Choose the better supported option.'
    if schema=='short2':
        instruction=SHORT_INSTRUCTION
        criteria={'replan':'Replace queued actions with a new plan now.',
                  'continue':'Keep executing the queued plan until the next check.'}
    return {'replanning':dict(type='choice',instructions=instruction,criteria=criteria)}


def variant_input(name,state):
    spec=VARIANTS[name]
    if spec['format']=='window':
        assert 'Decision window: next ' in state and 'No later queue commands are shown.' in state
    if spec['format']=='window3':assert 'Next-check command window only, including replacement delay.' in state
    transformed={'original':lambda s:s,'numbers':clean_numbers,'compact':compact_state,'window':lambda s:s,'window3':lambda s:s}[spec['format']](state)
    return transformed,questions_for(spec['schema']),spec
