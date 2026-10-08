"""Read-only completed pilot artifact audit; imports NumPy, no torch/model/GPU."""
from pathlib import Path
from datetime import datetime, timezone
import collections,hashlib,json,sys
import numpy as np

root=Path(sys.argv[1])
load=lambda p:json.loads(p.read_text())
manifest=load(root/'run_manifest.json')
rows=[];qual=[];checks=[];limitations=[];all_gates=[]
for item in load(root/'qualification.json'):
    h=item['horizon']
    with np.load(root/f'qualification_h{h}.npz') as data:
        assert data['actions'].shape==(h,7)
        assert data['normalized'].shape==(1,h,7)
        assert np.isfinite(data['actions']).all() and np.isfinite(data['normalized']).all()
        qual.append(dict(horizon=h,actions_shape=list(data['actions'].shape),normalized_shape=list(data['normalized'].shape),finite=True,
                         repeated_exact_runtime_assertion=item['repeated_exact'],archive_sha256=hashlib.sha256((root/f'qualification_h{h}.npz').read_bytes()).hexdigest(),
                         first_action=data['actions'][0].tolist(),last_action=data['actions'][-1].tolist()))
assert {x['horizon'] for x in qual}=={50,100,200}
checks.append('qualification archives have genuine Hx7/1xHx7 finite arrays for H50/100/200')
limitations.append('Repeated prediction b is not stored: exact repetition is supported by the executed runtime assertion and qualification metadata, not independently recalculated without GPU.')
cases={}
for case in sorted((root/'pilot').iterdir()):
    if not case.is_dir():continue
    result=load(case/'result.json');calls=load(case/'calls.json');timeline=load(case/'timeline.json');gates=load(case/'gate.json')
    with np.load(case/'trace.npz') as data:actions=data['actions'].copy();physics=data['physics'].copy()
    h=result['horizon'];method=result['method'];n=result['steps']
    assert len(actions)==len(physics)==len(timeline)==n and 1<=n<=230
    assert np.isfinite(actions).all() and np.isfinite(physics).all()
    assert result['success']==timeline[-1]['success']
    assert not any(t['terminated'] or t['truncated'] for t in timeline[:-1])
    assert n==230 or timeline[-1]['terminated'] or timeline[-1]['truncated']
    assert result['executed_action_sha256']==hashlib.sha256(actions.tobytes()).hexdigest()
    initial=np.load(case/'initial_state.npy')
    assert result['initial_state_sha256']==hashlib.sha256(initial.tobytes()).hexdigest()
    assert result['vla_calls']==len(calls) and result['nfe']==10*len(calls)
    assert result['native_generated_rows']==len(calls)*h
    assert result['unused_generated_rows']==sum(c['unused_rows'] for c in calls)
    assert result['gate_calls']==len(gates)
    for g in gates:
        assert g['status']=='ok' and g['cpu_fallback_count']==0 and not g['usage']['truncated'] and g['usage']['state_tokens_dropped']==0
        assert g['token_audit']['options']['options_distinct']==3
        probs=g['probabilities'];assert set(probs)=={'replan','continue','uncertain'}
        assert all(np.isfinite(x) and 0<=x<=1 for x in probs.values()) and abs(sum(probs.values())-1)<=.002
        assert g['choice']==(g['raw_choice'] if probs[g['raw_choice']]>=.7 else 'uncertain')
        assert g['state_sha256']==hashlib.sha256(g['state'].encode('utf-8')).hexdigest()
    all_gates.extend(gates)
    assert abs(result['prediction_seconds']-sum(c['predict_seconds'] for c in calls))<1e-8
    assert abs(result['gate_seconds']-sum(g['full_gate_seconds'] for g in gates))<1e-8
    for c in calls:
        cid=c['call_id']
        with np.load(case/f'chunk_{cid:03d}.npz') as chunk:
            assert chunk['actions'].shape==(h,7) and chunk['normalized'].shape==(1,h,7)
            assert np.isfinite(chunk['actions']).all() and np.isfinite(chunk['normalized']).all()
            used=[t for t in timeline if t['active_call']==cid]
            assert c['executed_rows']==[t['action_row'] for t in used]
            assert c['unused_rows']==h-len(used) and c['nfe']==10 and c['native_output_shape']==[h,7]
            for t in used:
                assert np.array_equal(actions[t['tick']],chunk['actions'][t['action_row']].astype(np.float64))
        if cid:
            if method=='vlash_style_k5':
                assert c['image_tick']==c['request_tick']-1
                assert c['state_tick']==c['request_tick']==c['actual_delivery_tick']
                assert c['current_state_alignment'] is True and c['executed_rows'][0]==0
            elif used:
                assert c['actual_delivery_tick']==c['request_tick']+1
                assert c['expired_prefix_rows']==1 and c['executed_rows'][0]==1
    request_ticks=[c['request_tick'] for c in calls]
    delivery_ticks=[c['actual_delivery_tick'] for c in calls if 'actual_delivery_tick' in c]
    if method=='smol70':assert request_ticks==list(range(0,n,3*h//10))
    elif method=='naive_k5':
        assert request_ticks==[0]+list(range(4,n,5))
        assert delivery_ticks==list(range(5,n,5))
    elif method=='vlash_style_k5':
        assert request_ticks==[0]+list(range(5,n,5))
        assert delivery_ticks==list(range(5,n,5))
    else:
        assert all(g['tick']%5==0 and g['choice'] in {'continue','replan','uncertain'} for g in gates)
    row=dict(horizon=h,method=method,success=result['success'],steps=n,vla_calls=len(calls),gate_calls=len(gates),gate_choices=result['gate_choices'],
             last_step=timeline[-1],request_ticks=request_ticks,delivery_ticks=delivery_ticks,
             initial_state_sha256=result['initial_state_sha256'],all_native_chunks_finite=True,trace_reconstructs_from_recorded_chunks=True,
             unused_generated_rows=result['unused_generated_rows'],pending_at_termination=result['pending_at_termination'],result_sha256=hashlib.sha256((case/'result.json').read_bytes()).hexdigest())
    rows.append(row);cases[(h,method)]=(row,actions,calls)
assert len(rows)==12 and len({r['initial_state_sha256'] for r in rows})==1
checks.extend(['all 12 completed pilot cases present, paired exact initial state hash','every stored chunk shape/finite and executed trace reconstructed from source chunk+row','success/termination/truncation recorded honestly; no action after termination','VLA calls/NFE/gate count/cost/unused rows reconcile','K5 delayed deliveries, naive row1, VLASH row0/image_tick/state_tick match contract'])
pairs=[]
for h in [50,100,200]:
    laya,la,lc=cases[(h,'laya')];smol,sa,sc=cases[(h,'smol70')]
    all_uncertain=set(laya['gate_choices'])<= {'uncertain'}
    if all_uncertain:
        assert np.array_equal(la,sa)
        assert laya['request_ticks']==smol['request_ticks'] and laya['delivery_ticks']==smol['delivery_ticks']
        assert laya['success']==smol['success']
    pairs.append(dict(horizon=h,laya_all_uncertain=all_uncertain,actions_equal=bool(np.array_equal(la,sa)),request_ticks_equal=laya['request_ticks']==smol['request_ticks'],delivery_ticks_equal=laya['delivery_ticks']==smol['delivery_ticks']))
checks.append('all-uncertain real LAYA/Smol70 pairs match actions and request/delivery ticks')
source_matches={}
for rel,digest in manifest['source_sha256'].items():
    source_matches[rel]=hashlib.sha256((root.parents[1]/rel).read_bytes()).hexdigest()==digest
assert all(source_matches.values())
checks.append('current scripts/source snapshots equal launch manifest source hashes')
assert manifest['original_checkpoint_config']['chunk_size']==50
limitations.append('No image/proprioception tensors are stored per request; real input content alignment is supported by reviewed scheduler code plus recorded ticks, not reconstructed from images.')
limitations.append('Full before/after frozen parameter verification remains pending until the complete Smol campaign terminates successfully.')
audit=dict(status='PASS_PILOT_ONLY',audited_utc=datetime.now(timezone.utc).isoformat(),root=str(root),cpu_only_read_only=True,qualification=qual,pilot_cases=rows,all_uncertain_pairs=pairs,
           checks=checks,limitations=limitations,source_manifest_matches=source_matches,parameter_sha256_before=manifest['parameter_sha256_before'],
           parameters_by_dtype=manifest['parameters_by_dtype'],original_training_chunk_size=manifest['original_checkpoint_config']['chunk_size'],
           laya_gate_audit=dict(total=len(all_gates),raw_choices=dict(collections.Counter(g['raw_choice'] for g in all_gates)),effective_choices=dict(collections.Counter(g['choice'] for g in all_gates)),
                                maximum_probability_range=[min(max(g['probabilities'].values()) for g in all_gates),max(max(g['probabilities'].values()) for g in all_gates)],
                                input_tokens_range=[min(g['usage']['input_tokens'] for g in all_gates),max(g['usage']['input_tokens'] for g in all_gates)],no_truncation_or_cpu_fallback=True),
           runtime_status_snapshot=load(root/'status.json'))
print(json.dumps(audit,indent=2))
