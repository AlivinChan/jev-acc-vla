"""Independent read-only NumPy audit of all completed original-run artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import collections,hashlib,json,math,sys
import numpy as np

root=Path(sys.argv[1]);load=lambda p:json.loads(p.read_text())
sha=lambda a:hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
close=lambda a,b:math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)
manifest=load(root/'run_manifest.json');status=load(root/'status.json')
rows=[json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines() if line.strip()]
key=lambda r:(r['phase'],r['task_id'],r['state_id'],r['horizon'],r['method'])
methods=['smol70','laya','naive_k5','vlash_style_k5'];horizons=[50,100,200]
expected={('main',tid,sid,h,m) for tid in range(10) for sid in range(3) for h in horizons for m in methods}
expected|={('pilot',0,0,h,m) for h in horizons for m in methods}
assert len(rows)==len({key(r) for r in rows})==372 and {key(r) for r in rows}==expected
assert [key(r) for r in rows]==[tuple(x) for x in load(root/'schedule.json')]
assert status==dict(phase='completed',completed=372,planned=372,frozen_parameters_unchanged=True)
assert manifest['frozen_parameters_unchanged'] is True
assert manifest['parameter_sha256_before']==manifest['parameter_sha256_after']=='b296dfca9e977fbe06d0f4a7971dbcfa368954e168c32f9c61295107f00eba82'
assert manifest['original_checkpoint_config']['chunk_size']==50
assert not list(root.rglob('failure.json'))
qual=load(root/'qualification.json');assert len(qual)==3 and {q['horizon'] for q in qual}==set(horizons)
for q in qual:
    h=q['horizon'];assert q['native_shape']==[h,7] and q['repeated_exact'] and q['finite']
    with np.load(root/f'qualification_h{h}.npz',allow_pickle=False) as data:
        assert data['actions'].shape==(h,7) and data['normalized'].shape==(1,h,7)
        assert np.isfinite(data['actions']).all() and np.isfinite(data['normalized']).all()
cases={};init_hashes=collections.defaultdict(set);gate_ids=set();all_gates=[];chunk_count=0;checked_actions=0
for result in rows:
    phase,tid,sid,h,method=key(result)
    case=root/phase/f't{tid:02d}_s{sid:02d}_h{h}_{method}'
    assert load(case/'result.json')==result
    calls=load(case/'calls.json');timeline=load(case/'timeline.json');gates=load(case/'gate.json')
    with np.load(case/'trace.npz',allow_pickle=False) as data:actions=data['actions'].copy();physics=data['physics'].copy()
    initial=np.load(case/'initial_state.npy',allow_pickle=False)
    n=result['steps'];assert 1<=n<=230 and len(timeline)==len(physics)==n and actions.shape==(n,7)
    assert np.isfinite(actions).all() and np.isfinite(physics).all() and np.isfinite(initial).all()
    assert sha(actions)==result['executed_action_sha256'] and sha(initial)==result['initial_state_sha256']
    init_hashes[(tid,sid)].add(result['initial_state_sha256'])
    assert result['success']==timeline[-1]['success']==any(t['success'] for t in timeline)
    assert not any(t['terminated'] or t['truncated'] for t in timeline[:-1])
    assert n==230 or timeline[-1]['terminated'] or timeline[-1]['truncated']
    assert result['pending_at_termination']==timeline[-1]['pending']
    assert result['vla_calls']==len(calls) and result['nfe']==len(calls)*10
    assert result['gate_calls']==len(gates) and collections.Counter(g['choice'] for g in gates)==result['gate_choices']
    assert close(result['prediction_seconds'],sum(c['predict_seconds'] for c in calls))
    assert close(result['gate_seconds'],sum(g['full_gate_seconds'] for g in gates))
    assert result['episode_wall_seconds']+1e-6>=result['prediction_seconds']+result['gate_seconds']
    assert result['real_time'] is False and result['logical_delay_steps']==1 and close(result['simulated_seconds'],n/20)
    assert result['native_generated_rows']==len(calls)*h and result['unused_generated_rows']==len(calls)*h-n
    used=collections.defaultdict(list);ticks=collections.defaultdict(list)
    for tick,item in enumerate(timeline):
        assert item['tick']==tick and 0<=item['active_call']<len(calls)
        assert 0<=item['action_row']<h and item['queue_remaining']==h-item['action_row']-1
        used[item['active_call']].append(item['action_row']);ticks[item['active_call']].append(tick)
    for cid,c in enumerate(calls):
        assert c['call_id']==cid and c['executed_rows']==used[cid] and c['nfe']==10 and c['native_output_shape']==[h,7]
        assert c['noise_seed']==20261007+tid*100000+sid*1000+c['request_tick']
        assert c['unused_rows']==h-len(used[cid]) and math.isfinite(c['predict_seconds']) and c['predict_seconds']>=0
        with np.load(case/f'chunk_{cid:03d}.npz',allow_pickle=False) as data:
            chunk,norm=data['actions'],data['normalized']
            assert chunk.shape==(h,7) and norm.shape==(1,h,7) and np.isfinite(chunk).all() and np.isfinite(norm).all()
            if ticks[cid]:assert np.array_equal(actions[ticks[cid]],chunk[used[cid]].astype(actions.dtype))
        if used[cid]:
            assert used[cid]==list(range(used[cid][0],used[cid][0]+len(used[cid])))
            if cid==0:assert used[cid][0]==ticks[cid][0]==c['request_tick']==c['intended_delivery_tick']==0
            else:
                assert c['actual_delivery_tick']==c['intended_delivery_tick']==ticks[cid][0]
                if method=='vlash_style_k5':
                    assert used[cid][0]==0 and c['image_tick']==ticks[cid][0]-1
                    assert c['state_tick']==c['request_tick']==ticks[cid][0] and c['current_state_alignment'] is True
                else:
                    assert used[cid][0]==1 and c['expired_prefix_rows']==1 and ticks[cid][0]==c['request_tick']+1
        else:assert cid==len(calls)-1 and c['intended_delivery_tick']>=n
        chunk_count+=1
    checked_actions+=n
    request_ticks=[c['request_tick'] for c in calls]
    delivery_ticks=[c['actual_delivery_tick'] for c in calls if 'actual_delivery_tick' in c]
    if method=='smol70':assert request_ticks==list(range(0,n,3*h//10))
    if method=='naive_k5':assert request_ticks==[0]+list(range(4,n,5)) and delivery_ticks==list(range(5,n,5))
    if method=='vlash_style_k5':assert request_ticks==[0]+list(range(5,n,5)) and delivery_ticks==list(range(5,n,5))
    if method!='laya':assert not gates
    for g in gates:
        assert g['request_id'] not in gate_ids;gate_ids.add(g['request_id'])
        assert g['status']=='ok' and not g['warmup'] and g['minimum_probability']==.7
        assert g['tick']>0 and g['tick']%5==0 and g['remaining']==timeline[g['tick']]['queue_remaining']+1
        assert g['cpu_fallback_count']==0 and not g['usage']['truncated'] and g['usage']['state_tokens_dropped']==0
        assert not g['usage'].get('options') and g['token_audit']['options']['options_distinct']==3
        assert not g['token_audit']['state']['truncated'] and g['token_audit']['state']['state_tokens_dropped']==0
        probs=g['probabilities'];assert set(probs)=={'replan','continue','uncertain'}
        assert all(math.isfinite(x) and 0<=x<=1 for x in probs.values()) and abs(sum(probs.values())-1)<=.002
        assert probs[g['raw_choice']]>=max(probs.values())-.0002
        assert g['choice']==(g['raw_choice'] if probs[g['raw_choice']]>=.7 else 'uncertain')
        assert g['state_sha256']==hashlib.sha256(g['state'].encode()).hexdigest()
        request=load(root/'laya_ipc/requests'/f"{g['request_id']}.json")
        response=load(root/'laya_ipc/responses'/f"{g['request_id']}.json")
        assert request['request_id']==g['request_id'] and request['state']==g['state']
        assert all(g[k]==v for k,v in response.items())
        assert g['prompt_sha256']==manifest['laya_ready']['prompt_sha256']
        assert g['full_gate_seconds']+1e-6>=g['ipc_outer_seconds']>=0
        matching=[c for c in calls if c['request_tick']==g['tick']]
        if g['choice']=='replan':assert len(matching)==1 and matching[0]['trigger']=='laya_replan'
        elif g['choice']=='continue':assert not matching
        else:assert bool(matching)==(g['remaining']<=.7*h)
    all_gates.extend(gates)
    cases[key(result)]=dict(result=result,requests=request_ticks,deliveries=delivery_ticks)
assert len(init_hashes)==30 and all(len(hashes)==1 for hashes in init_hashes.values())
pairs=[]
for tid in range(10):
    for sid in range(3):
        for h in horizons:
            smol=cases[('main',tid,sid,h,'smol70')];laya=cases[('main',tid,sid,h,'laya')]
            sr,lr=smol['result'],laya['result']
            assert set(lr['gate_choices'])=={'uncertain'}
            assert smol['requests']==laya['requests'] and smol['deliveries']==laya['deliveries']
            assert all(sr[k]==lr[k] for k in ['success','steps','executed_action_sha256','initial_state_sha256','vla_calls'])
            pairs.append(dict(task_id=tid,state_id=sid,horizon=h,all_uncertain=True,actions_requests_deliveries_success_equal=True))
assert manifest['total_vla_calls_including_qualification']==6+sum(r['vla_calls'] for r in rows)
assert close(manifest['total_prediction_seconds_including_qualification'],sum(q['warm_seconds']+q['hot_seconds'] for q in qual)+sum(r['prediction_seconds'] for r in rows))
warm=load(root/'laya_warmup.json');assert warm['warmup'] and warm['request_id']=='0000000'
stopped=load(root/'laya_ipc/stopped.json')
assert stopped['status']=='stopped' and stopped['reason']=='stop_requested' and stopped['requests_completed']==1+len(all_gates)
assert len(list((root/'laya_ipc/requests').glob('*.json')))==len(list((root/'laya_ipc/responses').glob('*.json')))==1+len(all_gates)
source_checks={}
for relative,expected_hash in manifest['source_sha256'].items():
    source_checks[relative]=hashlib.sha256((root.parents[1]/relative).read_bytes()).hexdigest()==expected_hash
assert all(source_checks.values())
installed_checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest['installed_source_sha256'].items()}
assert all(installed_checks.values())
groups=[]
for h in horizons:
    for method in methods:
        cell=[r for r in rows if r['phase']=='main' and r['horizon']==h and r['method']==method]
        assert len(cell)==30
        groups.append(dict(horizon=h,method=method,episodes=30,successes=sum(r['success'] for r in cell),steps=sum(r['steps'] for r in cell),
                           vla_calls=sum(r['vla_calls'] for r in cell),prediction_seconds=sum(r['prediction_seconds'] for r in cell),gate_calls=sum(r['gate_calls'] for r in cell),
                           gate_seconds=sum(r['gate_seconds'] for r in cell),episode_wall_seconds=sum(r['episode_wall_seconds'] for r in cell)))
main_gates=[g for r in rows if r['phase']=='main' and r['method']=='laya' for g in load(root/'main'/f"t{r['task_id']:02d}_s{r['state_id']:02d}_h{r['horizon']}_laya"/'gate.json')]
audit=dict(status='PASS_COMPLETE_ORIGINAL_SMOL_ONLY',audited_utc=datetime.now(timezone.utc).isoformat(),root=str(root),cpu_only_read_only=True,
           expected_episodes=372,validated_episodes=len(rows),phase_counts=dict(collections.Counter(r['phase'] for r in rows)),
           validated_action_rows=checked_actions,validated_chunks=chunk_count,paired_initial_states=30,all_uncertain_matched_pairs=len(pairs),pair_details=pairs,
           frozen_hash=manifest['parameter_sha256_before'],before_after_frozen_equal=True,parameters_by_dtype=manifest['parameters_by_dtype'],
           total_vla_calls_including_qualification=manifest['total_vla_calls_including_qualification'],total_gate_calls_including_warmup=stopped['requests_completed'],
           main_groups=groups,main_gate_effective_choices=dict(collections.Counter(g['choice'] for g in main_gates)),main_gate_raw_choices=dict(collections.Counter(g['raw_choice'] for g in main_gates)),
           main_gate_max_probability_range=[min(max(g['probabilities'].values()) for g in main_gates),max(max(g['probabilities'].values()) for g in main_gates)],
           source_checks=source_checks,installed_source_checks=installed_checks,
           checks=['372 unique expected cells and exact execution schedule','30 exact paired physics initial-state hashes','all H50/100/200 chunks finite and every executed row reconstructed','success/termination/truncation/no post-terminal actions','K5 schedules, naive row1, VLASH row0 and delayed-image/current-state ticks','90 real all-uncertain LAYA/Smol70 action hashes, call/delivery ticks, success/steps match','all gate state/probability/threshold/packing/IPC records reconcile','per-episode and campaign-wide call/NFE/time accounting','original and installed source fingerprints unchanged','terminal manifest before/after parameter fingerprint equal; no failure artifacts; IPC stopped cleanly'],
           limitations=['Stored hashes verify the runtime recorded frozen fingerprint check; no independent GPU model reload or weight recomputation performed.',
                        'Second qualification output was not stored; exact repetition rests on executed runtime assertions.',
                        'Input observation tensors are not archived per request; image/state alignment rests on source review and recorded ticks.',
                        'Official 30 and exploratory argmax 90 are outside this audit and require separate verification.',
                        'Reported costs are serial service wall times under shared GPU, not pure GPU-active seconds or real-time robot speedup.'])
print(json.dumps(audit,indent=2))
