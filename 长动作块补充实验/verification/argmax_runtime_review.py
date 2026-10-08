"""Independent completed tau=0 artifact audit: read-only NumPy, no model imports.

Usage: python argmax_runtime_review.py ARGMAX_SMOL_ROOT PRIMARY_SMOL_ROOT
Writes only JSON to stdout. Run only after the 90-episode campaign completes.
"""
from pathlib import Path
from datetime import datetime,timezone
import collections,hashlib,json,math,sys
import numpy as np


def audit(root,baseline):
    load=lambda p:json.loads(p.read_text())
    sha=lambda a:hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
    close=lambda a,b:math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)
    manifest=load(root/'run_manifest.json');status=load(root/'status.json');base_manifest=load(baseline/'run_manifest.json')
    assert status['phase']=='completed' and status['completed']==status['planned']==90 and status['frozen_parameters_unchanged'] is True
    assert manifest['minimum_probability']==0.0 and manifest['exploratory'] is True and manifest['phase']=='argmax'
    assert manifest['frozen_parameters_unchanged'] is True and base_manifest['frozen_parameters_unchanged'] is True
    fingerprints={manifest['parameter_sha256_before'],manifest['parameter_sha256_after'],base_manifest['parameter_sha256_before'],base_manifest['parameter_sha256_after']}
    assert len(fingerprints)==1 and len(next(iter(fingerprints)))==64
    before=load(root/'freeze_before.json');after=load(root/'freeze_after.json');assert before==after
    assert before['state_dict_sha256']==manifest['parameter_sha256_before']
    assert before['parameters_by_dtype']==manifest['parameters_by_dtype']==base_manifest['parameters_by_dtype']
    assert all(before[k] is True for k in ['all_parameters_frozen','all_parameter_grads_none','all_modules_eval'])
    assert before['training_steps']==before['optimizer_steps']==0
    assert manifest['original_checkpoint_config']==base_manifest['original_checkpoint_config']
    assert manifest['original_checkpoint_config']['chunk_size']==50
    assert manifest['laya_ready']['threshold']==0.0 and manifest['laya_ready']['frozen'] and manifest['laya_ready']['training'] is False
    assert manifest['laya_ready']['prompt_sha256']==base_manifest['laya_ready']['prompt_sha256']
    assert manifest['laya_ready']['source']==base_manifest['laya_ready']['source']
    assert manifest['laya_ready']['parameters_by_dtype']==base_manifest['laya_ready']['parameters_by_dtype']
    assert not list(root.rglob('failure.json'))
    rows=[json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines() if line.strip()]
    base_rows=[json.loads(line) for line in (baseline/'episodes.jsonl').read_text().splitlines() if line.strip()]
    key=lambda r:(r['phase'],r['task_id'],r['state_id'],r['horizon'],r['method'])
    expected={('argmax',tid,sid,h,'laya') for tid in range(10) for sid in range(3) for h in [50,100,200]}
    assert len(rows)==len({key(r) for r in rows})==90 and {key(r) for r in rows}==expected
    assert [key(r) for r in rows]==[tuple(item) for item in load(root/'schedule.json')]
    assert [(r['task_id'],r['state_id'],r['horizon']) for r in rows]==[(r['task_id'],r['state_id'],r['horizon']) for r in base_rows if r['phase']=='main' and r['method']=='laya']
    base_lookup={(r['task_id'],r['state_id'],r['horizon'],r['method']):r for r in base_rows if r['phase']=='main'}
    assert len(base_lookup)==360
    qual=load(root/'qualification.json');assert len(qual)==3 and {q['horizon'] for q in qual}=={50,100,200}
    for q in qual:
        h=q['horizon'];assert q['native_shape']==[h,7] and q['repeated_exact'] and q['finite']
        with np.load(root/f'qualification_h{h}.npz',allow_pickle=False) as data:
            assert data['actions'].shape==(h,7) and data['normalized'].shape==(1,h,7)
            assert np.isfinite(data['actions']).all() and np.isfinite(data['normalized']).all()
    checked_actions=0;chunk_count=0;gate_ids=set();all_gates=[];pair_details=[];trigger_counts=collections.Counter()
    for result in rows:
        _,tid,sid,h,method=key(result);n=result['steps']
        case=root/'argmax'/f't{tid:02d}_s{sid:02d}_h{h}_laya'
        base_case=baseline/'main'/f't{tid:02d}_s{sid:02d}_h{h}_smol70'
        counterpart=base_lookup[(tid,sid,h,'smol70')]
        assert load(case/'result.json')==result and load(base_case/'result.json')==counterpart
        calls=load(case/'calls.json');timeline=load(case/'timeline.json');gates=load(case/'gate.json')
        with np.load(case/'trace.npz',allow_pickle=False) as data:actions=data['actions'].copy();physics=data['physics'].copy()
        initial=np.load(case/'initial_state.npy',allow_pickle=False);base_initial=np.load(base_case/'initial_state.npy',allow_pickle=False)
        assert 1<=n<=230 and len(timeline)==len(physics)==n and actions.shape==(n,7)
        assert np.isfinite(actions).all() and np.isfinite(physics).all() and np.isfinite(initial).all()
        assert sha(actions)==result['executed_action_sha256']
        assert np.array_equal(initial,base_initial) and sha(initial)==result['initial_state_sha256']==counterpart['initial_state_sha256']
        with np.load(case/'chunk_000.npz',allow_pickle=False) as a,np.load(base_case/'chunk_000.npz',allow_pickle=False) as b:
            assert np.array_equal(a['actions'],b['actions']) and np.array_equal(a['normalized'],b['normalized'])
        assert result['success']==timeline[-1]['success']==any(t['success'] for t in timeline)
        assert not any(t['terminated'] or t['truncated'] for t in timeline[:-1])
        assert n==230 or timeline[-1]['terminated'] or timeline[-1]['truncated']
        assert result['pending_at_termination']==timeline[-1]['pending']
        assert result['vla_calls']==len(calls) and result['nfe']==len(calls)*10
        assert result['gate_calls']==len(gates) and collections.Counter(g['choice'] for g in gates)==result['gate_choices']
        assert close(result['prediction_seconds'],sum(c['predict_seconds'] for c in calls)) and close(result['gate_seconds'],sum(g['full_gate_seconds'] for g in gates))
        assert result['episode_wall_seconds']+1e-6>=result['prediction_seconds']+result['gate_seconds']
        assert result['real_time'] is False and result['logical_delay_steps']==1 and close(result['simulated_seconds'],n/20)
        assert result['native_generated_rows']==len(calls)*h and result['unused_generated_rows']==len(calls)*h-n
        used=collections.defaultdict(list);ticks=collections.defaultdict(list)
        for tick,item in enumerate(timeline):
            assert item['tick']==tick and 0<=item['active_call']<len(calls)
            assert 0<=item['action_row']<h and item['queue_remaining']==h-item['action_row']-1
            used[item['active_call']].append(item['action_row']);ticks[item['active_call']].append(tick)
        gate_by_tick={g['tick']:g for g in gates};assert len(gate_by_tick)==len(gates)
        assert set(gate_by_tick)=={tick for tick in range(5,n,5) if timeline[tick]['queue_remaining']+1>1}
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
                else:assert used[cid][0]==1 and c['expired_prefix_rows']==1 and c['actual_delivery_tick']==c['intended_delivery_tick']==ticks[cid][0]==c['request_tick']+1
            else:assert cid==len(calls)-1 and c['intended_delivery_tick']>=n
            trigger_counts[c['trigger']]+=1
            if cid==0:assert c['trigger']=='initial'
            elif c['trigger']=='mandatory_finite_queue':assert timeline[c['request_tick']]['queue_remaining']+1<=1 and c['request_tick'] not in gate_by_tick
            else:
                gate=gate_by_tick[c['request_tick']]
                assert (c['trigger']=='laya_replan' and gate['choice']=='replan') or (c['trigger']=='laya_uncertain_smol70_fallback' and gate['choice']=='uncertain' and gate['remaining']<=.7*h)
            chunk_count+=1
        checked_actions+=n
        for g in gates:
            assert g['request_id'] not in gate_ids;gate_ids.add(g['request_id'])
            assert g['status']=='ok' and not g['warmup'] and g['minimum_probability']==0.0 and g['choice']==g['raw_choice']
            assert g['tick']>0 and g['tick']%5==0 and g['remaining']==timeline[g['tick']]['queue_remaining']+1
            assert g['cpu_fallback_count']==0 and not g['usage']['truncated'] and g['usage']['state_tokens_dropped']==0
            assert not g['usage'].get('options') and g['token_audit']['options']['options_distinct']==3
            assert not g['token_audit']['state']['truncated'] and g['token_audit']['state']['state_tokens_dropped']==0
            probs=g['probabilities'];assert set(probs)=={'replan','continue','uncertain'}
            assert all(math.isfinite(x) and 0<=x<=1 for x in probs.values()) and abs(sum(probs.values())-1)<=.002
            assert probs[g['raw_choice']]>=max(probs.values())-.0002
            assert g['state_sha256']==hashlib.sha256(g['state'].encode()).hexdigest()
            request=load(root/'laya_ipc/requests'/f"{g['request_id']}.json");response=load(root/'laya_ipc/responses'/f"{g['request_id']}.json")
            assert request['request_id']==g['request_id'] and request['state']==g['state'] and all(g[k]==v for k,v in response.items())
            assert g['prompt_sha256']==manifest['laya_ready']['prompt_sha256'] and g['full_gate_seconds']+1e-6>=g['ipc_outer_seconds']>=0
            matching=[c for c in calls if c['request_tick']==g['tick']]
            if g['choice']=='replan':assert len(matching)==1 and matching[0]['trigger']=='laya_replan'
            elif g['choice']=='continue':assert not matching
            else:assert bool(matching)==(g['remaining']<=.7*h)
        all_gates.extend(gates)
        pair_details.append(dict(task_id=tid,state_id=sid,horizon=h,initial_state_equal=True,first_chunk_actions_and_normalized_equal=True,success_argmax=result['success'],success_smol70=counterpart['success']))
    assert manifest['total_vla_calls_including_qualification']==6+sum(r['vla_calls'] for r in rows)
    assert close(manifest['total_prediction_seconds_including_qualification'],sum(q['warm_seconds']+q['hot_seconds'] for q in qual)+sum(r['prediction_seconds'] for r in rows))
    warm=load(root/'laya_warmup.json');assert warm['warmup'] and warm['request_id']=='0000000' and warm['minimum_probability']==0.0 and warm['choice']==warm['raw_choice']
    stopped=load(root/'laya_ipc/stopped.json')
    assert stopped['status']=='stopped' and stopped['reason']=='stop_requested' and stopped['requests_completed']==manifest['total_gate_calls_including_warmup']==1+len(all_gates)
    assert len(list((root/'laya_ipc/requests').glob('*.json')))==len(list((root/'laya_ipc/responses').glob('*.json')))==1+len(all_gates)
    source_root=next(p for p in root.parents if (p/'scripts/main_experiment.py').exists())
    source_checks={rel:hashlib.sha256((source_root/rel).read_bytes()).hexdigest()==digest for rel,digest in manifest['source_sha256'].items()}
    assert all(source_checks.values())
    installed_checks={p:(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h if Path(p).is_file() else None) for p,h in manifest['installed_source_sha256'].items()}
    assert all(value is not False for value in installed_checks.values())
    assert manifest['installed_source_sha256']==base_manifest['installed_source_sha256']
    groups=[]
    for h in [50,100,200]:
        cell=[r for r in rows if r['horizon']==h];assert len(cell)==30
        comparisons=[]
        for method in ['smol70','laya','naive_k5','vlash_style_k5']:
            other=[base_lookup[(r['task_id'],r['state_id'],h,method)] for r in cell]
            comparisons.append(dict(baseline=method,paired=30,argmax_only=sum(a['success'] and not b['success'] for a,b in zip(cell,other)),baseline_only=sum(b['success'] and not a['success'] for a,b in zip(cell,other)),
                                    service_cost_ratio=sum(r['prediction_seconds']+r['gate_seconds'] for r in cell)/sum(r['prediction_seconds']+r['gate_seconds'] for r in other)))
        groups.append(dict(horizon=h,episodes=30,successes=sum(r['success'] for r in cell),steps=sum(r['steps'] for r in cell),vla_calls=sum(r['vla_calls'] for r in cell),prediction_seconds=sum(r['prediction_seconds'] for r in cell),
                           gate_calls=sum(r['gate_calls'] for r in cell),gate_seconds=sum(r['gate_seconds'] for r in cell),episode_wall_seconds=sum(r['episode_wall_seconds'] for r in cell),comparisons=comparisons))
    return dict(status='PASS_COMPLETE_ARGMAX_ONLY',audited_utc=datetime.now(timezone.utc).isoformat(),root=str(root),baseline=str(baseline),cpu_only_read_only=True,
                validated_episodes=len(rows),validated_action_rows=checked_actions,validated_chunks=chunk_count,matched_initial_states_and_first_chunks=len(pair_details),pair_details=pair_details,
                frozen_hash=manifest['parameter_sha256_before'],same_frozen_parameters_as_primary=True,parameters_by_dtype=manifest['parameters_by_dtype'],same_laya_source_and_prompt=True,
                total_vla_calls_including_qualification=manifest['total_vla_calls_including_qualification'],total_gate_calls_including_warmup=stopped['requests_completed'],
                groups=groups,gate_effective_choices=dict(collections.Counter(g['choice'] for g in all_gates)),gate_raw_choices=dict(collections.Counter(g['raw_choice'] for g in all_gates)),trigger_counts=dict(trigger_counts),
                gate_max_probability_range=[min(max(g['probabilities'].values()) for g in all_gates),max(max(g['probabilities'].values()) for g in all_gates)],source_checks=source_checks,installed_source_checks=installed_checks,
                checks=['90 unique planned cases and original relative ordering','same primary frozen SmolVLA state hash/dtypes and same Laya source/prompt','90 initial physics states and initial complete chunks exactly match primary Smol70',
                        'every native chunk finite and executed action row reconstructed; no index/queue overflow','raw argmax equals effective choice at tau0; semantic uncertain uses original fallback','all request triggers/gate opportunities/mandatory finite queue guard reconcile',
                        'no post-terminal action or hidden early truncation','every gate original IPC/state/probability/packing record matches','NFE/calls/unused rows/per-case and campaign costs reconcile','no failures; service stopped; source hashes match'],
                limitations=['Exploratory post-pilot ablation on the same exposed development states; not independent confirmation.',
                             'Cross-run wall costs can reflect shared-GPU interference; not real-time robot or GPU-active speedup.',
                             'Frozen identity verified from stored runtime before/after fingerprints, not an independent model reload.',
                             'Repeated qualification b was not archived; equal-repeat assertion cannot be independently recomputed here.',
                             'When auditing downloaded local artifacts, unavailable remote installed-source paths are null; their stored identities must still match the primary manifest.'])


if __name__=='__main__':
    assert len(sys.argv)==3,'Expected argmax smol root and original smol root'
    print(json.dumps(audit(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
