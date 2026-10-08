"""Fresh H50/K5 official VLASH reference; preserve the audited rollout and policy.

Copied from the original scripts/run_vlash_reproduction.py.
Only the output scope/default sample count and read-only instrumentation differ.
"""
import argparse, hashlib, json, pathlib, sys, time

ROOT=pathlib.Path('/mnt/4t/jev_vla_libero')
RUN_ROOT=ROOT/'experiments/long_chunk_laya'
ORIGINAL_HARNESS_SHA256='6d3dff20730a79faf5f05a5dc7fd00ffd220b98e55f976f337b9330d8a5c4883'
SOURCE=ROOT/'src/reproduction/vlash_d2da223e2251a4a5020b667e77cf8ad57e594144'
TRANSFORMERS=ROOT/'src/reproduction/transformers_dcddb970176382c0fcf4521b0c0e6fc15894dfe0'
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(TRANSFORMERS/'src'))

def frozen_snapshot(policy, torch):
    """Hash every distinct state tensor without touching RNG or tensor contents."""
    dtype_numel={};parameter_inventory=[]
    for name,p in policy.named_parameters():
        dtype=str(p.dtype);dtype_numel[dtype]=dtype_numel.get(dtype,0)+p.numel()
        parameter_inventory.append({'name':name,'shape':list(p.shape),'dtype':dtype,
                                    'requires_grad':p.requires_grad,'version':p._version,
                                    'grad_is_none':p.grad is None})
    digest=hashlib.sha256();seen={};tensor_hashes={}
    for name,tensor in policy.state_dict().items():
        identity=(tensor.data_ptr(),tuple(tensor.shape),str(tensor.dtype))
        if identity not in seen:
            raw=tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            seen[identity]=hashlib.sha256(memoryview(raw).cast('B')).hexdigest()
        tensor_hashes[name]=seen[identity]
        digest.update(json.dumps([name,list(tensor.shape),str(tensor.dtype),seen[identity]],
                                 separators=(',',':')).encode('utf-8'))
    return {'all_parameters_frozen':all(not p.requires_grad for p in policy.parameters()),
            'all_parameter_grads_none':all(p.grad is None for p in policy.parameters()),
            'all_modules_eval':all(not module.training for module in policy.modules()),
            'parameter_numel_by_dtype':dtype_numel,'parameters':parameter_inventory,
            'state_tensor_sha256':tensor_hashes,'state_dict_sha256':digest.hexdigest(),
            'training_steps':0,'optimizer_steps':0}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--suite',default='libero_spatial');parser.add_argument('--delay',type=int,default=1);parser.add_argument('--states',type=int,default=3);parser.add_argument('--out',required=True);args=parser.parse_args()
    assert 0<=args.delay<=3 and 1<=args.states<=50
    folder=pathlib.Path(args.out).resolve();assert folder.is_relative_to(RUN_ROOT.resolve()) and folder.name=='official_reference';folder.mkdir(parents=True,exist_ok=False)
    import numpy as np, torch
    from libero.libero import benchmark
    from lerobot.envs.configs import LiberoEnv
    from lerobot.utils.random_utils import set_seed
    import vlash.eval_libero as official
    from vlash.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.configs.policies import PreTrainedConfig
    from vlash.policies.factory import make_policy
    import transformers,lerobot,mujoco

    set_seed(42);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    model=ROOT/'models/vlash_pi05_libero_c85d5962a7b823f9f391f97d60227d6ca8bd7bc7'
    cfg=PreTrainedConfig.from_pretrained(str(model));assert isinstance(cfg,PI05Config);cfg.device='cuda';cfg.use_amp=False;cfg.n_action_steps=5;cfg.pretrained_path=str(model)
    assert cfg.chunk_size==50, 'Official reference must preserve native H=50'
    envcfg=LiberoEnv(task=args.suite)
    load_audit=[];original_load=torch.nn.Module.load_state_dict
    def audited_load(instance,*a,**kw):
        result=original_load(instance,*a,**kw)
        load_audit.append({'class':type(instance).__name__,'missing':result.missing_keys,'unexpected':result.unexpected_keys})
        return result
    torch.nn.Module.load_state_dict=audited_load
    try: policy=make_policy(cfg=cfg,env_cfg=envcfg)
    finally: torch.nn.Module.load_state_dict=original_load
    (folder/'load_audit.json').write_text(json.dumps(load_audit,indent=2))
    # Safetensors may omit aliases of a tensor loaded under another key. Prove alias identity, rather than waive missing weights.
    sd=policy.state_dict();missing={k for x in load_audit for k in x['missing']};aliases={};unresolved=[]
    for key in sorted(missing):
        matches=[k for k,v in sd.items() if k not in missing and v.data_ptr()==sd[key].data_ptr() and v.shape==sd[key].shape]
        if matches:aliases[key]=matches[0]
        else:unresolved.append(key)
    (folder/'missing_alias_audit.json').write_text(json.dumps({'proven_aliases':aliases,'unresolved':unresolved},indent=2))
    assert not unresolved and all(not x['unexpected'] for x in load_audit), 'Review checkpoint incompatibility before evaluation'
    policy.eval()
    for p in policy.parameters():p.requires_grad_(False)
    freeze_before=frozen_snapshot(policy,torch)
    assert freeze_before['all_parameters_frozen'] and freeze_before['all_parameter_grads_none'] and freeze_before['all_modules_eval']
    (folder/'freeze_before.json').write_text(json.dumps(freeze_before,indent=2))
    predict_records=[];predict_calls_by_batch={};original_predict=policy.predict_action_chunk
    def audited_predict(*a,**kw):
        batch=a[0] if a else kw['batch']
        batch_size=int(batch['observation.state'].shape[0])
        call_index=predict_calls_by_batch.get(batch_index,0)
        record={'call_index':len(predict_records),'rollout_batch':batch_index,
                'call_in_rollout':call_index,'control_step':call_index*cfg.n_action_steps,
                'batch_size':batch_size,'native_prediction_horizon':cfg.chunk_size,
                'returned_execution_horizon':cfg.n_action_steps,
                'all_model_parameters_frozen':True}
        predict_calls_by_batch[batch_index]=call_index+1
        # Synchronize around the original call so its wall time includes CUDA work.
        # This is timing instrumentation, not a different sampler or RNG stream.
        torch.cuda.synchronize();predict_start=time.perf_counter()
        try:
            result=original_predict(*a,**kw)
            torch.cuda.synchronize()
            record['wall_seconds']=time.perf_counter()-predict_start
            record['status']='completed';record['output_shape']=list(result.shape)
            return result
        except BaseException as exc:
            record['wall_seconds']=time.perf_counter()-predict_start
            record['status']='failed';record['error']=repr(exc)
            raise
        finally:
            predict_records.append(record)
            with (folder/'predict_calls.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
    policy.predict_action_chunk=audited_predict
    suite=benchmark.get_benchmark_dict()[args.suite](); tids=[0] if args.smoke else list(range(suite.n_tasks));count=2 if args.smoke else args.states
    schedule=official.schedule_envs(suite,tids,len(tids))
    schedule=[[(name,language,states[:count]) for name,language,states in batch] for batch in schedule]
    contract={'paper':'2512.01031v2','source_commit':'d2da223e2251a4a5020b667e77cf8ad57e594144','transformers_commit':TRANSFORMERS.name.split('_')[-1],
              'original_harness_sha256':ORIGINAL_HARNESS_SHA256,'reference_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              'checkpoint_repo':'mit-han-lab/vlash-pi05-libero-async5','checkpoint_revision':'c85d5962a7b823f9f391f97d60227d6ca8bd7bc7',
              'suite':args.suite,'delay_steps':args.delay,'prediction_horizon':cfg.chunk_size,'execution_horizon':5,'seed':42,'task_ids':tids,'initial_states_per_task':count,
              'scheduled_episodes':sum(len(s) for b in schedule for _,_,s in b),'batch_size':len(tids),'action_quant':1,'max_steps':{'libero_spatial':230,'libero_object':290,'libero_goal':310,'libero_10':530}[args.suite],
              'simulation_protocol':'official delayed image plus current proprioception; not measured physical async inference',
              'torch':torch.__version__,'transformers':transformers.__version__,'mujoco':mujoco.__version__,'python':sys.version,
              'parameter_count':sum(p.numel() for p in policy.parameters()),'load_audit':load_audit,'smoke_only':args.smoke,
              'freeze_snapshot_before':'freeze_before.json','training_steps':0,'optimizer_steps':0,
              'timing_scope':'CUDA-synchronized original policy.predict_action_chunk including input/output normalization; batched calls include already-done environment slots until the batch finishes',
              'deviations':['Python 3.12 versus README example 3.10','one GPU versus released eight-GPU launcher','video writing disabled; rollout policy/environment unchanged','fresh 3 initial states per task by default; not the full 50-state benchmark','read-only per-call timing and full model state hashes added']}
    (folder/'contract.json').write_text(json.dumps(contract,indent=2))
    original_rollout=official.rollout;records=[];batch_index=0
    def audited_rollout(*a,**kw):
        nonlocal batch_index
        env=a[0]
        before_ids=env.get_env_attr('init_state_id')
        before_resets=env.get_env_attr('reset_count')
        data=original_rollout(*a,**kw)
        trace=folder/('rollout_'+str(batch_index).zfill(3)+'.npz');np.savez_compressed(trace,**{k:v.cpu().numpy() for k,v in data.items() if k in {'action','reward','success','done'}})
        done=data['done'].cpu().numpy();success=data['success'].cpu().numpy()
        assert np.isfinite(data['action'].cpu().numpy()).all() and done.any(axis=1).all()
        for i in range(len(done)):
            first=int(np.flatnonzero(done[i])[0]); strict=bool(success[i,:first+1].any());published=bool(success[i,:min(first+2,len(success[i]))].any())
            # Wrapper resets once at construction without a seed, then uses this state and advances its id after each seeded reset.
            records.append({'batch':batch_index,'env_slot':i,'task_id':tids[i],'init_state_id':int(before_ids[i]),'reset_count_before':int(before_resets[i]),'seed':kw.get('seeds',[None]*len(done))[i],
                            'success_strict':strict,'success_official_mask':published,'first_done_index':first,'trace':str(trace),'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest()})
        batch_index+=1
        (folder/'episodes.json').write_text(json.dumps(records,indent=2))
        print(json.dumps({'completed_episodes':len(records),'strict_successes':sum(x['success_strict'] for x in records),'official_successes':sum(x['success_official_mask'] for x in records)}),flush=True)
        return data
    official.rollout=audited_rollout
    start=time.monotonic()
    with torch.no_grad():
        info=official.eval_policy(env_cfg=envcfg,policy=policy,schedule=schedule,start_seed=42,max_steps=contract['max_steps'],max_episodes_rendered=0,videos_dir=None,async_delay=args.delay,method=official.VLASHMethodConfig(),action_quant=1)
    evaluation_wall_seconds=time.monotonic()-start
    freeze_after=frozen_snapshot(policy,torch)
    (folder/'freeze_after.json').write_text(json.dumps(freeze_after,indent=2))
    assert freeze_after==freeze_before, 'Frozen official model state changed during evaluation'
    assert len(records)==contract['scheduled_episodes']
    official_rate=sum(x['success_official_mask'] for x in records)/len(records)*100
    assert abs(official_rate-info['overall']['pc_successes'])<1e-8
    (folder/'official_eval_info.json').write_text(json.dumps(info,indent=2))
    summary={'episodes':len(records),'successes':sum(x['success_strict'] for x in records),'pc_success':sum(x['success_strict'] for x in records)/len(records)*100,'official_pc_success':official_rate,
             'mask_disagreements':sum(x['success_strict']!=x['success_official_mask'] for x in records),'wall_seconds':evaluation_wall_seconds,'max_gpu_memory_allocated':torch.cuda.max_memory_allocated(),
             'prediction_horizon':cfg.chunk_size,'execution_horizon':cfg.n_action_steps,
             'predict_call_attempts':len(predict_records),'predict_calls':sum(x['status']=='completed' for x in predict_records),
             'predict_batch_elements':sum(x['batch_size'] for x in predict_records),
             'predict_wall_seconds':sum(x['wall_seconds'] for x in predict_records),
             'predict_timing_scope':contract['timing_scope'],
             'freeze_verified_unchanged':True,'state_dict_sha256':freeze_before['state_dict_sha256'],
             'parameter_numel_by_dtype':freeze_before['parameter_numel_by_dtype'],'training_steps':0,'optimizer_steps':0,
             'paper_reference_percent':{'libero_spatial':98.8,'libero_object':99.2,'libero_goal':96.7,'libero_10':94.4}[args.suite] if args.delay==1 else None,
             'matched_checkpoint_training_provenance':'author-published checkpoint; repository does not identify exact paper table checkpoint',
             'full_initial_state_evaluation':not args.smoke and args.states==50,'paper_table_match_verified':False}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
