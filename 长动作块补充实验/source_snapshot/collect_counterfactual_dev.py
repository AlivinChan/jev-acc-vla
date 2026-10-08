"""Paired development branch runner; offline injected delay, never a realtime benchmark."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,pickle,random,time,shutil
import numpy as np
import torch
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.configs import LiberoEnv as EnvConfig
from lerobot.envs import preprocess_observation,make_env_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.configs import RTCAttentionSchedule
from lerobot.utils.random_utils import set_seed

ROOT=Path('/mnt/4t/jev_vla_libero')
class InvalidCase(Exception):pass

def batchify(value):
    if isinstance(value,dict):return {k:batchify(v) for k,v in value.items()}
    return np.expand_dims(np.asarray(value),0)

def snapshot(env):
    base=env._env.env;controller=base.robots[0].controller
    fields={k:np.asarray(getattr(controller,k)).copy().reshape(-1) for k in ['goal_pos','goal_ori','ee_pos','ee_ori_mat'] if getattr(controller,k,None) is not None}
    objects={k:base.sim.data.body_xpos[v].copy().tolist() for k,v in sorted(base.obj_body_id.items())}
    rng=hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().tolist(),torch.cuda.get_rng_state().tolist()))).hexdigest()
    return {'physics':base.sim.get_state().flatten().copy(),'controller':fields,'objects':objects,'rng_sha256':rng}

def compare_snapshots(a,b):
    assert a['physics'].shape==b['physics'].shape and set(a['controller'])==set(b['controller'])
    diffs=[float(np.max(np.abs(a['physics']-b['physics'])))]
    diffs += [float(np.max(np.abs(a['controller'][k]-b['controller'][k]))) for k in a['controller']]
    assert a['objects'].keys()==b['objects'].keys()
    diffs += [float(np.max(np.abs(np.array(a['objects'][k])-np.array(b['objects'][k])))) for k in a['objects']]
    return {'max_abs_state_difference':max(diffs),'rng_equal':a['rng_sha256']==b['rng_sha256']}

class Engine:
    def __init__(self,config):
        self.config=config;self.suite=benchmark.get_benchmark_dict()[config['suite']]()
        sources=json.loads((ROOT/'configs/baseline_sources.json').read_text());self.model_path=ROOT/'models'/('smolvla_libero_'+sources['checkpoint_revision'])
        assert torch.cuda.mem_get_info()[0]>6*2**30
        cfg=SmolVLAConfig.from_pretrained(str(self.model_path));cfg.device='cuda'
        cfg.rtc_config=RTCConfig(enabled=True,prefix_attention_schedule=RTCAttentionSchedule.ZEROS,execution_horizon=config['delay_steps'],max_guidance_weight=10.0)
        self.policy=SmolVLAPolicy.from_pretrained(str(self.model_path),config=cfg,strict=True,local_files_only=True).eval().to('cuda')
        for p in self.policy.parameters():p.requires_grad_(False)
        self.pre,self.post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(self.model_path),preprocessor_overrides={'device_processor':{'device':'cuda'}})
        self.env_pre,_=make_env_pre_post_processors(env_cfg=EnvConfig(task=config['suite']),policy_cfg=cfg)
        self.cfg=cfg
    def restore(self,tid,sid,actions):
        set_seed(1000+sid)
        env=LiberoEnv(task_suite=self.suite,task_id=tid,task_suite_name=self.config['suite'],obs_type='pixels_agent_pos',observation_width=360,observation_height=360,episode_index=sid,control_mode='relative',hard_reset=True)
        try:
            obs,_=env.reset(seed=1000+sid)
            for action in actions:
                obs,_,term,trunc,_=env.step(action)
                if term or trunc:raise InvalidCase('task completed during branch-origin prefix')
            return env,obs
        except BaseException:env.close();raise
    def predict(self,obs,instruction,seed,previous=None):
        torch.cuda.synchronize();start=time.perf_counter()
        batch=preprocess_observation(batchify(obs));batch['task']=[instruction];batch=self.pre(self.env_pre(batch))
        generator=torch.Generator(device='cpu').manual_seed(seed)
        noise=torch.randn((1,50,self.cfg.max_action_dim),generator=generator,dtype=torch.float32).to('cuda')
        self.policy.reset();kwargs={}
        if previous is not None:kwargs={'prev_chunk_left_over':previous,'inference_delay':self.config['delay_steps'],'execution_horizon':self.config['delay_steps']}
        # RTC uses torch.enable_grad internally. inference_mode would disable its needed gradient path.
        with torch.no_grad():normalized=self.policy.predict_action_chunk(batch,noise=noise,**kwargs);actions=self.post(normalized.clone())
        torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
        arr=actions.detach().cpu().numpy()[0]
        assert arr.shape==(50,7) and np.isfinite(arr).all()
        return normalized.detach().clone(),arr,ms
    def refresh(self,env):
        base=env._env.env;base.sim.forward();base._update_observables(force=True)
        return env._format_raw_obs(base._get_observations(force_update=True))
    def perturb(self,env,obs,event,target,unrelated):
        if event=='normal_execution':return obs
        name=target if event=='target_shift' else unrelated;base=env._env.env
        obj=base.objects_dict[name]
        if base._check_grasp(base.robots[0].gripper,obj.contact_geoms):raise InvalidCase('currently grasped object cannot be teleported')
        if len(obj.joints)!=1:raise InvalidCase('object is not represented by one free joint')
        joint=obj.joints[0];qpos=base.sim.data.get_joint_qpos(joint).copy()
        if qpos.shape!=(7,):raise InvalidCase('object joint is not free')
        qpos[:3]+=np.array(self.config['shift_xyz_m']);base.sim.data.set_joint_qpos(joint,qpos);base.sim.data.set_joint_qvel(joint,np.zeros(6))
        obs=self.refresh(env)
        own={base.sim.model.geom_name2id(g) for g in obj.contact_geoms}
        others={base.sim.model.geom_name2id(g) for n,o in base.objects_dict.items() if n!=name for g in o.contact_geoms}
        for contact in base.sim.data.contact[:base.sim.data.ncon]:
            if contact.dist<-.002 and ((contact.geom1 in own and contact.geom2 in others) or (contact.geom2 in own and contact.geom1 in others)):
                raise InvalidCase('injected object deeply intersects another movable object')
        return obs

def state_text(instruction,anchor,expected,current,robot):
    names=sorted(current)
    rows=[]
    for name in names:
        xyz=[np.round(np.array(v[name]),3).tolist() for v in [anchor,expected,current]]
        display=name.replace('_',' ')
        rows.append(display+': '+json.dumps(xyz,separators=(',',':')))
    return ('Task: '+instruction+'\nMetres. Each row lists [when planned, expected now under committed actions, observed now]. Expected positions are privileged simulation estimates.\n'+'\n'.join(rows)+'\nEEF xyz: '+json.dumps(np.round(np.asarray(robot['eef']['pos']),3).tolist())+'; gripper: '+json.dumps(np.round(np.asarray(robot['gripper']['qpos']),3).tolist())+'\nOrdinary robot-induced progress is expected, including moving the target. Decide whether unexpected deviation requires replanning the finite remaining block.')

def noise_seed(tid,sid,anchor,tick):return 20261001+tid*100000+sid*1000+anchor*10+tick

def run_group(engine,folder,tid,sid,anchor,events,preflight):
    cfg=engine.config;source=json.loads((ROOT/'manifests/baseline_verification_summary.json').read_text());source_run=Path(source['run_root'])
    p=source_run/'traces'/f'task{tid}_episode{sid:02d}'
    source_meta=json.loads(Path(str(p)+'_summary.json').read_text());source_actions=np.load(str(p)+'_rollout.npz')['action'][0]
    if len(source_actions)<=anchor+cfg['chunk_prefix_steps']:raise InvalidCase('source episode too short for anchor')
    actual_prefix=source_actions[:anchor];env,obs=engine.restore(tid,sid,actual_prefix)
    try:
        instruction=env.task_description;anchor_snap=snapshot(env)
        seed=noise_seed(tid,sid,anchor,0);old_norm,old_actions,old_ms=engine.predict(obs,instruction,seed)
        if not preflight:
            repeated_norm,repeated,repeat_ms=engine.predict(obs,instruction,seed)
            assert np.max(np.abs(old_actions-repeated))<=1e-5,'Fixed-noise chunk not deterministic'
        k=cfg['chunk_prefix_steps']
        for action in old_actions[:k]:
            obs,_,term,trunc,_=env.step(action)
            if term or trunc:raise InvalidCase('task completed while committing initial prefix')
        expected_snap=snapshot(env);prefix=np.concatenate([actual_prefix,old_actions[:k]])
        base=env._env.env;target=base.obj_of_interest[0]
        assert target in base.objects_dict
        same_category=[name for name,obj in base.objects_dict.items() if name not in base.obj_of_interest and obj.category_name==base.objects_dict[target].category_name]
        if not same_category:raise InvalidCase('no unrelated object of the same category')
        unrelated=sorted(same_category)[0]
        if not preflight:
            costs=[]
            for index in range(3):
                norm,new,cost=engine.predict(obs,instruction,noise_seed(tid,sid,anchor,1),old_norm[:,k:]);costs.append(cost)
            assert costs[-1]<=cfg['delay_ms'],'RTC measured hot cost exceeds injected delay; revise contract before labels'
            preflight.update({'normal_chunk_deterministic':True,'normal_chunk_ms':[old_ms,repeat_ms],'rtc_ms_warmup2_then_measure1':costs,'delay_ms':cfg['delay_ms'],'fixed_noise_shape':[1,50,engine.cfg.max_action_dim],'rtc_prefix_normalized_max_abs_error':float((norm[:,:cfg['delay_steps']]-old_norm[:,k:k+cfg['delay_steps']]).abs().max().item()),'passed':True})
            (folder/'preflight.json').write_text(json.dumps(preflight,indent=2))
    finally:env.close()
    records=[]
    for event in events:
        cid=f'task{tid}_state{sid}_anchor{anchor}_{event}';case_dir=folder/'cases'/cid;case_dir.mkdir(parents=True)
        streams=[];snapshots=[];row={'case_id':cid,'task_id':tid,'initial_state_id':sid,'anchor_step':anchor,'event':event,'seed':1000+sid,'prefix_steps':len(prefix),'source_trace_prefix':str(p),'target_for_generation_only':target,'unrelated_for_generation_only':unrelated,'instruction':instruction,'status':'pending','label':None}
        try:
            for branch in ['continue','replan']:
                env,obs=engine.restore(tid,sid,prefix)
                try:
                    origin=snapshot(env);match=compare_snapshots(expected_snap,origin)
                    assert match['max_abs_state_difference']<=cfg['snapshot_tolerance'] and match['rng_equal'],'Branch pre-event origin mismatch'
                    obs=engine.perturb(env,obs,event,target,unrelated);start_snap=snapshot(env)
                    if snapshots:
                        match2=compare_snapshots(snapshots[0],start_snap)
                        assert match2['max_abs_state_difference']<=cfg['snapshot_tolerance'] and match2['rng_equal'],'Paired event origins mismatch'
                        row['branch_start_match']=match2
                    snapshots.append(start_snap)
                    if branch=='continue':
                        row['state_text']=state_text(instruction,anchor_snap['objects'],expected_snap['objects'],start_snap['objects'],obs['robot_state'])
                        row['observed_objects']=start_snap['objects'];row['expected_objects']=expected_snap['objects']
                    result={'success':False,'decision_window_success':False,'vla_calls':0,'predict_ms':[],'injection_mode':cfg['latency_mode']}
                    new_actions=None
                    if branch=='replan':
                        new_norm,new_actions,cost=engine.predict(obs,instruction,noise_seed(tid,sid,anchor,1),old_norm[:,k:]);result['vla_calls']+=1;result['predict_ms'].append(cost)
                        assert cost<=cfg['delay_ms'],'RTC call exceeds injected latency contract'
                        np.savez_compressed(case_dir/'replanned_chunk.npz',normalized=new_norm.cpu().numpy(),actions=new_actions)
                    d=cfg['delay_steps'];h=cfg['decision_window_steps'];assert h<=len(old_actions)-k and d<h
                    window=old_actions[k:k+h].copy() if branch=='continue' else np.concatenate([old_actions[k:k+d],new_actions[d:h]])
                    actions_taken=[];physics=[];objects=[];shared=[];tick=len(prefix);success=False
                    for index,action in enumerate(window):
                        actions_taken.append(action.copy());obs,_,term,trunc,info=env.step(action);tick+=1;success=bool(info['is_success']);snap=snapshot(env)
                        physics.append(snap['physics']);objects.append(snap['objects'])
                        if index<d:shared.append(snap['physics'])
                        if term or trunc:break
                    result['decision_window_success']=success
                    while not success and tick<cfg['max_episode_steps']:
                        _,recovery,cost=engine.predict(obs,instruction,noise_seed(tid,sid,anchor,tick));result['vla_calls']+=1;result['predict_ms'].append(cost)
                        for action in recovery[:cfg['recovery_action_steps']]:
                            if tick>=cfg['max_episode_steps']:break
                            actions_taken.append(action.copy());obs,_,term,trunc,info=env.step(action);tick+=1;success=bool(info['is_success']);snap=snapshot(env);physics.append(snap['physics']);objects.append(snap['objects'])
                            if term or trunc:break
                        if term or trunc:break
                    result.update(success=success,total_control_steps=tick,post_event_steps=len(actions_taken),wall_elapsed_is_not_realtime=True)
                    np.savez_compressed(case_dir/(branch+'_trace.npz'),executed_actions=np.array(actions_taken),physics_states=np.array(physics),committed_physics=np.array(shared),start_physics=start_snap['physics'])
                    (case_dir/(branch+'_objects.json')).write_text(json.dumps(objects))
                    (case_dir/(branch+'_result.json')).write_text(json.dumps(result,indent=2));streams.append(result);row[branch]=result
                finally:env.close()
            a=np.load(case_dir/'continue_trace.npz');b=np.load(case_dir/'replan_trace.npz')
            assert a['committed_physics'].shape==b['committed_physics'].shape
            diff=float(np.abs(a['committed_physics']-b['committed_physics']).max()) if a['committed_physics'].size else 0
            assert diff<=cfg['snapshot_tolerance'],'Shared committed-prefix divergence'
            committed=min(cfg['delay_steps'],len(a['executed_actions']),len(b['executed_actions']))
            assert np.array_equal(a['executed_actions'][:committed],b['executed_actions'][:committed])
            row['shared_prefix_max_state_difference']=diff
            c,s=streams[0]['success'],streams[1]['success']
            row['label']='replan' if s and not c else 'continue' if c and not s else 'uncertain'
            row['ambiguity']='both_success' if c and s else 'both_failure' if not c and not s else None
            row['status']='valid'
            np.savez_compressed(case_dir/'original_chunk_and_prefix.npz',old_normalized=old_norm.cpu().numpy(),old_actions=old_actions,executed_prefix=prefix)
        except InvalidCase as error:row.update(status='invalid',invalid_reason=str(error))
        (case_dir/'record.json').write_text(json.dumps(row,indent=2));records.append(row)
        with (folder/'records.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
        print('__CF_PROGRESS__ '+json.dumps({'case':cid,'status':row['status'],'label':row['label'],'ambiguity':row.get('ambiguity')}),flush=True)
    return records

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--config',default='counterfactual_dev_v1.json');parser.add_argument('--state-manifest');args=parser.parse_args();assert Path(args.config).name==args.config and (args.state_manifest is None or Path(args.state_manifest).name==args.state_manifest)
    config_path=ROOT/'configs'/args.config;cfg=json.loads(config_path.read_text());assert set(cfg['task_ids'])<={0,1} and set(cfg['initial_state_ids'])<={0,3} and set(cfg['anchor_steps'])<={20,40}
    folder=ROOT/'runs'/('counterfactual_'+('smoke_' if args.smoke else cfg['version'].replace('-','_')+'_')+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));folder.mkdir()
    (folder/'code').mkdir();shutil.copy2(__file__,folder/'code'/Path(__file__).name)
    manifest={'tier':'auxiliary/dev','mode':'smoke' if args.smoke else cfg['version'],'config':cfg,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_utc':datetime.now(timezone.utc).isoformat(),'formal_comparison':False,'dataset_split':'development IDs0,3 only','baseline_reference':'manifests/baseline_verification_summary.json','argv':__import__('sys').argv}
    (folder/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    state_path=ROOT/'manifests'/(args.state_manifest or ('counterfactual_smoke_state.json' if args.smoke else 'counterfactual_dev_state.json'))
    def state(phase,records,error=None):state_path.write_text(json.dumps({'phase':phase,'run_root':str(folder),'completed_cases':len(records),'error':error},indent=2))
    records=[];preflight={};state('loading',records)
    try:
        engine=Engine(cfg);state('running',records)
        tasks=[0] if args.smoke else cfg['task_ids'];sids=[0] if args.smoke else cfg['initial_state_ids'];anchors=[20] if args.smoke else cfg['anchor_steps']
        for tid in tasks:
            for sid in sids:
                for anchor in anchors:
                    try:
                        records.extend(run_group(engine,folder,tid,sid,anchor,cfg['events'],preflight))
                    except InvalidCase as error:
                        for event in cfg['events']:
                            row={'case_id':f'task{tid}_state{sid}_anchor{anchor}_{event}','task_id':tid,'initial_state_id':sid,'anchor_step':anchor,'event':event,'status':'invalid','label':None,'invalid_reason':str(error)}
                            case_dir=folder/'cases'/row['case_id'];case_dir.mkdir(parents=True,exist_ok=True);(case_dir/'record.json').write_text(json.dumps(row,indent=2));records.append(row)
                            with (folder/'records.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
                    state('running',records)
        summary={'run_root':str(folder),'phase':'completed','planned_cases':len(tasks)*len(sids)*len(anchors)*len(cfg['events']),'cases':len(records),'valid':sum(x['status']=='valid' for x in records),'invalid':sum(x['status']=='invalid' for x in records),'label_counts':{k:sum(x['label']==k for x in records if x['status']=='valid') for k in ['replan','continue','uncertain']},'ambiguity_counts':{k:sum(x.get('ambiguity')==k for x in records) for k in ['both_success','both_failure']},'label_source':'paired outcomes under delayed RTC first intervention and shared synchronous recovery','real_time':False,'Laya_executed':False,'preflight':preflight,'main_baseline_accepted':False}
        assert len(records)==summary['planned_cases']
        (folder/'summary.json').write_text(json.dumps(summary,indent=2));state_path.write_text(json.dumps(summary,indent=2));print('__CF_COMPLETED__ '+json.dumps(summary),flush=True)
    except BaseException as error:state('failed',records,repr(error));raise
if __name__=='__main__':main()
