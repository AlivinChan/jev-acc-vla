"""One skipped Smol70 call versus an exact baseline suffix, with the VLA frozen."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,signal,sys,time,traceback
import controller_runtime as runtime
from collect_counterfactual_dev import snapshot,compare_snapshots
from run_closed_loop import freeze
from supervise import allowed
BASE=runtime.BASE;ROOT=runtime.ROOT;np=runtime.np;torch=runtime.torch;atomic=runtime.atomic
OLD=BASE/'experiments/long_chunk_laya/attempt_v1/smol/main'


def branch(predictor,item,source,folder,choice):
    tid,sid,h,t=item['task'],item['state'],item['horizon'],item['tick']
    trace=np.load(source/'trace.npz');actions=trace['actions'];origin=item['active_origin'];period=item['period']
    env,obs=predictor.engine.restore(tid,sid,actions[:origin])
    instructions=predictor.engine.suite.get_task(tid).language
    anchor=runtime.object_positions(env)
    taken=[];physics=[];calls=[];success=False
    try:
        for action in actions[origin:t]:
            obs,_,term,trunc,info=env.step(action)
            assert not term and not trunc
        start=snapshot(env)
        assert np.array_equal(start['physics'],trace['physics'][t-1]),'Source prefix physics changed'
        old=np.load(source/f'chunk_{item["active_call_id"]:03d}.npz')['actions']
        queue=list(old[item['action_row']:].copy());pending=None
        state=runtime.state_text(instructions,obs,env,anchor,queue,t-origin,h,period)
        for tick in range(t,230):
            if (tick-t)%16==0:allowed()
            if pending is not None:
                assert pending['delivery']==tick
                queue=list(pending['actions'][1:].copy());pending=None
            if (tick==t and choice=='replan') or (tick>=t+period and (tick-t)%period==0):
                seed=20261007+tid*100000+sid*1000+tick
                arr,norm,cost=predictor.predict(obs,instructions,h,seed)
                if tick==t:
                    original=np.load(source/f'chunk_{item["source_call_id"]:03d}.npz')
                    assert np.array_equal(arr,original['actions']) and np.array_equal(norm,original['normalized']), 'Original intervention chunk changed'
                cid=len(calls);np.savez_compressed(folder/f'{choice}_chunk_{cid:03d}.npz',actions=arr,normalized=norm)
                calls.append(dict(tick=tick,seed=seed,nfe=10,seconds=cost,chunk_id=cid,delivery=tick+1))
                pending=dict(delivery=tick+1,actions=arr)
            assert queue,'Counterfactual queue exhausted'
            action=np.asarray(queue.pop(0),dtype=np.float64)
            taken.append(action.copy());obs,_,term,trunc,info=env.step(action)
            success=bool(info['is_success']);physics.append(env._env.env.sim.get_state().flatten().copy())
            if term or trunc:break
        taken=np.asarray(taken);physics=np.asarray(physics)
        result=dict(choice=choice,success=success,start_tick=t,total_steps=t+len(taken),post_steps=len(taken),
                    vla_calls=len(calls),prediction_seconds=sum(c['seconds'] for c in calls),
                    decision_window_success=success and len(taken)<=period,pending_at_end=pending is not None)
        if choice=='replan':
            original=json.loads((source/'result.json').read_text())
            assert np.array_equal(taken,actions[t:]),'R branch does not reproduce baseline actions'
            assert np.array_equal(physics,trace['physics'][t:]),'R branch does not reproduce baseline physics'
            assert result['success']==original['success'] and result['total_steps']==original['steps']
            result['baseline_suffix_exact']=True
        np.savez_compressed(folder/f'{choice}_trace.npz',actions=taken,physics=physics,start_physics=start['physics'])
        atomic(folder/f'{choice}_calls.json',calls);atomic(folder/f'{choice}_result.json',result)
        return result,start,state
    except BaseException as error:
        np.savez_compressed(folder/f'{choice}_partial.npz',actions=np.asarray(taken),physics=np.asarray(physics))
        atomic(folder/f'{choice}_failure.json',dict(error=repr(error),calls=calls,post_steps=len(taken)))
        raise
    finally:env.close()


def labels(c,r):
    if not c['success'] and not r['success']:
        return dict(necessity=None,utility=None,ambiguity='both_failure')
    necessity='continue' if c['success'] else 'replan'
    if c['success']!=r['success']:utility=necessity
    else:utility='replan' if r['vla_calls']<c['vla_calls'] else 'continue'
    return dict(necessity=necessity,utility=utility,ambiguity='both_success' if c['success'] and r['success'] else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--states',default='0,1,2');p.add_argument('--pilot',action='store_true')
    p.add_argument('--exclude-completed-run')
    args=p.parse_args();out=Path(args.out).resolve();assert out.is_relative_to(ROOT/'runs');out.mkdir(exist_ok=False)
    torch.set_num_threads(4);states=[int(s) for s in args.states.split(',')]
    source_schedule=ROOT/'fixtures/counterfactual_schedule.json'
    schedule=[r for r in json.loads(source_schedule.read_text()) if r['state'] in states]
    if args.pilot:schedule=[r for r in schedule if r['state']==0 and r['task'] in [0,1] and r['source_call_id']==1]
    excluded=[]
    if args.exclude_completed_run:
        name=args.exclude_completed_run
        assert name.isascii() and name.replace('_','').isalnum()
        prior=ROOT/'runs'/name
        assert json.loads((prior/'supervisor.json').read_text())['status']=='completed'
        prior_manifest=json.loads((prior/'data/manifest.json').read_text())
        assert prior_manifest['frozen_parameters_unchanged']
        assert prior_manifest['source_schedule_sha256']==hashlib.sha256(source_schedule.read_bytes()).hexdigest()
        prior_records={r['id']:r for r in (json.loads(l) for l in (prior/'data/records.jsonl').read_text().splitlines())}
        for item in schedule:
            if item['id'] in prior_records:
                assert all(prior_records[item['id']][k]==v for k,v in item.items())
                assert prior_records[item['id']]['branches']['replan']['baseline_suffix_exact']
                excluded.append(item['id'])
        schedule=[item for item in schedule if item['id'] not in excluded]
    assert schedule
    atomic(out/'schedule.json',schedule)
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),planned=len(schedule),states=states,pilot=args.pilot,
        source_schedule_sha256=hashlib.sha256(source_schedule.read_bytes()).hexdigest(),
        excluded_completed_run=args.exclude_completed_run,excluded_already_completed_ids=excluded,
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        label_protocol_sha256=hashlib.sha256((ROOT/'COUNTERFACTUAL_PROTOCOL.md').read_bytes()).hexdigest(),
        real_time=False,logical_delay_steps=1,frozen_VLA=True,optimizer_steps=0)
    atomic(out/'manifest.json',manifest);records=[]
    try:
        allowed();predictor=runtime.Predictor();before=freeze(predictor,predictor.before_hash)
        assert before['state_dict_sha256']=='b296dfca9e977fbe06d0f4a7971dbcfa368954e168c32f9c61295107f00eba82'
        atomic(out/'freeze_before.json',before)
        for index,item in enumerate(schedule):
            allowed();case=out/'cases'/item['id'];case.mkdir(parents=True)
            source=OLD/item['case_name']
            for name,digest in item['source_files'].items():
                assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,(item['id'],name)
            results={};starts=[];texts=[]
            order=['continue','replan'] if index%2==0 else ['replan','continue']
            for choice in order:
                result,start,state=branch(predictor,item,source,case,choice)
                results[choice]=result;starts.append(start);texts.append(state)
            match=compare_snapshots(starts[0],starts[1])
            assert match['max_abs_state_difference']==0 and match['rng_equal'] and texts[0]==texts[1]
            c=np.load(case/'continue_trace.npz');r=np.load(case/'replan_trace.npz')
            assert np.array_equal(c['actions'][0],r['actions'][0]) and np.array_equal(c['physics'][0],r['physics'][0])
            record=dict(**item,branches=results,labels=labels(results['continue'],results['replan']),
                        branch_order=order,start_match=match,shared_first_action_and_physics_exact=True,
                        state_text=texts[0],state_sha256=hashlib.sha256(texts[0].encode()).hexdigest())
            atomic(case/'record.json',record);records.append(record)
            with (out/'records.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            atomic(out/'status.json',dict(phase='running',completed=len(records),planned=len(schedule),latest_id=item['id']))
            print(json.dumps(dict(completed=len(records),planned=len(schedule),id=item['id'],
                                  labels=record['labels'],outcomes={k:v['success'] for k,v in results.items()})),flush=True)
        after=freeze(predictor,runtime.parameter_digest(predictor.engine.policy));assert before==after
        atomic(out/'freeze_after.json',after)
        manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),freeze_before=before,freeze_after=after,
            frozen_parameters_unchanged=True,total_vla_calls=predictor.calls,
            total_prediction_seconds=predictor.predict_seconds)
        atomic(out/'manifest.json',manifest);atomic(out/'status.json',dict(phase='completed',completed=len(records),planned=len(schedule)))
    except BaseException as error:
        atomic(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed=len(records)))
        raise


def interrupted(number,frame):raise SystemExit(f'Owned worker interrupted by signal {number}')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    main()
