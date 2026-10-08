"""One immutable schedule of paired frozen-VLA trials, with per-candidate gate cadence."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,random,signal,subprocess,sys,time,traceback
import mechanism_runtime as runtime
from supervise import allowed
from verified_runtime import VerifiedPredictor
ROOT=runtime.ROOT;BASE=runtime.BASE
SCRIPT_ROOT=Path(__file__).resolve().parent
np=runtime.np;torch=runtime.torch;atomic=runtime.atomic


class BinaryClient(runtime.LayaClient):
    def __init__(self,folder):
        self.folder=folder;folder.mkdir()
        (folder/'requests').mkdir();(folder/'responses').mkdir()
        self.log=(folder/'service.log').open('w')
        self.process=subprocess.Popen([str(BASE/'envs/laya/bin/python'),str(SCRIPT_ROOT/'binary_service.py'),
                                      '--folder',str(folder)],stdout=self.log,stderr=subprocess.STDOUT)
        self.index=0;self.variant='binary_short';started=time.monotonic()
        try:
            while not (folder/'ready.json').exists():
                if self.process.poll() is not None:raise RuntimeError('Binary LAYA failed to load')
                if time.monotonic()-started>180:raise TimeoutError('Binary LAYA loading timeout')
                time.sleep(.02)
            self.ready=json.loads((folder/'ready.json').read_text())
            assert self.ready['threshold']==0 and self.ready['binary_argmax'] and self.ready['frozen']
        except BaseException:
            self.close();raise

    def predict(self,state,warmup=False):
        index=f'{self.index:07d}';self.index+=1;started=time.perf_counter()
        atomic(self.folder/'requests'/f'{index}.json',dict(request_id=index,state=state,warmup=warmup,variant=self.variant))
        path=self.folder/'responses'/f'{index}.json'
        while not path.exists():
            if self.process.poll() is not None:raise RuntimeError('Binary LAYA service died')
            if time.perf_counter()-started>60:raise TimeoutError('Binary LAYA request timed out')
            time.sleep(.001)
        response=json.loads(path.read_text())
        assert response['status']=='ok',response
        assert response['choice']==response['raw_choice'] and response['choice'] in {'continue','replan'}
        assert response['variant']==self.variant and response['minimum_probability']==0
        response['ipc_outer_seconds']=time.perf_counter()-started
        return response


def configure_gate(gate,candidate):
    gate.variant=candidate['variant']
    readout=candidate.get('readout_model_id')
    assert readout is None or hasattr(gate,'bundle')
    gate.readout_model_id=readout


def freeze(predictor,fingerprint):
    policy=predictor.engine.policy
    result=dict(state_dict_sha256=fingerprint,all_frozen=all(not p.requires_grad for p in policy.parameters()),
                gradients_none=all(p.grad is None for p in policy.parameters()),
                modules_eval=all(not m.training for m in policy.modules()),optimizer_steps=0)
    assert result['all_frozen'] and result['gradients_none'] and result['modules_eval']
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--config',required=True)
    args=p.parse_args();out=Path(args.out).resolve();assert out.is_relative_to(ROOT/'runs')
    config_path=(ROOT/'configs'/args.config).resolve();assert config_path.parent==ROOT/'configs'
    cfg=json.loads(config_path.read_text());out.mkdir(exist_ok=False)
    for name,expected_hash in cfg.get('locked_source_sha256',{}).items():
        assert Path(name).name==name and name.endswith('.py')
        assert hashlib.sha256((SCRIPT_ROOT/name).read_bytes()).hexdigest()==expected_hash,('Locked execution source changed',name)
    logical_delay=cfg.get('logical_delay_steps',1)
    assert isinstance(logical_delay,int) and 1<=logical_delay<=5
    torch.set_num_threads(4)
    schedule=[]
    for tid in cfg['tasks']:
        for sid in cfg['states']:
            cells=[(cid,h) for cid,candidate in cfg['candidates'].items() for h in cfg['horizons']
                   if h in candidate.get('horizons',cfg['horizons']) and sid in candidate.get('states',cfg['states'])]
            random.Random(cfg['schedule_seed']+tid*100+sid).shuffle(cells)
            schedule.extend(dict(task=tid,state=sid,candidate=cid,horizon=h) for cid,h in cells)
    atomic(out/'schedule.json',schedule)
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SCRIPT_ROOT.glob('*.py')},
        started_utc=datetime.now(timezone.utc).isoformat(),planned=len(schedule),max_steps=230,
        privileged_gate_state=True,real_time=False,inference_overlap=False,logical_delay_steps=logical_delay,
        checkpoint_training_horizon=50,native_lengths=cfg['horizons'],no_optimizer=True,minimum_probability=0.0)
    atomic(out/'manifest.json',manifest)
    gate=None;predictor=None;results=[]
    try:
        allowed();predictor=VerifiedPredictor(out);before=freeze(predictor,predictor.before_hash)
        assert predictor.before_hash=='b296dfca9e977fbe06d0f4a7971dbcfa368954e168c32f9c61295107f00eba82'
        atomic(out/'freeze_before.json',before)
        env,obs=predictor.engine.restore(0,0,np.empty((0,7)));qualification=[]
        try:
            instruction=predictor.engine.suite.get_task(0).language
            for h in cfg['horizons']:
                a,n,cost=predictor.predict(obs,instruction,h,20261007)
                b,n2,cost2=predictor.predict(obs,instruction,h,20261007)
                assert np.array_equal(a,b) and np.array_equal(n,n2)
                qualification.append(dict(horizon=h,repeated_exact=True,warm_seconds=cost,hot_seconds=cost2))
            if cfg.get('model_bundle'):
                from readout_client import ReadoutClient
                bundle_path=(ROOT/cfg['model_bundle']).resolve()
                assert bundle_path.is_relative_to(ROOT/'models')
                gate=ReadoutClient(out/'laya_ipc',bundle_path)
            else:gate=BinaryClient(out/'laya_ipc')
            warm=runtime.state_text(instruction,obs,env,runtime.object_positions(env),list(a),0,h,logical_delay=logical_delay)
            warmed=set()
            for cid,candidate in cfg['candidates'].items():
                if not candidate['method'].startswith('laya'):continue
                key=(candidate['variant'],candidate.get('readout_model_id'))
                if key in warmed:continue
                configure_gate(gate,candidate)
                if candidate['variant'] in ['binary_window','binary_window3']:
                    from window_state import prepare_window_state
                    warm_state=prepare_window_state(warm,list(a),compact=candidate['variant']=='binary_window3')
                else:warm_state=warm
                atomic(out/f'warmup_{cid}.json',gate.predict(warm_state,warmup=True));warmed.add(key)
        finally:env.close()
        atomic(out/'qualification.json',qualification)
        manifest['laya_ready']=gate.ready;manifest['freeze_before']=before;atomic(out/'manifest.json',manifest)
        initials={};warmups=gate.index;qualification_calls=predictor.calls
        for index,item in enumerate(schedule):
            allowed();cid=item['candidate'];candidate=cfg['candidates'][cid]
            if candidate['method'].startswith('laya'):configure_gate(gate,candidate)
            interval=max(1,round(candidate['interval_ratio']*item['horizon'])) if 'interval_ratio' in candidate else candidate.get('interval',5)
            minimum_age=max(0,round(candidate['minimum_age_ratio']*item['horizon'])) if 'minimum_age_ratio' in candidate else candidate.get('minimum_age',0)
            assert isinstance(interval,int) and 1<=interval<=item['horizon']
            assert isinstance(minimum_age,int) and 0<=minimum_age<=item['horizon']
            if candidate['method']=='fixed_interval':assert interval<=item['horizon']-logical_delay
            qualification_calls += predictor.qualify_start(item["task"],item["state"],cfg["horizons"])
            predictor.begin_episode(item)
            result=runtime.run_episode(predictor,gate,out,cid,item['task'],item['state'],item['horizon'],candidate['method'],
                gate_interval=interval,minimum_age=minimum_age,logical_delay=logical_delay,
                alignment_mode=candidate.get("alignment_mode","current_state"),
                noise_clock=candidate.get("noise_clock","prediction"))
            predictor.end_episode()
            result['candidate_id']=cid;result['candidate']=candidate
            key=(item['task'],item['state'])
            if key in initials:assert initials[key]==result['initial_state_sha256']
            else:initials[key]=result['initial_state_sha256']
            baseline=BASE/f"experiments/long_chunk_laya/attempt_v1/smol/main/t{item['task']:02d}_s{item['state']:02d}_h{item['horizon']}_smol70"
            if baseline.exists():
                reference=json.loads((baseline/'result.json').read_text())
                assert reference['initial_state_sha256']==result['initial_state_sha256']
                original=np.load(baseline/'chunk_000.npz')
                current=np.load(Path(result['case_path'])/'chunk_000.npz')
                assert np.array_equal(original['actions'],current['actions']) and np.array_equal(original['normalized'],current['normalized'])
                result['original_start_and_chunk_exact']=True
            atomic(Path(result['case_path'])/'result.json',result);results.append(result)
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            atomic(out/'status.json',dict(phase='running',completed=len(results),planned=len(schedule),latest=result))
            print(json.dumps(dict(completed=len(results),planned=len(schedule),candidate=cid,task=item['task'],
                state=item['state'],horizon=item['horizon'],success=result['success'],calls=result['vla_calls'])),flush=True)
        after=freeze(predictor,runtime.parameter_digest(predictor.engine.policy));assert before==after
        assert predictor.calls==qualification_calls+sum(r['vla_calls'] for r in results)
        assert gate.index==warmups+sum(r['gate_calls'] for r in results)
        atomic(out/'freeze_after.json',after)
        manifest['initial_verification']=predictor.finalize_verification(len(results))
        manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),freeze_after=after,
            qualification_vla_calls=qualification_calls,warmup_gate_calls=warmups,
            total_vla_calls=predictor.calls,total_gate_calls=gate.index,frozen_parameters_unchanged=True)
        if cfg.get('model_bundle'):
            gate.close()
            stopped=json.loads((out/'laya_ipc/stopped.json').read_text())
            assert stopped['frozen_laya_unchanged'] and stopped['model_bundle_sha256']==gate.bundle_sha
            manifest['readout_service_shutdown']=stopped
        atomic(out/'manifest.json',manifest)
        atomic(out/'status.json',dict(phase='completed',completed=len(results),planned=len(schedule),frozen_parameters_unchanged=True))
    except BaseException as error:
        atomic(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed=len(results)))
        raise
    finally:
        if gate is not None:gate.close()


def interrupted(number,frame):raise SystemExit(f'Owned worker interrupted by signal {number}')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    main()
