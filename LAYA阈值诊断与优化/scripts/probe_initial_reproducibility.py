"""Instrument first inputs on the exact task0/state10 schedule, then replay saved observations."""
from pathlib import Path
import copy
import hashlib
import json
import signal
import sys
import numpy as np
import torch
import run_closed_loop as runner

ROOT = runner.ROOT
OUTPUT = Path(sys.argv[sys.argv.index('--out')+1]).resolve()
PROBE = OUTPUT/'input_probe'
instances = []
CURRENT = {'item':None, 'first':False, 'replay':None}
BasePredictor = runner.runtime.Predictor
base_episode = runner.runtime.run_episode


def flat_arrays(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key in sorted(value):
            result.update(flat_arrays(value[key], prefix+'/'+str(key)))
    elif isinstance(value, torch.Tensor):
        tensor=value.detach().cpu()
        # Capture numeric values in the actual dtype wherever numpy supports it.
        result[prefix] = tensor.float().numpy().copy() if tensor.dtype == torch.bfloat16 else tensor.numpy().copy()
    elif isinstance(value, (np.ndarray, int, float, bool)):
        result[prefix] = np.asarray(value).copy()
    elif isinstance(value, (list, tuple)):
        for index,item in enumerate(value):
            result.update(flat_arrays(item,prefix+'/'+str(index)))
    elif isinstance(value, str):
        result[prefix] = np.asarray(value)
    elif value is not None:
        raise TypeError((prefix,type(value)))
    return result


def receipt(arrays):
    return {key:dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key,value in sorted(arrays.items())}


class ProbePredictor(BasePredictor):
    def __init__(self):
        super().__init__()
        PROBE.mkdir(exist_ok=False)
        self.probe_records=[]
        self.saved=[]
        self.last_batch=None
        instances.append(self)
        original=self.engine.policy.predict_action_chunk

        def observe(batch, noise=None, **kwargs):
            if CURRENT['first'] or CURRENT['replay'] is not None:
                self.last_batch=receipt(flat_arrays(dict(batch=batch,noise=noise)))
            return original(batch,noise=noise,**kwargs)

        self.engine.policy.predict_action_chunk=observe
        restore=self.engine.restore

        def restore_observed(*args,**kwargs):
            env,obs=restore(*args,**kwargs)
            if CURRENT['item'] is not None:
                sim=env._env.env.sim
                arrays={n:np.asarray(getattr(sim.model,n)).copy() for n in
                    ['body_pos','body_quat','geom_pos','geom_quat','geom_rgba','mat_rgba','light_pos','cam_pos','cam_quat','qpos0']
                    if hasattr(sim.model,n)}
                CURRENT['model']=receipt(arrays)
            return env,obs

        self.engine.restore=restore_observed

    def predict(self,obs,instruction,horizon,seed):
        record_initial=CURRENT['first']
        before=copy.deepcopy(obs) if record_initial else None
        actions,normalized,elapsed=super().predict(obs,instruction,horizon,seed)
        if record_initial:
            item=CURRENT['item'];name=f'{len(self.saved):03d}_{item["candidate"]}_h{horizon}'
            np.savez_compressed(PROBE/f'{name}_obs.npz',**flat_arrays(before))
            np.savez_compressed(PROBE/f'{name}_prediction.npz',actions=actions,normalized=normalized)
            record=dict(name=name,**item,noise_seed=seed,raw_observation_before=receipt(flat_arrays(before)),
                raw_observation_after=receipt(flat_arrays(obs)),processed_model_input=self.last_batch,
                simulator_model=CURRENT['model'],prediction=receipt(dict(actions=actions,normalized=normalized)))
            self.probe_records.append(record)
            self.saved.append((record,before,instruction,horizon,seed,actions.copy(),normalized.copy()))
            (PROBE/'initials.json').write_text(json.dumps(self.probe_records,indent=2))
            CURRENT['first']=False
        return actions,normalized,elapsed


def episode(predictor,gate,out,phase,tid,sid,horizon,method,**kwargs):
    CURRENT.update(item=dict(candidate=phase,task=tid,state=sid,horizon=horizon),first=True)
    try:
        return base_episode(predictor,gate,out,phase,tid,sid,horizon,method,**kwargs)
    finally:
        CURRENT.update(item=None,first=False)


def main():
    runner.runtime.Predictor=ProbePredictor
    runner.runtime.run_episode=episode
    runner.main()
    assert len(instances)==1
    predictor=instances[0]
    assert len(predictor.saved)==21
    replays=[]
    for record,obs,instruction,horizon,seed,original_actions,original_normalized in predictor.saved:
        for repeat in range(2):
            runner.allowed()
            CURRENT['replay']=record['name']
            actions,normalized,cost=predictor.predict(copy.deepcopy(obs),instruction,horizon,seed)
            row=dict(name=record['name'],repeat=repeat,
                processed_input_exact=predictor.last_batch==record['processed_model_input'],
                actions_exact=bool(np.array_equal(actions,original_actions)),
                normalized_exact=bool(np.array_equal(normalized,original_normalized)),
                actions_max_abs=float(np.max(abs(actions-original_actions))),
                normalized_max_abs=float(np.max(abs(normalized-original_normalized))),
                prediction=receipt(dict(actions=actions,normalized=normalized)),seconds=cost)
            replays.append(row)
            with (PROBE/'replays.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
    after=runner.runtime.parameter_digest(predictor.engine.policy)
    assert after==predictor.before_hash
    by_horizon={}
    for h in [50,100,200]:
        rows=[r for r in predictor.probe_records if r['horizon']==h]
        reference=rows[0]
        by_horizon[h]=[dict(name=r['name'],raw_observation_exact=r['raw_observation_before']==reference['raw_observation_before'],
            observation_unmodified=r['raw_observation_before']==r['raw_observation_after'],
            processed_input_exact=r['processed_model_input']==reference['processed_model_input'],
            simulator_model_exact=r['simulator_model']==reference['simulator_model'],
            prediction_exact=r['prediction']==reference['prediction']) for r in rows]
    result=dict(episodes_instrumented=21,replay_predictions=len(replays),frozen_parameters_unchanged=True,
        frozen_vla_sha256=after,by_horizon=by_horizon,replays=replays,
        interpretation='Forensic probe with host-side input copies and tensor transfers; exclude all timings and task outcomes from efficacy cohorts. Six controller/runner source files unchanged; wrapper instrumented separately.')
    (PROBE/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(probe_completed=True,episodes=21,replays=42)),flush=True)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,runner.interrupted)
    main()
