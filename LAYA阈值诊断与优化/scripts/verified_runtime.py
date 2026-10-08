"""Initial-observation and first-prediction guards; controller semantics remain unchanged."""
from pathlib import Path
import copy
import hashlib
import json
import os
import time
import numpy as np
import torch
import controller_runtime as runtime
from supervise import allowed


def arrays(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key in sorted(value):
            result.update(arrays(value[key], prefix+'/'+str(key)))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        if tensor.dtype == torch.bfloat16:
            result[prefix+'@bfloat16_bits'] = tensor.view(torch.uint16).numpy().copy()
        else:
            result[prefix] = tensor.numpy().copy()
    elif isinstance(value, (np.ndarray, int, float, bool, str)):
        result[prefix] = np.asarray(value).copy()
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            result.update(arrays(item, prefix+'/'+str(i)))
    elif value is not None:
        raise TypeError((prefix, type(value)))
    return result


def receipt(values):
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                     sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in sorted(values.items())}


def environment_record():
    return dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                gpu_name=torch.cuda.get_device_name())


class VerifiedPredictor(runtime.Predictor):
    def __init__(self, out):
        super().__init__()
        self.folder = out/'initial_verification'
        self.folder.mkdir(exist_ok=False)
        self.references = {}
        self.active = None
        self.capture = False
        self.last_input_arrays = None
        self.qualification_records = []
        self.guard_records = []
        self.last_restore = None
        self.capture_seconds = 0.0
        runtime.atomic(self.folder/'environment.json', environment_record())
        original = self.engine.policy.predict_action_chunk

        def observed(batch, noise=None, **kwargs):
            if self.capture or self.active is not None:
                start = time.perf_counter()
                self.last_input_arrays = arrays(dict(batch=batch, noise=noise))
                self.capture_seconds = time.perf_counter()-start
            return original(batch, noise=noise, **kwargs)

        self.engine.policy.predict_action_chunk = observed
        restore = self.engine.restore

        def observed_restore(*args, **kwargs):
            env, obs = restore(*args, **kwargs)
            if self.active is not None:
                try:
                    self.last_restore = self.restore_receipt(env, obs)
                    reference = self.references[(self.active['task'], self.active['state'])]
                    if self.last_restore != reference['restore']:
                        self.save_failure('initial_restore_mismatch', obs, restore=self.last_restore,
                                          expected_restore=reference['restore'])
                        raise AssertionError('Initial raw observation / physics / simulator model mismatch')
                except BaseException:
                    env.close()
                    raise
            return env, obs

        self.engine.restore = observed_restore

    @staticmethod
    def restore_receipt(env, obs):
        sim = env._env.env.sim
        model = {n: np.asarray(getattr(sim.model, n)).copy() for n in
                 ['body_pos', 'body_quat', 'geom_pos', 'geom_quat', 'geom_rgba', 'mat_rgba',
                  'light_pos', 'cam_pos', 'cam_quat', 'qpos0'] if hasattr(sim.model, n)}
        return dict(observation=receipt(arrays(obs)), physics=receipt(arrays(sim.get_state().flatten())),
                    simulator_model=receipt(model))

    def qualify_start(self, tid, sid, horizons):
        key = (tid, sid)
        if key in self.references:
            return 0
        allowed()
        folder = self.folder/f't{tid:02d}_s{sid:02d}'
        folder.mkdir(exist_ok=False)
        before_calls = self.calls
        assert self.active is None
        env, obs = self.engine.restore(tid, sid, np.empty((0, 7)))
        try:
            base_restore = self.restore_receipt(env, obs)
            np.savez_compressed(folder/'raw_observation.npz', **arrays(obs))
            runtime.atomic(folder/'restore.json', base_restore)
            reference = dict(restore=base_restore, horizons={})
            self.capture = True
            instruction = self.engine.suite.get_task(tid).language
            for h in horizons:
                expected = None
                for repeat in range(2):
                    allowed()
                    a, n, seconds = self.predict(copy.deepcopy(obs), instruction, h, 20261007+tid*100000+sid*1000)
                    inp = receipt(self.last_input_arrays)
                    output = receipt(dict(actions=a, normalized=n))
                    name = f'h{h}_repeat{repeat}'
                    np.savez_compressed(folder/f'{name}_prediction.npz', actions=a, normalized=n)
                    if repeat == 0:
                        np.savez_compressed(folder/f'h{h}_processed_input.npz', **self.last_input_arrays)
                        expected = dict(processed_input=inp, prediction=output)
                    current = dict(processed_input=inp, prediction=output)
                    record = dict(task=tid, state=sid, horizon=h, repeat=repeat, seconds=seconds,
                                  capture_seconds=self.capture_seconds, **current, repeated_exact=current==expected)
                    self.qualification_records.append(record)
                    runtime.atomic(self.folder/'qualification_calls.json', self.qualification_records)
                    assert current == expected, 'Repeated reference prediction mismatch; abort, never retry/select'
                reference['horizons'][h] = expected
            assert self.restore_receipt(env, obs) == base_restore, 'Qualification mutated initial observation'
            self.references[key] = reference
            runtime.atomic(folder/'reference.json', reference)
        finally:
            self.capture = False
            env.close()
        return self.calls-before_calls

    def begin_episode(self, item):
        assert self.active is None and (item['task'], item['state']) in self.references
        self.active = dict(item)

    def end_episode(self):
        assert self.active is None, 'First-prediction guard was not executed'

    def save_failure(self, reason, obs, **fields):
        name = f'failure_{len(self.guard_records):04d}'
        np.savez_compressed(self.folder/f'{name}_raw_observation.npz', **arrays(obs))
        if self.last_input_arrays is not None:
            np.savez_compressed(self.folder/f'{name}_processed_input.npz', **self.last_input_arrays)
        runtime.atomic(self.folder/f'{name}.json', dict(reason=reason, active=self.active, **fields))

    def predict(self, obs, instruction, horizon, seed):
        initial_item = self.active
        a, n, seconds = super().predict(obs, instruction, horizon, seed)
        if initial_item is not None:
            key = (initial_item['task'], initial_item['state'])
            expected = self.references[key]['horizons'][horizon]
            current = dict(processed_input=receipt(self.last_input_arrays), prediction=receipt(dict(actions=a, normalized=n)))
            record = dict(**initial_item, noise_seed=seed, restore=self.last_restore, **current,
                          capture_seconds=self.capture_seconds, exact=current==expected,
                          checked_before_first_action=True)
            self.guard_records.append(record)
            with (self.folder/'guards.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            if current != expected:
                self.save_failure('first_prediction_mismatch', obs, actual=current, expected=expected)
                np.savez_compressed(self.folder/'failed_prediction.npz', actions=a, normalized=n)
                raise AssertionError('First prediction differs from repeated reference; abort before first action')
            self.active = None
        return a, n, seconds

    def finalize_verification(self, episodes):
        assert len(self.guard_records) == episodes and all(r['exact'] for r in self.guard_records)
        result = dict(starts=len(self.references), guarded_episodes=len(self.guard_records),
                      qualification_vla_calls=len(self.qualification_records),
                      qualification_prediction_seconds=sum(r['seconds'] for r in self.qualification_records),
                      episode_input_capture_seconds=sum(r['capture_seconds'] for r in self.guard_records),
                      input_capture_included_in_prediction_seconds=True,
                      disk_logging_and_qualification_excluded_from_controller_service=True,
                      exact=True, environment=environment_record(),
                      original_incident_root_cause='unresolved; safeguards do not establish a causal repair')
        runtime.atomic(self.folder/'summary.json', result)
        return result
