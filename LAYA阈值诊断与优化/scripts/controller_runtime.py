"""Frozen native-length SmolVLA comparison; explicit logical delay, unpaced simulation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, collections, copy, hashlib, json, os, platform, random
import subprocess, sys, time, traceback

BASE = Path('/mnt/4t/jev_vla_libero')
ROOT = BASE / 'experiments/laya_gate_optimization'
sys.path.insert(0, str(BASE / 'experiments/long_chunk_laya/source_snapshot'))
import numpy as np
import torch
from collect_counterfactual_dev import Engine, batchify, preprocess_observation

METHODS = ['smol70', 'laya', 'laya_skip1', 'laya_first', 'fixed_r60', 'fixed_interval', 'naive_k5', 'vlash_style_k5']
HORIZONS = [50, 100, 200]


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest_bytes(a):
    a = np.asarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def parameter_digest(policy):
    h = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Predictor:
    def __init__(self):
        cfg = json.loads((BASE / 'configs/counterfactual_dev_v1.json').read_text())
        self.engine = Engine(cfg)
        self.engine.cfg.rtc_config.enabled = False
        self.engine.cfg.num_steps = 10
        assert self.engine.policy.model.config is self.engine.cfg
        assert not self.engine.policy.training
        assert all(not p.requires_grad for p in self.engine.policy.parameters())
        self.before_hash = parameter_digest(self.engine.policy)
        self.calls = 0
        self.predict_seconds = 0.0

    def predict(self, obs, instruction, horizon, seed):
        engine = self.engine
        engine.cfg.chunk_size = horizon
        engine.cfg.n_action_steps = horizon
        engine.policy.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = preprocess_observation(batchify(obs))
        batch['task'] = [instruction]
        batch = engine.pre(engine.env_pre(batch))
        noise = torch.randn((1, horizon, engine.cfg.max_action_dim),
                            generator=torch.Generator(device='cpu').manual_seed(seed),
                            dtype=torch.float32).to('cuda')
        with torch.no_grad():
            normalized = engine.policy.predict_action_chunk(batch, noise=noise)
            actions = engine.post(normalized.clone())
        torch.cuda.synchronize()
        normalized_np = normalized.detach().cpu().numpy()
        arr = actions.detach().cpu().numpy()[0]
        elapsed = time.perf_counter() - start
        assert normalized_np.shape == (1, horizon, 7)
        assert arr.shape == (horizon, 7) and np.isfinite(arr).all()
        assert np.isfinite(normalized_np).all()
        self.calls += 1
        self.predict_seconds += elapsed
        return arr, normalized_np, elapsed


class LayaClient:
    def __init__(self, folder):
        self.folder = folder
        folder.mkdir()
        (folder / 'requests').mkdir()
        (folder / 'responses').mkdir()
        self.log = (folder / 'service.log').open('w')
        self.process = subprocess.Popen([str(BASE / 'envs/laya/bin/python'),
                                        str(ROOT / 'scripts/laya_service.py'),
                                        '--folder', str(folder)],
                                       stdout=self.log, stderr=subprocess.STDOUT)
        self.index = 0
        start = time.monotonic()
        while not (folder / 'ready.json').exists():
            if self.process.poll() is not None:
                raise RuntimeError('Laya service exited before ready; see service.log')
            if time.monotonic() - start > 180:
                raise TimeoutError('Laya loading timeout')
            time.sleep(.02)
        self.ready = json.loads((folder / 'ready.json').read_text())

    def predict(self, state, warmup=False):
        index = f'{self.index:07d}'
        self.index += 1
        start = time.perf_counter()
        atomic(self.folder / 'requests' / (index + '.json'),
               {'request_id': index, 'state': state, 'warmup': warmup})
        result_path = self.folder / 'responses' / (index + '.json')
        while not result_path.exists():
            if self.process.poll() is not None:
                raise RuntimeError('Laya service died during request')
            if time.perf_counter() - start > 60:
                raise TimeoutError('Laya request timeout')
            time.sleep(.001)
        response = json.loads(result_path.read_text())
        if response.get('status') == 'error':
            raise RuntimeError('Laya invalid response: ' + json.dumps(response))
        assert response['choice'] in ['continue', 'replan', 'uncertain']
        response['ipc_outer_seconds'] = time.perf_counter() - start
        return response

    def close(self):
        (self.folder / 'stop').touch()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()


def object_positions(env):
    base = env._env.env
    return {name: base.sim.data.body_xpos[index].copy()
            for name, index in sorted(base.obj_body_id.items())}


def state_text(instruction, obs, env, anchor, queue, age, horizon, gate_interval=5, logical_delay=1):
    round3 = lambda a: np.round(np.asarray(a, dtype=np.float64), 3).tolist()
    robot = obs['robot_state']
    lines = [f'Task: {instruction}',
             f'Native plan length {horizon}; age {age}; remaining {len(queue)} actions.',
             f'One action = one control step. Next gate check in at most {gate_interval} steps; '
             f'a requested replacement is used after {logical_delay} additional logical step'+('s.' if logical_delay!=1 else '.'),
             'Object coordinates are privileged simulator observations in metres, '
             'not predicted future states. Rows: [position at last plan request, current position].']
    for name, pos in object_positions(env).items():
        lines.append(name.replace('_', ' ') + ': ' + json.dumps(
            [round3(anchor[name]), round3(pos)], separators=(',', ':')))
    lines.append('Current EEF xyz: ' + json.dumps(round3(robot['eef']['pos']))
                 + '; gripper qpos: ' + json.dumps(round3(robot['gripper']['qpos'])))
    offsets = sorted(set([0, min(4, len(queue)-1), min(9, len(queue)-1),
                          min(19, len(queue)-1), len(queue)-1]))
    lines.append('Remaining planned commands at offsets ' + str(offsets)
                 + ' [dx,dy,dz,rotation3,gripper], controller units not metres: '
                 + json.dumps([round3(queue[i]) for i in offsets], separators=(',', ':')))
    lines.append('Ordinary robot and object motion can be task progress. '
                 'Decide whether the existing finite plan should continue to the next check '
                 'or a new VLA request is needed. No future outcome is provided.')
    return '\n'.join(lines)


def run_episode(predictor, gate, out, phase, tid, sid, horizon, method, gate_interval=5, minimum_age=0, logical_delay=1):
    assert isinstance(logical_delay,int) and 1<=logical_delay<=5<horizon
    case = out / phase / f't{tid:02d}_s{sid:02d}_h{horizon}_{method}'
    case.mkdir(parents=True)
    engine = predictor.engine
    env, obs = engine.restore(tid, sid, np.empty((0, 7)))
    instruction = engine.suite.get_task(tid).language
    initial = env._env.env.sim.get_state().flatten().copy()
    np.save(case / 'initial_state.npy', initial)
    records, calls, gate_rows, executed, physical, plans = [], [], [], [], [], []
    pending = None
    queue = []
    active_id = None
    origin = 0
    anchor = object_positions(env)
    last_request = 0
    success = False
    start = time.perf_counter()

    def request(input_obs, tick, reason, response_tick, is_aligned=False):
        seed = 20261007 + tid * 100000 + sid * 1000 + tick
        arr, norm, seconds = predictor.predict(input_obs, instruction, horizon, seed)
        call_id = len(calls)
        plans.append(arr.copy())
        np.savez_compressed(case / f'chunk_{call_id:03d}.npz', actions=arr, normalized=norm)
        row = dict(call_id=call_id, request_tick=tick, intended_delivery_tick=response_tick,
                   trigger=reason, noise_seed=seed, predict_seconds=seconds,
                   native_output_shape=list(arr.shape), nfe=10, executed_rows=[],
                   current_state_alignment=is_aligned)
        calls.append(row)
        return arr, call_id

    try:
        initial_actions, active_id = request(obs, 0, 'initial', 0)
        queue = list(initial_actions.copy())
        row_index = 0
        for tick in range(230):
            if pending is not None and tick == pending['delivery']:
                if method == 'vlash_style_k5':
                    mixed = copy.deepcopy(pending['obs'])
                    mixed['robot_state'] = copy.deepcopy(obs['robot_state'])
                    arr, active_id = request(mixed, tick, 'vlash_delayed_image_current_state', tick, True)
                    calls[active_id]['image_tick'] = pending['image_tick']
                    calls[active_id]['state_tick'] = tick
                    queue = list(arr.copy())
                    row_index = 0
                    origin = tick
                    anchor = object_positions(env)
                else:
                    arr, active_id = pending['actions'], pending['call_id']
                    queue = list(arr[logical_delay:].copy())
                    row_index = logical_delay
                    origin = pending['request_tick']
                    anchor = pending['anchor']
                    calls[active_id]['expired_prefix_rows'] = logical_delay
                calls[active_id]['actual_delivery_tick'] = tick
                pending = None

            trigger = None
            if pending is None and tick > 0:
                if len(queue) <= logical_delay:
                    trigger = 'mandatory_finite_queue'
                elif method == 'smol70' and len(queue) <= .7 * horizon:
                    trigger = 'remaining_ratio_le_0.7'
                elif method == 'fixed_interval' and tick-origin>=gate_interval:
                    trigger = 'fixed_request_interval'
                elif method in ['laya_skip1', 'fixed_r60'] and tick-origin >= 2*round(.3*horizon):
                    trigger = 'single_skip_recovery_at_0.6H'
                elif method == 'laya_first' and ((len(calls)==1 and tick>=2*round(.3*horizon)) or (len(calls)>1 and len(queue)<=.7*horizon)):
                    trigger = 'first_gate_recovery_then_smol70'
                elif (method == 'laya_skip1' and tick-origin == round(.3*horizon)) or (method == 'laya_first' and len(calls)==1 and tick==round(.3*horizon)):
                    gate_start = time.perf_counter()
                    interval = round(.3*horizon)
                    text = state_text(instruction, obs, env, anchor, queue, tick-origin, horizon, interval, logical_delay)
                    if getattr(gate,'variant',None) in ['binary_window','binary_window3']:
                        from window_state import prepare_window_state
                        text=prepare_window_state(text,queue,compact=gate.variant=='binary_window3')
                    answer = gate.predict(text)
                    assert answer['choice'] in {'continue', 'replan'}
                    answer.update(tick=tick, remaining=len(queue), plan_age=tick-origin,
                                  state=text, full_gate_seconds=time.perf_counter()-gate_start)
                    gate_rows.append(answer)
                    if answer['choice'] == 'replan':
                        trigger = 'laya_replan'
                elif method in ['naive_k5', 'vlash_style_k5'] and tick-last_request >= 5-logical_delay:
                    # Native naive request cadence is five steps. For VLASH the image
                    # capture precedes its next five-step boundary by logical_delay.
                    trigger = 'fixed_k5'
                elif method == 'laya' and tick % gate_interval == 0 and tick-origin >= minimum_age:
                    gate_start = time.perf_counter()
                    text = state_text(instruction, obs, env, anchor, queue, tick-origin, horizon, gate_interval, logical_delay)
                    if getattr(gate,'variant',None) in ['binary_window','binary_window3']:
                        from window_state import prepare_window_state
                        text=prepare_window_state(text,queue,compact=gate.variant=='binary_window3')
                    answer = gate.predict(text)
                    answer.update(tick=tick, remaining=len(queue), plan_age=tick-origin,
                                  state=text, full_gate_seconds=time.perf_counter()-gate_start)
                    gate_rows.append(answer)
                    if answer['choice'] == 'replan':
                        trigger = 'laya_replan'
                    elif answer['choice'] == 'uncertain' and len(queue) <= .7*horizon:
                        trigger = 'laya_uncertain_smol70_fallback'
            # This also handles d=5 image capture at tick0, before the first step.
            if method in ['naive_k5','vlash_style_k5'] and pending is None and tick-last_request == 5-logical_delay:
                trigger = 'vlash_capture_before_k5_boundary' if method=='vlash_style_k5' else 'fixed_k5'
            if trigger:
                if method == 'vlash_style_k5':
                    pending = dict(obs=copy.deepcopy(obs), image_tick=tick, delivery=tick+logical_delay)
                    last_request = tick+logical_delay
                else:
                    arr, cid = request(obs, tick, trigger, tick+logical_delay)
                    pending = dict(actions=arr, call_id=cid, request_tick=tick,
                                   delivery=tick+logical_delay, anchor=object_positions(env))
                    last_request = tick+logical_delay if method == 'naive_k5' else tick
            assert queue, 'Finite queue exhausted before valid delivery'
            action = np.asarray(queue.pop(0), dtype=np.float64)
            calls[active_id]['executed_rows'].append(row_index)
            executed.append(action.copy())
            obs, _, term, trunc, info = env.step(action)
            success = bool(info['is_success'])
            physical.append(env._env.env.sim.get_state().flatten().copy())
            records.append(dict(tick=tick, active_call=active_id, action_row=row_index,
                                queue_remaining=len(queue), pending=pending is not None,
                                success=success, terminated=bool(term), truncated=bool(trunc)))
            row_index += 1
            if term or trunc:
                break
        elapsed = time.perf_counter()-start
        for call in calls:
            call['unused_rows'] = horizon-len(call['executed_rows'])
        result = dict(phase=phase, task_id=tid, state_id=sid, horizon=horizon, method=method,
                      gate_interval=gate_interval, minimum_age=minimum_age,
                      success=success, steps=len(executed), vla_calls=len(calls), nfe=10*len(calls),
                      prediction_seconds=sum(c['predict_seconds'] for c in calls),
                      gate_calls=len(gate_rows), gate_seconds=sum(g['full_gate_seconds'] for g in gate_rows),
                      episode_wall_seconds=elapsed, simulated_seconds=len(executed)/20,
                      gate_choices=dict(collections.Counter(g['choice'] for g in gate_rows)),
                      initial_state_sha256=digest_bytes(initial),
                      executed_action_sha256=digest_bytes(np.asarray(executed)),
                      unused_generated_rows=sum(c['unused_rows'] for c in calls),
                      native_generated_rows=len(calls)*horizon,
                      real_time=False, logical_delay_steps=logical_delay,
                      pending_at_termination=pending is not None,
                      case_path=str(case))
        np.savez_compressed(case / 'trace.npz', actions=np.asarray(executed), physics=np.asarray(physical))
        atomic(case / 'calls.json', calls)
        atomic(case / 'gate.json', gate_rows)
        atomic(case / 'timeline.json', records)
        atomic(case / 'result.json', result)
        return result
    except BaseException as error:
        atomic(case / 'failure.json', dict(error=repr(error), calls=calls, steps=len(executed)))
        np.savez_compressed(case / 'partial_trace.npz', actions=np.asarray(executed), physics=np.asarray(physical))
        atomic(case / 'gate.json', gate_rows)
        raise
    finally:
        env.close()


