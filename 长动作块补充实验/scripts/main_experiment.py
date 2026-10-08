"""Frozen native-length SmolVLA comparison; logical d=1, unpaced simulation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, collections, copy, hashlib, json, os, platform, random
import subprocess, sys, time, traceback

BASE = Path('/mnt/4t/jev_vla_libero')
ROOT = BASE / 'experiments/long_chunk_laya'
sys.path.insert(0, str(ROOT / 'source_snapshot'))
import numpy as np
import torch
from collect_counterfactual_dev import Engine, batchify, preprocess_observation

METHODS = ['smol70', 'laya', 'naive_k5', 'vlash_style_k5']
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


def state_text(instruction, obs, env, anchor, queue, age, horizon):
    round3 = lambda a: np.round(np.asarray(a), 3).tolist()
    robot = obs['robot_state']
    lines = [f'Task: {instruction}',
             f'Native plan length {horizon}; age {age}; remaining {len(queue)} actions.',
             'One action = one control step. Next gate check in at most 5 steps; '
             'a requested replacement is used after 1 additional logical step.',
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


def run_episode(predictor, gate, out, phase, tid, sid, horizon, method):
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
                    queue = list(arr[1:].copy())
                    row_index = 1
                    origin = pending['request_tick']
                    anchor = pending['anchor']
                    calls[active_id]['expired_prefix_rows'] = 1
                calls[active_id]['actual_delivery_tick'] = tick
                pending = None

            trigger = None
            if pending is None and tick > 0:
                if len(queue) <= 1:
                    trigger = 'mandatory_finite_queue'
                elif method == 'smol70' and len(queue) <= .7 * horizon:
                    trigger = 'remaining_ratio_le_0.7'
                elif method in ['naive_k5', 'vlash_style_k5'] and tick-last_request >= 4:
                    # Native naive request cadence is five steps. For VLASH the image
                    # capture precedes its next five-step action-block boundary by one.
                    trigger = 'fixed_k5'
                elif method == 'laya' and tick % 5 == 0:
                    gate_start = time.perf_counter()
                    text = state_text(instruction, obs, env, anchor, queue, tick-origin, horizon)
                    answer = gate.predict(text)
                    answer.update(tick=tick, remaining=len(queue), plan_age=tick-origin,
                                  state=text, full_gate_seconds=time.perf_counter()-gate_start)
                    gate_rows.append(answer)
                    if answer['choice'] == 'replan':
                        trigger = 'laya_replan'
                    elif answer['choice'] == 'uncertain' and len(queue) <= .7*horizon:
                        trigger = 'laya_uncertain_smol70_fallback'
            # VLASH first boundary is tick5: capture its delayed image at tick4.
            if method == 'vlash_style_k5' and pending is None and tick-last_request == 4:
                trigger = 'vlash_capture_before_k5_boundary'
            if trigger:
                if method == 'vlash_style_k5':
                    pending = dict(obs=copy.deepcopy(obs), image_tick=tick, delivery=tick+1)
                    last_request = tick+1
                else:
                    arr, cid = request(obs, tick, trigger, tick+1)
                    pending = dict(actions=arr, call_id=cid, request_tick=tick,
                                   delivery=tick+1, anchor=object_positions(env))
                    last_request = tick+1 if method == 'naive_k5' else tick
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
                      success=success, steps=len(executed), vla_calls=len(calls), nfe=10*len(calls),
                      prediction_seconds=sum(c['predict_seconds'] for c in calls),
                      gate_calls=len(gate_rows), gate_seconds=sum(g['full_gate_seconds'] for g in gate_rows),
                      episode_wall_seconds=elapsed, simulated_seconds=len(executed)/20,
                      gate_choices=dict(collections.Counter(g['choice'] for g in gate_rows)),
                      initial_state_sha256=digest_bytes(initial),
                      executed_action_sha256=digest_bytes(np.asarray(executed)),
                      unused_generated_rows=sum(c['unused_rows'] for c in calls),
                      native_generated_rows=len(calls)*horizon,
                      real_time=False, logical_delay_steps=1,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    torch=torch.__version__, python=platform.python_version(),
                    gpu=torch.cuda.get_device_name(), logical_delay_steps=1,
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in list((ROOT/'scripts').glob('*.py'))+
                                   list((ROOT/'source_snapshot').glob('*.py'))},
                    suite='libero_spatial', tasks=list(range(10)), state_ids=[0, 1, 2],
                    horizons=HORIZONS, methods=METHODS, max_steps=230,
                    checkpoint_training_horizon=50, native_length_extrapolation=True,
                    real_time=False, no_optimizer=True, gate_interval=5,
                    inference_overlap=False, privileged_gate_state=True)
    atomic(out/'run_manifest.json', manifest)
    gate = None
    try:
        load_start = time.perf_counter()
        predictor = Predictor()
        manifest['smolvla_load_seconds'] = time.perf_counter()-load_start
        manifest['parameter_sha256_before'] = predictor.before_hash
        manifest['checkpoint_path'] = str(predictor.engine.model_path)
        manifest['original_checkpoint_config'] = json.loads((predictor.engine.model_path/'config.json').read_text())
        manifest['effective_settings'] = dict(rtc_enabled=False, num_flow_steps=10,
                                               native_lengths=HORIZONS, noise_dtype='torch.float32')
        manifest['installed_source_sha256'] = {}
        for name in ['lerobot.policies.smolvla.modeling_smolvla',
                     'lerobot.policies.smolvla.configuration_smolvla',
                     'lerobot.policies.smolvla.smolvlm_with_expert', 'lerobot.envs.libero']:
            module = sys.modules.get(name)
            if module is not None and getattr(module, '__file__', None):
                path = Path(module.__file__)
                manifest['installed_source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        dtype_counts = collections.Counter()
        for param in predictor.engine.policy.parameters():
            dtype_counts[str(param.dtype)] += param.numel()
        manifest['parameters_by_dtype'] = dict(dtype_counts)
        env, obs = predictor.engine.restore(0, 0, np.empty((0, 7)))
        qualification = []
        try:
            instruction = predictor.engine.suite.get_task(0).language
            for horizon in HORIZONS:
                a, n, cost = predictor.predict(obs, instruction, horizon, 20261007)
                b, n2, cost2 = predictor.predict(obs, instruction, horizon, 20261007)
                assert np.array_equal(a, b) and np.array_equal(n, n2)
                np.savez_compressed(out/f'qualification_h{horizon}.npz', actions=a, normalized=n)
                qualification.append(dict(horizon=horizon, native_shape=list(a.shape),
                                          repeated_exact=True, warm_seconds=cost,
                                          hot_seconds=cost2, finite=True))
            gate = LayaClient(out/'laya_ipc')
            warm_text = state_text(instruction, obs, env, object_positions(env),
                                   list(a), 0, HORIZONS[-1])
            warm = gate.predict(warm_text, warmup=True)
            atomic(out/'laya_warmup.json', warm)
            manifest['laya_ready'] = gate.ready
        finally:
            env.close()
        atomic(out/'qualification.json', qualification)
        atomic(out/'run_manifest.json', manifest)
        all_results = []
        schedule = [('pilot', 0, 0, h, m) for h in HORIZONS for m in METHODS]
        if not args.pilot_only:
            for tid in range(10):
                for sid in [0, 1, 2]:
                    cells = [(h, m) for h in HORIZONS for m in METHODS]
                    random.Random(20261007+tid*10+sid).shuffle(cells)
                    schedule.extend(('main', tid, sid, h, m) for h, m in cells)
        atomic(out/'schedule.json', schedule)
        initial_hashes = {}
        for index, (phase, tid, sid, horizon, method) in enumerate(schedule):
            if index == 12:
                assert len(all_results) == 12
                atomic(out/'pilot_passed.json', dict(episodes=12, mechanism_errors=0,
                                                    note='Software/pipeline qualification, not efficacy acceptance'))
            result = run_episode(predictor, gate, out, phase, tid, sid, horizon, method)
            key = (tid, sid)
            if key in initial_hashes:
                assert result['initial_state_sha256'] == initial_hashes[key], 'Paired initial state differs'
            else:
                initial_hashes[key] = result['initial_state_sha256']
            all_results.append(result)
            with (out/'episodes.jsonl').open('a') as f:
                f.write(json.dumps(result)+'\n')
            status = dict(phase='running', completed=index+1, planned=len(schedule),
                          latest=result, updated_utc=datetime.now(timezone.utc).isoformat())
            atomic(out/'status.json', status)
            print(json.dumps({'completed':index+1,'planned':len(schedule), **result}), flush=True)
        after_hash = parameter_digest(predictor.engine.policy)
        assert after_hash == predictor.before_hash
        assert all(not p.requires_grad for p in predictor.engine.policy.parameters())
        manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                        parameter_sha256_after=after_hash, frozen_parameters_unchanged=True,
                        total_vla_calls_including_qualification=predictor.calls,
                        total_prediction_seconds_including_qualification=predictor.predict_seconds)
        atomic(out/'run_manifest.json', manifest)
        atomic(out/'status.json', dict(phase='completed', completed=len(all_results),
                                      planned=len(schedule), frozen_parameters_unchanged=True))
    except BaseException as error:
        atomic(out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        if gate is not None:
            gate.close()


if __name__ == '__main__':
    main()
