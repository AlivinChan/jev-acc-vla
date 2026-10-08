"""Frozen controller plus a predeclared state-blind random single-skip baseline."""
from controller_runtime import *


def draw_skip(seed, tid, sid, decision_index, fraction):
    numerator, denominator = fraction
    assert isinstance(seed, int) and isinstance(decision_index, int) and decision_index >= 0
    assert isinstance(numerator, int) and isinstance(denominator, int) and 0 <= numerator <= denominator and denominator > 0
    key = f'laya_random_skip_v1|{seed}|{tid}|{sid}|{decision_index}'
    value = int.from_bytes(hashlib.sha256(key.encode('ascii')).digest()[:8], 'big')
    choice = 'continue' if value*denominator < numerator*(1 << 64) else 'replan'
    return dict(choice=choice, seed=seed, decision_index=decision_index,
                fraction=list(fraction), uniform_u64=value, draw_key=key,
                decision_rule='State-blind predeclared Bernoulli draw; not a LAYA prediction')


def run_episode(predictor, gate, out, phase, tid, sid, horizon, method, gate_interval=5, minimum_age=0, logical_delay=1, coin_seed=None, continue_fraction=None):
    assert isinstance(logical_delay,int) and 1<=logical_delay<=5<horizon
    if method == 'seeded_skip1':
        assert coin_seed in [11131, 22261] and continue_fraction is not None
    coin_rows = []
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
                elif method in ['laya_skip1', 'fixed_r60', 'seeded_skip1'] and tick-origin >= 2*round(.3*horizon):
                    trigger = 'single_skip_recovery_at_0.6H'
                elif method == 'laya_first' and ((len(calls)==1 and tick>=2*round(.3*horizon)) or (len(calls)>1 and len(queue)<=.7*horizon)):
                    trigger = 'first_gate_recovery_then_smol70'
                elif method == 'seeded_skip1' and tick-origin == round(.3*horizon):
                    decision_start = time.perf_counter()
                    answer = draw_skip(coin_seed, tid, sid, len(coin_rows), continue_fraction)
                    answer.update(tick=tick, plan_age=tick-origin, remaining=len(queue),
                                  full_decision_seconds=time.perf_counter()-decision_start)
                    coin_rows.append(answer)
                    if answer['choice'] == 'replan':
                        trigger = 'seeded_skip_replan'
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
                      gate_calls=len(gate_rows),
                      gate_seconds=sum(g['full_gate_seconds'] for g in gate_rows)+sum(g['full_decision_seconds'] for g in coin_rows),
                      laya_gate_seconds=sum(g['full_gate_seconds'] for g in gate_rows),
                      random_decisions=len(coin_rows), random_decision_seconds=sum(g['full_decision_seconds'] for g in coin_rows),
                      random_choices=dict(collections.Counter(g['choice'] for g in coin_rows)),
                      coin_seed=coin_seed, continue_fraction=continue_fraction,
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
        atomic(case / 'random.json', coin_rows)
        atomic(case / 'timeline.json', records)
        atomic(case / 'result.json', result)
        return result
    except BaseException as error:
        atomic(case / 'failure.json', dict(error=repr(error), calls=calls, steps=len(executed)))
        np.savez_compressed(case / 'partial_trace.npz', actions=np.asarray(executed), physics=np.asarray(physical))
        atomic(case / 'gate.json', gate_rows)
        atomic(case / 'random.json', coin_rows)
        raise
    finally:
        env.close()
