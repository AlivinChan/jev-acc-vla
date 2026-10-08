"""Actual-controller tests for stateless random skips, extremes and unchanged paths."""
from pathlib import Path
from types import SimpleNamespace
import ast
import collections
import copy
import hashlib
import json
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Sim:
    def __init__(self):
        self.tick = 0
    def get_state(self):
        return np.array([self.tick], dtype=float)


def observation(tick):
    return {'frame_tick': tick, 'robot_state': {'eef': {'pos': np.array([tick, 0., 0.])},
                                              'gripper': {'qpos': np.zeros(2)}}}


class Env:
    def __init__(self, terminal):
        self.sim, self.terminal = Sim(), terminal
        self._env = SimpleNamespace(env=SimpleNamespace(sim=self.sim))
    def step(self, action):
        self.sim.tick += 1
        return observation(self.sim.tick), 0, self.sim.tick == self.terminal, False, {'is_success': False}
    def close(self):
        pass


class Engine:
    suite = SimpleNamespace(get_task=lambda tid: SimpleNamespace(language='move block'))
    def __init__(self, terminal=230):
        self.terminal = terminal
    def restore(self, *args):
        return Env(self.terminal), observation(0)


class Predictor:
    def __init__(self, terminal=230):
        self.engine, self.inputs = Engine(terminal), []
    def predict(self, obs, instruction, horizon, seed):
        self.inputs.append((copy.deepcopy(obs), seed))
        values = np.arange(horizon, dtype=np.float32)+(seed % 997)*.00001+obs['frame_tick']*.1+obs['robot_state']['eef']['pos'][0]*.01
        actions = np.repeat(values[:, None], 7, axis=1)
        return actions, actions[None], .00001


class Gate:
    def __init__(self, choice):
        self.choice = choice
        self.calls = 0
    def predict(self, state):
        self.calls += 1
        return {'choice': self.choice}


def load_functions(name):
    core = ast.parse((ROOT/'scripts/controller_runtime.py').read_text(encoding='utf-8'))
    nodes = [n for n in core.body if isinstance(n, ast.FunctionDef) and n.name in {'atomic', 'digest_bytes', 'state_text'}]
    tree = ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'))
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'run_episode', 'draw_skip'}]
    scope = dict(Path=Path, np=np, collections=collections, copy=copy, hashlib=hashlib, json=json, time=time,
                 object_positions=lambda env: {'block': np.zeros(3)})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), scope)
    return scope


def compare_traces(a, b):
    with np.load(Path(a['case_path'])/'trace.npz') as x, np.load(Path(b['case_path'])/'trace.npz') as y:
        assert all(np.array_equal(x[k], y[k]) for k in ['actions', 'physics'])
    def calls(result):
        rows = json.loads((Path(result['case_path'])/'calls.json').read_text(encoding='utf-8'))
        return [{k: row.get(k) for k in ['request_tick', 'noise_seed', 'actual_delivery_tick', 'executed_rows']} for row in rows]
    assert calls(a) == calls(b)


def main():
    original = load_functions('controller_runtime.py')['run_episode']
    extended = load_functions('seeded_skip_runtime.py')
    run, draw = extended['run_episode'], extended['draw_skip']
    cases = []
    with tempfile.TemporaryDirectory(prefix='laya_seeded_skip_cpu_') as temporary:
        root = Path(temporary)
        for delay in [1, 4, 5]:
            for h in [50, 100, 200]:
                for fraction, expected in [([0, 1], 'smol70'), ([1, 1], 'fixed_r60')]:
                    phase = f'd{delay}_h{h}_{expected}'
                    # A None gate makes any accidental LAYA invocation fail immediately.
                    a = run(Predictor(), None, root, phase, 0, 0, h, 'seeded_skip1',
                            logical_delay=delay, coin_seed=11131, continue_fraction=fraction)
                    b = original(Predictor(), None, root, phase+'_reference', 0, 0, h, expected, logical_delay=delay)
                    compare_traces(a, b)
                    assert a['gate_calls'] == 0 and a['gate_choices'] == {} and a['laya_gate_seconds'] == 0
                    assert a['gate_seconds'] == a['random_decision_seconds'] >= 0
                    rows = json.loads((Path(a['case_path'])/'random.json').read_text(encoding='utf-8'))
                    assert len(rows) == a['random_decisions'] and all(x['plan_age'] == round(.3*h) for x in rows)
                    cases.append(dict(delay=delay, horizon=h, test='extreme fraction matches '+expected))
                for method in ['smol70', 'fixed_interval', 'naive_k5', 'vlash_style_k5', 'laya_skip1']:
                    phase = f'd{delay}_h{h}_{method}_unchanged'
                    a = run(Predictor(), Gate('continue'), root, phase, 0, 0, h, method, gate_interval=45, logical_delay=delay)
                    b = original(Predictor(), Gate('continue'), root, phase+'_reference', 0, 0, h, method,
                                 gate_interval=45, logical_delay=delay)
                    compare_traces(a, b)
                    assert a['random_decisions'] == a['random_decision_seconds'] == 0 and not a['random_choices']
                    cases.append(dict(delay=delay, horizon=h, test='original '+method+' unchanged'))
        for seed in [11131, 22261]:
            fraction = [93, 232]
            a = run(Predictor(), None, root, f'mixed_{seed}', 2, 10, 50, 'seeded_skip1',
                    logical_delay=4, coin_seed=seed, continue_fraction=fraction)
            b = run(Predictor(), None, root, f'mixed_repeat_{seed}', 2, 10, 50, 'seeded_skip1',
                    logical_delay=4, coin_seed=seed, continue_fraction=fraction)
            compare_traces(a, b)
            rows = json.loads((Path(a['case_path'])/'random.json').read_text(encoding='utf-8'))
            calls = json.loads((Path(a['case_path'])/'calls.json').read_text(encoding='utf-8'))
            assert {x['choice'] for x in rows} == {'continue', 'replan'}
            for index, row in enumerate(rows):
                key = f'laya_random_skip_v1|{seed}|2|10|{index}'
                number = int(hashlib.sha256(key.encode('ascii')).hexdigest()[:16], 16)
                expected = 'continue' if number*232 < 93*2**64 else 'replan'
                assert row['choice'] == expected and row['uniform_u64'] == number
                next_tick = row['tick']+(15 if row['choice'] == 'continue' else 0)
                if next_tick < 230:
                    assert any(c['request_tick'] == next_tick for c in calls)
            cases.append(dict(seed=seed, test='mixed deterministic sequence, full repeat, independent expected requests'))
        for terminal in [5, 15, 16, 20]:
            a = run(Predictor(terminal), None, root, f'terminal_{terminal}', 0, 0, 50, 'seeded_skip1',
                    logical_delay=5, coin_seed=11131, continue_fraction=[0, 1])
            b = original(Predictor(terminal), None, root, f'terminal_ref_{terminal}', 0, 0, 50, 'smol70', logical_delay=5)
            compare_traces(a, b)
            assert a['random_decisions'] == int(terminal > 15)
            cases.append(dict(terminal=terminal, test='early termination and paid pending calls match the reference'))
    # The same draw uses no H or state text; changing the fraction changes only the cutoff.
    x, y = draw(11131, 2, 10, 5, [93, 232]), draw(11131, 2, 10, 5, [12, 64])
    assert x['uniform_u64'] == y['uniform_u64'] and x['draw_key'] == y['draw_key']
    cases.append(dict(test='horizon-independent draw key; separate prescribed rational cutoff'))
    result = dict(checks=len(cases), errors=0, GPU_used=False, cases=cases,
                  source_sha256={name: hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
                                 for name in ['controller_runtime.py', 'seeded_skip_runtime.py']},
                  scope='Actual function ASTs in a visible seed/tick environment. GPU replay qualification remains required.')
    (ROOT/'checks/seeded_skip_cpu.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(checks=len(cases), errors=0, source_sha256=result['source_sha256'])))


if __name__ == '__main__':
    main()
