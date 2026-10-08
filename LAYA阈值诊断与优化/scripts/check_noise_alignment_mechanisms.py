"""Exercise actual controller ASTs using visible tick and seed dependent predictions."""
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
    def __init__(self):
        self.sim = Sim()
        self._env = SimpleNamespace(env=SimpleNamespace(sim=self.sim))
    def step(self, action):
        self.sim.tick += 1
        return observation(self.sim.tick), 0, self.sim.tick == 20, False, {'is_success': False}
    def close(self):
        pass


class Engine:
    suite = SimpleNamespace(get_task=lambda tid: SimpleNamespace(language='move block'))
    def restore(self, *args):
        return Env(), observation(0)


class Predictor:
    def __init__(self):
        self.engine = Engine()
        self.inputs = []
    def predict(self, obs, instruction, horizon, seed):
        self.inputs.append((copy.deepcopy(obs), seed))
        values = np.arange(horizon, dtype=np.float32) + (seed % 997)*.00001 + obs['frame_tick']*.1 + obs['robot_state']['eef']['pos'][0]*.01
        actions = np.repeat(values[:, None], 7, axis=1)
        return actions, actions[None], .001


def arrays(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            result.update(arrays(v, prefix+'/'+k))
        return result
    return {prefix: np.asarray(value).copy()}


def receipt(values):
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in values.items()}


def load_functions(name):
    originals = ast.parse((ROOT/'scripts/controller_runtime.py').read_text(encoding='utf-8'))
    nodes = [n for n in originals.body if isinstance(n, ast.FunctionDef) and n.name in {'atomic', 'digest_bytes', 'state_text'}]
    source = ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'))
    nodes += [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'run_episode']
    scope = dict(Path=Path, np=np, collections=collections, copy=copy, hashlib=hashlib, json=json, time=time,
                 object_positions=lambda env: {'block': np.zeros(3)}, arrays=arrays, receipt=receipt)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), scope)
    return scope['run_episode']


def main():
    original = load_functions('controller_runtime.py')
    extended = load_functions('mechanism_runtime.py')
    cases = []
    variants = [('naive', 'naive_k5', {}), ('original_vlash', 'vlash_style_k5', {}),
                ('matched_vlash', 'vlash_style_k5', dict(noise_clock='image')),
                ('stale_row0', 'vlash_style_k5', dict(noise_clock='image', alignment_mode='stale_state'))]
    with tempfile.TemporaryDirectory(prefix='laya_alignment_cpu_') as temporary:
        root = Path(temporary)
        for d in [1, 4, 5]:
            for h in [50, 100, 200]:
                results = {}
                for name, method, extra in variants:
                    predictor = Predictor()
                    result = extended(predictor, None, root, f'{name}_d{d}_h{h}', 0, 0, h, method, logical_delay=d, **extra)
                    folder = Path(result['case_path'])
                    calls = json.loads((folder/'calls.json').read_text(encoding='utf-8'))
                    for c, (obs, seed) in zip(calls[1:], predictor.inputs[1:]):
                        boundary = c['intended_delivery_tick']
                        image_tick = boundary-d
                        state_tick = boundary if method == 'vlash_style_k5' and name != 'stale_row0' else image_tick
                        seed_tick = boundary if name == 'original_vlash' else image_tick
                        assert obs['frame_tick'] == image_tick and obs['robot_state']['eef']['pos'][0] == state_tick
                        assert seed == c['noise_seed'] == 20261007+seed_tick and c['noise_seed_tick'] == seed_tick
                        if c.get('actual_delivery_tick') is not None:
                            assert c['actual_delivery_tick'] == boundary
                            assert c['executed_rows'][0] == (d if name == 'naive' else 0)
                        else:
                            assert boundary == 20 and not c['executed_rows'] and name == 'naive'
                    if not extra:
                        reference = original(Predictor(), None, root, f'reference_{name}_d{d}_h{h}', 0, 0, h, method, logical_delay=d)
                        with np.load(folder/'trace.npz') as a, np.load(Path(reference['case_path'])/'trace.npz') as b:
                            assert all(np.array_equal(a[k], b[k]) for k in ['actions', 'physics'])
                    results[name] = folder
                    cases.append(dict(delay=d, horizon=h, variant=name, all_update_clocks_and_rows_correct=True,
                                      default_path_unchanged=not extra))
                with np.load(results['naive']/'chunk_001.npz') as a, np.load(results['stale_row0']/'chunk_001.npz') as b:
                    assert all(np.array_equal(a[k], b[k]) for k in ['actions', 'normalized'])
                with np.load(results['naive']/'trace.npz') as a:
                    for folder in results.values():
                        with np.load(folder/'trace.npz') as b:
                            assert np.array_equal(a['actions'][:5], b['actions'][:5])
    result = dict(cases=cases, checks=len(cases), errors=0, GPU_used=False,
                  source_sha256={name: hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
                                 for name in ['controller_runtime.py', 'mechanism_runtime.py']},
                  scope='Actual functions, weight-free environment; real GPU pilot remains required')
    (ROOT/'checks/noise_alignment_cpu.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(checks=len(cases), errors=0, source_sha256=result['source_sha256'])))


if __name__ == '__main__':
    main()
