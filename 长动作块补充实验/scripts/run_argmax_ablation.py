"""Post-pilot exploratory LAYA tau=0 ablation, sharing the primary rollout code.

Runs 90 episodes (30 initial states x H50/100/200), plus six SmolVLA
qualification calls and one separately accounted Laya warmup. No new training.
Raw three-way ``uncertain`` still follows the original Smol70 fallback.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import collections
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import traceback

import main_experiment as primary

BASE = primary.BASE
ROOT = primary.ROOT
HORIZONS = primary.HORIZONS
atomic = primary.atomic
np = primary.np
torch = primary.torch


class ArgmaxLayaClient(primary.LayaClient):
    """Only the launched service and response contract differ from the parent."""
    def __init__(self, folder):
        self.folder = folder
        folder.mkdir()
        (folder / 'requests').mkdir()
        (folder / 'responses').mkdir()
        self.log = (folder / 'service.log').open('w')
        self.process = subprocess.Popen(
            [str(BASE / 'envs/laya/bin/python'), str(ROOT / 'scripts/laya_argmax_service.py'),
             '--folder', str(folder)], stdout=self.log, stderr=subprocess.STDOUT)
        self.index = 0
        start = time.monotonic()
        try:
            while not (folder / 'ready.json').exists():
                if self.process.poll() is not None:
                    raise RuntimeError('Argmax Laya service exited before ready; see service.log')
                if time.monotonic() - start > 180:
                    raise TimeoutError('Argmax Laya loading timeout')
                time.sleep(.02)
            self.ready = json.loads((folder / 'ready.json').read_text())
            assert self.ready['threshold'] == 0.0
            assert self.ready['frozen'] and self.ready['training'] is False
        except BaseException:
            self.close()
            raise

    def predict(self, state, warmup=False):
        response = super().predict(state, warmup=warmup)
        assert response['minimum_probability'] == 0.0, 'Wrong confidence threshold'
        assert response['choice'] == response['raw_choice'], 'Argmax choice was overridden'
        return response


def make_schedule():
    schedule = []
    for tid in range(10):
        for sid in [0, 1, 2]:
            # Preserve each initial state's relative H order from the primary run.
            cells = [(h, m) for h in HORIZONS for m in primary.METHODS]
            random.Random(20261007 + tid * 10 + sid).shuffle(cells)
            schedule.extend(('argmax', tid, sid, h, m) for h, m in cells if m == 'laya')
    assert len(schedule) == 90 and len(set(schedule)) == 90
    return schedule


def freeze_snapshot(predictor, fingerprint):
    policy = predictor.engine.policy
    dtype_counts = collections.Counter()
    parameter_count = 0
    for parameter in policy.parameters():
        dtype_counts[str(parameter.dtype)] += parameter.numel()
        parameter_count += parameter.numel()
    snapshot = dict(state_dict_sha256=fingerprint, parameters_by_dtype=dict(dtype_counts),
                    parameter_count=parameter_count,
                    all_parameters_frozen=all(not p.requires_grad for p in policy.parameters()),
                    all_parameter_grads_none=all(p.grad is None for p in policy.parameters()),
                    all_modules_eval=all(not module.training for module in policy.modules()),
                    training_steps=0, optimizer_steps=0)
    assert snapshot['all_parameters_frozen'] and snapshot['all_parameter_grads_none']
    assert snapshot['all_modules_eval']
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    assert out.is_relative_to(ROOT.resolve()) and out.name == 'smol'
    assert out.parent.name.startswith('argmax_attempt_')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    schedule = make_schedule()
    atomic(out / 'schedule.json', schedule)
    manifest = dict(
        started_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
        torch=torch.__version__, python=platform.python_version(),
        gpu=torch.cuda.get_device_name(), logical_delay_steps=1,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in list((ROOT / 'scripts').glob('*.py')) +
                       list((ROOT / 'source_snapshot').glob('*.py'))},
        suite='libero_spatial', tasks=list(range(10)), state_ids=[0, 1, 2],
        horizons=HORIZONS, methods=['laya'], phase='argmax', planned_episodes=90,
        qualification_vla_calls=6, qualification_nfe=60, warmup_gate_calls=1,
        minimum_probability=0.0, exploratory=True,
        trigger_basis='posthoc_pilot_threshold_abstention',
        ablation='Remove probability cutoff only; retain original three-way labels, prompt, and rollout',
        semantic_uncertain_behavior='Original remaining/H <= 0.7 fallback remains',
        max_steps=230, checkpoint_training_horizon=50, native_length_extrapolation=True,
        real_time=False, no_optimizer=True, gate_interval=5,
        inference_overlap=False, privileged_gate_state=True)
    atomic(out / 'run_manifest.json', manifest)
    atomic(out / 'status.json', dict(phase='qualifying', completed=0, planned=90,
                                   minimum_probability=0.0, exploratory=True))
    gate = None
    predictor = None
    all_results = []
    try:
        load_start = time.perf_counter()
        predictor = primary.Predictor()
        manifest['smolvla_load_seconds'] = time.perf_counter() - load_start
        manifest['parameter_sha256_before'] = predictor.before_hash
        manifest['checkpoint_path'] = str(predictor.engine.model_path)
        manifest['original_checkpoint_config'] = json.loads(
            (predictor.engine.model_path / 'config.json').read_text())
        manifest['effective_settings'] = dict(rtc_enabled=False, num_flow_steps=10,
                                               native_lengths=HORIZONS, noise_dtype='torch.float32',
                                               gate_minimum_probability=0.0)
        manifest['installed_source_sha256'] = {}
        for name in ['lerobot.policies.smolvla.modeling_smolvla',
                     'lerobot.policies.smolvla.configuration_smolvla',
                     'lerobot.policies.smolvla.smolvlm_with_expert', 'lerobot.envs.libero']:
            module = sys.modules.get(name)
            if module is not None and getattr(module, '__file__', None):
                source = Path(module.__file__)
                manifest['installed_source_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        before = freeze_snapshot(predictor, predictor.before_hash)
        manifest['parameters_by_dtype'] = before['parameters_by_dtype']
        atomic(out / 'freeze_before.json', before)
        atomic(out / 'run_manifest.json', manifest)
        env, obs = predictor.engine.restore(0, 0, np.empty((0, 7)))
        qualification = []
        try:
            instruction = predictor.engine.suite.get_task(0).language
            for horizon in HORIZONS:
                a, n, cost = predictor.predict(obs, instruction, horizon, 20261007)
                b, n2, cost2 = predictor.predict(obs, instruction, horizon, 20261007)
                assert np.array_equal(a, b) and np.array_equal(n, n2)
                np.savez_compressed(out / f'qualification_h{horizon}.npz', actions=a, normalized=n)
                qualification.append(dict(horizon=horizon, native_shape=list(a.shape),
                                          repeated_exact=True, warm_seconds=cost,
                                          hot_seconds=cost2, finite=True))
            assert predictor.calls == 6
            atomic(out / 'qualification.json', qualification)
            gate = ArgmaxLayaClient(out / 'laya_ipc')
            warm_text = primary.state_text(instruction, obs, env, primary.object_positions(env),
                                           list(a), 0, HORIZONS[-1])
            warm = gate.predict(warm_text, warmup=True)
            atomic(out / 'laya_warmup.json', warm)
            manifest['laya_ready'] = gate.ready
        finally:
            env.close()
        atomic(out / 'run_manifest.json', manifest)
        initial_hashes = {}
        for index, (phase, tid, sid, horizon, method) in enumerate(schedule):
            result = primary.run_episode(predictor, gate, out, phase, tid, sid, horizon, method)
            key = (tid, sid)
            if key in initial_hashes:
                assert result['initial_state_sha256'] == initial_hashes[key], 'Paired initial state differs'
            else:
                initial_hashes[key] = result['initial_state_sha256']
            all_results.append(result)
            with (out / 'episodes.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            atomic(out / 'status.json', dict(phase='running', completed=index + 1, planned=90,
                                             latest=result, minimum_probability=0.0, exploratory=True,
                                             updated_utc=datetime.now(timezone.utc).isoformat()))
            print(json.dumps({'completed': index + 1, 'planned': 90, **result}), flush=True)
        after_hash = primary.parameter_digest(predictor.engine.policy)
        after = freeze_snapshot(predictor, after_hash)
        atomic(out / 'freeze_after.json', after)
        assert before == after, 'Frozen SmolVLA changed during exploratory ablation'
        assert predictor.calls == 6 + sum(row['vla_calls'] for row in all_results)
        assert gate.index == 1 + sum(row['gate_calls'] for row in all_results)
        manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                        parameter_sha256_after=after_hash, frozen_parameters_unchanged=True,
                        total_vla_calls_including_qualification=predictor.calls,
                        total_prediction_seconds_including_qualification=predictor.predict_seconds,
                        total_gate_calls_including_warmup=gate.index)
        atomic(out / 'run_manifest.json', manifest)
        atomic(out / 'status.json', dict(phase='completed', completed=90, planned=90,
                                        frozen_parameters_unchanged=True,
                                        minimum_probability=0.0, exploratory=True))
    except BaseException as error:
        atomic(out / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                                          completed=len(all_results),
                                          vla_calls=None if predictor is None else predictor.calls))
        atomic(out / 'status.json', dict(phase='failed', completed=len(all_results), planned=90,
                                        minimum_probability=0.0, exploratory=True, error=repr(error)))
        raise
    finally:
        if gate is not None:
            gate.close()


if __name__ == '__main__':
    main()
