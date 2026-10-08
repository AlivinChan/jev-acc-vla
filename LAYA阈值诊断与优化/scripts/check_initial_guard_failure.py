"""CPU fault injection: reject changed input or output before returning an action block."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import numpy as np
import torch
import verified_runtime as verified


def main():
    assert not torch.cuda.is_initialized()
    records = []
    old = verified.runtime.Predictor.predict
    expected_input = {'/batch/observation.state': np.array([[0.25]], dtype=np.float32)}
    expected_action = np.zeros((50, 7), dtype=np.float32)
    expected_norm = expected_action[None].copy()
    try:
        for corruption in ['none', 'input', 'actions', 'normalized']:
            with tempfile.TemporaryDirectory(prefix='laya_guard_cpu_') as temp:
                predictor = object.__new__(verified.VerifiedPredictor)
                predictor.folder = Path(temp)
                predictor.guard_records = []
                predictor.last_restore = {'test_physics': 'identical'}
                predictor.capture_seconds = .001
                predictor.active = dict(task=0, state=10, horizon=50, candidate='fault_injection')
                predictor.references = {(0, 10): {'horizons': {50: dict(
                    processed_input=verified.receipt(expected_input),
                    prediction=verified.receipt(dict(actions=expected_action, normalized=expected_norm)))}}}

                def fake(self, obs, instruction, horizon, seed):
                    self.last_input_arrays = {k: v.copy() for k, v in expected_input.items()}
                    a, n = expected_action.copy(), expected_norm.copy()
                    if corruption == 'input':
                        self.last_input_arrays['/batch/observation.state'][0, 0] += .01
                    elif corruption == 'actions':
                        a[0, 0] += .01
                    elif corruption == 'normalized':
                        n[0, 0, 0] += .01
                    return a, n, .2

                verified.runtime.Predictor.predict = fake
                returned = False
                raised = False
                try:
                    predictor.predict({'image': np.zeros((2, 2, 3), dtype=np.uint8)}, 'test', 50, 20271007)
                    returned = True
                except AssertionError as error:
                    assert 'First prediction differs' in str(error)
                    raised = True
                if corruption == 'none':
                    assert returned and not raised and predictor.active is None
                else:
                    assert raised and not returned and predictor.active is not None
                    assert (predictor.folder/'failed_prediction.npz').exists()
                    failure = json.loads((predictor.folder/'failure_0001.json').read_text())
                    assert failure['reason'] == 'first_prediction_mismatch'
                    assert (predictor.folder/'failure_0001_raw_observation.npz').exists()
                    assert (predictor.folder/'failure_0001_processed_input.npz').exists()
                records.append(dict(corruption=corruption, returned=returned, rejected=raised, correct=True))
    finally:
        verified.runtime.Predictor.predict = old
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(checks=4, errors=0, cuda_initialized=False, cases=records,
                          verified_runtime_sha256=hashlib.sha256(Path(verified.__file__).read_bytes()).hexdigest(),
                          test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
