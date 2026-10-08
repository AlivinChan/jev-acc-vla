"""Verify the installed action mask's directional dependency on CPU, without loading weights."""
from pathlib import Path
import ast
import argparse
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=ROOT / 'evidence/vla_utils.py')
args = parser.parse_args()
path = args.source
torch.set_num_threads(1)
source = path.read_text(encoding='utf-8')
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'make_att_2d_masks')
scope = dict(torch=torch, Tensor=torch.Tensor)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
assert not torch.cuda.is_initialized()
masks = {}
checks = []
for h in [50, 100, 200]:
    masks[h] = scope['make_att_2d_masks'](torch.ones((1, h), dtype=torch.bool), torch.ones((1, h)))
    assert torch.equal(masks[h][0], torch.tril(torch.ones((h, h), dtype=torch.bool)))
    checks.append(dict(horizon=h, allowed_action_dependencies='current and earlier rows only', future_rows_visible=0))
for shorter, longer in [(50, 100), (50, 200), (100, 200)]:
    assert torch.equal(masks[shorter], masks[longer][:, :shorter, :shorter])
    assert not masks[longer][:, :shorter, shorter:].any()
    checks.append(dict(shorter=shorter, longer=longer, common_action_mask_exact=True,
                       appended_actions_visible_to_common_prefix=False))
assert not torch.cuda.is_initialized()
result = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), torch_version=torch.__version__,
              checks=checks, errors=0, cuda_initialized=False,
              scope='Actual installed make_att_2d_masks with the all-one suffix mask from embed_suffix. This proves mask directionality, not bitwise invariance of the whole floating-point network or long-tail action quality.')
(ROOT / 'checks/action_attention_mask.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result))
