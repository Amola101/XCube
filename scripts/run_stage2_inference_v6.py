"""
Same idea as run_stage2_inference_v5.py, but for v6 (material conditioning,
retrained on the new corr_medium/Step1 dataset instead of the original
TEUNet dataset -- see PROGRESS.md). Baseline comparison is against Step1's
own prediction (corr_medium's analog of "TEUNet's raw output"), not TEUNet.

Since corr_medium's test.lst isn't index-tiered the same fixed way the
original TEUNet split was, this computes each sample's own Dice score
between Step1's prediction and GT to bucket it into good/moderate/poor,
matching the exact thresholds used when the dataset was built.

Usage:
    python scripts/run_stage2_inference_v6.py [indices...]
"""
import sys
sys.path.insert(0, '/home/ameliacatala/Documents/XCube')
import pickle
import importlib
from pathlib import Path

import torch

from xcube.utils import exp
from xcube.data.base import DatasetSpec as DS

_orig_load = torch.load
def _trusted_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _trusted_load

custom_pickle = pickle
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "featurevdb._Cpp":
            module = "fvdb._Cpp"
        return super().find_class(module, name)
custom_pickle.Unpickler = CustomUnpickler

VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/Diffusion_stage2_v6_material/version_0')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr_corr_medium')

# Fixed, already-fully-written checkpoint file -- NOT last.ckpt, since
# training is still running in the background and actively rewriting that
# file; reading it mid-write could load a corrupted/partial checkpoint.
args = sys.argv[1:]
ckpt_args = [a for a in args if a.endswith('.ckpt')]
idx_args = [a for a in args if not a.endswith('.ckpt')]

if ckpt_args:
    CKPT_PATH = ckpt_args[0]
else:
    ckpts = sorted((VERSION_DIR / 'checkpoints').glob('epoch=*.ckpt'), key=lambda p: p.stat().st_mtime)
    CKPT_PATH = str(ckpts[-1])
print('Using checkpoint:', CKPT_PATH)

model_args = exp.parse_config_yaml(VERSION_DIR / 'hparams.yaml')
net_module = importlib.import_module("xcube.models." + model_args.model).Model
model = net_module.load_from_checkpoint(CKPT_PATH, hparams=model_args)
model = model.cuda().eval()

test_lst = DATA_DIR / 'test.lst'
stems = [s for s in test_lst.read_text().split('\n') if s]
print(f'{len(stems)} test samples total.\n')

indices = [int(a) for a in idx_args] if idx_args else list(range(len(stems)))

def grid_iou(gt, pd):
    idx = pd.ijk_to_index(gt.ijk)
    upi = (pd.num_voxels + gt.num_voxels).cpu().numpy().tolist()
    inter = torch.sum(idx[0].jdata >= 0).item()
    return inter / (upi[0] - inter + 1e-6)

def grid_dice(gt, pd):
    idx = pd.ijk_to_index(gt.ijk)
    upi = (pd.num_voxels + gt.num_voxels).cpu().numpy().tolist()
    inter = torch.sum(idx[0].jdata >= 0).item()
    return 2 * inter / (upi[0] + 1e-6)

def assign_tier(score):
    if score > 0.8:
        return 'good'
    elif score >= 0.5:
        return 'moderate'
    return 'poor'

results = []
for i in indices:
    stem = stems[i]
    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    step1_grid = sample['input_grid'].to('cuda')
    step1_material = sample['input_material'].to('cuda')

    tier = assign_tier(grid_dice(gt_grid, step1_grid))

    with torch.no_grad():
        cond_latent = model.encode_cond_grid(step1_grid, step1_material)
        res, output_x = model.evaluation_api(
            batch={DS.COND_PC: step1_grid, DS.COND_MATERIAL: step1_material}, grids=cond_latent.grid,
            use_ddim=True, ddim_step=100, use_ema=False)
    pred_grid = res.structure_grid[0]

    iou_pred = grid_iou(gt_grid, pred_grid)
    iou_step1 = grid_iou(gt_grid, step1_grid)
    results.append((tier, iou_pred, iou_step1))
    print(f'[{tier:8s}] {stem:24s} IoU(pred,GT)={iou_pred:.3f}  IoU(Step1,GT)={iou_step1:.3f}  diff={iou_pred-iou_step1:+.3f}')

print()
for tier in ['good', 'moderate', 'poor']:
    sub = [(p, t) for tr, p, t in results if tr == tier]
    if not sub:
        continue
    avg_pred = sum(p for p, t in sub) / len(sub)
    avg_step1 = sum(t for p, t in sub) / len(sub)
    print(f'{tier:8s} (n={len(sub)}): avg IoU(pred,GT)={avg_pred:.3f}  avg IoU(Step1,GT)={avg_step1:.3f}  diff={avg_pred-avg_step1:+.3f}')

avg_pred_all = sum(p for _, p, t in results) / len(results)
avg_step1_all = sum(t for _, p, t in results) / len(results)
print(f'\nOVERALL (n={len(results)}): avg IoU(pred,GT)={avg_pred_all:.3f}  avg IoU(Step1,GT)={avg_step1_all:.3f}  diff={avg_pred_all-avg_step1_all:+.3f}')
