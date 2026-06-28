"""
Quick no-retrain experiment: dilate TEUNet's grid (pad with a margin of
neighboring cells) before encoding/feeding it to the diffusion model, instead
of using its exact (possibly broken) footprint. Tests whether this gives the
decoder enough room to recover structure TEUNet's reconstruction lost.

Usage:
    python scripts/test_dilation_fix.py [kernel_size]
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

VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/Diffusion_stage2/version_2')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr')
KERNEL_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 3

model_args = exp.parse_config_yaml(VERSION_DIR / 'hparams.yaml')
net_module = importlib.import_module("xcube.models." + model_args.model).Model
model = net_module.load_from_checkpoint(VERSION_DIR / 'checkpoints' / 'last.ckpt', hparams=model_args)
model = model.cuda().eval()

stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
print(f'Dilation kernel_size={KERNEL_SIZE}\n')

# Same representative spread across tiers used in the earlier preliminary check.
indices = [0, 30, 60, 90, 120, 130, 145, 160, 175, 185, 195]
tier_of = lambda i: 'good' if i < 126 else ('moderate' if i < 172 else 'poor')

def grid_iou(gt, pd):
    idx = pd.ijk_to_index(gt.ijk)
    upi = (pd.num_voxels + gt.num_voxels).cpu().numpy().tolist()
    inter = torch.sum(idx[0].jdata >= 0).item()
    return inter / (upi[0] - inter + 1e-6)

results = []
for i in indices:
    stem = stems[i]
    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    teunet_grid = sample['input_grid'].to('cuda')
    dilated_grid = teunet_grid.conv_grid(KERNEL_SIZE, 1)

    with torch.no_grad():
        # Baseline: no dilation (what we already measured before).
        cond_latent = model.vae._encode({DS.INPUT_PC: teunet_grid}, use_mode=True)
        res, _ = model.evaluation_api(
            batch={DS.COND_PC: teunet_grid}, grids=cond_latent.grid,
            use_ddim=True, ddim_step=100, use_ema=False)
        pred_grid = res.structure_grid[0]

        # Dilated: encode/condition/topology all from the padded TEUNet grid.
        cond_latent_dil = model.vae._encode({DS.INPUT_PC: dilated_grid}, use_mode=True)
        res_dil, _ = model.evaluation_api(
            batch={DS.COND_PC: dilated_grid}, grids=cond_latent_dil.grid,
            use_ddim=True, ddim_step=100, use_ema=False)
    pred_grid_dil = res_dil.structure_grid[0]

    iou_teunet = grid_iou(gt_grid, teunet_grid)
    iou_pred = grid_iou(gt_grid, pred_grid)
    iou_pred_dil = grid_iou(gt_grid, pred_grid_dil)
    tier = tier_of(i)
    results.append((tier, iou_teunet, iou_pred, iou_pred_dil))
    print(f'[{tier:8s}] {stem:20s} TEUNet={iou_teunet:.3f}  baseline_pred={iou_pred:.3f}  dilated_pred={iou_pred_dil:.3f}  '
          f'dilation_diff={iou_pred_dil-iou_pred:+.3f}')

print()
for tier in ['good', 'moderate', 'poor']:
    sub = [(t, p, d) for tr, t, p, d in results if tr == tier]
    if not sub:
        continue
    avg_t = sum(t for t, p, d in sub) / len(sub)
    avg_p = sum(p for t, p, d in sub) / len(sub)
    avg_d = sum(d for t, p, d in sub) / len(sub)
    print(f'{tier:8s} (n={len(sub)}): TEUNet={avg_t:.3f}  baseline_pred={avg_p:.3f}  dilated_pred={avg_d:.3f}  dilation_diff={avg_d-avg_p:+.3f}')
