"""
Same idea as visualize_stage2_multi.py, but for v6 (material conditioning,
corr_medium/Step1 dataset instead of TEUNet). Baseline column is Step1's own
prediction, not TEUNet's -- Step1's own failure modes have never actually
been looked at visually before now (the earlier Pattern A-D audit was
TEUNet-specific and should NOT be assumed to carry over here).

By default renders every "poor" tier test sample (computed on the fly via
per-sample Dice, since corr_medium's test.lst isn't fixed-index-tiered the
same way the original TEUNet split was) plus a couple of good/moderate ones
for context.

Usage:
    python scripts/visualize_stage2_multi_v6.py [idx1 idx2 idx3 ...]
"""
import sys
sys.path.insert(0, '/home/ameliacatala/Documents/XCube')
import pickle
import importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Fixed, already-fully-written checkpoint -- not last.ckpt, since training is
# still actively running and rewriting that file.
ckpts = sorted((VERSION_DIR / 'checkpoints').glob('epoch=*.ckpt'), key=lambda p: p.stat().st_mtime)
CKPT_PATH = str(ckpts[-1])
print('Using checkpoint:', CKPT_PATH)

model_args = exp.parse_config_yaml(VERSION_DIR / 'hparams.yaml')
net_module = importlib.import_module("xcube.models." + model_args.model).Model
model = net_module.load_from_checkpoint(CKPT_PATH, hparams=model_args)
model = model.cuda().eval()

stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]

def to_ijk(grid):
    return grid.ijk[0].jdata.cpu().numpy()

def to_dense(ijk, mins, shape):
    arr = np.zeros(shape, dtype=bool)
    local = ijk - mins
    valid = np.all((local >= 0) & (local < np.array(shape)), axis=1)
    local = local[valid]
    arr[local[:, 0], local[:, 1], local[:, 2]] = True
    return arr

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

if len(sys.argv) > 1:
    indices = [int(a) for a in sys.argv[1:]]
    tiers = {}
    for i in indices:
        stem = stems[i]
        sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
        tiers[i] = assign_tier(grid_dice(sample['target_grid'], sample['input_grid']))
else:
    # Scan every test sample's Step1-vs-GT Dice to find all "poor" tier ones
    # (cheap: no model inference, just occupancy overlap), plus a couple of
    # good/moderate for context.
    print('Scanning all test samples to find poor-tier cases...')
    tiers = {}
    for i, stem in enumerate(stems):
        sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
        tiers[i] = assign_tier(grid_dice(sample['target_grid'], sample['input_grid']))
    poor_indices = [i for i, t in tiers.items() if t == 'poor']
    good_indices = [i for i, t in tiers.items() if t == 'good'][:2]
    mod_indices = [i for i, t in tiers.items() if t == 'moderate'][:2]
    indices = good_indices + mod_indices + poor_indices
    print(f'Found {len(poor_indices)} poor-tier samples; rendering {len(indices)} total '
          f'({len(good_indices)} good, {len(mod_indices)} moderate, {len(poor_indices)} poor).')

fig = plt.figure(figsize=(15, 5 * len(indices)))
col_titles = ['Step1 Input (flawed)', 'XCube Stage 2 v6 Output', 'Ground Truth']

for row, idx in enumerate(indices):
    stem = stems[idx]
    tier = tiers[idx]
    print(f'[{row+1}/{len(indices)}] sample #{idx} ({tier}): {stem}')

    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    step1_grid = sample['input_grid'].to('cuda')
    step1_material = sample['input_material'].to('cuda')

    with torch.no_grad():
        cond_latent = model.encode_cond_grid(step1_grid, step1_material)
        res, output_x = model.evaluation_api(
            batch={DS.COND_PC: step1_grid, DS.COND_MATERIAL: step1_material}, grids=cond_latent.grid,
            use_ddim=True, ddim_step=100, use_ema=False)
    pred_grid = res.structure_grid[0]

    step1_ijk = to_ijk(step1_grid)
    pred_ijk = to_ijk(pred_grid)
    gt_ijk = to_ijk(gt_grid)

    mins = gt_ijk.min(axis=0)
    maxs = gt_ijk.max(axis=0)
    shape = tuple(maxs - mins + 1)

    for col, (ijk, base_title) in enumerate(zip([step1_ijk, pred_ijk, gt_ijk], col_titles)):
        ax = fig.add_subplot(len(indices), 3, row * 3 + col + 1, projection='3d')
        ax.voxels(to_dense(ijk, mins, shape), edgecolor='k', linewidth=0.1)
        title = base_title if row == 0 else ''
        ax.set_title(f'{title}\n[{tier}] idx={idx} {ijk.shape[0]} voxels'.strip())
        ax.set_box_aspect(shape)
        ax.view_init(elev=20, azim=-60)

plt.tight_layout()
out_path = '/home/ameliacatala/Documents/XCube/stage2_v6_visual_comparison_multi.png'
plt.savefig(out_path, dpi=130)
print('Saved to', out_path)
