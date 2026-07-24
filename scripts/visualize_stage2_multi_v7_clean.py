"""
Clean, chrome-free tier comparison for the Stage 2 diffusion model (v7,
corr_medium/Step1 dataset, corrupted conditioning -- the latest, most
complete Stage 2 attempt on this data). One representative good/moderate/
poor tier sample each, rendered in the same style as
visualize_vae_reconstruction.py (white background, no axes/grid, consistent
color, voxel-count captions) so it composes cleanly into a diagram.

Usage:
    python scripts/visualize_stage2_multi_v7_clean.py [good_idx moderate_idx poor_idx]
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

VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/Diffusion_stage2_v7_corrupted/version_0')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr_corr_medium')

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

def grid_iou(gt, pd):
    idx = pd.ijk_to_index(gt.ijk)
    upi = (pd.num_voxels + gt.num_voxels).cpu().numpy().tolist()
    inter = torch.sum(idx[0].jdata >= 0).item()
    return inter / (upi[0] - inter + 1e-6)

def assign_tier(score):
    if score > 0.8:
        return 'good'
    elif score >= 0.5:
        return 'moderate'
    return 'poor'

def render_clean(ax, dense, shape, color='#1f77b4', edge_color='k'):
    ax.voxels(dense, facecolors=color, edgecolor=edge_color, linewidth=0.1)
    ax.set_box_aspect(shape)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    ax.set_facecolor('white')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor('white')
        pane.set_edgecolor('white')
        pane.set_alpha(1.0)

if len(sys.argv) > 3:
    chosen = {'good': int(sys.argv[1]), 'moderate': int(sys.argv[2]), 'poor': int(sys.argv[3])}
else:
    print('Scanning test samples to find one representative good/moderate/poor example...')
    chosen = {}
    for i, stem in enumerate(stems):
        sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
        tier = assign_tier(grid_dice(sample['target_grid'], sample['input_grid']))
        if tier not in chosen:
            chosen[tier] = i
        if len(chosen) == 3:
            break
    print('Chosen indices:', chosen)

tiers_order = ['good', 'moderate', 'poor']
out_dir = Path('/home/ameliacatala/Documents/XCube')

fig = plt.figure(figsize=(15, 5 * 3), facecolor='white')
col_titles = ['Step1 input (conditioning)', 'Stage 2 diffusion output', 'Ground truth']

for row, tier in enumerate(tiers_order):
    idx = chosen[tier]
    stem = stems[idx]
    print(f'[{tier}] sample #{idx}: {stem}')

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

    step1_ijk, pred_ijk, gt_ijk = to_ijk(step1_grid), to_ijk(pred_grid), to_ijk(gt_grid)
    mins = gt_ijk.min(axis=0)
    maxs = gt_ijk.max(axis=0)
    shape = tuple(maxs - mins + 1)

    iou_step1 = grid_iou(gt_grid, step1_grid)
    iou_pred = grid_iou(gt_grid, pred_grid)
    ious = [iou_step1, iou_pred, 1.0]

    for col, (ijk, base_title, iou) in enumerate(zip([step1_ijk, pred_ijk, gt_ijk], col_titles, ious)):
        ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection='3d')
        render_clean(ax, to_dense(ijk, mins, shape), shape)
        title = base_title if row == 0 else ''
        iou_str = '' if col == 2 else f'  (IoU={iou:.2f})'
        ax.set_title(f'{title}\n[{tier}] {ijk.shape[0]} voxels{iou_str}'.strip(), fontsize=10)

plt.tight_layout()
out_path = out_dir / 'stage2_v7_tier_comparison_clean.png'
plt.savefig(out_path, dpi=150, facecolor='white')
print('Saved to', out_path)
