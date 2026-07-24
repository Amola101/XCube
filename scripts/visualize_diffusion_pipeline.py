"""
Diffusion-model analog of visualize_vae_reconstruction.py: clean, chrome-free
3D renders of one GPR pipe sample as it passes through the trained Stage 2
diffusion model (v7, corr_medium/Step1, corrupted conditioning) -- the
flawed Step1 prediction that's fed in as conditioning, the encoded
conditioning latent (colored by its own feature content, same PCA->HSV
technique as the VAE embedding panel), the coarse structure the
diffusion-guided decode predicts first, the final fine-resolution output,
and (new relative to the VAE version, since here input != target) the real
ground truth for comparison. Same test sample (idx=0, 16,105 GT voxels) as
visualize_vae_reconstruction.py's default, for direct visual continuity.

Usage:
    python scripts/visualize_diffusion_pipeline.py [test_set_index]
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
from matplotlib.colors import hsv_to_rgb

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

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
stem = stems[idx]
print('Visualizing sample:', stem)

sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
gt_grid = sample['target_grid'].to('cuda')
step1_grid = sample['input_grid'].to('cuda')
step1_material = sample['input_material'].to('cuda')

with torch.no_grad():
    cond_latent = model.encode_cond_grid(step1_grid, step1_material)
    res, output_x = model.evaluation_api(
        batch={DS.COND_PC: step1_grid, DS.COND_MATERIAL: step1_material}, grids=cond_latent.grid,
        use_ddim=True, ddim_step=100, use_ema=False)
tree = res.structure_grid  # 0 = finest (final output), 1 = coarser

def to_ijk(grid):
    return grid.ijk[0].jdata.cpu().numpy()

before_ijk = to_ijk(step1_grid)            # Step1's flawed input (conditioning)
during_ijk = to_ijk(tree[1])               # coarse structure the diffusion-guided decode predicts first
after_ijk = to_ijk(tree[0])                # final fine-resolution Stage 2 output
gt_ijk = to_ijk(gt_grid)                   # real ground truth, for comparison
embedding_ijk = to_ijk(cond_latent.grid)   # conditioning latent's own coarse topology
embedding_feat = cond_latent.feature.jdata.cpu().numpy()  # [N, latent_dim] per-voxel codes

feat_centered = embedding_feat - embedding_feat.mean(axis=0, keepdims=True)
_, _, vt = np.linalg.svd(feat_centered, full_matrices=False)
pca3 = feat_centered @ vt[:3].T

def norm01(x, lo=2, hi=98):
    p_lo, p_hi = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)

hue = norm01(pca3[:, 0])
sat = 0.45 + 0.45 * norm01(pca3[:, 1])
val = 0.75 + 0.20 * norm01(pca3[:, 2])
embedding_rgb = hsv_to_rgb(np.stack([hue, sat, val], axis=-1))

# Shared bounding box (fine-grid units), sized from GT+input combined so
# neither the (possibly larger/offset) Step1 footprint nor GT gets clipped.
during_ijk_fine = during_ijk * 2
embedding_ijk_fine = embedding_ijk * 2

all_fine_ijk = np.concatenate([before_ijk, after_ijk, gt_ijk, during_ijk_fine, embedding_ijk_fine], axis=0)
mins = all_fine_ijk.min(axis=0)
maxs = all_fine_ijk.max(axis=0)
shape = tuple(maxs - mins + 1)

def to_dense(ijk, cell=1):
    arr = np.zeros(shape, dtype=bool)
    local = ijk - mins
    valid = np.all((local >= 0) & (local < np.array(shape)), axis=1)
    local = local[valid]
    if cell > 1:
        for dx in range(cell):
            for dy in range(cell):
                for dz in range(cell):
                    off = local + np.array([dx, dy, dz])
                    ok = np.all(off < np.array(shape), axis=1)
                    o = off[ok]
                    arr[o[:, 0], o[:, 1], o[:, 2]] = True
    else:
        arr[local[:, 0], local[:, 1], local[:, 2]] = True
    return arr

def to_dense_colored(ijk, rgb, cell=1):
    occ = np.zeros(shape, dtype=bool)
    colors = np.ones(shape + (4,), dtype=np.float32)
    local = ijk - mins
    valid = np.all((local >= 0) & (local < np.array(shape)), axis=1)
    local = local[valid]
    rgb = rgb[valid]
    for dx in range(cell):
        for dy in range(cell):
            for dz in range(cell):
                off = local + np.array([dx, dy, dz])
                ok = np.all(off < np.array(shape), axis=1)
                o = off[ok]
                occ[o[:, 0], o[:, 1], o[:, 2]] = True
                colors[o[:, 0], o[:, 1], o[:, 2], :3] = rgb[ok]
    return occ, colors

def render_clean(ax, dense, color='#c3aee0', edge_color='#9c7fc0'):
    ax.voxels(dense, facecolors=color, edgecolor=edge_color, linewidth=0.15)
    ax.set_box_aspect(shape)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    ax.set_facecolor('white')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor('white')
        pane.set_edgecolor('white')
        pane.set_alpha(1.0)

def render_clean_colored(ax, occ, colors, edge_color='#00000055'):
    ax.voxels(occ, facecolors=colors, edgecolor=edge_color, linewidth=0.15)
    ax.set_box_aspect(shape)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    ax.set_facecolor('white')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor('white')
        pane.set_edgecolor('white')
        pane.set_alpha(1.0)

# Step1's input gets a different fill color (amber, not the standard tube
# purple) so it visually reads as "the flawed hint," not as if it were
# already correct tube structure -- distinguishes it at a glance from GT/output.
panels = [
    ('before', before_ijk, 1, '#e0b25c', '#c08f3a'),
    ('during', during_ijk_fine, 2, '#c3aee0', '#9c7fc0'),
    ('after', after_ijk, 1, '#c3aee0', '#9c7fc0'),
    ('gt', gt_ijk, 1, '#c3aee0', '#9c7fc0'),
]

out_dir = Path('/home/ameliacatala/Documents/XCube')

for name, ijk, cell, color, edge in panels:
    fig = plt.figure(figsize=(5, 5), facecolor='white')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    render_clean(ax, to_dense(ijk, cell=cell), color=color, edge_color=edge)
    out_path = out_dir / f'diffusion_{name}.png'
    plt.savefig(out_path, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'{name}: {ijk.shape[0]} voxels -> saved to {out_path}')

occ, colors = to_dense_colored(embedding_ijk_fine, embedding_rgb, cell=2)
fig = plt.figure(figsize=(5, 5), facecolor='white')
ax = fig.add_subplot(1, 1, 1, projection='3d')
render_clean_colored(ax, occ, colors)
embedding_path = out_dir / 'diffusion_embedding.png'
plt.savefig(embedding_path, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f'embedding: {embedding_ijk.shape[0]} conditioning-latent voxels ({embedding_feat.shape[1]}-dim, PCA->RGB) -> saved to {embedding_path}')

def grid_iou_ijk(a_ijk, b_ijk):
    a_set = set(map(tuple, a_ijk.tolist()))
    b_set = set(map(tuple, b_ijk.tolist()))
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0

print(f'\nIoU(Step1 input, GT)  = {grid_iou_ijk(before_ijk, gt_ijk):.3f}')
print(f'IoU(Stage2 output, GT) = {grid_iou_ijk(after_ijk, gt_ijk):.3f}')

fig = plt.figure(figsize=(25, 5), facecolor='white')
titles = ['Step1 input\n(conditioning)', 'Encoded condition\n(latent embedding)',
          'Coarse structure\n(half resolution)', 'Stage 2 output\n(final)', 'Ground truth']
for i, (ax_data, title) in enumerate(zip(
        [(to_dense(before_ijk, 1), '#e0b25c', '#c08f3a'), None,
         (to_dense(during_ijk_fine, 2), '#c3aee0', '#9c7fc0'),
         (to_dense(after_ijk, 1), '#c3aee0', '#9c7fc0'),
         (to_dense(gt_ijk, 1), '#c3aee0', '#9c7fc0')], titles)):
    ax = fig.add_subplot(1, 5, i + 1, projection='3d')
    if ax_data is None:
        render_clean_colored(ax, occ, colors)
    else:
        dense, color, edge = ax_data
        render_clean(ax, dense, color=color, edge_color=edge)
plt.tight_layout()
combined_path = out_dir / 'diffusion_pipeline_visual.png'
plt.savefig(combined_path, dpi=130, facecolor='white')
plt.close(fig)
print('Combined comparison saved to', combined_path)
