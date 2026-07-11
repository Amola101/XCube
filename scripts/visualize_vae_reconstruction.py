"""
Saves clean, chrome-free 3D renders of one GPR pipe sample as it passes
through the trained Stage 1 VAE: the ground-truth input, the coarse
intermediate structure the network predicts first, and the final
fine-resolution reconstruction it decodes back out to. Plain white
background, no axes/grid/titles -- meant to be dropped straight into a
figure/diagram.

Usage:
    python scripts/visualize_vae_reconstruction.py [test_set_index]
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

VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/VAE_stage1_corr_medium/version_1')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr_corr_medium')

model_args = exp.parse_config_yaml(VERSION_DIR / 'hparams.yaml')
net_module = importlib.import_module("xcube.models." + model_args.model).Model
model = net_module.load_from_checkpoint(VERSION_DIR / 'checkpoints' / 'last.ckpt', hparams=model_args)
model = model.cuda().eval()

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
stem = stems[idx]
print('Visualizing sample:', stem)

sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
gt_grid = sample['target_grid'].to('cuda')
gt_material = sample['target_material'].to('cuda')

with torch.no_grad():
    batch = {DS.INPUT_PC: gt_grid, DS.INPUT_MATERIAL: gt_material}
    out = model(batch, {})
    # The actual compressed latent code (the VAE bottleneck / "embedding") --
    # distinct from tree[1] below, which is the decoder's own first-stage
    # structural prediction *from* this latent, not the latent itself.
    latent = model._encode({DS.INPUT_PC: gt_grid, DS.INPUT_MATERIAL: gt_material}, use_mode=True)
tree = out['tree']  # dict keyed by tree depth: 0 = finest, 1 = coarser (tree_depth=2)

def to_ijk(grid):
    return grid.ijk[0].jdata.cpu().numpy()

before_ijk = to_ijk(gt_grid)               # ground-truth input
during_ijk = to_ijk(tree[1])               # coarse structure predicted first
after_ijk = to_ijk(tree[0])                # final fine-resolution reconstruction
embedding_ijk = to_ijk(latent.grid)        # same coarse topology as the latent lives on
embedding_feat = latent.feature.jdata.cpu().numpy()  # [N, latent_dim] per-voxel codes

# Reduce the multi-channel latent to 3 dimensions via PCA, so each voxel's
# color reflects its actual latent feature content rather than being a flat
# placeholder color. Mapped through HSV (not raw RGB) with saturation/value
# floors so it stays vivid and bright regardless of how the latent's variance
# happens to be distributed -- raw PCA->RGB tends to look dark/muddy since
# most points cluster near the middle of the range.
from matplotlib.colors import hsv_to_rgb

feat_centered = embedding_feat - embedding_feat.mean(axis=0, keepdims=True)
_, _, vt = np.linalg.svd(feat_centered, full_matrices=False)
pca3 = feat_centered @ vt[:3].T

def norm01(x, lo=2, hi=98):
    # Percentile-based normalization (not raw min/max) so a couple of outlier
    # voxels don't compress everything else into a narrow, muddy band.
    p_lo, p_hi = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)

hue = norm01(pca3[:, 0])
sat = 0.45 + 0.45 * norm01(pca3[:, 1])   # 0.45-0.90: always fairly vivid
val = 0.75 + 0.20 * norm01(pca3[:, 2])   # 0.75-0.95: always bright, never dark
embedding_rgb = hsv_to_rgb(np.stack([hue, sat, val], axis=-1))

# Shared bounding box (fine-grid units) so all three panels render at the
# same physical scale. The coarse ("during") grid is in half-resolution
# units, so its ijk is scaled up by 2x to align with the fine grid's frame.
during_ijk_fine = during_ijk * 2

mins = before_ijk.min(axis=0)
maxs = before_ijk.max(axis=0)
shape = tuple(maxs - mins + 1)

def to_dense(ijk, cell=1):
    arr = np.zeros(shape, dtype=bool)
    local = ijk - mins
    valid = np.all((local >= 0) & (local < np.array(shape)), axis=1)
    local = local[valid]
    if cell > 1:
        # Fill the whole cell-sized block each coarse voxel covers, so it
        # reads as visibly blockier rather than a sparse sub-sampled dot.
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
    """Like to_dense, but also returns a per-voxel RGBA color volume built
    from an [N, 3] color array aligned with ijk (one color per input voxel,
    replicated across its cell-sized block)."""
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
    """Voxel render with no axes, no grid, no panes -- just the shape on white,
    with edges toned to match the fill instead of a harsh black grid."""
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
    """Same clean chrome-free style, but each voxel gets its own RGB color
    (used for the latent-embedding render, colored by PCA of its features)."""
    ax.voxels(occ, facecolors=colors, edgecolor=edge_color, linewidth=0.15)
    ax.set_box_aspect(shape)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    ax.set_facecolor('white')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor('white')
        pane.set_edgecolor('white')
        pane.set_alpha(1.0)

panels = [
    ('before', before_ijk, 1),
    ('during', during_ijk_fine, 2),
    ('after', after_ijk, 1),
]

out_dir = Path('/home/ameliacatala/Documents/XCube')

# Individual clean images, one per panel -- for dropping into a figure/diagram.
for name, ijk, cell in panels:
    fig = plt.figure(figsize=(5, 5), facecolor='white')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    render_clean(ax, to_dense(ijk, cell=cell))
    out_path = out_dir / f'vae_reconstruction_{name}.png'
    plt.savefig(out_path, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'{name}: {ijk.shape[0]} voxels -> saved to {out_path}')

# Embedding panel: same coarse topology as "during" (2x scale to align with
# the shared bounding box), but colored by the actual latent feature content.
embedding_ijk_fine = embedding_ijk * 2
occ, colors = to_dense_colored(embedding_ijk_fine, embedding_rgb, cell=2)
fig = plt.figure(figsize=(5, 5), facecolor='white')
ax = fig.add_subplot(1, 1, 1, projection='3d')
render_clean_colored(ax, occ, colors)
embedding_path = out_dir / 'vae_reconstruction_embedding.png'
plt.savefig(embedding_path, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f'embedding: {embedding_ijk.shape[0]} latent voxels ({embedding_feat.shape[1]}-dim, PCA->RGB) -> saved to {embedding_path}')

# Combined side-by-side (same clean style, no titles/suptitle) for a quick
# at-a-glance comparison: before -> embedding -> during -> after.
fig = plt.figure(figsize=(20, 5), facecolor='white')
ax = fig.add_subplot(1, 4, 1, projection='3d')
render_clean(ax, to_dense(before_ijk, cell=1))
ax = fig.add_subplot(1, 4, 2, projection='3d')
render_clean_colored(ax, occ, colors)
ax = fig.add_subplot(1, 4, 3, projection='3d')
render_clean(ax, to_dense(during_ijk_fine, cell=2))
ax = fig.add_subplot(1, 4, 4, projection='3d')
render_clean(ax, to_dense(after_ijk, cell=1))
plt.tight_layout()
combined_path = out_dir / 'vae_reconstruction_visual.png'
plt.savefig(combined_path, dpi=130, facecolor='white')
plt.close(fig)
print('Combined comparison saved to', combined_path)
