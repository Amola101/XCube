"""
Saves a side-by-side 3D picture comparing TEUNet's input, the trained Stage 2
model's output, and the ground truth, for one real test sample.

Usage:
    python scripts/visualize_stage2_sample.py [test_set_index]

test_set_index is 0-197 (0-125 = "good" tier, 126-171 = "moderate", 172-197 = "poor").
Defaults to 175 (a "poor" tier sample) if not given.
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

VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/Diffusion_stage2/version_2')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr')

model_args = exp.parse_config_yaml(VERSION_DIR / 'hparams.yaml')
net_module = importlib.import_module("xcube.models." + model_args.model).Model
model = net_module.load_from_checkpoint(VERSION_DIR / 'checkpoints' / 'last.ckpt', hparams=model_args)
model = model.cuda().eval()

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 175
stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
stem = stems[idx]
print('Visualizing sample:', stem)

sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
gt_grid = sample['target_grid'].to('cuda')
teunet_grid = sample['input_grid'].to('cuda')

with torch.no_grad():
    cond_latent = model.vae._encode({DS.INPUT_PC: teunet_grid}, use_mode=True)
    res, output_x = model.evaluation_api(
        batch={DS.COND_PC: teunet_grid}, grids=cond_latent.grid,
        use_ddim=True, ddim_step=100, use_ema=False)
pred_grid = res.structure_grid[0]

def to_ijk(grid):
    return grid.ijk[0].jdata.cpu().numpy()

teunet_ijk = to_ijk(teunet_grid)
pred_ijk = to_ijk(pred_grid)
gt_ijk = to_ijk(gt_grid)

# Shared bounding box (based on GT, which spans the full 64x64x48 grid) so all
# three panels render at the same physical scale and are visually comparable.
mins = gt_ijk.min(axis=0)
maxs = gt_ijk.max(axis=0)
shape = tuple(maxs - mins + 1)

def to_dense(ijk):
    arr = np.zeros(shape, dtype=bool)
    local = ijk - mins
    # TEUNet/prediction can occupy voxels slightly outside GT's own bounding
    # box -- clip those out rather than crash, instead of assuming a perfect fit.
    valid = np.all((local >= 0) & (local < np.array(shape)), axis=1)
    local = local[valid]
    arr[local[:, 0], local[:, 1], local[:, 2]] = True
    return arr

fig = plt.figure(figsize=(15, 5))
titles = ['TEUNet Input (flawed)', 'XCube Stage 2 Output', 'Ground Truth']
for i, (ijk, title) in enumerate(zip([teunet_ijk, pred_ijk, gt_ijk], titles)):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    ax.voxels(to_dense(ijk), edgecolor='k', linewidth=0.1)
    ax.set_title(f'{title}\n({ijk.shape[0]} voxels)')
    ax.set_box_aspect(shape)
    ax.view_init(elev=20, azim=-60)

plt.suptitle(f'Sample: {stem}')
plt.tight_layout()
out_path = '/home/ameliacatala/Documents/stage2_visual_comparison.png'
plt.savefig(out_path, dpi=130)
print('Saved to', out_path)
