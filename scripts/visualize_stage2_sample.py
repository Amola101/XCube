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

def to_points(grid):
    xyz = grid.grid_to_world(grid.ijk.float())
    return xyz[0].jdata.cpu().numpy()

teunet_pts = to_points(teunet_grid)
pred_pts = to_points(pred_grid)
gt_pts = to_points(gt_grid)

fig = plt.figure(figsize=(15, 5))
titles = ['TEUNet Input (flawed)', 'XCube Stage 2 Output', 'Ground Truth']
for i, (pts, title) in enumerate(zip([teunet_pts, pred_pts, gt_pts], titles)):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2)
    ax.set_title(f'{title}\n({pts.shape[0]} voxels)')
    ax.set_xlim(gt_pts[:, 0].min(), gt_pts[:, 0].max())
    ax.set_ylim(gt_pts[:, 1].min(), gt_pts[:, 1].max())
    ax.set_zlim(gt_pts[:, 2].min(), gt_pts[:, 2].max())

plt.suptitle(f'Sample: {stem}')
plt.tight_layout()
out_path = '/home/ameliacatala/Documents/stage2_visual_comparison.png'
plt.savefig(out_path, dpi=130)
print('Saved to', out_path)
