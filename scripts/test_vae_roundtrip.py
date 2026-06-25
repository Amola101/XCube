"""
Diagnostic: passes TEUNet's flawed grid through the frozen Stage 1 VAE's own
encoder + decoder directly (no diffusion, no noise) and compares the result to
ground truth. This isolates whether the VAE's own representational capacity is
the bottleneck, separately from the diffusion model's conditioning/denoising.

Usage:
    python scripts/test_vae_roundtrip.py
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

VAE_CONFIG = Path('/home/ameliacatala/Documents/XCube/configs/gpr/gpr_vae.yaml')
VAE_CKPT = Path('/home/ameliacatala/Documents/checkpoints/gpr/VAE_stage1/version_4/checkpoints/last.ckpt')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr')

model_args = exp.parse_config_yaml(VAE_CONFIG)
net_module = importlib.import_module("xcube.models." + model_args.model).Model
vae = net_module.load_from_checkpoint(VAE_CKPT, hparams=model_args)
vae = vae.cuda().eval()

stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
print(f'{len(stems)} test samples. Tiers (by index): 0-125 good, 126-171 moderate, 172-197 poor.\n')

def grid_iou(gt, pd):
    idx = pd.ijk_to_index(gt.ijk)
    upi = (pd.num_voxels + gt.num_voxels).cpu().numpy().tolist()
    inter = torch.sum(idx[0].jdata >= 0).item()
    return inter / (upi[0] - inter + 1e-6)

tier_of = lambda i: 'good' if i < 126 else ('moderate' if i < 172 else 'poor')

results = []
for i, stem in enumerate(stems):
    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    teunet_grid = sample['input_grid'].to('cuda')

    with torch.no_grad():
        latent = vae._encode({DS.INPUT_PC: teunet_grid}, use_mode=True)
        res = vae.unet.FeaturesSet()
        res, output_x = vae.unet.decode(res, latent, is_testing=True)
    roundtrip_grid = res.structure_grid[0]

    iou_roundtrip = grid_iou(gt_grid, roundtrip_grid)
    iou_teunet = grid_iou(gt_grid, teunet_grid)
    tier = tier_of(i)
    results.append((tier, iou_roundtrip, iou_teunet))
    if i % 20 == 0:
        print(f'[{tier:8s}] {i:3d}/{len(stems)}  IoU(VAE roundtrip,GT)={iou_roundtrip:.3f}  IoU(TEUNet,GT)={iou_teunet:.3f}')

print()
for tier in ['good', 'moderate', 'poor']:
    sub = [(r, t) for tr, r, t in results if tr == tier]
    if not sub:
        continue
    avg_r = sum(r for r, t in sub) / len(sub)
    avg_t = sum(t for r, t in sub) / len(sub)
    print(f'{tier:8s} (n={len(sub)}): avg IoU(VAE roundtrip,GT)={avg_r:.3f}  avg IoU(TEUNet,GT)={avg_t:.3f}  diff={avg_r-avg_t:+.3f}')

avg_r_all = sum(r for _, r, t in results) / len(results)
avg_t_all = sum(t for _, r, t in results) / len(results)
print(f'\nOVERALL (n={len(results)}): avg IoU(VAE roundtrip,GT)={avg_r_all:.3f}  avg IoU(TEUNet,GT)={avg_t_all:.3f}  diff={avg_r_all-avg_t_all:+.3f}')
