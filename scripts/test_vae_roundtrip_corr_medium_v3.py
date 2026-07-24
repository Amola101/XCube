"""
Same diagnostic as test_vae_roundtrip_corr_medium_v2.py, but pointed at the
Stage 1 VAE trained with the footprint-erosion fix (PROGRESS.md, 2026-07-21:
`coarse_erosion_prob` in xcube/modules/autoencoding/sunet.py's decode(),
config configs/gpr/gpr_vae_corr_medium_v3_erosion.yaml). Combines
coarse_dilation_kernel=3 (same margin size as the v2 script's version_3 real
kernel=3 run, 0.300 overall round-trip IoU) with train-time erosion of a
fraction of the coarsest grid's own boundary voxels before re-padding, so
that margin contains a genuine mix of erased-but-real cells and
genuinely-outside cells during training, instead of the always-100%-fake
margin the plain dilation attempts used.

Usage:
    python scripts/test_vae_roundtrip_corr_medium_v3.py [version]
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

VAE_CONFIG = Path('/home/ameliacatala/Documents/XCube/configs/gpr/gpr_vae_corr_medium_v3_erosion.yaml')
_version = sys.argv[1] if len(sys.argv) > 1 else 'version_0'
VAE_CKPT_DIR = Path(f'/home/ameliacatala/Documents/checkpoints/gpr/VAE_stage1_corr_medium_v3_erosion/{_version}/checkpoints')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr_corr_medium')

ckpts = sorted(VAE_CKPT_DIR.glob('epoch=*.ckpt'), key=lambda p: p.stat().st_mtime)
VAE_CKPT = ckpts[-1]
print('Using checkpoint:', VAE_CKPT)

model_args = exp.parse_config_yaml(VAE_CONFIG)
net_module = importlib.import_module("xcube.models." + model_args.model).Model
vae = net_module.load_from_checkpoint(VAE_CKPT, hparams=model_args)
vae = vae.cuda().eval()
print('coarse_dilation_kernel =', getattr(vae.unet, 'coarse_dilation_kernel', 1))
print('coarse_erosion_prob =', getattr(vae.unet, 'coarse_erosion_prob', 0.0),
      '(should have NO effect here -- eval() sets self.training=False, so decode() never erodes at test time)')

stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
print(f'{len(stems)} test samples.\n')

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
for i, stem in enumerate(stems):
    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    step1_grid = sample['input_grid'].to('cuda')
    step1_material = sample['input_material'].to('cuda')

    tier = assign_tier(grid_dice(gt_grid, step1_grid))

    with torch.no_grad():
        latent = vae._encode({DS.INPUT_PC: step1_grid, DS.INPUT_MATERIAL: step1_material}, use_mode=True)
        res = vae.unet.FeaturesSet()
        res, output_x = vae.unet.decode(res, latent, is_testing=True)
    roundtrip_grid = res.structure_grid[0]

    iou_roundtrip = grid_iou(gt_grid, roundtrip_grid)
    iou_step1 = grid_iou(gt_grid, step1_grid)
    results.append((tier, iou_roundtrip, iou_step1))
    if i % 50 == 0:
        print(f'[{tier:8s}] {i:3d}/{len(stems)}  IoU(VAE roundtrip,GT)={iou_roundtrip:.3f}  IoU(Step1,GT)={iou_step1:.3f}')

print()
for tier in ['good', 'moderate', 'poor']:
    sub = [(r, t) for tr, r, t in results if tr == tier]
    if not sub:
        continue
    avg_r = sum(r for r, t in sub) / len(sub)
    avg_t = sum(t for r, t in sub) / len(sub)
    print(f'{tier:8s} (n={len(sub)}): avg IoU(VAE roundtrip,GT)={avg_r:.3f}  avg IoU(Step1,GT)={avg_t:.3f}  diff={avg_r-avg_t:+.3f}')

avg_r_all = sum(r for _, r, t in results) / len(results)
avg_t_all = sum(t for _, r, t in results) / len(results)
print(f'\nOVERALL (n={len(results)}): avg IoU(VAE roundtrip,GT)={avg_r_all:.3f}  avg IoU(Step1,GT)={avg_t_all:.3f}  diff={avg_r_all-avg_t_all:+.3f}')
