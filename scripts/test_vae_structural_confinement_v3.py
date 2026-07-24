"""
Same structural-confinement diagnostic as test_vae_structural_confinement_v2.py,
but pointed at the Stage 1 VAE trained with the footprint-erosion fix
(PROGRESS.md, 2026-07-21: coarse_erosion_prob in xcube/modules/autoencoding/
sunet.py's decode(), config configs/gpr/gpr_vae_corr_medium_v3_erosion.yaml).

Reachability is still measured against the coarse_dilation_kernel-padded grid
(erosion only happens train-time, gated on self.training; eval() disables it,
so at test time decode() sees the real, unmodified Step1 coarse footprint --
same reachable region definition as the v2 script).

Usage:
    python scripts/test_vae_structural_confinement_v3.py [version] [n_samples]

n_samples defaults to the full test set (999 for corr_medium).
"""
import sys
sys.path.insert(0, '/home/ameliacatala/Documents/XCube')
import pickle
import importlib
from pathlib import Path

import fvdb
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

COARSE_DILATION_KERNEL = getattr(vae.unet, 'coarse_dilation_kernel', 1)
print('coarse_dilation_kernel =', COARSE_DILATION_KERNEL)
print('coarse_erosion_prob =', getattr(vae.unet, 'coarse_erosion_prob', 0.0),
      '(should have NO effect here -- eval() sets self.training=False)')

# coarsest decode level for tree_depth=2 / num_blocks=2 is feat_depth=1 -> 2x downsample,
# matching how base_loss.py computes struct-acc-1 (gt_grid.coarsened_grid(2 ** feat_depth)).
COARSE_FACTOR = 2

stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else len(stems)
stems = stems[:n_samples]
print(f'{len(stems)} test samples.\n')

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

results = []
total_escaped = 0  # sanity check accumulator -- must stay exactly 0

for i, stem in enumerate(stems):
    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    step1_grid = sample['input_grid'].to('cuda')
    step1_material = sample['input_material'].to('cuda')

    tier = assign_tier(grid_dice(gt_grid, step1_grid))

    with torch.no_grad():
        latent = vae._encode({DS.INPUT_PC: step1_grid, DS.INPUT_MATERIAL: step1_material}, use_mode=True)
        reach_grid = latent.grid
        if COARSE_DILATION_KERNEL > 1:
            margin = (COARSE_DILATION_KERNEL - 1) // 2
            reach_grid = fvdb.GridBatch(device=latent.grid.device)
            reach_grid.set_from_ijk(
                latent.grid.ijk,
                pad_min=[-margin] * 3,
                pad_max=[margin] * 3,
                voxel_sizes=latent.grid.voxel_sizes,
                origins=latent.grid.origins,
            )
        res = vae.unet.FeaturesSet()
        res, output_x = vae.unet.decode(res, latent, is_testing=True)
    roundtrip_grid = res.structure_grid[0]
    roundtrip_coarse = roundtrip_grid.coarsened_grid(COARSE_FACTOR)

    gt_coarse = gt_grid.coarsened_grid(COARSE_FACTOR)
    total_gt_coarse = gt_coarse.total_voxels
    if total_gt_coarse == 0:
        continue

    reach_idx = reach_grid.ijk_to_index(gt_coarse.ijk).jdata
    reach_mask = reach_idx >= 0
    n_reachable = reach_mask.sum().item()
    n_unreachable = total_gt_coarse - n_reachable

    reachable_ijk = gt_coarse.ijk.r_masked_select(reach_mask)
    unreachable_ijk = gt_coarse.ijk.r_masked_select(~reach_mask)

    n_recovered = 0
    if reachable_ijk.jdata.shape[0] > 0:
        rec_idx = roundtrip_coarse.ijk_to_index(reachable_ijk).jdata
        n_recovered = (rec_idx >= 0).sum().item()

    n_escaped = 0
    if unreachable_ijk.jdata.shape[0] > 0:
        esc_idx = roundtrip_coarse.ijk_to_index(unreachable_ijk).jdata
        n_escaped = (esc_idx >= 0).sum().item()
    total_escaped += n_escaped

    iou_roundtrip = grid_iou(gt_grid, roundtrip_grid)
    iou_step1 = grid_iou(gt_grid, step1_grid)

    pct_unreachable = n_unreachable / total_gt_coarse
    recovery_rate = n_recovered / n_reachable if n_reachable > 0 else float('nan')

    results.append((tier, pct_unreachable, recovery_rate, n_escaped, iou_roundtrip, iou_step1))

    if i % 50 == 0:
        print(f'[{tier:8s}] {i:3d}/{len(stems)}  '
              f'unreachable={pct_unreachable:.3f}  recovery_in_reach={recovery_rate:.3f}  '
              f'escaped={n_escaped}  IoU(rt,gt)={iou_roundtrip:.3f}  IoU(step1,gt)={iou_step1:.3f}')

print()
print('=' * 100)
for tier in ['good', 'moderate', 'poor']:
    sub = [r for r in results if r[0] == tier]
    if not sub:
        continue
    avg_unreachable = sum(r[1] for r in sub) / len(sub)
    recov = [r[2] for r in sub if r[2] == r[2]]  # drop nan
    avg_recovery = sum(recov) / len(recov) if recov else float('nan')
    max_escaped = max(r[3] for r in sub)
    avg_iou_rt = sum(r[4] for r in sub) / len(sub)
    avg_iou_s1 = sum(r[5] for r in sub) / len(sub)
    print(f'{tier:8s} (n={len(sub)}): avg %GT-coarse-unreachable={avg_unreachable:.3f}  '
          f'avg recovery-within-reachable={avg_recovery:.3f}  max escaped(should be 0)={max_escaped}  '
          f'avg IoU(roundtrip,gt)={avg_iou_rt:.3f}  avg IoU(step1,gt)={avg_iou_s1:.3f}')

avg_unreachable_all = sum(r[1] for r in results) / len(results)
recov_all = [r[2] for r in results if r[2] == r[2]]
avg_recovery_all = sum(recov_all) / len(recov_all) if recov_all else float('nan')

print()
print(f'OVERALL (n={len(results)}): avg %GT-coarse-unreachable={avg_unreachable_all:.3f}  '
      f'avg recovery-within-reachable={avg_recovery_all:.3f}')
print(f'Sanity check -- total voxels the decoder produced OUTSIDE its architecturally '
      f'reachable region (dilated by coarse_dilation_kernel={COARSE_DILATION_KERNEL}), summed '
      f'across all {len(results)} samples (must be exactly 0 if the confinement theory holds '
      f'as stated): {total_escaped}')
