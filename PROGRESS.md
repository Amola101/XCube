# GPR Diffusion-Restoration Project — Progress Log

## Goal

A conditional diffusion model that denoises random noise into the correct
ground-truth (GT) tube/pipe shape, conditioned on TEUNet's flawed reconstruction
of the same subsurface GPR scan. Two stages:

- **Stage 1**: a VAE learns a latent space of real tube geometry by
  self-reconstructing GT shapes.
- **Stage 2**: a conditional diffusion model denoises into that latent space,
  conditioned on TEUNet's (imperfect) output.

The research theme specifically calls for diffusion rather than plain
regression, since regression tends to average over multiple plausible
corrections into one blurry compromise, while diffusion can commit to one
sharp, plausible answer per sample.

## Source data

`/home/ameliacatala/Documents/preprocess/transfer_data/{teunet,gt}/` — 1973
paired patches (TEUNet probability grid + binary GT grid), 64x64x48 voxels,
~5mm resolution. Dice score distribution: good (>0.8): 1257, moderate
(0.5-0.8): 455, poor (<0.5): 261.

## Environment setup

- Working conda env: **`preproc`** (not `base`, not `gpr310` — those had an
  unrelated PyPI package literally named `fvdb`, a FAISS-based vector DB tool,
  shadowing the real NVIDIA fVDB library).
- The publicly installable `fvdb_core` package is missing `fvdb.nn.VDBTensor`,
  required throughout XCube. Fixed by building the real fVDB from source, from
  a specific historical PR of `AcademySoftwareFoundation/openvdb`
  (`pull/1808/head`, branch `feature/fvdb`), cloned into `openvdb/` (gitignored,
  not part of this repo — a separate third-party library).
- Building fVDB from source needed 4 patches for libtorch-version drift
  (`torch::linalg::inv` removed, a CuBLAS reduction option enum change, a
  duplicate pybind11 type caster, and CUDA include-path environment variables).
- **Required env var**: `LD_PRELOAD=.../envs/preproc/lib/libstdc++.so.6` (system
  libstdc++ is too old). Originally set via `LD_LIBRARY_PATH` instead, which
  caused a hard-to-diagnose `CUBLAS_STATUS_NOT_INITIALIZED` bug by shadowing
  torch's own bundled cuBLAS. This is now set **automatically** on every
  `conda activate preproc`, via a fixed conda activation hook script
  (`envs/preproc/etc/conda/activate.d/libstdcxx.sh`) — no longer needs to be
  set manually per command.
- CUDA 12.4 JIT compilation needs `PATH=/usr/local/cuda-12.4/bin:$PATH` and
  `CPATH=/usr/local/cuda-12.4/targets/x86_64-linux/include:$CPATH`.

## Preprocessing (`datagen/preprocess_gpr.py`)

Converts the raw TEUNet/GT `.h5` pairs into `.pkl` files (fvdb `GridBatch`
objects) plus stratified train/val/test split lists (80/10/10 per dice tier).

Bugs fixed:
- The real fVDB build has no `GridBatch.from_ijk(...)` classmethod (that's the
  newer `fvdb_core` API) — replaced with `GridBatch()` + `.set_from_ijk(...)`.
- Grid origin convention: XCube expects `origins = voxel_size / 2`, not
  `[0, 0, 0]`. Mismatched origins caused an assertion failure in
  `base_loss.py` during Stage 1 training.

Final dataset: `/home/ameliacatala/Documents/preprocess/data_full/gpr/` — all
1973 samples processed successfully, 0 skipped, all spot-checks passed
(including "moderate"/"poor" tier samples).

## Stage 1: VAE (`configs/gpr/gpr_vae.yaml`, `xcube/models/autoencoder.py`)

Self-reconstructs GT shapes to learn a latent space of valid tube geometry.
Network scaled down from XCube's default (Waymo-scale) sizes to fit GPR's
tiny 64x64x48 grids: `cut_ratio=4`, `f_maps=32`, `c_dim=32`.

Bug fixed: Lightning's automatic `batch_size` inference for `self.log(...)`
fails when a batch contains only custom fvdb objects (no plain tensors) —
fixed by passing `batch_size=out['gt_grid'].grid_count` explicitly everywhere.

**Result (100 epochs, full 1973-sample dataset):**
- Validation loss: 1.53 → 0.27
- Structure accuracy (voxel-level filled/empty correctness): 97.3% → 99.68%
- Curve flattened in the second half — 100 epochs was sufficient, not wasteful.
- Checkpoint: `checkpoints/gpr/VAE_stage1/version_4/checkpoints/last.ckpt`

## Stage 2: Conditional Diffusion (`configs/gpr/gpr_diffusion.yaml`, `xcube/models/diffusion.py`)

### Conditioning design

XCube's diffusion code always treats "the main shape being learned" (read from
`DS.INPUT_PC`) and "the conditioning hint" as two separate fields. GPR's
dataset originally only exposed two fields (`INPUT_PC`, `GT_DENSE_PC`),
neither cleanly mapping to "GT is the target, TEUNet is a separate hint." Fix:

1. Added a new field `DS.COND_PC` (`xcube/data/base.py`) for "a second,
   separate conditioning grid."
2. `xcube/data/gpr.py`: populates `COND_PC` with TEUNet's grid
   (`input_data['input_grid']`), independent of the `input_key` used for the
   main `INPUT_PC` field (which the diffusion config sets to `"target_grid"`,
   i.e. GT, since `extract_latent` always treats `INPUT_PC` as the thing to
   noise/denoise).
3. `xcube/models/diffusion.py`: added `use_cond_grid_concat_cond` — encodes
   TEUNet's grid (`batch[DS.COND_PC]`) through the **same frozen Stage 1 VAE
   encoder** used for the main latent, aligns it onto the noisy latent's own
   sparse grid via `fill_to_grid` (TEUNet's grid generally occupies different
   voxels than GT), then concatenates it as extra channels before denoising.
   Added in 4 places: hparams default, `_forward_cond` (training/sampling),
   `get_dataset_spec`, and `evaluation_api` (real inference, no GT available).

### Other bugs fixed along the way

- Missing dependency `torch_scatter` — not available as a prebuilt wheel for
  this torch/CUDA combo (too new), built from source via pip using the same
  CUDA toolchain env vars as the fVDB build.
- Same Lightning `batch_size` inference bug as Stage 1, fixed the same way
  (`out['log_batch_size'] = bsz` in `forward()`, passed through
  `train_val_step`).
- `torch.load` in PyTorch >=2.6 defaults to `weights_only=True`, which blocks
  loading our own checkpoints (they contain an OmegaConf settings object).
  Fixed in 3 places: `diffusion.py`'s VAE loader, `train.py`'s manual resume
  checkpoint load, and `train.py`'s `trainer.fit(..., ckpt_path=...)` call
  (via a temporarily-scoped `torch.load` monkeypatch, since that one happens
  deep inside the pytorch-lightning library). Safe since it's always our own
  locally-trained checkpoint, never a downloaded one.

### Training results

Ran in two stages (50 epochs, then resumed to 100) on the full dataset:

- Epochs 0-50: validation loss dropped from ~0.99 (a freshly-initialized
  diffusion model's loss is expected to start near 1.0, the variance of
  the noise it's learning to predict) down to a plateau around 0.30.
- Epochs 50-100 (resumed from the epoch-50 checkpoint, no retraining from
  scratch needed): loss stayed in the same ~0.25-0.40 noisy plateau — little
  further improvement, but also **no overfitting** at any point: training and
  validation loss tracked closely together the entire 100 epochs.
- Practical lesson: this setup converges by ~epoch 25-30; the remaining ~75
  epochs of compute mostly just confirmed stability.
- Checkpoint: `checkpoints/gpr/Diffusion_stage2/version_2/checkpoints/last.ckpt`
- Curve plots: `stage1_training_curve.png`,
  `stage2_diffusion_curve.png` (50 epochs), `stage2_diffusion_curve_100ep.png`
  (full 100 epochs) in `~/Documents/`.

## Testing / Evaluation (`scripts/run_stage2_inference.py`)

Ran the trained Stage 2 model on real test samples, comparing its output
against ground truth using IoU (intersection-over-union of occupied voxels),
and against TEUNet's raw output as a baseline.

**Preliminary check (11 samples spread across tiers)** suggested a promising
pattern: roughly even on "good"/"moderate" tiers, but a meaningful improvement
on "poor" tier (+0.038 avg IoU vs. TEUNet baseline).

**Full test set (all 198 samples, run 2026-06-25)** — the real, trustworthy
number; raw output saved in `scripts/results/stage2_full_test_198samples.txt`:

| Tier | n | Avg IoU (model output vs. truth) | Avg IoU (TEUNet vs. truth) | Difference |
|---|---|---|---|---|
| Good | 126 | 0.802 | 0.816 | -0.013 |
| Moderate | 46 | 0.572 | 0.580 | -0.008 |
| Poor | 26 | 0.090 | 0.085 | +0.004 |
| **Overall** | **198** | **0.655** | **0.665** | **-0.010** |

**Honest conclusion: the small-sample improvement on "poor" tier did not hold
up at full scale** (+0.038 on n=3 shrank to +0.004 on n=26) — at full scale,
the model is essentially on par with, or very slightly behind, just using
TEUNet's raw output directly, across every tier. Several poor-tier samples
show TEUNet finding zero overlapping voxels with ground truth at all — cases
where TEUNet's reconstruction failed almost completely, leaving little for the
diffusion model's conditioning signal to work from.

This is a genuine negative result for the current setup, not yet a successful
improvement over baseline. Worth investigating next: the diffuser network was
deliberately minimal for a first working version (`model_channels=32`,
`channel_mult=[1,2]`, `num_res_blocks=1`, no attention) — more capacity may be
needed; the conditioning signal (TEUNet's grid encoded through the frozen VAE
encoder, concatenated as extra channels) may be too coarse/lossy; and the
training loss plateaued by ~epoch 25-30, which may indicate a capacity
ceiling rather than an undertrained model.

## Visualization (`scripts/visualize_stage2_sample.py`)

Side-by-side 3D scatter plot (TEUNet input / model output / ground truth) for
any chosen test sample, saved to `stage2_visual_comparison.png`. Shows the
actual occupied-voxel point clouds and their counts.

## Status as of today

Stage 1 and Stage 2 both trained and validated. Real inference confirmed
working, with an encouraging (if small-sample) sign that the model helps most
on the hardest cases. Not yet done: full 198-sample test-set evaluation for a
statistically solid number; further visual/qualitative inspection.
