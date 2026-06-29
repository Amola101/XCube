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
improvement over baseline.

## Diagnosis (`scripts/test_vae_roundtrip.py`)

To find the actual cause, ran a diagnostic: pass TEUNet's grid through the
frozen Stage 1 VAE's encoder + decoder directly — zero noise, zero diffusion
process at all — and compare to ground truth the same way.

| Tier | TEUNet baseline | VAE round-trip only (no diffusion) | Full diffusion model |
|---|---|---|---|
| Good | 0.816 | 0.808 | 0.802 |
| Moderate | 0.580 | 0.580 | 0.572 |
| Poor | 0.085 | 0.088 | 0.090 |
| **Overall** | **0.665** | **0.661** | **0.655** |

**The VAE round-trip alone performs almost identically to the full trained
diffusion model.** The diffusion process isn't doing meaningful denoising —
it's behaving like it learned to mostly pass the condition straight through.

Confirmed visually too (`scripts/visualize_stage2_sample.py`, now rendering
actual solid voxel cubes instead of scattered dots — much clearer): on a
poor-tier sample, TEUNet's input gets the pipe's right-hand section right but
the left-hand section disintegrates into a scattered, broken cluster. The
model's output keeps the good section, adds a modest number of voxels
overall, but the broken left-hand section stays just as scattered — it never
reorganizes that region into the smooth tube ground truth actually has.

**Root-cause theory (structural confinement):** the VAE's decoder
(`sunet.py:467-512`) grows structure level-by-level, only ever subdividing
cells that are *already part of* the coarse footprint it's handed — it can
never invent occupied space outside that starting footprint (like zooming
into a map: you can add detail inside a region you're already looking at, but
can't discover a city that wasn't on the map at all). During training,
`extract_latent` always uses `DS.INPUT_PC` = ground truth (`input_key:
"target_grid"`), so the model only ever practices "refine an already-correct
neighborhood" — it never sees a *wrong* starting footprint during training.
At real test time, the starting footprint instead comes from encoding
TEUNet's own (possibly wrong) grid. On good/moderate tiers TEUNet's footprint
roughly overlaps GT's, so this barely bites; on poor tier, where TEUNet's
footprint diverges most, the model is structurally boxed into TEUNet's wrong
neighborhood and can't escape it — matching the measured results exactly.

**Fix attempt 1 (tried 2026-06-27): test-time dilation — did not work.**
`scripts/test_dilation_fix.py` dilates TEUNet's grid via fvdb's
`GridBatch.conv_grid(kernel_size, stride=1)` before encoding/conditioning,
giving the decoder a wider candidate region to grow structure into. Tested
both a 1-voxel margin (`kernel_size=3`) and a 3-voxel margin (`kernel_size=7`)
on the same 11-sample tier spread — neither changed results meaningfully
(poor tier: 0.230 baseline vs 0.229 dilated at kernel=7; essentially flat
across all tiers, no improvement at any margin tested).

**Why this null result is itself informative**: it rules out "not enough
room" as the mechanism, and sharpens the diagnosis — the model doesn't just
need more space, it never learned *how* to use an uncertain/wide candidate
region productively. It only ever practiced refining a region it could
already trust was exactly correct (GT's). Given extra room at test time, it
has no learned behavior for filling it in, so it mostly predicts "not
occupied" regardless. This confirms the real fix has to change what the model
practices on during training, not just what it's given at test time.

**Fix attempt 2 (tried 2026-06-27/28): retrain on TEUNet's dilated footprint
— also did NOT work.** Implemented in `xcube/models/diffusion.py`: new
hparams `train_cond_footprint` and `cond_grid_dilation_kernel`, a shared
`encode_cond_grid()` helper (dilates via `conv_grid` then encodes through the
frozen VAE, used consistently in training and at real inference), and a new
branch in `forward()` that uses the dilated-TEUNet encode as the structural
topology + noising target (GT's true features aligned onto it via
`fill_to_grid`) instead of GT's own footprint. New config
`configs/gpr/gpr_diffusion_v2.yaml` (`cond_grid_dilation_kernel: 5`, ~2-voxel
margin), trained 50 epochs on the full dataset (`checkpoints/gpr/Diffusion_stage2_v2/version_2`,
val loss plateaued ~0.28-0.33, comparable to v1). Full 198-sample evaluation
(`scripts/run_stage2_inference_v2.py`):

| Tier | n | v2 IoU vs GT | TEUNet IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 126 | 0.800 | 0.816 | -0.016 |
| Moderate | 46 | 0.565 | 0.580 | -0.015 |
| Poor | 26 | 0.089 | 0.085 | +0.004 |
| **Overall** | **198** | **0.652** | **0.665** | **-0.013** |

**Essentially identical to v1** (-0.010 → -0.013 overall, poor tier +0.004
unchanged). Two structural-footprint fixes in a row (test-time dilation, and
now retraining on a dilated footprint) produced no improvement at all.

**Updated theory**: combined with the earlier VAE-round-trip finding (frozen
VAE encode/decode alone, zero diffusion, already matches the full model's
performance), the evidence now points less at "the decoder doesn't have
enough room" and more at **the diffusion model learning to just pass the
conditioning hint straight through rather than doing real generative
correction** — changing the footprint doesn't matter if the model was never
forced to rely on anything besides copying its hint in the first place.

**Fix attempt 3 (tried 2026-06-28/29): classifier-free conditioning dropout
— also did NOT meaningfully change anything.** `use_classifier_free` and
`classifier_free_prob` already existed as hparams (unused before); the
dropout mechanism (`conduct_classifier_free`) was already fully wired into
the `use_cond_grid_concat_cond` branch. New config
`configs/gpr/gpr_diffusion_v3.yaml` (built on v1's plain footprint, not v2's
dilated one, to isolate dropout as the only new variable), trained 50 epochs
full dataset (`checkpoints/gpr/Diffusion_stage2_v3/version_0`, val loss
plateau ~0.30-0.34, same range as v1/v2). Full 198-sample evaluation
(`scripts/run_stage2_inference_v3.py`):

| Tier | n | v3 IoU vs GT | TEUNet IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 126 | 0.806 | 0.816 | -0.010 |
| Moderate | 46 | 0.570 | 0.580 | -0.010 |
| Poor | 26 | 0.090 | 0.085 | +0.005 |
| **Overall** | **198** | **0.657** | **0.665** | **-0.008** |

**The real finding now is the consistency itself**: three structurally
different fixes (test-time dilation, retraining on a dilated footprint,
classifier-free dropout) all land within noise of each other (-0.010, -0.013,
-0.008 overall; poor tier always +0.004 to +0.005). None of our three
theories about the specific mechanism turned out to be the deciding factor.
This looks less like "haven't found the right tweak" and more like this
conditioning setup (concat-based, single frozen-VAE encode of TEUNet's grid,
small network: `model_channels=32`, `channel_mult=[1,2]`, `num_res_blocks=1`,
no attention) has a real ceiling around 0.65-0.66 overall IoU — just shy of
TEUNet's own baseline — regardless of these three training-time
interventions.

**Not yet tried, would need a genuinely different angle rather than another
training-time tweak**:
- Increase network capacity (more channels/res blocks/attention) — current
  network is deliberately tiny (scaled down from XCube's Waymo-scale
  defaults); it's possible it's simply too small to learn anything beyond
  near-copy-through, independent of training regime.
- Reconsider whether per-voxel IoU vs. TEUNet is even the right comparison —
  TEUNet's own reconstruction is itself a strong, already-trained baseline;
  beating it on the same input information may require the diffusion model
  to access something TEUNet doesn't (e.g. the raw GPR scan itself, not just
  TEUNet's already-collapsed binary/probability grid).
- Try a fundamentally different conditioning mechanism (e.g. cross-attention
  instead of channel-concat) rather than another concat-based variant.

(Separately, also worth trying: `use_classifier_free` is currently `false` in
the config — enabling it randomly hides the conditioning signal during
training ~10% of the time, which could counter a possible "shortcut" effect
from concatenating TEUNet's hint very strongly at every denoising step. Not
mutually exclusive with the structural-confinement fixes above.)

## Visualization (`scripts/visualize_stage2_sample.py`)

Side-by-side 3D **voxel-cube** rendering (TEUNet input / model output / ground
truth) for any chosen test sample, saved to `stage2_visual_comparison.png`.
Originally used scattered dots (`ax.scatter`), which made even the ground
truth look like noise; switched to matplotlib's `ax.voxels()` to draw actual
filled cubes per occupied cell, with a shared bounding box/scale and a fixed
camera angle — now clearly shows real pipe shapes.

## Status as of today (2026-06-25)

Stage 1 and Stage 2 both trained. Real inference and a full 198-sample
evaluation confirmed Stage 2 does **not** yet beat the TEUNet baseline, and
the VAE round-trip diagnostic + structural-confinement theory above explains
why. Not yet done: trying either candidate fix (test-time dilation, or
retraining with TEUNet's footprint included), and re-evaluating afterward.
