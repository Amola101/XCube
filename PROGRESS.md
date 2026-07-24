# GPR Diffusion-Restoration Project — Progress Log

## Goal

A conditional diffusion model that denoises random noise into the correct
ground-truth (GT) tube/pipe shape, conditioned on an upstream model's flawed
reconstruction of the same subsurface GPR scan. Two stages:

- **Stage 1**: a VAE learns a latent space of real tube geometry by
  self-reconstructing GT shapes.
- **Stage 2**: a conditional diffusion model denoises into that latent space,
  conditioned on the upstream model's (imperfect) output.

The research theme specifically calls for diffusion rather than plain
regression, since regression tends to average over multiple plausible
corrections into one blurry compromise, while diffusion can commit to one
sharp, plausible answer per sample.

**This file is split into two parts by data source, because they behave
differently and should not be conflated:**

- **Part 1 (archived)** — the original TEUNet dataset. No longer in use as of
  2026-07-07. Kept as a full record of 6 fix attempts and why they were
  abandoned, but nothing in Part 1 should be assumed to carry over to Part 2
  without separately checking it.
- **Part 2 (current/active)** — the corr_medium/Step1 dataset. This is the
  dataset and checkpoints actually in use now. **Read this part first** for
  current status.

## Environment setup

(Applies to both parts — general project environment, not data-specific.)

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
- **As of ~2026-07-08**: the JIT CUDA extension build (`ext/common`, used by
  `color_util.py`) started failing with `nvcc` rejecting the conda env's own
  bundled `gcc` (14.3.0 — CUDA 12.4 supports up to 13) as the host compiler,
  even though a working cached `.so` already existed. Root cause not fully
  pinned down, but coincides with the shared machine's disk-space issue being
  resolved by another user (see the aside at the bottom of this file) —
  possibly a system change invalidated cached intermediate build objects.
  **Fixed by forcing `CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12`** (system
  compilers, CUDA-12.4-compatible) when invoking `train.py`. Needed for any
  training run on this machine going forward.

---

# PART 1 — TEUNet era (ARCHIVED, no longer in use)

> **Everything in this part uses the original TEUNet dataset, which was
> retired 2026-07-07 in favor of corr_medium/Step1 (Part 2).** This part is
> kept as a complete record of what was tried and why it was abandoned — 6
> fix attempts, a full poor-tier visual audit, and a firm "informational
> bottleneck" diagnosis. **Do not treat conclusions here as automatically
> true for Step1 data** — when checked directly (Part 2), Step1's own failure
> modes turned out to look meaningfully different from TEUNet's.

## Source data (TEUNet)

`/home/ameliacatala/Documents/preprocess/transfer_data/{teunet,gt}/` — 1973
paired patches (TEUNet probability grid + binary GT grid), 64x64x48 voxels,
~5mm resolution. Dice score distribution: good (>0.8): 1257, moderate
(0.5-0.8): 455, poor (<0.5): 261.

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

### v1 training results

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
**This finding turned out to generalize beyond TEUNet — see Part 2's fix
attempt 1, where the same signature reappeared on Step1 data.**

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

## TEUNet fix attempt 1 (tried 2026-06-27): test-time dilation — did not work

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

## TEUNet fix attempt 2 (tried 2026-06-27/28): retrain on TEUNet's dilated footprint — also did NOT work

Implemented in `xcube/models/diffusion.py`: new hparams `train_cond_footprint`
and `cond_grid_dilation_kernel`, a shared `encode_cond_grid()` helper (dilates
via `conv_grid` then encodes through the frozen VAE, used consistently in
training and at real inference), and a new branch in `forward()` that uses
the dilated-TEUNet encode as the structural topology + noising target (GT's
true features aligned onto it via `fill_to_grid`) instead of GT's own
footprint. New config `configs/gpr/gpr_diffusion_v2.yaml`
(`cond_grid_dilation_kernel: 5`, ~2-voxel margin), trained 50 epochs on the
full dataset (`checkpoints/gpr/Diffusion_stage2_v2/version_2`, val loss
plateaued ~0.28-0.33, comparable to v1). Full 198-sample evaluation
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

## TEUNet fix attempt 3 (tried 2026-06-28/29): classifier-free conditioning dropout — also did NOT meaningfully change anything

`use_classifier_free` and `classifier_free_prob` already existed as hparams
(unused before); the dropout mechanism (`conduct_classifier_free`) was
already fully wired into the `use_cond_grid_concat_cond` branch. New config
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

**Note (important context, understood only later — see Part 2): this
dropout trains "generate a pipe from nothing," a different skill from
"given a wrong-but-present hint, fix it."** It never actually confronted the
network with a hint that was *present but untrustworthy* — exactly the
situation the corrupted-conditioning idea in Part 2 targets directly.

## TEUNet fix attempt 4 (tried 2026-06-30): free structural generation at inference — catastrophic failure, IoU ≈ 0

Implemented in `scripts/run_stage2_inference_v4.py` (uses the v1 checkpoint,
no retraining). Instead of encoding TEUNet's grid and passing its topology as
`grids`, manually constructs a fully-dense coarse grid matching the VAE's
bottom level (`feat_depth = tree_depth-1 = 1`, `gap_stride = 2`,
`voxel_bound = [32, 32, 24]`, `voxel_sizes = voxel_size * gap_stride`,
`origins = voxel_sizes / 2`) and passes that as `grids`, while still passing
TEUNet's grid as a conditioning hint via `batch={DS.COND_PC: teunet_grid}`.
Full 198-sample eval (`scripts/results/stage2_full_test_v4_198samples.txt`):

| Tier | n | v4 IoU vs GT | TEUNet IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 126 | 0.004 | 0.816 | −0.811 |
| Moderate | 46 | 0.002 | 0.580 | −0.578 |
| Poor | 26 | 0.009 | 0.085 | −0.077 |
| **Overall** | **198** | **0.004** | **0.665** | **−0.661** |

The model filled the entire domain with a solid block (~165k voxels)
regardless of sample — visible in `~/Documents/stage2_visual_comparison_multi_v4.png`.
Root cause: the decoder was only ever trained to subdivide GT-shaped sparse
grids; given a fully-dense starting grid it has no learned behavior for
pruning and activates every voxel. This definitively rules out structural
confinement as the fixable bottleneck — the model had full freedom and
performed catastrophically worse, not better.

## Poor-Tier Visual Audit — TEUNet data (2026-07-02)

All 26 poor-tier test samples (indices 172–197) rendered side-by-side
(TEUNet input / v1 model output / ground truth) in four batches saved to
`~/Documents/poor_batch{1-4}_*.png`. Four distinct failure patterns
identified — **these "Pattern A-D" labels are specific to TEUNet's failure
modes; do not assume they describe Step1's data (they don't — see Part 2)**:

**Pattern A — Complete spatial miss (IoU = 0.000): 10 of 26 samples (38%)**
TEUNet places voxels in entirely the wrong physical location or finds
near-nothing. Model output stays at 0.000 — no correction is possible
because the conditioning signal carries zero spatial information about the
pipe's actual location. Includes cases (#176, #177, #183, #191) where
TEUNet outputs a thick rectangular slab in a small corner of the domain
while GT has two large parallel cylinders spanning the full length.

**Pattern B — Over-segmentation / wrong shape (IoU ≈ 0.05): 4 of 26 (15%)**
TEUNet finds a large amorphous blob roughly where the pipe is but produces
the wrong shape entirely (flat slab instead of cylinder). Model copies this
with marginal voxel count changes and no structural correction.

**Pattern C — Fragmentation (IoU = 0.10–0.30): 4 of 26 (15%)**
TEUNet finds voxels in the right region but they are scattered and
disconnected. **The only category where the model shows any meaningful
improvement**: #175 (0.242→0.295), #185 (0.241→0.296). Slight
consolidation of the fragmented signal occurs, but never a full clean
tube reconstruction.

**Pattern D — Complex geometry (IoU = 0.08–0.29): 8 of 26 (31%)**
GT contains multi-pipe arrangements, L/T junctions, or curved paths.
TEUNet partially captures these but gets the topology wrong. Model output
is near-identical to TEUNet (±0.01 IoU). Best cases in the entire poor
tier (#187 at 0.301, #192 at 0.278) fall here — TEUNet was close but
noise prevented it crossing the 0.3 threshold.

**Key finding**: Pattern A (38%) is entirely out of reach for any model
that conditions only on TEUNet's output — there is no information in the
conditioning signal about where the pipe is. Patterns B and D suggest the
bottleneck is shape/topology learning capacity. Only Pattern C shows the
model can do anything useful. Accessing pre-TEUNet signal (raw GPR scan or
intermediate representation) is the only realistic path to fixing Pattern A
— **this specific hope is what motivated requesting corr_medium; see Part 2
for how that actually turned out.**

## Visualization scripts (TEUNet era)

| Script | Purpose |
|---|---|
| `scripts/visualize_stage2_sample.py` | Single sample: TEUNet / v1 output / GT |
| `scripts/visualize_stage2_multi.py` | Multi-sample batch: TEUNet / v1 output / GT |
| `scripts/visualize_stage2_multi_v4.py` | Same but uses free-gen coarse grid (fix 4) |

All use matplotlib `ax.voxels()` for filled-cube rendering with a shared
bounding box from GT and a fixed camera angle (elev=20, azim=-60).

## TEUNet fix attempt 5 (tried 2026-07-03): bigger network (capacity) — still no improvement

Tested the "network is too small" theory: `configs/gpr/gpr_diffusion_v4.yaml`
keeps v1's plain GT footprint (no dilation, no classifier-free dropout, no
attention) but quadruples rough capacity — `model_channels` 32→64, one more
depth level (`channel_mult` [1,2]→[1,2,4]), `num_res_blocks` 1→2. Trained 50
epochs (`batch_size: 2`, ~19hr on the RTX 2000 Ada). Full 198-sample eval
(`scripts/results/stage2_full_test_v4capacity_198samples.txt`):

| Tier | n | v4-capacity IoU vs GT | TEUNet IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 126 | 0.780 | 0.816 | −0.036 |
| Moderate | 46 | 0.527 | 0.580 | −0.053 |
| Poor | 26 | 0.089 | 0.085 | +0.003 |
| **Overall** | **198** | **0.630** | **0.665** | **−0.035** |

Not only did more capacity fail to beat TEUNet's baseline, it landed
*below* the ~0.65-0.66 overall IoU that all three training-regime variants
(v1/v2/v3) converged to — network size was not the bottleneck. Combined
with the audit's finding that 38% of poor-tier failures are complete
spatial misses with zero recoverable signal in TEUNet's output, this
closes out the architectural/training-regime angle entirely.

## Post-processing test (tried 2026-07-03): morphological closing on model output

Cheap, no-retraining check: does bridging small gaps (morphological closing,
`scipy.ndimage.binary_closing`, 1-2 iterations) on the v1 model's *output*
voxels recover any IoU, especially on poor-tier fragmented samples?
`scripts/test_postprocess_closing.py`, full 198-sample run
(`scripts/results/postprocess_closing_198samples.txt`): **essentially zero
change everywhere** (good +0.000, moderate -0.000, poor +0.000 to +0.001).
The model's fragmented outputs aren't "almost right, just needs bridging" —
confirms the gaps are too large/structurally different for cheap geometric
cleanup, consistent with an information gap rather than a fixable shape
defect.

## TEUNet fix attempt 6 (tried 2026-07-06): hardness-based loss reweighting — also did NOT help

Theory: with every training sample weighted equally, the ~64% "good" tier
(where blind copy-through of TEUNet's hint is already near-optimal)
dominates the average loss, leaving little gradient incentive to learn real
correction behavior on the ~13% "poor" tier where it actually matters. Kept
v1's exact architecture and plain GT footprint (capacity and structural
fixes already ruled out) and changed only the loss: `configs/gpr/gpr_diffusion_v5.yaml`
sets `use_hardness_reweight: true`, `hardness_scale: 3.0`. New
`hardness_scale` param on `GPRDataset` (`xcube/data/gpr.py`) computes
`weight = 1 + hardness_scale * (1 - IoU(TEUNet, GT))` per sample (range
~1.16-3.61 across a random subset, mean ~2.05 — sensible spread, not
degenerate); `use_hardness_reweight` in `xcube/models/diffusion.py` switches
`compute_loss` from a flat mean MSE to a per-voxel-weighted mean (voxel
weight looked up via `jidx` from its sample's `DS.LOSS_WEIGHT`). Trained 50
epochs (`checkpoints/gpr/Diffusion_stage2_v5/version_0`, ~8hr — faster than
v4-capacity since this reused v1's smaller network). Full 198-sample eval
(`scripts/results/stage2_full_test_v5reweight_198samples.txt`):

| Tier | n | v5-reweight IoU vs GT | TEUNet IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 126 | 0.807 | 0.816 | −0.009 |
| Moderate | 46 | 0.573 | 0.580 | −0.007 |
| Poor | 26 | 0.089 | 0.085 | +0.004 |
| **Overall** | **198** | **0.658** | **0.665** | **−0.007** |

Overall lands within the same noise band as v1/v2/v3 (0.655/0.652/0.657) —
no regression like v4-capacity, but no real gain either. **Poor tier is the
key number: +0.004, identical to v1's +0.004 and v2's +0.004/v3's +0.005**
— despite up to ~3.6x more training loss weight on exactly these samples,
zero additional correction ability resulted. Strong evidence the model
isn't failing to prioritize hard cases (which reweighting would fix); it's
that TEUNet's signal genuinely contains no recoverable information for
those cases, so no amount of training emphasis can manufacture a correction
signal that isn't there.

## TEUNet era conclusion (status as of 2026-07-06)

Six fix attempts completed (three training-regime variants, one structural
free-gen experiment, one capacity increase, one loss-reweighting) plus a
full poor-tier visual audit and a post-processing gap-bridging test. All six
training variants land at or below the ~0.65-0.66 ceiling, just shy of
TEUNet's own 0.665 baseline, and the post-processing test rules out cheap
geometric cleanup too. Combined with the audit finding 38% of poor-tier
samples are complete spatial misses with zero recoverable signal, the
conclusion held firmly: the bottleneck is informational, not architectural,
training-regime, loss-weighting, or post-processing based.

**This is where TEUNet-data work stopped.** The team requested raw GPR scan
access (pre-TEUNet signal) as the one remaining untried angle. What actually
arrived in response is a different dataset from a different upstream model
(Step1/corr_medium) — not raw signal, and not TEUNet. See Part 2.

---

# PART 2 — Step1/corr_medium era (CURRENT, active dataset)

> **This is the dataset and checkpoints in active use.** TEUNet (Part 1) is
> retired. Fix attempts here are numbered fresh (Step1 fix attempt 1, 2, ...)
> — separately from Part 1's 6 TEUNet attempts — specifically so the two
> don't get conflated. As of this writing, only 2 fix attempts have actually
> been tried on this data (material conditioning, corrupted conditioning);
> none of Part 1's dilation/capacity/reweighting fixes have been re-tested
> here, and shouldn't be assumed to transfer.

## Why the switch: A-scan access declined, corr_medium arrived instead

The team's request (end of Part 1) was for **raw GPR scan access** — the
true rawest signal here would be the **A-scan** (the raw radar waveform,
pre-*any* neural network). **The data owner declined to provide A-scan
access, for reasons not stated.** This request is considered closed —
not a live option going forward.

What arrived instead (2026-07-07) was `corr_medium`: a GT/prediction pair
from a **Step1 model** — a different, earlier-pipeline-stage model, not
TEUNet, and not raw signal. It's another model's processed output, same
category of thing as TEUNet's output was, just from a different model and a
different (larger, differently-balanced) sample set.

## New data source: corr_medium Step1 predictions (arrived 2026-07-07)

`corr_medium_gt_voxel_radius.h5` and `corr_medium_step1_pred_voxel_radius.h5`,
10,000 paired samples, in
`/home/ameliacatala/Documents/corr_medium_gt_voxel_radius/`.

### Preprocessing (`datagen/preprocess_gpr_corr_medium.py`)

Converts this pair into the exact same `.pkl` format `preprocess_gpr.py`
produces, so it drops into the existing `GPRDataset`/Stage 1/Stage 2 code
unchanged. Key differences from the original TEUNet source format:

- Single `.h5` file per split holding all 10,000 samples (vs. one `.h5` file
  per sample originally).
- Volume axis order in the source is (D,H,W) = (48,64,64); transposed to
  (X,Y,Z) = (64,64,48) to match `GPRDataset`'s existing convention.
- Occupancy (`pipe_mask`) is already boolean on both GT and prediction sides
  — no probability threshold needed. The prediction's `confidence` field
  (dense probability at occupied voxels) fills the role TEUNet's continuous
  `input_prob` played before.
- Isotropic 0.005m voxel spacing (vs. TEUNet's near-isotropic but not exactly
  equal per-axis spacing).

Same per-sample Dice-score tiering and 80/10/10 stratified split as the
original pipeline.

**Result**: 10,000/10,000 samples processed, 0 skipped, all spot-checks
passed. Output: `/home/ameliacatala/Documents/preprocess/data_full/gpr_corr_medium/0.005/`
(6.7G), splits `train.lst`/`val.lst`/`test.lst` (8,002/999/999 entries).

Tier breakdown — notably different balance from the original TEUNet dataset:

| Tier | corr_medium (n=10,000) | Original TEUNet (n=1,973) |
|---|---|---|
| Good (>0.8) | 6,143 (61.4%) | 1,257 (63.7%) |
| Moderate (0.5-0.8) | 3,725 (37.3%) | 455 (23.1%) |
| Poor (<0.5) | 132 (1.3%) | 261 (13.2%) |

Poor tier is far smaller proportionally here (1.3% vs. 13.2%).

**Important caveat (from `README_corr_medium_voxel_radius.md`, packaged with
the source data): this is NOT a harder/OOD signal.** The dataset is
explicitly medium-difficulty and **in-distribution** relative to the Step1
model's own training set (checkpoint `gpr_topo_step1/runs/real_v9/best.pt`) —
the README states plainly: *"the prediction quality is expected to look
strong. A diffusion model may not necessarily improve these results much."*
The README also names two specific known failure modes in this data — pipes
near a patch's bottom sometimes missed entirely, and pipes cut by the patch
boundary becoming poor-quality half-pipes.

The README also documents two per-voxel fields present in both H5 files:
`radius_m` (physical pipe radius per voxel) and `material` (pipe material
class per voxel). `radius_m` was skipped (`pipe_mask` is already a
radius-expanded mask, so it's largely redundant with occupancy shape);
`material` became the material-conditioning fix attempt below.

**Verified 2026-07-08**: checked a random 200-sample subset directly — worst
Dice was 0.39, zero samples at exactly 0.000, consistent with the README's
overall worst of 0.234. **Confirms no TEUNet-style "Pattern A" complete
misses in this dataset** — Step1's failure modes are categorically
different from TEUNet's (further confirmed visually below).

## Material conditioning (implemented 2026-07-08)

`corr_medium`'s H5 files also carry a per-voxel `material` class (int16, `-1`
outside `pipe_mask`; observed classes `{0,1,2,3}` across GT+prediction).

**Key finding that reshaped the implementation**: traced how conditioning
actually reaches the model and found `confidence`/`input_prob` — present
since the original TEUNet-era pipeline — was **never actually consumed**. It's
wired through `DS.INPUT_INTENSITY`, gated by VAE hparam `use_input_intensity`
(`false` in `gpr_vae.yaml`), and even when enabled, Stage 2's
`encode_cond_grid()` (which reuses the **frozen** Stage 1 VAE encoder to
encode the conditioning grid) never passed intensity data to it at all —
only grid positions. This meant the frozen encoder's first layer (`mix_fc`)
was never trained to accept any extra per-voxel channel, so material
conditioning requires retraining Stage 1's encoder with the new input
dimension, not just a Stage-2-only change. Since Stage 1 needed retraining
anyway (new corr_medium data), this is folded into that retrain rather than
being extra overhead.

### Design

Mirrors the existing `use_input_semantic` pattern in
`xcube/modules/autoencoding/base_encoder.py` (categorical class index through
a learned `nn.Embedding`, concatenated onto the position-embedded features
before `mix_fc`) rather than one-hot encoding manually.

- `xcube/data/base.py`: two new `DatasetSpec` entries —
  `INPUT_MATERIAL` (aligned with whatever `INPUT_PC` currently is: GT's
  material when `input_key="target_grid"`, the Step1 prediction's material
  otherwise) and `COND_MATERIAL` (always the Step1 prediction's material,
  aligned with `COND_PC`, since Stage 2 encodes that grid separately from
  `INPUT_PC` through the same frozen encoder).
- `xcube/modules/autoencoding/base_encoder.py`: new `use_input_material`,
  `num_material`, `dim_material` hparams; `nn.Embedding(num_material,
  dim_material)` on the raw class index, concatenated into `unet_feat`.
- `xcube/models/autoencoder.py`: `get_dataset_spec()` requests
  `DS.INPUT_MATERIAL` when the flag is set.
- `xcube/data/gpr.py`: `_get_item` populates `DS.INPUT_MATERIAL` from
  `target_material` or `input_material` in the `.pkl` depending on
  `input_key`, and `DS.COND_MATERIAL` always from `input_material`.
- `xcube/models/diffusion.py`: new `use_cond_material` hparam;
  `encode_cond_grid()` takes an optional `cond_material` argument and passes
  `DS.INPUT_MATERIAL` through to the frozen VAE's `_encode` call alongside
  `DS.INPUT_PC`; all three call sites (`_forward_cond`, `forward`'s
  `train_cond_footprint` branch, `evaluation_api`) updated; `get_dataset_spec`
  requests `DS.COND_MATERIAL` when the flag is set.
- `datagen/preprocess_gpr_corr_medium.py`: `build_sample()` now also extracts
  `target_material` and `input_material`, stored as `int8`.
- New configs: `configs/gpr/gpr_vae_corr_medium.yaml` (Stage 1, retrained on
  corr_medium, `use_input_material: true`, `num_material: 4`,
  `dim_material: 8`) and `configs/gpr/gpr_diffusion_v6_material.yaml`
  (Stage 2, `use_cond_material: true`, otherwise v1's plain
  architecture/footprint).

## Stage 1 VAE retrain: corr_medium + material (finished 2026-07-09)

`checkpoints/gpr/VAE_stage1_corr_medium/version_1`, 50 epochs on the full
10,000-sample corr_medium dataset with material conditioning. Validation
loss 0.85 → 0.31 (train 1.34 → 0.27), structure accuracy (finest tree level,
`struct-acc-0`) 98.8% → 99.6%, still trending slightly at epoch 49 (not
fully flattened like the original 100-epoch TEUNet-era Stage 1 run). The
coarser tree level (`struct-acc-1`) sits at a trivial flat 100% throughout.
Curve plots: `vae_stage1_corr_medium_training.png`.

Reconstruction visuals (`scripts/visualize_vae_reconstruction.py`): clean
white-background renders of one sample's ground-truth input, the VAE's
actual latent embedding (16-dim per voxel, PCA→RGB via HSV mapping so it
stays bright/readable rather than muddy), the coarse depth-1 structure, and
the final depth-0 reconstruction — `vae_reconstruction_{before,embedding,
during,after}.png` plus a combined `vae_reconstruction_visual.png`.
Reconstruction closely matches the input, consistent with measured accuracy.

**Environment note**: needed the `CC=gcc-12 CXX=g++-12` fix described at the
top of this file.

## Step1 fix attempt 1 (checked 2026-07-11): material conditioning (v6) — did NOT help

`configs/gpr/gpr_diffusion_v6_material.yaml`, using the corr_medium+material
VAE above. Smoke-tested (5 train/2 val batches) before the full run — clean.
Trained 50 epochs (`checkpoints/gpr/Diffusion_stage2_v6_material/version_0`,
~41hr at ~13.75 it/s / 40,260 steps/epoch — corr_medium's 5x larger sample
count vs. TEUNet is the main driver). Training itself froze silently for
~11 hours partway through (a background data-loading worker died after a
transient shared-memory error and the main process hung waiting for a batch
that would never arrive — a known PyTorch multi-worker deadlock, not
specific to this project); caught by checking log-file timestamps rather
than trusting that the process was still alive, killed, and resumed cleanly
from the last good checkpoint (epoch 14) with no meaningful progress lost.

Evaluated at **epoch 46 of 50** (new `scripts/run_stage2_inference_v6.py`,
adapted from the TEUNet-era `run_stage2_inference_v5.py`: corr_medium's
999-sample test set instead of TEUNet's 198, baseline is Step1's own
prediction instead of TEUNet's, and per-sample Dice-based tiering computed
on the fly since corr_medium's `test.lst` isn't index-tiered the same fixed
way TEUNet's was). Full 999-sample eval
(`scripts/results/stage2_full_test_v6material_epoch46_999samples.txt`):

| Tier | n | v6 IoU vs GT | Step1 IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 614 | 0.742 | 0.767 | −0.025 |
| Moderate | 372 | 0.537 | 0.556 | −0.018 |
| Poor | 13 | 0.289 | 0.297 | −0.007 |
| **Overall** | **999** | **0.660** | **0.682** | **−0.023** |

**Negative across every tier, including poor** — actually a meaningfully
*worse* pattern than any TEUNet-era attempt, where poor tier was always the
one bright spot (+0.004 to +0.005 there; −0.007 here). This resolved an open
question: whether the TEUNet-era "model just copies its conditioning
through" diagnosis (built entirely on TEUNet-conditioned training) would
transfer to Step1's different, less catastrophic error profile. It does.
Training was left running toward epoch 50 in the background (converges by
~epoch 25-30 historically, so this read was expected to hold).

## Visual audit of Step1's own failure modes (2026-07-11)

**This was the first time Step1's actual failures were looked at directly —
they had been wrongly assumed to look like TEUNet's Pattern A-D (Part 1)
before this.** New `scripts/visualize_stage2_multi_v6.py`: scans all 999 test
samples' Step1-vs-GT Dice on the fly to find every "poor" tier case (13
found), renders all 13 plus 2 good/2 moderate for context (Step1 input / v6
output / GT), saved to `stage2_v6_visual_comparison_multi.png`.

**Finding 1 (confirms the copy-through diagnosis, now visible directly, not
just inferred from IoU numbers): in every single row, the v6 model's output
voxel count and shape track Step1's input almost exactly** — including on
rows where Step1's shape is completely wrong.

**Finding 2 (Step1's error profile is genuinely different from TEUNet's —
the reason "stop assuming Pattern A/B/C/D applies here" matters):**
- **Wrong shape, similar voxel count** (idx=987: Step1 has 9,474 voxels vs.
  GT's 10,271 — close in size, but Step1's shape is a blocky, disconnected
  cluster where GT is a set of clean parallel cylinders. Same at idx=986).
  Step1 found *something* of roughly the right size, just organized wrong —
  unlike TEUNet's Pattern A, where the conditioning signal often carried
  zero spatial information at all.
- **Large undercounts of a bigger true structure** (idx=996: Step1 has
  16,714 voxels vs. GT's 45,612 — less than half; idx=997: 8,696 vs.
  12,218). Step1 found a small piece of a much larger junction and stopped.
- **A few genuinely close cases** (idx=988, idx=990) — topology mostly
  right, minor branch/length differences.

**No TEUNet-style complete misses (zero-overlap cases) appeared anywhere in
this sample.** Step1 reliably finds *something* real; the failure mode is
wrong organization or incompleteness, not "found nothing."

**Implication for what to try next**: since v6 copies Step1's output almost
exactly even when Step1's shape is plausible-but-wrong (idx=987), a training
approach that specifically teaches "the hint can be the right size/region
but structurally wrong" is a better-targeted fix for *this* data's failure
mode than anything tried on TEUNet data — this is the reasoning behind Step1
fix attempt 2, below.

## Step1 fix attempt 2 (launched 2026-07-11): corrupted conditioning (v7) — in progress

**Theory**: on most training examples (in both TEUNet's and Step1's data),
the conditioning hint is already close to GT, so "just copy the hint"
already scores well on average — the network never had training pressure to
learn real correction behavior. Classifier-free dropout (Part 1, TEUNet fix
attempt 3) tested a related but different idea — sometimes removing the
hint *entirely* — and didn't help; that trains "generate from nothing," not
"given a wrong hint, fix it." This is the first attempt that makes the hint
*reliably wrong* (not just occasionally absent) on every training example,
so copy-through can no longer be a safe default strategy.

**Implementation** — `xcube/data/gpr.py`, new `GPRDataset` constructor
kwargs `use_corrupted_cond`, `cond_corrupt_drop_prob` (default 0.15),
`cond_corrupt_add_ratio` (default 0.15), `cond_corrupt_jitter_range`
(default 2 voxels): a new `_corrupt_cond_grid()` method randomly drops a
fraction of `COND_PC`'s voxels, then adds jittered-copy noise voxels near
the kept ones (simulating false-positive/misplaced detections), rebuilding a
new `fvdb.GridBatch` via `set_from_ijk` (matching the original grid's own
`voxel_sizes`/`origins`) and a correspondingly resized material tensor.
**Gated to the training split only** (`self.use_corrupted_cond = use_corrupted_cond
and split == 'train'`, checked in code, not just by only setting the flag in
`train_kwargs`) — validation and test always see the real, uncorrupted
Step1 prediction, so evaluation stays honest and comparable to prior
attempts. New config `configs/gpr/gpr_diffusion_v7_corrupted.yaml` — same
architecture, dataset, and material conditioning as v6, adding corrupted
conditioning as the one new variable.

**Verified directly (not just "didn't crash") before launching**: ran the
dataset in isolation and confirmed (a) corrupted voxel counts differ
meaningfully from clean ones (~10% net change per sample, consistent with
15% drop + 15% jittered-add with some overlap from deduplication), (b)
material stays perfectly aligned 1:1 with grid voxel count in both the
clean and corrupted cases, and (c) `use_corrupted_cond` is force-disabled on
a `val`-split dataset even if passed `True`, confirming the safety gate
works. Smoke-tested end-to-end (5 train/2 val batches) — clean.

Launched the full 50-epoch run in the user's own terminal (nohup,
`~/v7_corrupted_training.log`), `checkpoints/gpr/Diffusion_stage2_v7_corrupted/version_0`,
started 2026-07-11 while v6's own training still had ~1-1.5hrs left on the
same GPU (VRAM headroom is ample — ~1.4GB/16GB used by v6 alone — so no risk
of the two runs interfering beyond a modest, temporary compute-sharing
slowdown to both). Same architecture/dataset size as v6, so **~41hr expected
for the full 50 epochs**.

## VAE round-trip check on corr_medium (2026-07-13): confirms the diffusion training loop is not the bottleneck

**Context**: back in the TEUNet era (Part 1), a diagnostic found that skipping
the entire diffusion process — running TEUNet's flawed grid through the
frozen Stage 1 VAE's encoder+decoder directly, zero noise, zero denoising —
scored almost identically to the full trained diffusion model. That was the
strongest evidence the diffusion model wasn't doing real corrective work.
This check had never been re-run on corr_medium/Step1 data; v6/v7's design
assumed it carried over without verifying it.

New script `scripts/test_vae_roundtrip_corr_medium.py` (adapted from the
TEUNet-era `test_vae_roundtrip.py`, pointed at the corr_medium VAE checkpoint,
including material conditioning the same way v6/v7 do). Full 999-sample
corr_medium test set, on-the-fly Dice-based tiering:

| Tier | n | VAE round-trip only (no diffusion) | Step1 baseline | Diff |
|---|---|---|---|---|
| Good | 614 | 0.744 | 0.767 | −0.023 |
| Moderate | 372 | 0.544 | 0.556 | −0.011 |
| Poor | 13 | 0.293 | 0.297 | −0.004 |
| **Overall** | **999** | **0.664** | **0.682** | **−0.019** |

Compare directly to v6's full trained diffusion model (already logged above):

| Tier | v6 full model | VAE round-trip only | Gap |
|---|---|---|---|
| Good | 0.742 | 0.744 | +0.002 |
| Moderate | 0.537 | 0.544 | +0.007 |
| Poor | 0.289 | 0.293 | +0.004 |
| **Overall** | **0.660** | **0.664** | **+0.004** |

**Confirmed: the same signature found on TEUNet data reproduces on
corr_medium/Step1 data.** Skipping the diffusion model entirely gives results
statistically indistinguishable from (marginally better than) the full
trained v6 model. **The diffusion training loop is not the bottleneck — the
ceiling is set by the frozen Stage 1 VAE's encode/decode capacity.** This
retroactively explains why all 8 Stage-2-training-side fix attempts (6
TEUNet-era + material conditioning + corrupted conditioning) landed in the
same narrow band: none of them could have worked, because none touched the
actual bottleneck.

## Structural confinement diagnostic on corr_medium (2026-07-14): confirmed as a real, tier-dependent ceiling

Directly tests the "structural confinement" theory (Part 1) at the Stage 1
VAE level, independent of Stage 2/diffusion entirely. The theory: the VAE
decoder (`sunet.py`'s `decode()`) starts from the coarsest level's structure —
exactly `latent.grid`, the grid produced by encoding whatever input it's
given — and can only ever *keep or prune* cells already present in that
coarse grid; sparse convs can't activate a coarse cell that wasn't already
there. So any part of GT's true structure whose coarse (2x) parent cell is
entirely absent from Step1's own coarse footprint is architecturally
impossible for the decoder to recover, no matter how it was trained.

New script `scripts/test_vae_structural_confinement.py`, using fvdb's
`GridBatch.coarsened_grid()` (the same helper `base_loss.py` uses to compute
per-level structure accuracy, so this matches the network's own notion of
"coarse level" exactly) to measure, per test sample: how much of GT's coarse
structure is even reachable at all (i.e. overlaps `latent.grid`, Step1's own
coarse footprint), how much of that reachable part the decoder actually
recovers, and — as a hard correctness check of the theory itself — whether
the decoder ever produces structure outside the reachable region at all.

**Sanity check (proof, not measurement): 0 escaped voxels across all 999
samples.** The decoder never once produced structure outside its
architecturally reachable region — confinement is an exact, absolute
property of this architecture, not a tendency.

**Cost of confinement, by tier** (full 999-sample test set,
`scripts/results/structural_confinement_999samples.txt`):

| Tier | n | % of GT structure architecturally unreachable | Recovery rate within reachable room |
|---|---|---|---|
| Good | 614 | 1.9% | 95.9% |
| Moderate | 372 | 13.2% | 94.0% |
| Poor | 13 | **48.2%** | 91.5% |

**On poor-tier samples, nearly half of GT's true structure sits in a coarse
region Step1's own prediction never hinted at — the decoder has zero chance
of recovering it regardless of training, because it's architecturally boxed
out of that space before Stage 2 even runs.** Moderate tier shows a smaller
but still real 13% ceiling. This directly explains why all 8 Stage-2
training-side fixes landed flat: they were tuning a stage that was never the
bottleneck.

**Important nuance**: recovery-within-reachable-room is already high (92-96%)
across every tier — the decoder isn't wasting the room it does have. This
means simply widening the input footprint at test time (TEUNet fix attempts
1-2, Part 1 — both failed) isn't sufficient by itself, because the decoder
was never *trained* to trust an uncertain wider region; it has no learned
behavior for productively using room it never practiced with. **The real fix
has to change how Stage 1 is trained** — teaching the VAE's own coarse-level
structure prediction to productively use a wider, less literal candidate
region — not a Stage 2 change and not just a test-time dilation.

## v7 (corrupted conditioning) finished; Stage 1 coarse-dilation VAE finished but was a no-op — real dilation implemented and relaunched (2026-07-15)

**v7 finished its full 50 epochs** (`checkpoints/gpr/Diffusion_stage2_v7_corrupted/version_0`,
`Trainer.fit stopped: max_epochs=50 reached`). Not yet run through the full
999-sample evaluation script — still pending, expected to land ~flat per the
VAE round-trip finding, but not yet a confirmed number.

**The Stage 1 coarse-dilation VAE retrain (`VAE_stage1_corr_medium_v2_coarse_dilation/version_0`,
described in the previous entry) also finished its 50 epochs, but evaluating
it exposed a critical bug: its "dilation" never actually happened.**

`decode()`'s dilation step (`xcube/modules/autoencoding/sunet.py`) used
`GridBatch.conv_grid(kernel_size, stride=1)`, believing it widens the active
coarse footprint by a margin. **Verified directly this session (isolated
test, independent of any checkpoint): `conv_grid(k, 1)` returns a grid with
the exact same active voxel coordinates as the input, for every kernel size
tried (1, 2, 3, 5, 7) — 5,326 voxels in, 5,326 voxels out, identical
coordinates.** Per fvdb's own docstring, `conv_grid` computes the output
structure of a same-position convolution (aggregating neighbor *features*
into existing voxels), not a morphological dilation that adds new voxels —
the wrong primitive for this purpose.

**This invalidates the "widening didn't help" conclusion in three places**,
not just this one:
- TEUNet fix attempt 1 (test-time dilation, Part 1) — used the identical
  `conv_grid` call.
- TEUNet fix attempt 2 (retrain on a "dilated" footprint, Part 1) — same call.
- This Stage 1 coarse-dilation retrain (`version_0`) — same call, so its
  round-trip/confinement numbers (0.655 overall IoU, 1.9/13.2/48.2%
  unreachable by tier) came out statistically identical to the pre-dilation
  VAE not because widening the room doesn't help, but because no widening
  ever occurred — `version_0` trained on the exact same footprint as the
  original VAE, just with different random seed/epoch noise on top. None of
  these three results should be read as evidence against the underlying idea.

**Fix**: `set_from_ijk`'s `pad_min`/`pad_max` args are the real per-voxel
dilation primitive in this codebase (already used correctly elsewhere, e.g.
`xcube/modules/autoencoding/losses/nksr_loss.py`'s `_get_svh_samples`, and
confirmed in fvdb's C++ source, `GridBatch.cpp`'s
`buildPaddedGridFromCoords`). Replaced the `conv_grid` call in `decode()`
with building a new `GridBatch` via `set_from_ijk(x.grid.ijk, pad_min=[-m]*3,
pad_max=[m]*3, ...)` where `m = (coarse_dilation_kernel - 1) // 2`. **Verified
this actually grows the grid**: same 5,326-voxel test sample went to 9,745
voxels with `margin=1` (`kernel_size=3`). Smoke-tested (5 train/2 val
batches) — clean, loss decreasing normally.

**Relaunched the Stage 1 retrain with the real fix**
(`checkpoints/gpr/VAE_stage1_corr_medium_v2_coarse_dilation/version_1`,
started 2026-07-15 ~10:35, same config/kernel size as `version_0`, ~20hr
expected based on the original corr_medium VAE's runtime for the same
dataset size). `version_0` (the no-op run) is left in place for reference,
not deleted, but should not be used for any further comparison — treat it as
equivalent to "no dilation" going forward, not as a real data point on the
dilation idea.

## Two follow-up diagnostics before committing to the retrain (2026-07-15): margin size is too small, and coarse-level pruning is a rubber stamp

Before launching the real-dilation retrain, two questions were raised about
the fix's scope: (1) what if the missing structure sits *farther away* than
the dilation margin reaches, and (2) what if the decoder also needs to
*remove* wrong candidate blocks, not just gain new ones. Built two new
diagnostics to check both empirically, using the original (non-dilated,
already-verified) corr_medium VAE checkpoint — these are geometric/behavioral
questions about the existing data and model, independent of whichever
dilation fix is being tested.

**`scripts/test_unreachable_distance.py`** — for every GT coarse cell found
unreachable in the structural-confinement test, measures the (Chebyshev)
distance to the nearest cell in Step1's own coarse footprint, bucketed against
plausible dilation margins:

| Tier | margin=1 (kernel=3, current retrain) | margin=2 (kernel=5) | margin=3 (kernel=7) | margin=5 (kernel=11) |
|---|---|---|---|---|
| Good | 44.9% | 61.1% | 74.0% | 90.6% |
| Moderate | 24.7% | 41.0% | 55.1% | 76.6% |
| Poor | **10.3%** | 22.7% | 35.3% | 54.3% |
| Overall | 26.8% | 42.6% | 56.4% | 76.8% |

**The margin the currently-running retrain uses (`coarse_dilation_kernel: 3`,
i.e. margin=1) only reaches ~10% of poor tier's missing structure.** Even a
much larger margin=5 (kernel=11) caps out at ~54% on poor tier — the missing
structure there is often not a "near miss," consistent with the README's
noted failure modes (pipe missed entirely near patch bottom, pipe cut by
patch boundary). Good tier's misses are mostly near (90.6% reachable by
margin=5), so the fix's value is real but tier-dependent — it should help
good/moderate tier's residual gap far more than poor tier's.

**`scripts/test_false_positive_pruning.py`** — checks whether the decoder
actually drops coarse candidate cells from Step1's own footprint that don't
correspond to real GT structure. **Result: literally 0/2,160,938 false-positive
coarse cells were dropped at the coarse level, across all 999 test
samples (100.000% survival) — verified not to be a measurement bug (`res.
structure_grid[1]` is coordinate-for-coordinate identical to the input
`latent.grid` for every sample checked).** Root cause is the same one already
identified for the "add" side: Stage 1 trains by self-reconstructing GT, so
the coarse footprint it's handed during training is always exactly correct —
the coarsest-level `struct_conv` has literally never seen a training example
where the right answer was "discard this candidate," so it converged to
always keep everything.

**This sounds worse than it turns out to be in practice**: a follow-up check
(30-sample spot check, not yet the full 999) found that even though the
coarse level formally "keeps" every false-positive candidate, only **28.6%**
of those regions end up with any actually-occupied voxel in the *final*,
finest-resolution output — the finer decode levels do real (if imperfect)
cleanup downstream of the coarse rubber-stamp. So "remove wrong blocks" isn't
fully broken, just handled later and leakily (roughly 1 in 4 false-positive
regions survives into the final shape) rather than at the coarse gate where
it architecturally could be cheaper/cleaner to do.

## Kernel size decided (kernel=7), and why `confidence`/`radius_m` were NOT added to this retrain (2026-07-15)

Chose **`coarse_dilation_kernel: 7`** (margin=3, ~35% poor-tier / ~74%
good-tier reach per the distance table above) over margin=1 (too small,
~10% poor-tier reach) or margin=5 (~54% poor-tier reach, but a bigger,
less-tested jump in untrusted candidate room, closer to the regime that
collapsed to ~0 IoU in TEUNet fix attempt 4). Config
(`configs/gpr/gpr_vae_corr_medium_v2_coarse_dilation.yaml`) updated and
re-smoke-tested (5 train/2 val batches) — clean, similar speed to kernel=3
(~2.5 it/s vs. ~2.7 it/s, negligible overhead from the extra candidate cells).

**Also considered, this session: should the retrain also add `confidence`
(Step1's per-voxel prediction confidence, already extracted into the `.pkl`
as `input_prob`) or `radius_m` (per-voxel pipe radius) as extra VAE inputs?**
Decided **no** to both, for this retrain specifically:

- **`radius_m`**: unchanged from the original decision when material
  conditioning was added (see "Material conditioning" section above) — it
  only describes the thickness of voxels *already known* to be pipe, and
  carries no signal about where structure is missing or which coarse
  candidates are false. Not relevant to either open problem (add or remove).
- **`confidence`**: a more interesting candidate than radius, since it's
  continuous and could plausibly signal "trust this region" / "this is a
  shaky guess" — directly relevant to both the add and remove problems. But
  it is **not a simple flag flip** the way material was. `use_input_intensity`
  / `DS.INPUT_INTENSITY` already exist in the code (`xcube/modules/
  autoencoding/base_encoder.py`, `xcube/data/gpr.py`) but are gated off
  (`use_input_intensity: false`) and were never actually exercised even when
  nominally on (same finding as the pre-existing material section above).
  **The deeper problem**: ground truth has no natural "confidence" value —
  it's simply true. Material had a natural GT-side counterpart (`target_
  material`, a real physical attribute), so Stage 1's self-reconstruction
  training gave the material embedding real, varying signal to learn from.
  Confidence doesn't: feeding Stage 1 training a constant placeholder (e.g.
  "always fully confident") for GT would train the intensity channel on a
  value that never varies, which teaches the network nothing about it —
  **the same "flag is wired up but never meaningfully exercised" failure
  shape already found twice this session** (the no-op `conv_grid` dilation,
  and the coarse-level rubber-stamp pruning). Bolting a similarly-hollow fix
  onto this retrain would conflate two hypotheses in one ~20hr run and risk
  a third false-negative result. **Decision: hold confidence conditioning as
  its own, separately-designed experiment** (most likely needs to enter
  through Stage 2's non-frozen diffusion layers directly, rather than through
  the frozen Stage-1 encoder, which has no honest way to train on it) —
  not attempted yet, do not conflate with the dilation retrain below.

## Kernel=7 result (`version_1`), a checkpoint-path bug found while investigating it, corrected numbers, and the balance_struct_loss fix's real (negative) result (evaluated 2026-07-16/17)

`version_1` finished all 50 epochs cleanly. **First evaluation numbers logged
here were wrong**, due to a bug in the diagnostic scripts themselves —
corrected below.

**The bug**: `test_vae_roundtrip_corr_medium_v2.py` and
`test_vae_structural_confinement_v2.py` had their checkpoint directory
hardcoded to `version_0/checkpoints` when originally written, and this was
never updated after `version_1` (or later `version_2`) was trained. So every
prior run of these two scripts against "kernel=7" was actually loading
**`version_0`'s weights (trained under the old no-op `conv_grid` bug, i.e.
never practiced any real dilation at all) combined with kernel=7 hparams
applied only at inference time** — the exact same failure mode as the
TEUNet-era test-time-only dilation experiment (Part 1, fix attempt 1), not a
real test of a checkpoint actually trained with real dilation. Confirmed
directly: on one sample, `version_0`'s weights under kernel=7 runtime hparams
produce a massive **over-fill** (115,144 voxels vs. GT's 16,105, coarsest
level keeps 100% of the 21,518-cell dilated candidate pool — the same
rubber-stamp behavior as always, just applied to a 4x bigger pool), a
completely different failure shape from what was reported (which described
under-filling). Both scripts fixed to take the version directory as an
argv parameter (`python scripts/test_vae_roundtrip_corr_medium_v2.py version_1`,
`version_2`, etc.) instead of a hardcoded path, defaulting to `version_2`.

**Corrected numbers, re-run directly against each real checkpoint by path:**

| Metric | kernel=1 (original) | kernel=7, no balance (`version_1`, corrected) | kernel=7 + balance (`version_2`, corrected) |
|---|---|---|---|
| Round-trip IoU, overall | 0.664 | 0.131 | 0.117 |
| Round-trip IoU, poor tier | 0.293 | 0.087 | 0.077 |
| Poor-tier unreachable % (ceiling) | 48.2% | 26.7% | 26.7% |
| Recovery within reachable room, overall | ~93-96% | **30.8%** | **27.1%** |

(The single-sample ad-hoc check from the previous session — 3,248 kept out of
21,518 candidates for `version_1` — was done by explicit hardcoded path, not
through the buggy script, so it was correct all along and is consistent with
these corrected aggregate numbers.)

**The `balance_struct_loss` fix made no meaningful difference — if anything,
very slightly worse** (recovery-within-reach 30.8% → 27.1%, round-trip IoU
0.131 → 0.117, both within noise of each other but showing no sign of
recovery). The class-imbalance theory (dilation floods the coarsest level
with ~4x more empty candidates, plain cross-entropy has no way to account for
it) was a reasonable, well-motivated hypothesis, and the mechanism itself
(inverse-frequency weighting, verified to actually engage via the config —
`balance_struct_loss: true` resolves correctly through `DictConfig.get()`,
this was checked directly to rule out yet another silent-flag failure) works
as designed. **It just isn't the fix for this collapse.** The real cause is
still unresolved, but two real, correctly-measured training attempts now
agree: giving this network's coarse-level decision a ~4x wider, mostly-empty
candidate pool during training collapses its judgment (recovery-within-reach
~93-96% → ~27-31%) regardless of whether the resulting class imbalance in the
loss is corrected.

## Clarified: confidence-conditioning does NOT substitute for padding on the confinement question (2026-07-17)

Important correction to the previous entry's framing, raised directly by the
user: confinement is an exact architectural property (0 escaped voxels, always)
— the decoder can only keep-or-discard cells already in its candidate pool, and
**no amount of extra per-voxel signal (confidence or otherwise) lets it produce
structure in a region that was never offered as a candidate at all.**
Confidence-conditioning targets a different problem (judging what's
trustworthy among reachable cells); it is not a substitute fix for the
confinement ceiling specifically. If closing that ceiling is the goal, some
form of candidate-pool widening is architecturally required — full stop.

**What the two failed attempts actually ruled out is narrower than "widening
doesn't work"**: both used `kernel_size=7` (margin=3, ~4x candidate growth in
one jump), and both collapsed the network's judgment. **A small, genuinely-
real widening has never actually been tested** — the original `kernel_size=3`
run (`version_0`) was the no-op `conv_grid` bug, not real dilation at all.

**Decision: test real `kernel_size=3` next** (margin=1, ~1.5-2x candidate
growth, much gentler than kernel=7's ~4x) — on the theory that the judgment
collapse was driven by the size of the jump, not by widening per se. Smaller
predicted ceiling gain (~10% of poor tier's missing structure recoverable,
vs. kernel=7's ~35%), but a real, previously-untested data point.
`balance_struct_loss` deliberately left **off** for this run, to isolate
margin size as the only new variable (kernel=7 already showed balancing
doesn't clearly help either way, so conflating it here would muddy the
result). Config (`configs/gpr/gpr_vae_corr_medium_v2_coarse_dilation.yaml`)
updated, smoke-tested (5 train/2 val batches) — clean.

## Real kernel=3 result (`version_3`, evaluated 2026-07-20): smaller collapse, but still a net regression — structural-dilation path closed for real

`version_3` (real `kernel_size=3`, no loss balancing) finished cleanly.
Evaluated with the corrected, argv-parameterized scripts:

| Metric | Original (no dilation) | kernel=7 (2 attempts) | kernel=3, real (`version_3`) |
|---|---|---|---|
| Round-trip IoU, overall | 0.664 | 0.117-0.131 | **0.300** |
| Round-trip IoU, poor tier | 0.293 | 0.077-0.087 | **0.166** |
| Poor-tier unreachable % (ceiling) | 48.2% | 26.7% | 41.4% (close to the ~43% predicted) |
| Recovery within reachable room, overall | ~93-96% | 27.1-30.8% | **56.6%** |

**Clear dose-response confirmed**: a gentler margin causes a smaller collapse
(56.6% recovery vs. kernel=7's 27-31%) and a smaller ceiling improvement
(poor-tier unreachable 48.2%→41.4%, vs. kernel=7's →26.7%) — both exactly as
the distance analysis and the "size of the jump matters" theory predicted.
**But even this smallest real margin still nets clearly worse than not
dilating at all** (overall IoU 0.300 vs. 0.664; poor tier 0.166 vs. 0.293).
Three real attempts now (kernel=7 alone, kernel=7+balance, kernel=3 alone)
all land net-negative vs. the undilated baseline.

**Per the decision rule set before this run: this closes the structural-
dilation path.** The evidence now says the problem is "widening the coarse
candidate pool during training at all," for this network/training setup, not
a matter of finding the right margin size or the right loss weighting.
**The original, undilated VAE (`VAE_stage1_corr_medium/version_1`, 0.664
overall round-trip IoU) remains the one in active use** — none of the three
dilation attempts (`version_0`-`version_3` under `VAE_stage1_corr_medium_v2_
coarse_dilation`) should be used going forward; kept only as a documented
record of what was tried.

**The confinement ceiling itself (up to 48.2% of poor-tier GT structure
architecturally unreachable) remains real, quantified, and — for now —
unresolved.** No fix attempted has closed it without a net-negative
trade-off. This is an accepted, documented limitation of the current
approach, not a solved problem — confidence-conditioning (next priority) is
a genuinely different fix for a different part of the pipeline (judgment on
already-reachable cells), not a substitute resolution for this ceiling.

## Footprint-erosion fix (`version_0` under `VAE_stage1_corr_medium_v3_erosion`, trained + evaluated 2026-07-21/22): dramatically smaller collapse, but still net-negative overall — poor tier reaches parity

**Theory** (raised directly by the user, reasoning from the "always-fake
margin" diagnosis above): all three prior dilation attempts padded a margin
around the coarsest grid's EXACTLY-correct footprint (Stage 1
self-reconstructs GT), so every margin cell was fake in every single
training example — the coarsest `struct_conv` had no incentive to do
anything but learn "always discard the margin," which is useless at real
inference where Step1's own footprint sometimes correctly omits structure
and sometimes doesn't. Fix: **erode** a fraction of the coarsest grid's own
boundary voxels (train-time only) *before* `coarse_dilation_kernel` re-pads
a margin around what's left, so that margin now contains a genuine mix of
erased-but-real cells (should be added back) and genuinely-outside cells
(should stay discarded) — a real judgment call instead of a rigged one.

**Implementation** — `xcube/modules/autoencoding/sunet.py`, new
`StructPredictionNet` hparam `coarse_erosion_prob` and method
`_erode_coarse_grid()`: for each batch sample independently (via the grid's
own `jidx`, so samples never leak into each other's neighbor checks),
identifies "boundary" coarse voxels (at least one empty face-neighbor) and
randomly drops a fraction of them, using `JaggedTensor.r_masked_select` +
`GridBatch.fill_to_grid` (the same safe feature-reprojection idiom the
existing dilation code already uses) rather than manual reindexing. Called
at the top of `decode()`, gated on `self.training` (so eval/inference,
which always calls `.eval()` first, is completely unaffected — verified by
checking `net.training` directly after `.eval()`). New config
`configs/gpr/gpr_vae_corr_medium_v3_erosion.yaml`: `coarse_dilation_kernel:
3` (same margin as the already-measured `version_3` real-kernel=3 run, to
isolate erosion as the one new variable) + `coarse_erosion_prob: 0.3`.

**Verified directly before training** (learning from the earlier `conv_grid`
no-op bug — never trust a mechanism just because training doesn't crash): on
a synthetic two-sample batch, erosion shrank each sample's grid
independently (216→146 and 64→37 voxels, confirming no cross-sample
leakage), was an exact no-op at `coarse_erosion_prob=0` (so it cannot affect
any other existing config), and 100% of eroded voxels were recoverable
again within the re-dilated margin. `.eval()` correctly set
`net.training=False`. Smoke-tested end-to-end (5 train/2 val batches) —
clean, loss decreasing normally.

Trained the full 50 epochs (`checkpoints/gpr/VAE_stage1_corr_medium_v3_erosion/
version_0`, training loss 30.5 → 0.563). Evaluated with new
argv-parameterized scripts mirroring the `_v2` ones exactly
(`scripts/test_vae_roundtrip_corr_medium_v3.py`,
`scripts/test_vae_structural_confinement_v3.py`), full 999-sample test set:

| Metric | Original (no dilation) | kernel=3 alone (`version_3`) | kernel=3 + erosion (this run) |
|---|---|---|---|
| Round-trip IoU, overall | 0.664 | 0.300 | **0.587** |
| Round-trip IoU, poor tier | 0.293 | 0.166 | **0.285** |
| Poor-tier unreachable % (ceiling) | 48.2% | 41.4% | 41.4% (unchanged, as expected — erosion doesn't change what dilation can reach, only how well the network judges it) |
| Recovery within reachable room, overall | ~93-96% | 56.6% | **85.5%** |

**Two real findings here, not one:**

1. **The always-fake-margin mechanism was real and was a major driver of the
   collapse, not the whole story.** Recovery-within-reach jumped from 56.6%
   (kernel=3 alone) to 85.5% (kernel=3 + erosion) — most of the way back to
   the undilated baseline's ~93-96%, and overall round-trip IoU recovered
   from 0.300 to 0.587 (closing roughly 79% of the gap to 0.664). This is a
   categorically different result from every prior dilation attempt, all of
   which landed in a narrow 0.117-0.300 band regardless of margin size or
   loss balancing.
2. **Poor tier — the tier the whole confinement ceiling was originally
   measured on — is now within noise of the undilated baseline** (0.285 vs.
   0.293, a −0.008 difference, essentially flat) while simultaneously having
   a wider theoretically-reachable region than baseline (41.4% vs. 48.2%
   unreachable). That combination (same practical accuracy, larger reachable
   ceiling) did not happen in any prior dilation attempt.
3. **Still net-negative overall** (0.587 vs. 0.664, a real −0.077 gap,
   concentrated in good/moderate tiers) — erosion narrowed the collapse
   substantially but did not fully close it. `coarse_erosion_prob=0.3` was a
   reasonable starting guess, not a tuned value, so it's not yet known
   whether a different erosion strength would close more of the remaining
   gap or whether 85.5%/56.6%-style recovery is close to this mechanism's
   ceiling.

**Decision (2026-07-22, user call): stop here rather than tune
`coarse_erosion_prob` further.** `coarse_erosion_prob=0.3` was a first
guess, not a tuned value, so a different strength might close more of the
remaining gap — but the user chose to treat this as the closing result for
the structural-dilation/erosion family rather than spend another ~20hr
retrain chasing it. **This closes the dilation/erosion line of fixes.** The
original, undilated VAE (`VAE_stage1_corr_medium/version_1`, 0.664 overall
round-trip IoU) remains the one in active use; `version_0` under
`VAE_stage1_corr_medium_v3_erosion` is kept only as a documented record —
notably a much stronger record than the plain-dilation attempts, since it
isolates and confirms the always-fake-margin mechanism as real, even though
it doesn't change which checkpoint is in active use.

**Net takeaway for this whole family (plain dilation → erosion+dilation)**:
the coarse-level confinement ceiling (up to 48.2% of poor-tier GT structure
architecturally unreachable) is real and quantified; widening the candidate
pool during training can now be shown to fail for a *specific, confirmed*
reason (always-fake margin, not an unfixable structural property of the
network) rather than an unexplained collapse — but confirming the mechanism
did not, on its own, turn out to be sufficient to make widening net-positive
at the one erosion strength tested. Poor tier reaching parity while gaining
reachable room is the one genuinely new, positive data point from this
whole family; it just wasn't enough to carry the overall number past
baseline.

## Reporting visuals for Stage 2 diffusion (v7), and three real single-sample data points (2026-07-22/23)

Not a new fix attempt — a documentation/reporting pass, built on the request
to have shareable visuals of the current Stage 2 diffusion model analogous
to the existing Stage 1 VAE reconstruction diagram
(`scripts/visualize_vae_reconstruction.py`). All renders are from real
inference on the trained `Diffusion_stage2_v7_corrupted/version_0` checkpoint
(corr_medium, material + corrupted conditioning) — no mockups.

New scripts:
- `scripts/plot_diffusion_v7_training_curve.py`: reads v7's own TensorBoard
  event files directly (two of them, merged — training paused ~11hrs
  mid-run after a dataloader worker crash and resumed, see the 2026-07-11
  entry) and plots train/val loss vs. epoch. **Result: train loss
  0.474→0.382, val loss 0.440→0.368 over 50 epochs, train and val tracking
  closely the whole run (no overfitting).**
- `scripts/visualize_stage2_multi_v7_clean.py`: one representative
  good/moderate/poor tier sample each (Step1 input / Stage 2 output / GT),
  real DDIM-100 inference per sample. Iterated once on styling per user
  feedback: blue voxel fill (matplotlib default, matching the original
  `visualize_stage2_multi_v6.py` look) with column titles above each panel,
  not the initial purple/below-panel-caption style first tried.
- `scripts/visualize_diffusion_pipeline.py`: diffusion analog of
  `visualize_vae_reconstruction.py` — same test sample (idx 0, 16,105 GT
  voxels, the same one already used in the existing VAE diagram, chosen for
  direct visual continuity between the two diagrams). Renders Step1's input
  (amber, marked visually as "the flawed hint" rather than trusted
  structure), the encoded conditioning latent (PCA→HSV colored, same
  technique as the VAE embedding panel), the coarse half-resolution
  structure the diffusion-guided decode predicts first, the final Stage 2
  output, and (new relative to the VAE version, since here input != target)
  ground truth for comparison.

**The one real numeric data point from this pass**: on the good-tier sample
used in the pipeline diagram, IoU(Step1 input, GT) = 0.717 vs. IoU(Stage 2
output, GT) = 0.722 — a +0.005 nudge. Consistent with, not contradicting,
the standing "copies its conditioning through" diagnosis (VAE round-trip
check, 2026-07-13): a single good-tier sample where the hint was already
close to correct is exactly where copy-through-with-marginal-adjustment is
expected to look almost identical to the baseline. **This is one sample, not
a tier average — v7 still has not been run through the full 999-sample
evaluation** (still item 2 in the priority list below).

All three visuals assembled into one shareable report:
https://claude.ai/code/artifact/601876f7-0ac5-40a1-9a2c-92125123ac16
(private Claude Artifact, not part of the git repo).

## Current status / next steps

As of 2026-07-23: the structural-dilation/erosion line of fixes on Stage 1
is closed by user decision (see above) — the footprint-erosion variant is
the strongest result in that family (poor tier reached parity with
baseline) but still net-negative overall, and further tuning was decided
against. The original, undilated VAE remains the one in active use. v7
(corrupted conditioning, Stage 2) still finished-but-unevaluated on the full
test set (only single-sample spot checks exist so far — see above).

Priority order:
1. **Design confidence-conditioning properly** (Stage 2's non-frozen
   diffusion layers, per the 2026-07-15 discussion — ground truth has no
   natural confidence value, so this cannot live in Stage 1's frozen encoder
   the way material does). Targets judgment on reachable cells (including
   the 28.6% false-positive-pruning leak, logged 2026-07-15) — a real,
   different problem, explicitly not a fix for the confinement ceiling.
2. Evaluate v7 (corrupted conditioning) on the full 999-sample test set for
   completeness — expected to land ~flat, but not yet measured.
3. If the project's goal allows it, treat the rigorous multi-attempt
   elimination itself (6 TEUNet + 2 Step1 Stage-2 attempts + 4 Step1 Stage-1
   structural attempts, a confirmed frozen-VAE ceiling, and multiple
   now-quantified confinement/imbalance mechanisms) as a defensible research
   outcome on its own, even absent a fix that fully closes the gap.

**Not an option going forward**: requesting A-scan/raw-waveform access —
declined once already, treated as closed.

---

## Aside: shared-machine disk-space incident (2026-07-08)

Unrelated to the research itself, but affected ability to run jobs during
this session: the shared machine's root filesystem hit 926G/926G (15M free),
traced to ~750G used by another user's data that `du`/`lsof` under a
non-root account couldn't fully diagnose. Freed ~12G in the meantime via
safe local cleanup (removed unused `herb-phenology` conda env, stale
training logs, an unrelated project folder). The other user's data was later
cleared on their end, restoring ~462G free. Not logged further here since
it's infra, not modeling — noted only because it explains why the full
corr_medium preprocessing run initially failed and had to be redone, and
coincides with the `nvcc`/gcc-14 build issue noted in Environment Setup.
