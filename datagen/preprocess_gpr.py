"""
preprocess_gpr.py

Converts paired TEUNet / ground-truth HDF5 matrices into the .pkl format
expected by the XCube training pipeline, then writes train/val/test split lists.
"""

import argparse
import random
import sys
from pathlib import Path

import fvdb
import h5py
import numpy as np
import torch

# Project constants
VOXEL_SIZE       = [0.005079, 0.005079, 0.005106]  # meters, X Y Z (non-isotropic)
GRID_DIMS        = (64, 64, 48)                     # voxels, X Y Z
TEUNET_THRESHOLD = 0.5                              # >= means occupied


# Step 1: Loading raw matrices

def load_pair(teunet_path: Path, gt_path: Path):
    """Return (teunet_array, gt_array) or raise on malformed files."""
    with h5py.File(teunet_path, 'r') as f:
        teunet = f['teunet'][:]   # (64, 64, 48) float16, values in [0, 1]
    with h5py.File(gt_path, 'r') as f:
        gt = f['gt'][:]           # (64, 64, 48) uint8, values in {0, 1}
    return teunet, gt


# Step 2 + 3: Voxel extraction and GridBatch construction

def dice_score(teunet: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """Dice coefficient between thresholded TEUNet output and binary GT."""
    teunet_bin = teunet >= threshold
    gt_bin = gt == 1
    intersection = (teunet_bin & gt_bin).sum()
    denom = teunet_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 0.0
    return float(2 * intersection / denom)


def build_sample(teunet: np.ndarray, gt: np.ndarray, threshold: float):
    """
    Convert raw matrices to an fvdb-based sample dict.
    Returns None (with a reason string) if the sample should be skipped.
    """
    # Validate shape
    if teunet.shape != GRID_DIMS or gt.shape != GRID_DIMS:
        return None, (
            f"shape mismatch: teunet={teunet.shape}, gt={gt.shape}, "
            f"expected {GRID_DIMS}"
        )

    # GT occupied positions
    gt_ijk = np.argwhere(gt == 1).astype(np.int32)           # (N, 3)
    if gt_ijk.shape[0] == 0:
        return None, "GT has no occupied voxels"

    # TEUNet occupied positions
    mask = teunet >= threshold # boolean mask of shape (64, 64, 48)
    input_ijk = np.argwhere(mask).astype(np.int32)           # (M, 3)
    if input_ijk.shape[0] == 0:
        return None, "TEUNet output has no occupied voxels after thresholding"

    gt_ijk_t = torch.tensor(gt_ijk,    dtype=torch.int32)
    input_ijk_t = torch.tensor(input_ijk, dtype=torch.int32)

    # Continuous TEUNet probability values at occupied positions ---
    teunet_float = teunet.astype(np.float32)
    teunet_vals = teunet_float[mask]                        # (M,)
    teunet_vals = torch.tensor(teunet_vals).unsqueeze(-1)   # (M, 1)

    #  Build sparse GridBatch objects
    gt_grid = fvdb.GridBatch.from_ijk(
        fvdb.JaggedTensor(gt_ijk_t),
        voxel_sizes=VOXEL_SIZE,
        origins=[0.0, 0.0, 0.0],
    )
    input_grid = fvdb.GridBatch.from_ijk(
        fvdb.JaggedTensor(input_ijk_t),
        voxel_sizes=VOXEL_SIZE,
        origins=[0.0, 0.0, 0.0],
    )

    sample = {
        'target_grid': gt_grid,     # GridBatch: ground-truth occupied voxels
        'input_grid': input_grid,   # GridBatch: TEUNet-thresholded occupied voxels
        'input_prob': teunet_vals,  # (M, 1) tensor: TEUNet probability at each input voxel
    }
    return sample, None

# Step 5: Stratified split

def assign_tier(score: float) -> str:
    if score > 0.8:
        return 'good'
    elif score >= 0.5:
        return 'moderate'
    return 'poor'


def stratified_split(stems_and_scores: list, seed: int = 42):
    """
    stems_and_scores: list of (stem, dice_score)
    Returns (train_stems, val_stems, test_stems).
    Each tier is split 80 / 10 / 10.
    """
    tiers = {'good': [], 'moderate': [], 'poor': []} # A dictionary that initializes empty lists for each tier
    for stem, score in stems_and_scores:
        tiers[assign_tier(score)].append(stem)

    train, val, test = [], [], [] # Lists to hold the stems for each split
    rng = random.Random(seed)

    for tier_name, names in tiers.items():
        rng.shuffle(names)
        n = len(names)
        n_val  = max(1, round(n * 0.1)) if n >= 3 else 0
        n_test = max(1, round(n * 0.1)) if n >= 3 else 0
        n_train = n - n_val - n_test

        train.extend(names[:n_train])
        val.extend(names[n_train:n_train + n_val])
        test.extend(names[n_train + n_val:])

        print(
            f"  tier={tier_name:8s}  total={n:4d}  "
            f"train={len(names[:n_train]):4d}  "
            f"val={n_val:3d}  test={n - n_train - n_val:3d}"
        )

    return train, val, test


# Spot-check: reload a few .pkl files after the run

REQUIRED_KEYS = {'target_grid', 'input_grid', 'input_prob'}

def spot_check(pkl_dir: Path, stems: list, n: int = 5):
    sample_stems = random.sample(stems, min(n, len(stems)))
    print(f"\nSpot-checking {len(sample_stems)} .pkl files...")
    all_ok = True
    for stem in sample_stems:
        path = pkl_dir / f"{stem}.pkl"
        try:
            obj = torch.load(path, weights_only=False)
            missing = REQUIRED_KEYS - set(obj.keys())
            if missing:
                print(f"  FAIL {stem}: missing keys {missing}")
                all_ok = False
                continue
            prob = obj['input_prob']
            if prob.dim() != 2 or prob.shape[1] != 1:
                print(f"  FAIL {stem}: input_prob shape {prob.shape}")
                all_ok = False
                continue
            print(f"  OK   {stem}  (M={prob.shape[0]}, N_gt={obj['target_grid'].total_voxels})")
        except Exception as e:
            print(f"  FAIL {stem}: {e}")
            all_ok = False
    return all_ok


# Validation helpers

def validate_pairing(teunet_dir: Path, gt_dir: Path):
    """Raise if stems don't match between the two directories."""
    teunet_stems = {p.stem for p in teunet_dir.glob("*.h5")}
    gt_stems     = {p.stem for p in gt_dir.glob("*.h5")}

    only_teunet = teunet_stems - gt_stems
    only_gt     = gt_stems - teunet_stems

    if only_teunet:
        raise ValueError(
            f"{len(only_teunet)} stem(s) exist in teunet/ but not gt/: "
            f"{sorted(only_teunet)[:5]} ..."
        )
    if only_gt:
        raise ValueError(
            f"{len(only_gt)} stem(s) exist in gt/ but not teunet/: "
            f"{sorted(only_gt)[:5]} ..."
        )
    return sorted(teunet_stems)


# Main

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess GPR HDF5 pairs into XCube-compatible .pkl files."
    )
    p.add_argument('--teunet_dir', required=True, type=Path,
                   help="Directory containing TEUNet .h5 files")
    p.add_argument('--gt_dir',     required=True, type=Path,
                   help="Directory containing ground-truth .h5 files")
    p.add_argument('--output_dir', required=True, type=Path,
                   help="Root output directory (resolution subdir created automatically)")
    p.add_argument('--threshold',  type=float, default=TEUNET_THRESHOLD,
                   help=f"Occupancy threshold for TEUNet (default {TEUNET_THRESHOLD})")
    p.add_argument('--seed',       type=int,   default=42,
                   help="Random seed for stratified split (default 42)")
    p.add_argument('--limit',      type=int,   default=None,
                   help="Process only the first N pairs (for test runs)")
    return p.parse_args()


def main():
    args = parse_args()

    teunet_dir = args.teunet_dir
    gt_dir     = args.gt_dir
    resolution = "0.005"
    pkl_dir    = args.output_dir / "gpr" / resolution
    pkl_dir.mkdir(parents=True, exist_ok=True)

    print(f"TEUNet dir : {teunet_dir}")
    print(f"GT dir     : {gt_dir}")
    print(f"Output dir : {pkl_dir}")
    print(f"Threshold  : {args.threshold}")
    print(f"Seed       : {args.seed}\n")

    # --- Validate pairing ---
    try:
        stems = validate_pairing(teunet_dir, gt_dir)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(stems)} paired samples.\n")

    if args.limit is not None:
        stems = stems[:args.limit]
        print(f"Limiting to first {len(stems)} samples (--limit {args.limit}).\n")

    # --- Process each pair ---
    stems_and_scores = []
    skipped          = 0

    for stem in stems:
        teunet_path = teunet_dir / f"{stem}.h5"
        gt_path     = gt_dir     / f"{stem}.h5"

        try:
            teunet, gt = load_pair(teunet_path, gt_path)
        except Exception as e:
            print(f"  WARN  skipping {stem}: failed to load -- {e}")
            skipped += 1
            continue

        sample, reason = build_sample(teunet, gt, args.threshold)
        if sample is None:
            print(f"  WARN  skipping {stem}: {reason}")
            skipped += 1
            continue

        score = dice_score(teunet, gt, args.threshold)
        torch.save(sample, pkl_dir / f"{stem}.pkl")
        stems_and_scores.append((stem, score))

    saved = len(stems_and_scores)
    print(f"\nSaved {saved} .pkl files, skipped {skipped}.\n")

    if saved == 0:
        print("ERROR: no samples were saved -- nothing to split.", file=sys.stderr)
        sys.exit(1)

    # --- Stratified split ---
    print("Stratified split:")
    train_stems, val_stems, test_stems = stratified_split(stems_and_scores, seed=args.seed)

    lst_dir = args.output_dir / "gpr"
    for split_name, split_stems in [
        ("train", train_stems),
        ("val",   val_stems),
        ("test",  test_stems),
    ]:
        lst_path = lst_dir / f"{split_name}.lst"
        lst_path.write_text("\n".join(split_stems) + "\n")
        print(f"  wrote {lst_path}  ({len(split_stems)} entries)")

    # Sanity-check: no overlap between splits
    train_set = set(train_stems)
    val_set   = set(val_stems)
    test_set  = set(test_stems)
    overlaps  = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlaps:
        print(f"\nWARN: {len(overlaps)} stems appear in more than one split: {overlaps}")
    else:
        print("\nSplit overlap check: OK (no overlap)")

    # --- Spot-check ---
    all_ok = spot_check(pkl_dir, [s for s, _ in stems_and_scores], n=5)
    if not all_ok:
        print("\nWARN: one or more spot-checks failed -- inspect the files above.")
    else:
        print("\nAll spot-checks passed.")


if __name__ == "__main__":
    main()
