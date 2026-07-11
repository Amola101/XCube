# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import numpy as np
import torch
from loguru import logger

from xcube.data.base import DatasetSpec as DS
from xcube.data.base import RandomSafeDataset

import fvdb
# Original: fvdb._Cpp.SparseGridBatch = fvdb._Cpp.GridBatch
# Guarded because fvdb_core 0.4.2 no longer exposes a `_Cpp` submodule (GridBatch lives at fvdb.GridBatch directly).
if hasattr(fvdb, "_Cpp"):
    fvdb._Cpp.SparseGridBatch = fvdb._Cpp.GridBatch

import pickle
custom_pickle = pickle
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "featurevdb._Cpp":
            module = "fvdb._Cpp"
        return super().find_class(module, name)
custom_pickle.Unpickler = CustomUnpickler


class GPRDataset(RandomSafeDataset):
    """
    Pairs a TEUNet sparse reconstruction (input_grid) with its ground-truth
    sparse occupancy (target_grid), produced by preprocess_gpr.py.
    """

    def __init__(self, base_path, split, resolution, spec=None,
                 random_seed=0, hparams=None, skip_on_error=False,
                 custom_name="gpr", duplicate_num=1, input_key="input_grid",
                 hardness_scale=0.0,
                 use_corrupted_cond=False, cond_corrupt_drop_prob=0.15,
                 cond_corrupt_add_ratio=0.15, cond_corrupt_jitter_range=2,
                 **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name
        self.resolution = resolution
        self.split = split
        self.spec = spec if spec is not None else [DS.INPUT_PC, DS.GT_DENSE_PC]
        # Stage 1 (VAE pretraining) sets input_key="target_grid" so the VAE
        # self-reconstructs GT shapes. Stage 2 (diffusion) uses the default
        # "input_grid" since TEUNet's output becomes a conditioning signal instead.
        self.input_key = input_key
        # Hardness-reweighting (see PROGRESS.md): 0.0 disables it (default, matches
        # v1-v4 behavior exactly). When > 0, _get_item computes a per-sample loss
        # weight from how little TEUNet's grid overlaps GT's -- samples where TEUNet
        # is very wrong (the ~13% "poor" tier) get up-weighted, since with equal
        # per-sample weighting the ~64% "good" tier (where blind copy-through is
        # already near-optimal) dominates the average loss and gives the model
        # little incentive to learn real correction behavior.
        self.hardness_scale = hardness_scale

        # Corrupted-conditioning (see PROGRESS.md): v1-v6 all left the conditioning
        # hint (COND_PC) untouched during training, and on most training examples
        # it's already close to GT -- so "just copy the hint" already scores well
        # on average, and the network never had a training-time reason to learn
        # real correction behavior (confirmed by classifier-free dropout in fix
        # attempt 3 NOT helping, and by material conditioning in fix attempt 7
        # also not helping). This deliberately damages COND_PC's voxels (random
        # drop + jittered-copy noise) on EVERY training example -- but never on
        # val/test, gated by self.split below, so evaluation always reflects the
        # real, uncorrupted Step1/TEUNet prediction -- so blind copy-through can
        # no longer be a safe default strategy during training.
        self.use_corrupted_cond = use_corrupted_cond and split == 'train'
        self.cond_corrupt_drop_prob = cond_corrupt_drop_prob
        self.cond_corrupt_add_ratio = cond_corrupt_add_ratio
        self.cond_corrupt_jitter_range = cond_corrupt_jitter_range

        split_file = os.path.join(base_path, (split + '.lst'))
        with open(split_file, 'r') as f:
            stems = f.read().split('\n')
        if '' in stems:
            stems.remove('')
        self.all_items = [os.path.join(base_path, str(resolution), "%s.pkl" % s) for s in stems]

        logger.info(f"GPRDataset: {len(self.all_items)} items")
        self.hparams = hparams
        self.duplicate_num = duplicate_num

    def __len__(self):
        return len(self.all_items) * self.duplicate_num

    def get_name(self):
        return f"{self.custom_name}-{self.split}"

    def get_short_name(self):
        return self.custom_name

    def _corrupt_cond_grid(self, grid, material, rng):
        """Randomly drops some of grid's occupied voxels and adds jittered-copy
        noise voxels nearby (simulating false positives/misplaced detections),
        so the conditioning hint is reliably wrong on every training example
        rather than only on the naturally-occurring hard tier."""
        ijk = grid.ijk[0].jdata.cpu().numpy()
        mat = material.cpu().numpy()
        n = ijk.shape[0]
        if n == 0:
            return grid, material

        keep_mask = rng.rand(n) >= self.cond_corrupt_drop_prob
        kept_ijk, kept_mat = ijk[keep_mask], mat[keep_mask]

        n_add = int(len(kept_ijk) * self.cond_corrupt_add_ratio)
        if n_add > 0 and len(kept_ijk) > 0:
            src_idx = rng.randint(0, len(kept_ijk), size=n_add)
            jitter = rng.randint(-self.cond_corrupt_jitter_range,
                                  self.cond_corrupt_jitter_range + 1, size=(n_add, 3))
            all_ijk = np.concatenate([kept_ijk, kept_ijk[src_idx] + jitter], axis=0)
            all_mat = np.concatenate([kept_mat, kept_mat[src_idx]], axis=0)
        else:
            all_ijk, all_mat = kept_ijk, kept_mat

        all_ijk, unique_idx = np.unique(all_ijk, axis=0, return_index=True)
        all_mat = all_mat[unique_idx]

        if all_ijk.shape[0] == 0:
            # Degenerate case (shouldn't happen with sane drop/add settings) --
            # fall back to the uncorrupted grid rather than crash on an empty one.
            return grid, material

        new_grid = fvdb.GridBatch()
        new_grid.set_from_ijk(
            fvdb.JaggedTensor(torch.from_numpy(all_ijk.astype(np.int32))),
            voxel_sizes=grid.voxel_sizes,
            origins=grid.origins,
        )
        new_material = torch.from_numpy(all_mat)
        return new_grid, new_material

    def _get_item(self, data_id, rng):
        item_path = self.all_items[data_id % len(self.all_items)]
        input_data = torch.load(item_path, pickle_module=custom_pickle)

        data = {}
        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = item_path

        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = input_data[self.input_key]

        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = input_data['target_grid']

        if DS.INPUT_INTENSITY in self.spec:
            data[DS.INPUT_INTENSITY] = input_data['input_prob']

        # Material, per-voxel, aligned to whichever grid is DS.INPUT_PC right now:
        # GT's own material when self-reconstructing GT (input_key="target_grid",
        # Stage 1 and Stage 2's main latent), the Step1 prediction's material
        # otherwise. Mirrors the input_key branch above exactly.
        if DS.INPUT_MATERIAL in self.spec:
            material_key = 'target_material' if self.input_key == 'target_grid' else 'input_material'
            data[DS.INPUT_MATERIAL] = input_data[material_key]

        # Stage 2 (diffusion) needs both grids at once: the GT shape (as INPUT_PC,
        # via input_key="target_grid" in the diffusion config) to learn/denoise, and
        # TEUNet's flawed grid as a separate conditioning hint -- always from
        # 'input_grid' regardless of self.input_key, since that's specifically
        # TEUNet's reconstruction in the saved .pkl.
        cond_grid = input_data['input_grid']
        cond_material = input_data['input_material']
        if self.use_corrupted_cond:
            cond_grid, cond_material = self._corrupt_cond_grid(cond_grid, cond_material, rng)

        if DS.COND_PC in self.spec:
            data[DS.COND_PC] = cond_grid

        # Material aligned with COND_PC specifically (always the prediction's
        # material, regardless of input_key) -- needed since Stage 2 encodes
        # COND_PC through the frozen VAE separately from INPUT_PC.
        if DS.COND_MATERIAL in self.spec:
            data[DS.COND_MATERIAL] = cond_material

        if self.hardness_scale > 0:
            teunet_ijk = set(map(tuple, input_data['input_grid'].ijk[0].jdata.cpu().numpy().tolist()))
            gt_ijk = set(map(tuple, input_data['target_grid'].ijk[0].jdata.cpu().numpy().tolist()))
            iou = len(teunet_ijk & gt_ijk) / (len(teunet_ijk | gt_ijk) + 1e-6)
            data[DS.LOSS_WEIGHT] = 1.0 + self.hardness_scale * (1.0 - iou)

        return data
