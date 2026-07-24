"""
Plots the Stage 2 diffusion model's train/val loss vs. epoch, for the
corr_medium/Step1-era model (v7_corrupted -- the latest, most complete Stage 2
attempt on this data: corr_medium dataset, material conditioning, and
corrupted conditioning all combined). Reads directly from the run's own
TensorBoard event files (two of them -- training was killed and resumed once
mid-run after a dataloader worker deadlock, see PROGRESS.md 2026-07-11).

Usage:
    python scripts/plot_diffusion_v7_training_curve.py
"""
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = '/home/ameliacatala/Documents/checkpoints/gpr/Diffusion_stage2_v7_corrupted/version_0'
files = sorted(glob.glob(f'{LOG_DIR}/events.out.tfevents.*'))
print('Reading event files:', files)

train_pts, val_pts, epoch_pts = [], [], []
for f in files:
    ea = EventAccumulator(f, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags()['scalars']
    if 'train_loss/sum' in tags:
        train_pts += [(e.step, e.value) for e in ea.Scalars('train_loss/sum')]
    if 'val_loss' in tags:
        val_pts += [(e.step, e.value) for e in ea.Scalars('val_loss')]
    if 'epoch' in tags:
        epoch_pts += [(e.step, e.value) for e in ea.Scalars('epoch')]

# De-duplicate by step (resume overlaps the last checkpointed step) and sort.
train_pts = sorted(dict(train_pts).items())
val_pts = sorted(dict(val_pts).items())
epoch_pts = sorted(dict(epoch_pts).items())

epoch_steps = np.array([s for s, _ in epoch_pts])
epoch_vals = np.array([v for _, v in epoch_pts])

def step_to_epoch(step):
    idx = np.searchsorted(epoch_steps, step, side='right') - 1
    idx = np.clip(idx, 0, len(epoch_vals) - 1)
    return epoch_vals[idx]

train_epochs = np.array([step_to_epoch(s) for s, _ in train_pts])
train_vals = np.array([v for _, v in train_pts])

# Average training loss within each epoch (it's logged every ~20 steps, not
# once per epoch, so raw per-step values are too noisy to compare directly
# against the once-per-epoch validation loss).
max_epoch = int(train_epochs.max())
train_epoch_avg = np.array([train_vals[train_epochs == e].mean() for e in range(max_epoch + 1)])

val_epochs = np.array([step_to_epoch(s) for s, _ in val_pts])
val_vals = np.array([v for _, v in val_pts])
val_order = np.argsort(val_epochs)
val_epochs, val_vals = val_epochs[val_order], val_vals[val_order]

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='white')
ax.plot(range(max_epoch + 1), train_epoch_avg, color='#7c5cbf', linewidth=2, label='Train loss (epoch avg)')
ax.plot(val_epochs, val_vals, color='#e0793c', linewidth=2, marker='o', markersize=3.5, label='Validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Stage 2 Diffusion — corr_medium/Step1 (v7, corrupted conditioning)')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#eeeeee', linewidth=0.8)
ax.set_axisbelow(True)

out_path = Path('/home/ameliacatala/Documents/XCube/diffusion_v7_training_curve.png')
plt.tight_layout()
plt.savefig(out_path, dpi=180, facecolor='white')
print('Saved to', out_path)
print(f'Train loss: epoch0={train_epoch_avg[0]:.3f} -> epoch{max_epoch}={train_epoch_avg[-1]:.3f}')
print(f'Val loss:   epoch0={val_vals[0]:.3f} -> epoch{int(val_epochs[-1])}={val_vals[-1]:.3f}')
