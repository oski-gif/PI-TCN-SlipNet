"""
BOUNDED SIDE-DIAGNOSTIC, not part of the production pipeline.

Directly visualizing raw signals aligned to slip onset (for slip trials)
vs. aligned to the start of the pull phase (for stable trials, as a fair
"equivalent stress event" reference, since they have no real onset) --
per ChatGPT's highest-priority recommendation: look at the actual curves
before trusting any more aggregate-statistic formulas.

Plots mean +/- std across many trials for: summed normal force, summed
tangential force, summed torque_mag, and pull_force, in a window from
-150 to +30 relative steps around each trial's reference point.
"""
import sys
sys.path.insert(0, "../../src/models")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcn import SlipDataset

ds = SlipDataset("../../Data/pitcn_dataset_full_random_split.npz", "train")
normal_idx = [i for i, n in enumerate(ds.feature_names) if n.endswith("_normal")]
tangential_idx = [i for i, n in enumerate(ds.feature_names) if n.endswith("_tangential")]
torque_idx = [i for i, n in enumerate(ds.feature_names) if n.endswith("_torque_mag")]
pull_idx = ds.feature_names.index("pull_force")
phase_idx = ds.feature_names.index("phase")

PRE, POST = 150, 30
WINDOW = PRE + POST

def extract(fm, ref):
    """fm: (T, F) raw feature matrix. ref: reference index (onset or pull-start).
    Returns (WINDOW, 4) array of [normal, tangential, torque, pull], or None
    if there isn't enough real data on either side of ref."""
    lo, hi = ref - PRE, ref + POST
    if lo < 0 or hi > fm.shape[0]:
        return None
    seg = fm[lo:hi]
    return np.stack([
        seg[:, normal_idx].sum(axis=1),
        seg[:, tangential_idx].sum(axis=1),
        seg[:, torque_idx].sum(axis=1),
        seg[:, pull_idx],
    ], axis=1)

slip_traces, stable_traces = [], []
for i in range(len(ds)):
    x, y_trial, y_series, seq_len = ds[i]
    L = seq_len.item()
    fm = x[:L].numpy()
    ys = y_series[:L].numpy()
    phase = fm[:, phase_idx]

    if y_trial.item() == 1:
        onset = np.argmax(ys == 1)
        if ys[onset] != 1:
            continue
        tr = extract(fm, onset)
        if tr is not None:
            slip_traces.append(tr)
    else:
        pull_start = np.argmax(phase >= 2)
        if phase[pull_start] < 2:
            continue
        tr = extract(fm, pull_start)
        if tr is not None:
            stable_traces.append(tr)

    if len(slip_traces) >= 400 and len(stable_traces) >= 400:
        break

slip_traces = np.stack(slip_traces)
stable_traces = np.stack(stable_traces)
print(f"slip trials used: {len(slip_traces)}  |  stable trials used: {len(stable_traces)}")

t_axis = np.arange(-PRE, POST)
titles = ["Summed normal force", "Summed tangential force", "Summed torque_mag", "pull_force"]

fig, axes = plt.subplots(4, 1, figsize=(9, 14), sharex=True)
for i, ax in enumerate(axes):
    s_mean, s_std = slip_traces[:, :, i].mean(axis=0), slip_traces[:, :, i].std(axis=0)
    b_mean, b_std = stable_traces[:, :, i].mean(axis=0), stable_traces[:, :, i].std(axis=0)
    ax.plot(t_axis, s_mean, label="slip trials (aligned to onset)", color="crimson")
    ax.fill_between(t_axis, s_mean - s_std, s_mean + s_std, color="crimson", alpha=0.15)
    ax.plot(t_axis, b_mean, label="stable trials (aligned to pull-start)", color="steelblue")
    ax.fill_between(t_axis, b_mean - b_std, b_mean + b_std, color="steelblue", alpha=0.15)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(titles[i])
    ax.legend(fontsize=8)
axes[-1].set_xlabel("timesteps relative to onset (slip) / pull-start (stable)")
plt.tight_layout()
out_path = "../../Data/_diagnostic_onset_visualization.png"
plt.savefig(out_path, dpi=110)
print(f"Saved plot to {out_path}")
