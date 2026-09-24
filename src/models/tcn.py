import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset



# Excluded from the model's input on purpose, decided 2026-09-22: the slip
# label is computed by thresholding displacement, so displacement itself
# (and slip_speed, its derivative) isn't correlated evidence -- it's close
# to the label in disguise. obj_x/y/z is real object pose, not a tactile
# measurement, so it's excluded for the same "would this be available from
# fingertip sensing alone" reason. pull_force/phase are kept: those are
# commanded/known by the controller, not measurements of the object's
# response, so they aren't leakage in the same sense.
LEAKAGE_FEATURES = {"obj_x", "obj_y", "obj_z", "displacement", "slip_speed"}


class SlipDataset(Dataset):
    """Wraps one split (train/val/test) of the PI-TCN dataset exported by
    Synthetic data creation/Simulation/export_data.py as a PyTorch Dataset.

    Returns both a per-trial label (y_trial) and a per-timestep label
    (y_series) for each sample -- whether the model predicts once per trial
    or at every timestep is an architecture decision, not a data-loading
    one, so both are handed back and the choice is made downstream.
    """

    def __init__(self, npz_path, split):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        data = np.load(npz_path, allow_pickle=True)
        all_feature_names = data["feature_names"].tolist()
        keep_idx = [i for i, n in enumerate(all_feature_names) if n not in LEAKAGE_FEATURES]
        self.feature_names = [all_feature_names[i] for i in keep_idx]
        self.X = data[f"X_{split}"][:, :, keep_idx]
        self.y_trial = data[f"y_{split}"]
        self.y_series = data[f"y_series_{split}"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y_trial = torch.tensor(self.y_trial[idx], dtype=torch.long)
        y_series = torch.from_numpy(self.y_series[idx]).long()
        return x, y_trial, y_series

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation)
