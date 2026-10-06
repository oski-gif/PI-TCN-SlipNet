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
        self.seq_len = data[f"seq_len_{split}"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y_trial = torch.tensor(self.y_trial[idx], dtype=torch.long)
        y_series = torch.from_numpy(self.y_series[idx]).long()
        # The trial's real length before padding -- needed to mask out
        # padded timesteps when computing per-timestep metrics (F1,
        # accuracy), otherwise thousands of trivial all-zero "stable"
        # padded steps get scored alongside real ones.
        seq_len = torch.tensor(self.seq_len[idx], dtype=torch.long)
        return x, y_trial, y_series, seq_len

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Conv1d's `padding` argument adds zeros to BOTH sides of the input
        # equally -- so by itself this is a normal, non-causal conv, not a
        # causal one yet. Concretely, with kernel_size=3 and dilation=1,
        # padding = (3-1)*1 = 2. A 5-timestep input [0,1,2,3,4] becomes
        # [pad,pad, 0,1,2,3,4, pad,pad] (9 positions) before convolving, and
        # a kernel_size=3 filter sliding across 9 positions produces 7
        # outputs, not 5. The LAST 2 of those 7 outputs were computed using
        # the RIGHT-side padding -- meaning they "saw" fake future timesteps
        # that don't exist in a real signal. forward() below trims exactly
        # `padding` values off the end to throw those away, which is the
        # step that actually makes this causal instead of just padded.
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation)
        self.trim = (kernel_size - 1) * dilation

    def forward(self, x):
        # This layer now always expects (batch, channels, time) in and out
        # -- no transpose here. Whatever stacks these (TCNBlock) is
        # responsible for the one transpose needed to get raw (batch, time,
        # channels) data into that shape before it ever reaches here.
        # Calling self.conv(x) -- not self.conv.forward(x) -- is what
        # actually runs the convolution: it triggers Conv1d's own internal
        # forward automatically, the exact same auto-call mechanism that
        # will later run THIS class's forward when something calls model(x).
        # Using the example above, x goes from 5 real timesteps to 7 here:
        # 5 legitimate outputs plus the 2 that peeked at the right padding.
        x = self.conv(x)
        # Cuts off the last `self.trim` outputs -- the future-peeking ones
        # from the example above -- leaving only genuinely causal ones.
        # Guarded by `> 0` because of a real Python gotcha: when trim is exactly 0 (kernel_size=1), `x[:, :, :-0]` does NOT mean "remove
        # nothing" -- -0 equals plain 0, so it slices to `x[:, :, :0]`, an EMPTY tensor. The final `return` is required too: without it,
        # forward() implicitly returns None, and whatever calls this layer would silently get nothing back instead of the actual result.
        if self.trim > 0:
            x = x[:, :, :-self.trim]
        return x

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None


    def forward(self, x):
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        x = self.conv(x)
        x = x + residual
        x = self.relu(x)
        return x


class TCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            layer_in_channels = in_channels if i == 0 else hidden_channels
            dilation = 2 ** i
            self.blocks.append(TCNBlock(layer_in_channels, hidden_channels, kernel_size, dilation))
        self.output_layer = nn.Conv1d(hidden_channels, 2, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x