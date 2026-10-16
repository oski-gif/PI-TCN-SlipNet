"""
Physics-Informed Temporal Convolutional Network (PI-TCN) for slip
classification.

CausalConv1d/TCNBlock/TCN below are copied verbatim from tcn.py as the
starting architecture -- same causal padding/trim, same residual-block
stacking, already verified correct there. SlipDataset is NOT duplicated
here; it's imported from tcn.py since nothing about data loading changes
yet. The physics-informed additions (Coulomb-ratio/margin feature,
cone-loss term) come next, on top of this unmodified starting point.
"""

import torch
import torch.nn as nn

from tcn import SlipDataset

import numpy as np

class PITCNDataset(SlipDataset):
    def __init__(self, npz_path, split):
        super().__init__(npz_path, split)
        data = np.load(npz_path, allow_pickle=True)
        object_friction = data[f"object_friction_sliding_{split}"]
        pad_friction = data[f"pad_friction_mean_{split}"]
        # Verified MuJoCo rule: effective friction is the element-wise MAX
        # of the two touching geoms' coefficients.
        self.mu_combined = np.maximum(object_friction, pad_friction)

    def __getitem__(self, idx):
        x, y_trial, y_series, seq_len = super().__getitem__(idx)
        mu = self.mu_combined[idx]
        mu_col = torch.full((x.shape[0], 1), mu, dtype=x.dtype)
        x = torch.cat([x, mu_col], dim=1)
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