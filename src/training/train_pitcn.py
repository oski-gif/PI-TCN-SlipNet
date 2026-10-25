# No __init__.py anywhere under src/, so this is a plain path hack rather
# than a package import -- same pattern used in every verification script
# while building tcn.py.
import sys
sys.path.insert(0, "../models")

import torch
from torch.utils.data import DataLoader
from PI_TCN import TCN, PITCNDataset
import numpy as np

# DataLoader wraps SlipDataset and hands back ready-made batches instead of
# looping through trials by hand. shuffle=True for train so the model
# doesn't learn anything from trial ORDER; shuffle=False for val so runs
# stay reproducible/comparable across epochs.
train_ds = PITCNDataset("../../Data/pitcn_dataset_full_random_split.npz", "train")
val_ds = PITCNDataset("../../Data/pitcn_dataset_full_random_split.npz", "val")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)


# device picks the RTX 3050 if available, falls back to CPU otherwise --
# .to(device) on the model moves its weights there; any tensor that
# interacts with the model later (input batches) needs the same .to(device)
# or PyTorch will error about tensors living on different devices.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = TCN(in_channels=25, hidden_channels=32, kernel_size=3, num_layers=7).to(device)

# class_weights is the inverse-frequency weighting export_data.py already
# computed from the TRAIN split only (never val/test) -- passing it to
# CrossEntropyLoss keeps the rarer class from being drowned out during
# training without discarding any trials the way hard balancing would.
data = np.load("../../Data/pitcn_dataset_full_random_split.npz")
class_weights = torch.tensor(data["class_weights"], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 40

# The actual training loop. Per epoch: one full pass over every TRAIN batch
# (model.train() matters more once dropout/batchnorm exist, but it's
# correct practice regardless), then one full pass over VAL with no
# weight updates, just to see how the model's doing on unseen data.
#
# Inside the train loop, the four lines zero_grad -> forward -> backward
# -> step are the whole learning mechanism: zero_grad clears gradients
# left over from the previous batch (PyTorch accumulates them by default,
# so skipping this would silently add each batch's gradient onto the
# last one); the model call is the forward pass; loss.backward() walks
# every operation that touched `loss` and computes how much each weight
# contributed to it; optimizer.step() is the actual update, using exactly
# what backward() just computed.
#
# `_` in the unpacking is seq_len (from SlipDataset) -- not needed here,
# only in evaluate.py where padded timesteps have to be masked out before
# scoring. torch.no_grad() during validation skips tracking gradients
# that would never be used anyway, which is faster and uses less memory.
for epoch in range(num_epochs):
    model.train()
    for x, y_trial, y_series, _ in train_loader:
        x, y_series = x.to(device), y_series.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y_series)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y_trial, y_series, _ in val_loader:
            x, y_series = x.to(device), y_series.to(device)
            outputs = model(x)
            loss = criterion(outputs, y_series)
            val_loss += loss.item()
    print(f"Epoch {epoch}: val_loss = {val_loss / len(val_loader):.4f}")

# state_dict() is just the learned weights, not the architecture -- loading
# this back later (see evaluate.py) requires building a TCN with the exact
# same constructor arguments first, then loading these values into it.
torch.save(model.state_dict(), "model_b.pt")
print("Saved model weights to model_b.pt")

