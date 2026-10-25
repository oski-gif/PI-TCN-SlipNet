# Same path hack as train.py -- no __init__.py under src/, so this makes
# tcn.py importable without turning src/ into a real Python package.
import sys
sys.path.insert(0, "../models")

import numpy as np
import torch
from torch.utils.data import DataLoader
from PI_TCN import TCN, PITCNDataset
from sklearn.metrics import f1_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TEST, not train/val -- this is the held-out split that was never touched
# during training or hyperparameter decisions, so its score is an honest
# read of real performance. shuffle=False since order doesn't matter for
# evaluation and keeping it off makes runs reproducible.
test_ds = PITCNDataset("../../Data/pitcn_dataset_full_random_split.npz", "test")
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

val_ds = PITCNDataset("../../Data/pitcn_dataset_full_random_split.npz", "val")
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# The architecture has to be reconstructed with the EXACT SAME constructor
# arguments used during training before the saved weights mean anything --
# state_dict() only holds learned values, not the shape they belong to.
# model.eval() turns off training-only behavior (matters more once
# dropout/batchnorm exist, but it's correct practice regardless).
model = TCN(in_channels=25, hidden_channels=32, kernel_size=3, num_layers=7).to(device)
model.load_state_dict(torch.load("model_b.pt", map_location=device))
model.eval()

def get_probs_and_labels(loader):
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y_trial, y_series, seq_len in loader:
            x, y_series = x.to(device), y_series.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            for i in range(x.shape[0]):
                L = seq_len[i].item()
                all_probs.append(probs[i, :L].cpu())
                all_labels.append(y_series[i, :L].cpu())
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()

val_probs, val_labels = get_probs_and_labels(val_loader)

best_threshold, best_f1 = 0.5, 0.0
for threshold in np.arange(0.05, 0.96, 0.05):
    preds = (val_probs >= threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1, best_threshold = f1, threshold
print(f"Best threshold on val: {best_threshold:.2f} (val F1={best_f1:.4f})")

test_probs, test_labels = get_probs_and_labels(test_loader)
test_preds = (test_probs >= best_threshold).astype(int)
print("F1 at tuned threshold:", f1_score(test_labels, test_preds))
print("Accuracy at tuned threshold:", accuracy_score(test_labels, test_preds))
print("F1 at default 0.5 (for comparison):", f1_score(test_labels, (test_probs >= 0.5).astype(int)))
