# Same path hack as train.py -- no __init__.py under src/, so this makes
# tcn.py importable without turning src/ into a real Python package.
import sys
sys.path.insert(0, "../models")

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

# The architecture has to be reconstructed with the EXACT SAME constructor
# arguments used during training before the saved weights mean anything --
# state_dict() only holds learned values, not the shape they belong to.
# model.eval() turns off training-only behavior (matters more once
# dropout/batchnorm exist, but it's correct practice regardless).
model = TCN(in_channels=25, hidden_channels=32, kernel_size=3, num_layers=7).to(device)
model.load_state_dict(torch.load("model_b.pt", map_location=device))
model.eval()

all_preds = []
all_labels = []

# torch.no_grad() because nothing here needs gradients -- this is pure
# inference, not training. outputs.argmax(dim=1) turns the model's raw
# 2-class logits into an actual predicted class (0 or 1) at every
# timestep, picking whichever of the 2 is larger; dim=1 because that's
# the classes axis in the (batch, 2, time) output shape.
#
# The inner loop is the part that actually matters for correctness: it
# goes trial-by-trial within each batch and keeps only the first `L`
# (real, non-padded) timesteps for that trial, using seq_len from
# SlipDataset. Without this, thousands of trivial all-zero padded
# timesteps would get scored as "correctly predicted stable" alongside
# the real data, inflating both F1 and accuracy.
with torch.no_grad():
    for x, y_trial, y_series, seq_len in test_loader:
        x, y_series = x.to(device), y_series.to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=1)

        for i in range(x.shape[0]):
            L = seq_len[i].item()
            all_preds.append(preds[i, :L].cpu())
            all_labels.append(y_series[i, :L].cpu())

# torch.cat glues every trial's (variable-length, padding already
# stripped) predictions/labels from every batch into one flat array --
# which is the shape sklearn's metrics expect.
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print("F1 score:", f1_score(all_labels, all_preds))
print("Accuracy:", accuracy_score(all_labels, all_preds))
