"""
Read-only check of how much the TCN test metrics depend on trials where the
object had already fallen before the pull (results.md, "Formation-drop recording
added"). Scores existing checkpoints on the whole test split, split by whether the
object fell more than 5 cm during ramp/settle.

Uses the same metric as src/training/evaluate.py and eval_tcn_threshold.py: F1
pooled over every real timestep, with padding removed using seq_len. Subset numbers
are only produced after the recorded full-test metrics reproduce and the computed
falls and labels match raw trial files. Nothing is tuned here, and no checkpoint or
dataset is written.

Reading the output:
- Removing easy negatives changes the base rate, so F1 can fall for that reason alone.
- Recall and specificity do not depend on the base rate, but the subsets also contain
  different objects and grips, so they can shift with composition too.
- Pooled specificity includes every ramp and settle timestep, which is always labelled
  no-slip.
- The intervals resample trials, because timesteps within a trial are strongly
  correlated. They cover test-set sampling only, not run-to-run training variance.

Usage (with no MuJoCo job running):
    python src/analysis/tcn_f1_without_drops.py --gate-only --older-a-checkpoint PATH
    python src/analysis/tcn_f1_without_drops.py --older-a-checkpoint PATH

Rows are named by checkpoint hash and measured F1, because the filename model_a.pt does
not identify a run: it holds the F1 0.691 restore run, not the 0.773 run. That row is
required, and the run stops if it does not reproduce. Any other row that does not
reproduce is reported and left out of the subset tables.
"""
import argparse
import hashlib
import json
import os
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))
from PI_TCN import TCN as TCN_B, PITCNDataset
from tcn import TCN as TCN_A, SlipDataset

DATASET = os.path.join(ROOT, "Data", "pitcn_dataset_full_random_split.npz")
MANIFEST = os.path.join(ROOT, "Data", "manifest_full_merged.json")
TRAINING_DIR = os.path.join(ROOT, "src", "training")
OUT_JSON = os.path.join(ROOT, "reports", "tcn_f1_without_drops.json")

DROP_FALL_M = 0.05  # same value as FORMATION_DROP_THRESHOLD, fixed in advance, not tuned
EXPECTED_TEST_TRIALS = 2628
RAW_CHECKS_PER_GROUP = 20
RAW_FALL_TOLERANCE_M = 1e-5
# Fixed before any checkpoint was measured. Covers values logged to 3 decimals plus
# CPU/GPU float differences, and is well below the gaps between 0.691, 0.722 and 0.773.
GATE_TOLERANCE = 1e-3
BOOTSTRAP_RESAMPLES = 2000

# The exact value eval_tcn_threshold.py's search reached when it recorded 0.30.
TUNED_THRESHOLD = float(np.arange(0.05, 0.96, 0.05)[5])

# Values logged in results.md before this check existed: F1, then accuracy where logged.
LOGGED_A_RESTORE_RUN = (0.691, 0.887)
LOGGED_A_ORIGINAL_RUN = (0.7728929388214366, 0.8989912209391691)
LOGGED_B_AT_05 = (0.772205528667876, None)
LOGGED_B_TUNED = (0.7985883040854257, 0.9082381685225329)


def fail(message):
    print(f"STOPPED, no subset numbers produced: {message}")
    sys.exit(2)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_arrays():
    with np.load(DATASET, allow_pickle=True) as data:
        names = [str(n) for n in data["feature_names"]]
        X = data["X_test"]
        return {"z": X[:, :, names.index("obj_z")], "phase": X[:, :, names.index("phase")],
                "padding_clean": [not np.any(X[i, L:]) for i, L in enumerate(data["seq_len_test"])],
                "seq_len": data["seq_len_test"], "y": data["y_test"],
                "y_series": data["y_series_test"], "object": data["object_test"],
                "squeeze": data["squeeze_test"], "pull": data["pull_multiple_test"],
                "mass": data["obj_mass_test"]}


def formation_falls(test):
    """Fall from the first real timestep to the lowest point in ramp/settle, per trial,
    on real timesteps only, after checking that the stored values are not normalized."""
    n_trials = len(test["seq_len"])
    if n_trials != EXPECTED_TEST_TRIALS:
        fail(f"expected {EXPECTED_TEST_TRIALS} test trials, found {n_trials}")
    falls = np.zeros(n_trials, dtype=np.float64)
    starts = np.zeros(n_trials, dtype=np.float64)
    seen_phases = set()
    for i, L in enumerate(test["seq_len"]):
        trial_z, trial_phase = test["z"][i, :L], test["phase"][i, :L]
        seen_phases.update(np.unique(trial_phase).tolist())
        formation = trial_phase < 2
        if L < 1 or not formation[0]:
            fail(f"trial {i} does not start in ramp/settle (seq_len={L})")
        if not test["padding_clean"][i]:
            fail(f"trial {i} has non-zero values in its padding")
        starts[i] = trial_z[0]
        falls[i] = trial_z[0] - trial_z[formation].min()
    if not seen_phases <= {0.0, 1.0, 2.0}:
        fail(f"phase values look normalized: {sorted(seen_phases)[:8]}")
    if starts.min() < 0.2 or starts.max() > 0.3:
        fail(f"first obj_z outside the raw 0.245 m spawn range: {starts.min()} to {starts.max()}")
    print(f"obj_z at first timestep: {starts.min():.5f} to {starts.max():.5f} m; phases {sorted(seen_phases)}")
    return falls


def confirm_against_raw(test, falls):
    """Recompute fall and labels from raw trial files for a seeded sample of test trials."""
    def key(obj, squeeze, pull, mass):
        return (str(obj), float(np.float32(squeeze)), float(np.float32(pull)), float(np.float32(mass)))

    with open(MANIFEST) as f:
        by_key = {}
        for entry in json.load(f):
            if entry["ever_contacted"]:
                k = key(entry["object_name"], entry["squeeze_intensity"],
                        entry["pull_multiple"], entry["obj_mass"])
                by_key.setdefault(k, []).append(entry)

    dropped = falls > DROP_FALL_M
    if min(dropped.sum(), (~dropped).sum()) < RAW_CHECKS_PER_GROUP:
        fail(f"too few trials to check: {dropped.sum()} dropped, {(~dropped).sum()} not")
    rng = random.Random(0)
    sample = (rng.sample(np.flatnonzero(dropped).tolist(), RAW_CHECKS_PER_GROUP)
              + rng.sample(np.flatnonzero(~dropped).tolist(), RAW_CHECKS_PER_GROUP))
    worst = 0.0
    for i in sample:
        L = test["seq_len"][i]
        matches = by_key.get(key(test["object"][i], test["squeeze"][i], test["pull"][i], test["mass"][i]), [])
        if len(matches) != 1:
            fail(f"test trial {i} matched {len(matches)} raw trials, expected exactly 1")
        with np.load(os.path.join(ROOT, matches[0]["data_path"].replace("\\", "/")), allow_pickle=True) as raw:
            names = [str(n) for n in raw["feature_names"]]
            fm = raw["feature_matrix"].astype(np.float32)
            raw_label = int(raw["trial_slip_label"])
            raw_series = raw["slip_label_series"]
        if len(fm) != L:
            fail(f"test trial {i}: seq_len {L} but raw file has {len(fm)} rows")
        raw_z, raw_phase = fm[:, names.index("obj_z")], fm[:, names.index("phase")]
        raw_fall = float(raw_z[0]) - float(raw_z[raw_phase < 2].min())
        if raw_label != int(test["y"][i]) or not np.array_equal(raw_series, test["y_series"][i, :L]):
            fail(f"test trial {i}: labels differ from raw file")
        worst = max(worst, abs(raw_fall - falls[i]))
        if abs(raw_fall - falls[i]) > RAW_FALL_TOLERANCE_M:
            fail(f"test trial {i}: fall {falls[i]} but raw file gives {raw_fall}")
    print(f"Raw check: {len(sample)} test trials match raw files on fall, length and labels "
          f"(worst fall difference {worst:.2e} m)")


def predict_test(dataset_cls, model_cls, checkpoint, in_channels, test):
    dataset = dataset_cls(DATASET, "test")
    model = model_cls(in_channels=in_channels, hidden_channels=32, kernel_size=3, num_layers=7)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    probs, argmax, labels = [], [], []
    with torch.no_grad():
        for x, _, y_series, seq_len in DataLoader(dataset, batch_size=32, shuffle=False):
            outputs = model(x)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1]
            batch_argmax = outputs.argmax(dim=1)
            for i in range(x.shape[0]):
                L = seq_len[i].item()
                probs.append(batch_probs[i, :L].clone().numpy())
                argmax.append(batch_argmax[i, :L].clone().numpy())
                labels.append(y_series[i, :L].clone().numpy())
    # Order matters: the fall mask comes from the dataset arrays, so every trial's
    # labels must match them exactly, or subsets would attach to the wrong trials.
    for i, trial_labels in enumerate(labels):
        if not np.array_equal(trial_labels, test["y_series"][i, :test["seq_len"][i]]):
            fail(f"{checkpoint}: trial {i}'s labels differ from the dataset arrays")
    return {"probs": probs, "argmax": argmax, "labels": labels}


def trial_predictions(output, threshold):
    if threshold is None:
        return [p.astype(int) for p in output["argmax"]]
    return [(p >= threshold).astype(int) for p in output["probs"]]


def trial_counts(labels, preds):
    """Rows of tp, fp, tn, fn per trial."""
    return np.array([[int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()),
                      int(((y == 0) & (p == 0)).sum()), int(((y == 1) & (p == 0)).sum())]
                     for y, p in zip(labels, preds)], dtype=np.int64)


def rates(tp, fp, tn, fn):
    def ratio(a, b):
        return a / b if b else float("nan")

    return {"precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "accuracy": ratio(tp + tn, tp + fp + tn + fn),
            "positive_timestep_rate": ratio(tp + fn, tp + fp + tn + fn)}


def summarize(counts):
    tp, fp, tn, fn = (int(v) for v in counts.sum(axis=0))
    result = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, **rates(tp, fp, tn, fn), "ci95": {}}
    rng = np.random.default_rng(0)
    draws = {name: np.zeros(BOOTSTRAP_RESAMPLES) for name in ("recall", "specificity", "f1")}
    for b in range(BOOTSTRAP_RESAMPLES):
        resampled = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        r = rates(*resampled.tolist())
        for name in draws:
            draws[name][b] = r[name]
    for name, values in draws.items():
        result["ci95"][name] = [float(v) for v in np.nanpercentile(values, [2.5, 97.5])]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--older-a-checkpoint", default=None,
                    help="Earlier 22-channel checkpoint, tested against the logged 0.773 run. "
                         "It gets subset tables only if it reproduces that run.")
    ap.add_argument("--gate-only", action="store_true",
                    help="Stop after the full-test gate. Writes nothing and prints no subsets.")
    args = ap.parse_args()
    torch.set_num_threads(2)

    a_path = os.path.join(TRAINING_DIR, "model_a.pt")
    b_path = os.path.join(TRAINING_DIR, "model_b.pt")
    a_kind = (SlipDataset, TCN_A, 22)
    b_kind = (PITCNDataset, TCN_B, 25)
    # Checkpoint, kind, threshold (None = argmax, as evaluate.py does), logged values, required.
    rows = [(a_path, a_kind, None, LOGGED_A_RESTORE_RUN, True),
            (b_path, b_kind, 0.5, LOGGED_B_AT_05, False),
            (b_path, b_kind, TUNED_THRESHOLD, LOGGED_B_TUNED, False)]
    if args.older_a_checkpoint:
        rows.append((args.older_a_checkpoint, a_kind, None, LOGGED_A_ORIGINAL_RUN, False))

    test = load_test_arrays()
    falls = formation_falls(test)
    confirm_against_raw(test, falls)
    outputs = {}
    hashes = {}
    for path, (dataset_cls, model_cls, channels), _, _, _ in rows:
        if path not in outputs:
            outputs[path] = predict_test(dataset_cls, model_cls, path, channels, test)
            hashes[path] = file_sha256(path)

    gate = []
    counts_by_row = {}
    required_failed = False
    for path, (_, _, channels), threshold, (logged_f1, logged_accuracy), required in rows:
        labels = outputs[path]["labels"]
        preds = trial_predictions(outputs[path], threshold)
        counts = trial_counts(labels, preds)
        measured_f1 = float(f1_score(np.concatenate(labels), np.concatenate(preds)))
        tp, fp, tn, fn = counts.sum(axis=0).tolist()
        measured = rates(tp, fp, tn, fn)
        if not np.isclose(measured["f1"], measured_f1):
            fail("confusion-count F1 disagrees with sklearn")
        ok = abs(measured_f1 - logged_f1) <= GATE_TOLERANCE and (
            logged_accuracy is None or abs(measured["accuracy"] - logged_accuracy) <= GATE_TOLERANCE)
        decision = "argmax" if threshold is None else f"threshold {threshold:.2f}"
        name = f"{hashes[path][:8]} F1={measured_f1:.3f} ({channels}ch, {decision})"
        gate.append({"row": name, "checkpoint": os.path.relpath(path, ROOT), "sha256": hashes[path],
                     "logged_f1": logged_f1, "logged_accuracy": logged_accuracy,
                     "measured_f1": measured_f1, "measured_accuracy": measured["accuracy"], "ok": ok})
        logged = f"logged F1 {logged_f1:.6f}" + (f", accuracy {logged_accuracy:.6f}" if logged_accuracy else "")
        print(f"Gate, {name}: {logged}; measured F1 {measured_f1:.6f}, accuracy "
              f"{measured['accuracy']:.6f}, {'OK' if ok else 'FAILED'}")
        if ok:
            counts_by_row[name] = counts
        elif required:
            required_failed = True

    provenance = {"dataset_sha256": file_sha256(DATASET), "gate_tolerance": GATE_TOLERANCE}
    if args.gate_only:
        print("Gate-only run: nothing written.")
        return
    if required_failed:
        with open(OUT_JSON, "w") as f:
            json.dump({"status": "gate_failed", "gate": gate, **provenance}, f, indent=2)
        fail("model_a.pt did not reproduce its logged F1 and accuracy")
    for row in gate:
        if not row["ok"]:
            print(f"Left out of the subset tables, did not reproduce: {row['row']}")

    trial_labels = test["y"]
    dropped = falls > DROP_FALL_M
    subsets = {"all trials": np.ones(len(falls), dtype=bool), "no drop": ~dropped, "drop": dropped}
    report = {"status": "ok", "drop_fall_m": DROP_FALL_M,
              "metric": "per-timestep, pooled, padding removed", **provenance,
              "gate": gate, "subsets": {}, "conditions": {}}
    print(f"\nFall > {DROP_FALL_M} m before the pull, whole test split:")
    for name, mask in subsets.items():
        by_label = {str(lab): int(((trial_labels == lab) & mask).sum()) for lab in (0, 1)}
        report["subsets"][name] = {"trials": int(mask.sum()), "trials_by_label": by_label}
        print(f"  {name:11s} trials {int(mask.sum()):5d}   no-slip {by_label['0']:5d}   slip {by_label['1']:5d}")

    for name, counts in counts_by_row.items():
        report["conditions"][name] = {}
        print(f"\n{name}")
        print(f"  {'subset':11s} {'tp':>9s} {'fp':>9s} {'tn':>9s} {'fn':>9s} {'pos.rate':>8s} {'prec':>6s} "
              f"{'recall (95% CI)':>21s} {'spec (95% CI)':>21s} {'f1 (95% CI)':>21s}")
        for subset, mask in subsets.items():
            m = summarize(counts[mask])
            report["conditions"][name][subset] = m

            def with_ci(metric):
                low, high = m["ci95"][metric]
                return f"{m[metric]:.3f} ({low:.3f}-{high:.3f})"

            print(f"  {subset:11s} {m['tp']:9d} {m['fp']:9d} {m['tn']:9d} {m['fn']:9d} "
                  f"{m['positive_timestep_rate']:8.3f} {m['precision']:6.3f} "
                  f"{with_ci('recall'):>21s} {with_ci('specificity'):>21s} {with_ci('f1'):>21s}")

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
