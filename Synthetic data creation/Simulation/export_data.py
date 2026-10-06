"""
Consolidate per-trial .npz files (from generate_trials.py) into a single
training-ready dataset for the PI-TCN.

Two modes:

1. Standard random split (--split random): shuffles trials, splits into
   train/val/test by fraction. Fine for a first pass, but per the ISEF
   feedback discussion, this risks the model just memorizing patterns
   specific to the objects it's seen -- doesn't prove generalization.

2. Leave-one-object-out (--split leave-one-out --holdout-object cone):
   ALL trials for the named object go to test; the other 4 objects' trials
   are split into train/val. This is what actually lets you claim
   "generalizes to unseen object geometry" rather than just "fits objects
   it was trained on" -- run this 5 times (once per holdout object) for
   the full generalization experiment GPT's feedback recommended.

Since trials can have slightly different lengths (contact happens at
different steps per trial), sequences are padded/truncated to a fixed
length for stacking into one array.

Usage:
    # Standard split
    python export_dataset.py --manifest data/raw_trials/manifest_balanced.json \\
        --out data/dataset_random_split.npz --split random

    # Leave-one-out, for the generalization experiment (run once per object)
    python export_dataset.py --manifest data/raw_trials/manifest_full.json \\
        --out data/dataset_loo_cone.npz --split leave-one-out --holdout-object cone
"""
import argparse
import json

import numpy as np


# Scalar per-trial metadata carried through to the exported dataset beyond
# the original handful of fields. FIX: this used to only be squeeze_intensity
# + onset_step -- everything else (friction coefficients, pull/mass, etc.)
# was silently unreachable from the exported .npz even though every source
# trial file records it, which would have quietly blocked any physics-
# informed feature/loss (e.g. Coulomb ratio needs object_friction_sliding
# and pad_friction_mean) from ever reaching the training data.
EXTRA_SCALAR_FIELDS = [
    "pull_multiple", "pull_force_max", "obj_mass",
    "object_friction_sliding", "object_friction_torsional", "object_friction_rolling",
    "pad_friction_mean", "pad_friction_min",
    "ever_contacted", "grasp_force_at_settle",
]


def load_trial(path):
    with np.load(path, allow_pickle=True) as d:
        feature_matrix = d["feature_matrix"]
        trial = {
            "feature_matrix": feature_matrix,
            "feature_names": d["feature_names"].tolist(),
            "slip_label_series": d["slip_label_series"],
            "trial_slip_label": int(d["trial_slip_label"]),
            "onset_step": int(d["onset_step"]),
            "object_name": str(d["object_name"]),
            "squeeze_intensity": float(d["squeeze_intensity"]),
            # The trial's REAL length before pad_or_truncate pads it out to
            # target_len with zeros. Without this, nothing downstream (e.g.
            # computing F1/accuracy) can tell real timesteps apart from
            # padding -- scoring over the padded region would mix in
            # thousands of trivial all-zero "obviously stable" timesteps
            # and inflate metrics in a way that doesn't reflect genuine
            # per-timestep performance.
            "seq_len": int(feature_matrix.shape[0]),
        }
        for field in EXTRA_SCALAR_FIELDS:
            val = d[field]
            trial[field] = bool(val) if field == "ever_contacted" else float(val)
        return trial


def pad_or_truncate(matrix, target_len):
    T, F = matrix.shape
    if T == target_len:
        return matrix
    if T > target_len:
        return matrix[:target_len]
    padded = np.zeros((target_len, F), dtype=matrix.dtype)
    padded[:T] = matrix
    return padded


def build_arrays(entries, target_len, feature_names_ref):
    X = np.zeros((len(entries), target_len, len(feature_names_ref)), dtype=np.float32)
    y_trial = np.zeros(len(entries), dtype=np.int64)
    y_series = np.zeros((len(entries), target_len), dtype=np.int64)
    object_names = []
    squeeze_intensities = np.zeros(len(entries), dtype=np.float32)
    onset_steps = np.zeros(len(entries), dtype=np.int64)
    seq_lens = np.zeros(len(entries), dtype=np.int64)
    extras = {field: np.zeros(len(entries), dtype=np.float32) for field in EXTRA_SCALAR_FIELDS}

    for i, e in enumerate(entries):
        trial = load_trial(e["data_path"])
        if trial["feature_names"] != feature_names_ref:
            raise ValueError(
                f"Feature order mismatch in {e['data_path']} -- all trials "
                f"must come from the same generate_trials.py run/version.")
        X[i] = pad_or_truncate(trial["feature_matrix"], target_len)
        y_trial[i] = trial["trial_slip_label"]
        y_series[i] = pad_or_truncate(
            trial["slip_label_series"].reshape(-1, 1), target_len).reshape(-1)
        object_names.append(trial["object_name"])
        squeeze_intensities[i] = trial["squeeze_intensity"]
        onset_steps[i] = trial["onset_step"]
        seq_lens[i] = trial["seq_len"]
        for field in EXTRA_SCALAR_FIELDS:
            extras[field][i] = trial[field]

    return X, y_trial, y_series, np.array(object_names), squeeze_intensities, onset_steps, seq_lens, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["random", "leave-one-out"], default="random")
    ap.add_argument("--holdout-object", nargs="+", default=None,
                     help="Required for --split leave-one-out. One or more "
                          "object names -- ALL trials for EVERY named object "
                          "go to test, train/val are drawn only from the "
                          "remaining objects. Despite the mode's name, this "
                          "isn't limited to holding out exactly one: pass "
                          "multiple names (e.g. --holdout-object cone sphere "
                          "l_block dumbbell hexagonal_prism) to test "
                          "generalization to several unseen object "
                          "geometries at once. Names must match OBJECT_LIBRARY "
                          "exactly (15 objects total as of this dataset).")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-no-contact", action="store_true",
                     help="Keep trials where the object was never touched "
                          "(ever_contacted=False). Off by default: an "
                          "untouched object still gets labeled slip/stable "
                          "based on whether it drifted past the threshold "
                          "under gravity/pull alone, which means the SAME "
                          "all-zero sensor reading ends up mapped to both "
                          "classes -- a direct label contradiction, not "
                          "just noise. Only pass this if you specifically "
                          "want to study no-contact trials as their own "
                          "thing.")
    args = ap.parse_args()

    # FIX: train_frac + val_frac > 1 used to silently produce an empty (or
    # negative-size, which numpy slicing just clips to empty too) test set
    # with no warning -- you'd only notice much later when test metrics
    # looked degenerate. Fail loudly instead.
    if args.train_frac < 0 or args.val_frac < 0:
        raise ValueError("--train-frac and --val-frac must both be non-negative.")
    if args.train_frac + args.val_frac > 1.0:
        raise ValueError(
            f"--train-frac ({args.train_frac}) + --val-frac ({args.val_frac}) "
            f"exceeds 1.0 -- would leave an empty or negative-size test set.")

    with open(args.manifest) as f:
        manifest = json.load(f)

    if not args.include_no_contact:
        n_before = len(manifest)
        manifest = [m for m in manifest if m["ever_contacted"]]
        n_dropped = n_before - len(manifest)
        if n_dropped:
            print(f"Dropped {n_dropped} no-contact trials ({n_before} -> "
                  f"{len(manifest)}) -- pass --include-no-contact to keep them.")

    if len(manifest) == 0:
        raise ValueError("Manifest is empty -- nothing to export.")

    # Peek at the first trial to get the fixed feature order.
    first_trial = load_trial(manifest[0]["data_path"])
    feature_names_ref = first_trial["feature_names"]

    # FIX: target_len used to be whichever trial happened to be manifest[0]'s
    # own length -- since ramp_steps is randomized per trial (see
    # generate_trials.py), trial lengths vary (1700-1900 steps in this
    # dataset), so that arbitrary choice silently TRUNCATED most other
    # trials. Truncation isn't just lossy here: slip onset is often late in
    # the hold phase, so a truncated trial could carry a "slip" label with
    # the actual slip evidence cut out of what the model ever sees (verified:
    # 29 trials in the initial 3120-trial balanced set had onset_step past
    # the old arbitrary cutoff). Using the true max length instead means
    # every trial is only ever padded, never truncated.
    target_len = max(
        int(np.load(m["data_path"], allow_pickle=True)["feature_matrix"].shape[0])
        for m in manifest
    )
    print(f"Using target_len={target_len} (max trial length in manifest, "
          f"so nothing gets truncated)")

    rng = np.random.default_rng(args.seed)

    if args.split == "random":
        indices = rng.permutation(len(manifest))
        n_train = int(len(indices) * args.train_frac)
        n_val = int(len(indices) * args.val_frac)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        train_entries = [manifest[i] for i in train_idx]
        val_entries = [manifest[i] for i in val_idx]
        test_entries = [manifest[i] for i in test_idx]

    else:  # leave-one-out (generalized to leave-SEVERAL-out -- see --holdout-object help)
        if args.holdout_object is None:
            raise ValueError("--holdout-object is required for --split leave-one-out")
        holdout_set = set(args.holdout_object)
        all_objects = set(m["object_name"] for m in manifest)
        unknown = holdout_set - all_objects
        if unknown:
            raise ValueError(f"--holdout-object name(s) not found in the manifest: "
                              f"{sorted(unknown)} -- check they match OBJECT_LIBRARY "
                              f"exactly. Objects present: {sorted(all_objects)}")
        test_entries = [m for m in manifest if m["object_name"] in holdout_set]
        remaining = [m for m in manifest if m["object_name"] not in holdout_set]
        if len(test_entries) == 0:
            raise ValueError(f"No trials found for holdout object(s) {sorted(holdout_set)} "
                              f"-- check the name(s) match OBJECT_LIBRARY exactly.")
        rng.shuffle(remaining)
        # FIX: was `val_frac / (1 - (1 - train_frac - val_frac))`, which is
        # just a needlessly roundabout way to write `val_frac / (train_frac +
        # val_frac)` -- and divides by zero when both are 0. Same value,
        # clearer, and guarded.
        denom = args.train_frac + args.val_frac
        if denom <= 0:
            raise ValueError("--train-frac + --val-frac must be > 0 for --split leave-one-out.")
        n_val = int(len(remaining) * (args.val_frac / denom))
        val_entries = remaining[:n_val]
        train_entries = remaining[n_val:]
        print(f"Holding out {sorted(holdout_set)} entirely for test "
              f"({len(test_entries)} trials). Train/val drawn from the other "
              f"{len(all_objects) - len(holdout_set)} objects "
              f"({sorted(all_objects - holdout_set)}).")

    print(f"train={len(train_entries)}  val={len(val_entries)}  test={len(test_entries)}")

    X_train, y_train, ys_train, obj_train, sq_train, onset_train, seqlen_train, extras_train = build_arrays(train_entries, target_len, feature_names_ref)
    X_val, y_val, ys_val, obj_val, sq_val, onset_val, seqlen_val, extras_val = build_arrays(val_entries, target_len, feature_names_ref)
    X_test, y_test, ys_test, obj_test, sq_test, onset_test, seqlen_test, extras_test = build_arrays(test_entries, target_len, feature_names_ref)

    # Class weights computed from TRAIN only (never val/test -- those must
    # stay an honest, unweighted picture of the real class distribution for
    # evaluation to mean anything). Standard inverse-frequency weighting:
    # weight_c = n_total / (n_classes * count_c), so the rarer class gets
    # weighted up. Use this as the `weight` arg to your loss (e.g.
    # nn.CrossEntropyLoss(weight=torch.tensor(class_weights))) instead of
    # throwing away majority-class trials the way hard balancing does.
    n_classes = 2
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1  # guard against a class with zero train trials
    class_weights = len(y_train) / (n_classes * counts)

    extra_arrays = {}
    for field in EXTRA_SCALAR_FIELDS:
        extra_arrays[f"{field}_train"] = extras_train[field]
        extra_arrays[f"{field}_val"] = extras_val[field]
        extra_arrays[f"{field}_test"] = extras_test[field]

    np.savez_compressed(
        args.out,
        feature_names=np.array(feature_names_ref),
        X_train=X_train, y_train=y_train, y_series_train=ys_train,
        object_train=obj_train, squeeze_train=sq_train, onset_train=onset_train,
        seq_len_train=seqlen_train,
        X_val=X_val, y_val=y_val, y_series_val=ys_val,
        object_val=obj_val, squeeze_val=sq_val, onset_val=onset_val,
        seq_len_val=seqlen_val,
        X_test=X_test, y_test=y_test, y_series_test=ys_test,
        object_test=obj_test, squeeze_test=sq_test, onset_test=onset_test,
        seq_len_test=seqlen_test,
        class_weights=class_weights,
        split_mode=args.split,
        holdout_object=args.holdout_object or "",
        **extra_arrays,
    )
    print(f"Saved dataset to {args.out}")
    print(f"  X shape per split: (n_trials, {target_len} timesteps, {len(feature_names_ref)} features)")
    print(f"  train class counts: stable={int(counts[0])} slip={int(counts[1])}  "
          f"class_weights={class_weights.tolist()}")


if __name__ == "__main__":
    main()