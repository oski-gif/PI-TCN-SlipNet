"""
BOUNDED DIAGNOSTIC. Tests whether a LOWER decide_action() trigger
threshold specifically rescues the detection-miss failure group
(results.md's "Failure-mode breakdown" + "Detection-miss root cause"
entries, 2026-10-30: 6-10 of 27 failures had max_slip_probability staying
under the current 0.3 trigger, mostly on mesh-kind objects, so
decide_action() essentially never fired) -- without creating new false-
positive corrections (and the instability/effort cost that comes with
them) on trials that are currently stable. Monkey-patches
closed_loop_sim.decide_action in memory only, same convention as
diagnostic_hysteresis.py / diagnostic_policy_variants.py -- never writes
to the real file.

Same full 45-trial benchmark (15 objects x 3 seeds, realistic 500/300
ramp/settle timing, squeeze_intensity=0.95, Model A/22ch) as every prior
closed-loop comparison, so this is directly comparable to the existing
40.0% baseline and every other control-policy diagnostic in results.md.
"""
import numpy as np
import closed_loop_sim as cls

STEP_FRACTION = cls.STEP_FRACTION  # 0.35, unchanged -- isolate ONE variable


def make_lower_threshold_policy(threshold):
    def policy(slip_probability, driven_targets, **kwargs):
        if slip_probability <= threshold:
            return {}
        actions = {}
        for aid, (lo, hi, target) in driven_targets.items():
            step = STEP_FRACTION * slip_probability * (hi - target)
            actions[aid] = target + step
        return actions
    return policy


def run_batch(args, objects, n, label, policy=None):
    results = []
    for obj_name in objects:
        for seed in range(n):
            if policy is not None:
                cls.decide_action = policy
            try:
                r = cls.run_trial(args, obj_name, seed, args.max_steps,
                                  use_viewer=False, verbose=False)
                results.append(r)
            except Exception as e:
                print(f"  [{label}] {obj_name} seed={seed} FAILED: {e}")
    n_ok = len(results)
    n_slipped = sum(r["slipped"] for r in results)
    rate = 100 * (n_ok - n_slipped) / n_ok if n_ok else 0.0
    print(f"[{label}] {n_ok - n_slipped}/{n_ok} held securely ({rate:.1f}%)")
    return rate, results


DETECTION_MISS_OBJECTS = {"hexagonal_prism", "square_pyramid", "cone",
                          "triangular_prism", "l_block"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=2500)
    ap.add_argument("--n-trials-per-object", type=int, default=3)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.15])
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    # Explicit device="cpu" -- NOT auto-detect. This script was run once
    # concurrently with the (CPU-only) RL training and hit a CUDA DLL
    # init failure at nearly the same moment a training worker died in
    # an unexplained way (results.md, 2026-10-30) -- auto-detect would
    # try to touch CUDA even though every training worker explicitly
    # avoids it for exactly this kind of multi-process contention risk.
    # Forcing CPU here makes this script actually safe to run alongside
    # training, matching every worker's own device policy.
    cls.load_model(args.checkpoint, args.in_channels, device="cpu")
    tuned_policy = cls.decide_action

    cls.decide_action = tuned_policy
    _, baseline_results = run_batch(args, objects, args.n_trials_per_object,
                                    "BASELINE (threshold=0.3, current tuned policy)")
    baseline_miss_held = sum(
        1 for r in baseline_results
        if r["object"] in DETECTION_MISS_OBJECTS and not r["slipped"])
    baseline_miss_total = sum(1 for r in baseline_results if r["object"] in DETECTION_MISS_OBJECTS)
    print(f"  -> detection-miss-prone objects only: "
          f"{baseline_miss_held}/{baseline_miss_total} held at baseline")

    for thr in args.thresholds:
        policy = make_lower_threshold_policy(thr)
        _, results = run_batch(args, objects, args.n_trials_per_object,
                               f"threshold={thr}", policy=policy)
        miss_held = sum(1 for r in results
                        if r["object"] in DETECTION_MISS_OBJECTS and not r["slipped"])
        miss_total = sum(1 for r in results if r["object"] in DETECTION_MISS_OBJECTS)
        other_held = sum(1 for r in results
                         if r["object"] not in DETECTION_MISS_OBJECTS and not r["slipped"])
        other_total = sum(1 for r in results if r["object"] not in DETECTION_MISS_OBJECTS)
        print(f"  -> detection-miss-prone objects: {miss_held}/{miss_total} held "
              f"(baseline was {baseline_miss_held}/{baseline_miss_total})")
        print(f"  -> everything else: {other_held}/{other_total} held "
              f"(check this didn't regress vs. baseline)")

    cls.decide_action = tuned_policy


if __name__ == "__main__":
    main()
