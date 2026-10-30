"""
BOUNDED DIAGNOSTIC, not part of the production pipeline. Tests alternative
decide_action() formulas by monkey-patching closed_loop_sim.decide_action
in memory for the duration of each test batch, WITHOUT ever writing to
closed_loop_sim.py -- the user's actual, persisted control policy is never
touched by this script. Purpose: generate real, comparable evidence for
which policy design is actually worth implementing for real, not guess.

Hypothesis under test: the current policy's
    step = STEP_FRACTION * probability * (hi - target)
shrinks toward zero as target approaches hi, so corrections have less and
less room to work with exactly when the grip is already tight and still
slipping. Testing a FIXED-size correction (not scaled by remaining room)
as the alternative.
"""
import sys

import numpy as np

import closed_loop_sim as cls


def make_fixed_step_policy(fixed_fraction, threshold):
    """fixed_fraction: fraction of the actuator's FULL range (hi - lo),
    not remaining room, applied every time probability exceeds threshold.
    Clamped to hi so it never exceeds the real actuator limit."""
    def policy(slip_probability, driven_targets, **kwargs):
        # **kwargs absorbs run_trial()'s feats= kwarg (added later for a
        # per-finger-localized policy variant) -- unused here, same
        # behavior as before.
        if slip_probability <= threshold:
            return {}
        actions = {}
        for aid, (lo, hi, target) in driven_targets.items():
            new_target = min(target + fixed_fraction * (hi - lo), hi)
            actions[aid] = new_target
        return actions
    return policy


def run_batch(args, objects, n_trials_per_object, label):
    results = []
    for obj_name in objects:
        for seed in range(n_trials_per_object):
            try:
                r = cls.run_trial(args, obj_name, seed, args.max_steps,
                                  use_viewer=False, verbose=False)
                results.append(r)
            except Exception as e:
                print(f"  [{label}] {obj_name} seed={seed} FAILED: {e}")
    n = len(results)
    n_slipped = sum(r["slipped"] for r in results)
    success_rate = 100 * (n - n_slipped) / n if n else 0.0
    mean_corrections = np.mean([r["n_corrections"] for r in results]) if results else 0.0
    print(f"[{label}] {n - n_slipped}/{n} held securely ({success_rate:.1f}%)  "
          f"mean_n_corrections={mean_corrections:.2f}")
    return success_rate, results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--n-trials-per-object", type=int, default=2)
    ap.add_argument("--objects", nargs="+", default=None)
    args = ap.parse_args()

    objects = args.objects or [o["name"] for o in cls.OBJECT_LIBRARY]
    cls.load_model(args.checkpoint, args.in_channels)

    # Baseline: current real policy, unmodified (whatever's actually in
    # closed_loop_sim.py right now -- threshold 0.3, STEP_FRACTION 0.35).
    run_batch(args, objects, args.n_trials_per_object, "current (proportional, threshold=0.3)")

    # Variant: fixed-size correction, same trigger threshold, a few
    # magnitudes to compare.
    original_policy = cls.decide_action
    for fixed_fraction in [0.05, 0.10, 0.20]:
        cls.decide_action = make_fixed_step_policy(fixed_fraction, threshold=0.3)
        run_batch(args, objects, args.n_trials_per_object,
                  f"fixed-step={fixed_fraction} (threshold=0.3)")
    cls.decide_action = original_policy


if __name__ == "__main__":
    main()
