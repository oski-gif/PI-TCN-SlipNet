"""
BOUNDED DIAGNOSTIC. Clean, apples-to-apples comparison under REALISTIC
timing (500/300 ramp/settle, the actual shared default) only:
  (a) baseline: squeeze_intensity=0.8, ORIGINAL policy settings
      (threshold=0.5, STEP_FRACTION=0.15), monkey-patched here since the
      real decide_action() now has the tuned settings
  (b) tuned: squeeze_intensity=0.95, the REAL current decide_action()
      (threshold=0.3, STEP_FRACTION=0.35), unmodified
Never writes to closed_loop_sim.py.
"""
import numpy as np
import closed_loop_sim as cls


def original_policy(slip_probability, driven_targets):
    if slip_probability <= 0.5:
        return {}
    actions = {}
    for aid, (lo, hi, target) in driven_targets.items():
        step = 0.15 * slip_probability * (hi - target)
        actions[aid] = target + step
    return actions


def run_batch(args, objects, n, label):
    results = []
    for obj_name in objects:
        for seed in range(n):
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=2500)
    ap.add_argument("--n-trials-per-object", type=int, default=3)
    args = ap.parse_args()

    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    cls.load_model(args.checkpoint, args.in_channels)

    tuned_policy = cls.decide_action

    args.squeeze_intensity = 0.8
    cls.decide_action = original_policy
    run_batch(args, objects, args.n_trials_per_object, "BASELINE: squeeze=0.8, original policy")

    args.squeeze_intensity = 0.95
    cls.decide_action = tuned_policy
    run_batch(args, objects, args.n_trials_per_object, "TUNED: squeeze=0.95, tuned policy")

    cls.decide_action = tuned_policy


if __name__ == "__main__":
    main()
