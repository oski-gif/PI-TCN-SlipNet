"""
BOUNDED DIAGNOSTIC, not part of the production pipeline. Tests a
per-finger-LOCALIZED correction against the current tuned (global-scalar)
policy, monkey-patching closed_loop_sim.decide_action in memory only --
the real, persisted decide_action() in closed_loop_sim.py is never touched
by this script.

Hypothesis under test (research-agent finding, see "Neural network
approach comparison.md" section 4.2.2): decide_action() applies the
IDENTICAL correction to every actuator from one pooled slip_probability,
even though per-fingertip touch/normal/tangential/torque_mag features
already exist in the pipeline (both as model input and, now, as the
`feats` row this script reads). Three independent GLOBAL-policy changes
(squeeze_intensity sweep, fixed-step correction, hysteresis) already
plateaued at 40% -- this tests whether the lever is actually about WHICH
finger gets corrected, not how hard or how often.

Per-finger signal: for each fingertip, ratio_i = tangential_i / |normal_i|
(a cheap, no-retraining-needed proxy for "how close to the Coulomb slip
cone this finger's own contact currently looks," using features the model
already has). weight_i = clip(ratio_i / mean(ratio), 0.5, 2.0) -- fingers
whose own ratio is above the 5-finger average get amplified correction,
fingers below it get damped, clipped so no finger ever gets zero or a
wildly oversized step. The existing global slip_probability gate/threshold
and STEP_FRACTION are left EXACTLY as in the real decide_action() -- this
isolates ONE variable (per-finger weighting) against the same baseline
every prior diagnostic in this file's family has used, same discipline as
diagnostic_hysteresis.py / diagnostic_policy_variants.py.

Benchmark: same 45-trial setup (15 objects x 3 seeds, realistic 500/300
ramp/settle timing, squeeze_intensity=0.95) as every prior closed-loop
comparison logged in results.md.
"""
import numpy as np

import closed_loop_sim as cls

STEP_FRACTION = cls.STEP_FRACTION  # 0.35, same as the real tuned policy
THRESHOLD = 0.3  # same gate as the real tuned policy
WEIGHT_CLIP = (0.5, 2.0)
EPS = 1e-6


def make_per_finger_policy(n_fingers):
    def policy(slip_probability, driven_targets, feats=None, **kwargs):
        if slip_probability <= THRESHOLD or feats is None:
            return {}

        ratios = []
        for i in range(n_fingers):
            normal = feats[4 * i + 1]
            tangential = feats[4 * i + 2]
            ratios.append(tangential / max(abs(normal), EPS))
        mean_ratio = float(np.mean(ratios)) if ratios else EPS
        weights = [float(np.clip(r / max(mean_ratio, EPS), *WEIGHT_CLIP))
                   for r in ratios]

        actions = {}
        # driven_targets is keyed by actuator id but built in run_trial()
        # by iterating args.actuators in order -- consecutive (mcp, pip)
        # pairs per finger (same convention as rl_env.py's finger_pairs).
        # dict insertion order is preserved (Python 3.7+), so enumerating
        # here reliably maps actuator position -> finger index.
        for i, (aid, (lo, hi, target)) in enumerate(driven_targets.items()):
            finger_idx = i // 2
            w = weights[finger_idx]
            step = STEP_FRACTION * slip_probability * w * (hi - target)
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=2500)
    ap.add_argument("--n-trials-per-object", type=int, default=3)
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    cls.load_model(args.checkpoint, args.in_channels)
    tuned_policy = cls.decide_action
    n_fingers = len(args.actuators) // 2

    cls.decide_action = tuned_policy
    run_batch(args, objects, args.n_trials_per_object,
              "CURRENT tuned policy (global scalar)")

    per_finger_policy = make_per_finger_policy(n_fingers)
    run_batch(args, objects, args.n_trials_per_object,
              "PER-FINGER localized correction", policy=per_finger_policy)

    cls.decide_action = tuned_policy


if __name__ == "__main__":
    main()
