"""
BOUNDED DIAGNOSTIC. Tests a smoothed/hysteresis policy variant (per the
research agent's finding: decide_action() currently reacts to one
instantaneous probability with a flat threshold, no smoothing, despite a
300-step history buffer sitting unused) against the current tuned policy,
both at squeeze_intensity=0.95, both under REALISTIC timing. Monkey-patched
only -- never writes to closed_loop_sim.py.
"""
import numpy as np
import closed_loop_sim as cls


def make_hysteresis_policy(engage_threshold, disengage_threshold, alpha, step_fraction):
    """alpha: EMA smoothing factor (higher = less smoothing, more reactive
    to the latest value). state is a per-trial-reset dict, not a module
    global, so trials never leak state into each other."""
    state = {"smoothed": None, "engaged": False}

    def reset():
        state["smoothed"] = None
        state["engaged"] = False

    def policy(slip_probability, driven_targets):
        if state["smoothed"] is None:
            state["smoothed"] = slip_probability
        else:
            state["smoothed"] = alpha * slip_probability + (1 - alpha) * state["smoothed"]

        if not state["engaged"]:
            if state["smoothed"] > engage_threshold:
                state["engaged"] = True
        else:
            if state["smoothed"] < disengage_threshold:
                state["engaged"] = False

        if not state["engaged"]:
            return {}
        actions = {}
        for aid, (lo, hi, target) in driven_targets.items():
            step = step_fraction * state["smoothed"] * (hi - target)
            actions[aid] = target + step
        return actions

    return policy, reset


def run_batch(args, objects, n, label, policy_and_reset=None):
    results = []
    for obj_name in objects:
        for seed in range(n):
            if policy_and_reset is not None:
                policy, reset = policy_and_reset
                reset()
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
    return rate


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

    # Re-confirm the current tuned policy under this run's conditions (same
    # objects/seeds/timing as the hysteresis test, for a fair comparison).
    cls.decide_action = tuned_policy
    run_batch(args, objects, args.n_trials_per_object, "current tuned policy (no smoothing)")

    policy_and_reset = make_hysteresis_policy(
        engage_threshold=0.3, disengage_threshold=0.15, alpha=0.3, step_fraction=0.35)
    run_batch(args, objects, args.n_trials_per_object,
              "hysteresis (engage=0.3, disengage=0.15, EMA alpha=0.3)",
              policy_and_reset=policy_and_reset)

    cls.decide_action = tuned_policy


if __name__ == "__main__":
    main()
