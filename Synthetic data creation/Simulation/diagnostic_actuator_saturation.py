"""
BOUNDED DIAGNOSTIC. Directly confirms (or refutes) the actuator-saturation
hypothesis for the "saturation despite correct detection" failure group
identified in results.md's "Failure-mode breakdown" entry (2026-10-30):
21 of 27 failed trials in the 40.0% clean comparison had high
max_slip_probability (0.92-1.0) and high n_corrections (mostly 250-330+)
yet still slipped 7x-33x past threshold. That conclusion was circumstantial
(high trigger rate + high confidence + failure anyway) -- this script adds
a per-step logging hook (closed_loop_sim.py's new step_callback= parameter
on run_trial(), added alongside this script) to directly read each driven
actuator's data.ctrl value against its ctrlrange [lo, hi] bounds, every
step, for specific trials already identified as being in the saturation
group. Never writes to closed_loop_sim.py's control logic itself.

Metric: fraction of ctrlrange "closed" = (ctrl - lo) / (hi - lo), in [0, 1].
1.0 means the actuator is pinned at its maximum -- no physical room left
for any further correction, regardless of how strongly decide_action()
wants to push it.
"""
import numpy as np
import closed_loop_sim as cls

# (object, seed) pairs from the saturation group in results.md's
# failure-mode breakdown, cross-referenced directly against
# Data/closed_loop_scenes/batch_results.jsonl.
SATURATION_TRIALS = [
    ("cube", 1), ("cube", 2),
    ("ellipsoid", 0), ("ellipsoid", 2),
    ("cylindrical_prism", 0), ("cylindrical_prism", 1), ("cylindrical_prism", 2),
]


def make_logger():
    """Returns (callback, get_summary). Records, per trial, the mean and
    max "fraction of ctrlrange closed" across all driven actuators, at
    every step during phase 2 (monitoring/pull) only -- phase 0/1 is the
    scripted approach/settle, not the window where saturation matters."""
    log = {"phase2_fracs": [], "phase2_steps": 0, "pinned_steps": 0}

    def callback(step, phase, driven, data, disp):
        if phase != 2:
            return
        fracs = []
        for aid, (lo, hi, target) in driven.items():
            ctrl = float(data.ctrl[aid])
            frac = (ctrl - lo) / (hi - lo) if hi > lo else 0.0
            fracs.append(frac)
        mean_frac = float(np.mean(fracs))
        max_frac = float(np.max(fracs))
        log["phase2_fracs"].append((step, mean_frac, max_frac, disp))
        log["phase2_steps"] += 1
        # "Pinned" = every driven actuator within 1% of its ctrlrange max.
        if min(fracs) >= 0.99:
            log["pinned_steps"] += 1

    def get_summary():
        if not log["phase2_fracs"]:
            return log, None
        fracs_mean = [f[1] for f in log["phase2_fracs"]]
        fracs_max = [f[2] for f in log["phase2_fracs"]]
        return log, {
            "phase2_steps": log["phase2_steps"],
            "pinned_steps": log["pinned_steps"],
            "pinned_fraction": log["pinned_steps"] / log["phase2_steps"],
            "mean_frac_closed_avg": float(np.mean(fracs_mean)),
            "max_frac_closed_final": fracs_max[-1],
            "mean_frac_closed_final": fracs_mean[-1],
        }

    return callback, get_summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=2500)
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    cls.load_model(args.checkpoint, args.in_channels, device="cpu")  # see diagnostic_lower_threshold.py's note on why this must be explicit

    for obj_name, seed in SATURATION_TRIALS:
        callback, get_summary = make_logger()
        try:
            result = cls.run_trial(args, obj_name, seed, args.max_steps,
                                   use_viewer=False, verbose=False,
                                   step_callback=callback)
        except Exception as e:
            print(f"{obj_name:20s} seed={seed}  FAILED: {type(e).__name__}: {e}")
            continue
        _, summary = get_summary()
        if summary is None:
            print(f"{obj_name:20s} seed={seed}  no phase-2 steps recorded (unexpected)")
            continue
        overshoot = result["max_displacement"] / result["threshold"]
        print(f"{obj_name:20s} seed={seed}  slipped={result['slipped']}  "
              f"overshoot={overshoot:.2f}x  n_corrections={result['n_corrections']}  "
              f"max_slip_p={result['max_slip_probability']:.3f}")
        print(f"{'':20s}         phase2_steps={summary['phase2_steps']}  "
              f"pinned_steps={summary['pinned_steps']} "
              f"({100*summary['pinned_fraction']:.1f}% of phase 2) "
              f"mean_frac_closed_avg={summary['mean_frac_closed_avg']:.4f}  "
              f"frac_closed_at_end(mean/max)="
              f"{summary['mean_frac_closed_final']:.4f}/{summary['max_frac_closed_final']:.4f}")


if __name__ == "__main__":
    main()
