"""
Batch closed-loop trial runner -- Claude-written infrastructure, same
model-vs-closed-loop split as closed_loop_sim.py. Calls run_trial() from
closed_loop_sim.py many times, headless, across multiple objects and
seeds, to get real closed-loop performance statistics for a trained model
instead of watching one trial at a time.

Loads the checkpoint ONCE (load_model()), then loops trials -- each trial
gets its own fresh scene (new seed -> new friction/mass/geometry jitter,
same as generate_trials.py), with the control policy (decide_action())
actively running throughout.

Results are written to --out-json incrementally, after EVERY trial, not
just at the end. A trial that raises is logged as an error row. After 3
failures in a row the process exits non-zero, and a rerun retries the failed
trials first, so a trial that always fails halts the run for a person to look
at instead of being dropped from the rate.

Usage:
    python batch_closed_loop_trials.py \\
        --mjcf third_party/orcahand_description/v2/scene_slip.xml \\
        --fingertips ... --actuators ... --mesh-dir assets \\
        --checkpoint ../../src/training/model_b.pt --in-channels 25 \\
        --squeeze-intensity 0.95 --n-trials-per-object 3 --max-steps 2300 \\
        --out-json ../../Data/some_new_results.jsonl

--max-steps is required and counts TOTAL steps including ramp and settle. 2300
matches the RL evaluations (ramp + settle + 1500); results.md, "Baseline
comparability check", records what going without this cost. --out-json is
resumable: rows already written are kept, and a file written with a different
config is refused rather than overwritten or mixed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import closed_loop_sim
from closed_loop_sim import (
    FORMATION_DROP_THRESHOLD, OBJECT_LIBRARY, add_common_args, file_sha256,
    load_model, read_rows_for_resume, run_trial,
)

MAX_CONSECUTIVE_FAILURES = 3


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, max_steps_default=None)
    ap.set_defaults(squeeze_intensity=None)
    ap.add_argument("--n-trials-per-object", type=int, default=3,
                    help="Trials per object in OBJECT_LIBRARY, each with a "
                         "different seed -- total trials = this x 15 objects.")
    ap.add_argument("--objects", nargs="+", default=None,
                    help="Subset of object names to test (default: all 15 "
                         "in OBJECT_LIBRARY).")
    ap.add_argument("--out-json", required=True,
                    help="Write results here incrementally, one line of "
                         "JSON per trial, including this run's config. A trial "
                         "that raises is logged as an error row, and the "
                         "process exits non-zero if any trial is missing, so "
                         "rerunning the same command retries it.")
    args = ap.parse_args()
    if args.max_steps is None:
        ap.error("--max-steps is required: it is total steps including ramp and "
                 "settle, and it changes the result. 2300 matches the RL evaluations.")
    if args.squeeze_intensity is None:
        ap.error("--squeeze-intensity is required: every cited baseline and RL evaluation uses 0.95.")

    objects = args.objects or [o["name"] for o in OBJECT_LIBRARY]
    out_json = args.out_json
    full_pull_steps = args.max_steps - args.ramp_steps - args.settle_steps - args.pull_ramp_steps
    if full_pull_steps <= 0:
        ap.error(f"--max-steps {args.max_steps} never reaches full pull.")

    # Round-tripped through JSON so tuples compare equal to rows read back on resume.
    run_config = json.loads(json.dumps({
        "controller": "hand_tuned", "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint), "in_channels": args.in_channels,
        "code_sha256": file_sha256(closed_loop_sim.__file__),
        "objects": objects, "n_trials_per_object": args.n_trials_per_object,
        "mjcf": args.mjcf, "mjcf_sha256": file_sha256(args.mjcf), "mesh_dir": args.mesh_dir,
        "fingertips": args.fingertips, "actuators": args.actuators,
        "normal_axis": args.normal_axis,
        "max_steps_total": args.max_steps, "ramp_steps": args.ramp_steps,
        "settle_steps": args.settle_steps, "pull_ramp_steps": args.pull_ramp_steps,
        "full_pull_steps": full_pull_steps,
        "pull_multiple": args.pull_multiple, "squeeze_intensity": args.squeeze_intensity,
        "object_pos": args.object_pos, "pad_friction_range": args.pad_friction_range,
        "orientation_tilt_deg_max": args.orientation_tilt_deg_max,
        "formation_drop_threshold": FORMATION_DROP_THRESHOLD}))
    print(f"Run config: {run_config}", flush=True)

    completed = {}
    if os.path.exists(out_json):
        for row in read_rows_for_resume(out_json, run_config):
            if "error" not in row:
                completed[(row["object"], row["seed"])] = row
        print(f"Resuming: {len(completed)} completed trials found in {out_json}", flush=True)

    load_model(args.checkpoint, args.in_channels)

    results = []
    errors = []
    start = time.time()
    total = len(objects) * args.n_trials_per_object
    done = 0
    consecutive_failures = 0

    with open(out_json, "a") as f:
        for obj_name in objects:
            for seed in range(args.n_trials_per_object):
                done += 1
                if (obj_name, seed) in completed:
                    results.append(completed[(obj_name, seed)])
                    continue
                try:
                    r = run_trial(args, obj_name, seed, args.max_steps,
                                  use_viewer=False, verbose=False)
                except Exception as e:
                    print(f"  [{done}/{total}] object={obj_name:20s} seed={seed}  "
                          f"FAILED: {type(e).__name__}: {e}", flush=True)
                    errors.append({"object": obj_name, "seed": seed,
                                  "error": f"{type(e).__name__}: {e}"})
                    f.write(json.dumps({**run_config, "object": obj_name, "seed": seed,
                                        "error": str(e)}) + "\n")
                    f.flush()
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"Stopping after {consecutive_failures} consecutive failures -- "
                              f"rerun the same command to resume.", flush=True)
                        sys.exit(3)
                    continue
                if r["steps_completed"] != args.max_steps:
                    print(f"Trial {obj_name}/{seed} stopped at step {r['steps_completed']} of "
                          f"{args.max_steps}, not saving it. Exiting.", flush=True)
                    sys.exit(3)
                consecutive_failures = 0
                results.append(r)
                f.write(json.dumps({**run_config, **r}) + "\n")
                f.flush()
                print(f"  [{done}/{total}] object={obj_name:20s} seed={seed}  "
                      f"slipped={r['slipped']}  dropped={r['dropped_during_formation']}  "
                      f"max_disp={r['max_displacement']:.4f}  "
                      f"max_p={r['max_slip_probability']:.4f}  "
                      f"n_corrections={r['n_corrections']}", flush=True)

    elapsed = time.time() - start
    n_ok = len(results)
    print(f"\n{n_ok}/{total} trials available, {elapsed:.1f}s this process, {len(errors)} failed")
    print(f"Results written incrementally to {out_json}")

    if not results:
        print("No successful trials -- nothing to summarize.")
        sys.exit(3)

    def held_slip_only(r):
        return not r["slipped"]

    def held_rl_scoring(r):
        return not r["slipped"] and not r["dropped_during_formation"]

    def rate(rows, held):
        n_held = sum(held(r) for r in rows)
        return f"{n_held}/{len(rows)} ({100 * n_held / len(rows):.1f}%)"

    print(f"\nLabel: hand-tuned, {args.max_steps} total steps, {full_pull_steps} at full pull, "
          f"squeeze {args.squeeze_intensity}")
    print(f"Overall, slip only (no drop rule): {rate(results, held_slip_only)}")
    print(f"Overall, RL scoring (not slipped, not dropped): {rate(results, held_rl_scoring)}")
    dropped = [r for r in results if r["dropped_during_formation"]]
    print(f"Dropped during formation: {len(dropped)}/{n_ok}, of which held under "
          f"slip-only scoring: {sum(held_slip_only(r) for r in dropped)}")
    print(f"Overall mean max_slip_probability: {np.mean([r['max_slip_probability'] for r in results]):.4f}")
    print(f"Overall mean n_corrections: {np.mean([r['n_corrections'] for r in results]):.2f}")

    print("\nPer-object breakdown:")
    for obj_name in objects:
        obj_results = [r for r in results if r["object"] == obj_name]
        if not obj_results:
            continue
        print(f"  {obj_name:20s}  slip-only {rate(obj_results, held_slip_only)}  "
              f"RL-scoring {rate(obj_results, held_rl_scoring)}  "
              f"mean_max_disp={np.mean([r['max_displacement'] for r in obj_results]):.4f}")

    if n_ok != total:
        print(f"WARNING: INCOMPLETE -- {n_ok}/{total} trials. Rerun to retry before citing.")
        sys.exit(3)


if __name__ == "__main__":
    main()
