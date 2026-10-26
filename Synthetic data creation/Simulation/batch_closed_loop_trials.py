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
just at the end -- if one trial crashes (MuJoCo contact divergence, a
transient lookup failure, etc.), it's logged and skipped rather than
losing every already-computed result in the batch.

Usage:
    python batch_closed_loop_trials.py \\
        --mjcf third_party/orcahand_description/v2/scene_slip.xml \\
        --fingertips ... --actuators ... --mesh-dir assets \\
        --checkpoint ../../src/training/model_b.pt --in-channels 25 \\
        --n-trials-per-object 3 --max-steps 1500
"""
import argparse
import json
import time

import numpy as np

from closed_loop_sim import (
    OBJECT_LIBRARY, add_common_args, load_model, run_trial,
)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--n-trials-per-object", type=int, default=3,
                    help="Trials per object in OBJECT_LIBRARY, each with a "
                         "different seed -- total trials = this x 15 objects.")
    ap.add_argument("--objects", nargs="+", default=None,
                    help="Subset of object names to test (default: all 15 "
                         "in OBJECT_LIBRARY).")
    ap.add_argument("--out-json", default=None,
                    help="Write results here incrementally, one line of "
                         "JSON per trial, as they complete -- default: "
                         "<out-dir>/batch_results.jsonl. A trial that "
                         "raises is logged as an error entry and skipped, "
                         "not allowed to lose the rest of the batch.")
    args = ap.parse_args()

    objects = args.objects or [o["name"] for o in OBJECT_LIBRARY]
    out_json = args.out_json or f"{args.out_dir}/batch_results.jsonl"

    load_model(args.checkpoint, args.in_channels)

    results = []
    errors = []
    start = time.time()
    total = len(objects) * args.n_trials_per_object
    done = 0

    with open(out_json, "w") as f:
        for obj_name in objects:
            for seed in range(args.n_trials_per_object):
                done += 1
                try:
                    r = run_trial(args, obj_name, seed, args.max_steps,
                                  use_viewer=False, verbose=False)
                except Exception as e:
                    print(f"  [{done}/{total}] object={obj_name:20s} seed={seed}  "
                          f"FAILED: {type(e).__name__}: {e}")
                    errors.append({"object": obj_name, "seed": seed,
                                  "error": f"{type(e).__name__}: {e}"})
                    f.write(json.dumps({"object": obj_name, "seed": seed, "error": str(e)}) + "\n")
                    f.flush()
                    continue
                results.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"  [{done}/{total}] object={obj_name:20s} seed={seed}  "
                      f"slipped={r['slipped']}  max_disp={r['max_displacement']:.4f}  "
                      f"max_p={r['max_slip_probability']:.4f}  "
                      f"n_corrections={r['n_corrections']}")

    elapsed = time.time() - start
    n_ok = len(results)
    print(f"\n{n_ok}/{total} trials completed in {elapsed:.1f}s "
          f"({elapsed/max(n_ok,1):.2f}s/trial), {len(errors)} failed")
    print(f"Results written incrementally to {out_json}")

    if not results:
        print("No successful trials -- nothing to summarize.")
        return

    n_slipped = sum(r["slipped"] for r in results)
    print(f"\nOverall: {n_ok - n_slipped}/{n_ok} held securely "
          f"({100*(n_ok-n_slipped)/n_ok:.1f}% success rate)")
    print(f"Overall mean max_slip_probability: {np.mean([r['max_slip_probability'] for r in results]):.4f}")
    print(f"Overall mean n_corrections: {np.mean([r['n_corrections'] for r in results]):.2f}")

    print("\nPer-object breakdown:")
    for obj_name in objects:
        obj_results = [r for r in results if r["object"] == obj_name]
        if not obj_results:
            continue
        n_obj_slipped = sum(r["slipped"] for r in obj_results)
        n_obj = len(obj_results)
        print(f"  {obj_name:20s}  {n_obj - n_obj_slipped}/{n_obj} held securely  "
              f"mean_max_disp={np.mean([r['max_displacement'] for r in obj_results]):.4f}")


if __name__ == "__main__":
    main()
