"""
READ-ONLY DIAGNOSTIC (2026-11-04). Records the friction each trial actually got. Runs no
physics: it only builds each trial's scene with the same seed, which writes a scene file to
--out-dir, so point that somewhere disposable. Used to ask whether outcomes depend on friction
under each contact setting, because Coulomb friction says a higher limit should hold better.

MuJoCo uses the larger of the two geoms' friction values for a contact at equal priority, so a
fingertip's effective limit on the object is max(object friction, that fingertip's pad friction).

Run from this directory, with the usual scene arguments plus --seed-range and --out-json.
"""
import argparse
import json
import os

import numpy as np

import closed_loop_sim as cls


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--seed-range", nargs=2, type=int, action="append", required=True,
                    metavar=("START", "STOP"), help="Half-open; can be given more than once.")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out_json):
        for line in open(args.out_json, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["object"], r["seed"]))
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    with open(args.out_json, "a", encoding="utf-8", newline="\n") as out:
        for start, stop in args.seed_range:
            for obj in objects:
                for seed in range(start, stop):
                    if (obj, seed) in done:
                        continue
                    _, _, actual = cls.build_scene(args.mjcf, args.mesh_dir, args.fingertips, obj, args.object_pos,
                                                   tuple(args.pad_friction_range), args.orientation_tilt_deg_max,
                                                   seed, args.out_dir)
                    object_mu = float(actual["friction"].split()[0])
                    pads = {k: float(v) for k, v in actual["pad_friction"].items()}
                    tip_mu = [max(object_mu, pads[n]) for n in args.fingertips if n in pads]
                    out.write(json.dumps({"object": obj, "seed": seed, "object_mu": object_mu, "pad_mu": pads,
                                          "tip_mu": tip_mu, "tip_mu_mean": float(np.mean(tip_mu)),
                                          "tip_mu_min": float(np.min(tip_mu))}) + "\n")
                    out.flush()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
