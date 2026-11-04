"""
READ-ONLY DIAGNOSTIC (2026-11-04). Runs the unchanged hand-tuned controller under
the contact settings tested in diagnostic_contact_physics.py, so a sensitivity
result for RL has a baseline beside it. Changes no file other than its own output.

run_trial() compiles its own model, so for this process only, closed_loop_sim's
reference to the mujoco module is replaced by a stand-in that sets the solver options
on each compiled model. Nothing else in the module changes. The "default" variant
sets nothing and must reproduce the stored 2300-step rows.

Run from this directory, with the usual scene arguments plus --checkpoint,
--in-channels, --squeeze-intensity 0.95, --variants, --n-trials-per-object,
--seed-start and --out-json. Uses --max-steps 2300 by default, RL's horizon.
"""
import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch loads, same as train_rl_policy.py

from types import SimpleNamespace

import mujoco
import torch

import closed_loop_sim as cls
from diagnostic_contact_physics import VARIANTS, contact_summary, full_load_summary
from generate_trials import OBJECT_BODY_NAME


class MujocoWithOptions:
    """Stands in for the mujoco module inside closed_loop_sim."""

    def __init__(self, options):
        self._options = options
        self.MjModel = self  # so MjModel.from_xml_path() below is what run_trial() calls

    def from_xml_path(self, path):
        model = mujoco.MjModel.from_xml_path(path)
        for name, value in self._options.items():
            setattr(model.opt, name, value)
        self.last_model = model  # what recorder() reads forces from; it only reads
        return model

    def __getattr__(self, name):
        return getattr(mujoco, name)


def recorder(stand_in, fingertips, samples):
    """step_callback for run_trial(). Every 50 physics steps of the pull it records
    the same readings as diagnostic_contact_physics.py. Reads state, changes nothing."""
    def callback(step, phase, driven, data, disp):
        if phase != 2 or step % 50 != 0:
            return
        model = stand_in.last_model
        obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
        tips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in fingertips]
        view = SimpleNamespace(model=model, data=data, obj_body_id=obj)
        samples.append({"rl_step": step // 5, "displacement": float(disp),
                        "object_pos": [float(v) for v in data.xpos[obj]],
                        "tip_pos": [[float(v) for v in data.xpos[t]] for t in tips],
                        "contact": contact_summary(view)})
    return callback


def compare_with_stored(rows, stored_path):
    stored = {}
    for line in open(stored_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            stored[(r["object"], int(r["seed"]))] = r
    checked = mismatches = 0
    for r in rows:
        s = stored.get((r["object"], r["seed"]))
        if s is None:
            continue
        checked += 1
        same = (r["slipped"] == s["slipped"] and r["dropped_during_formation"] == s["dropped_during_formation"]
                and r["n_corrections"] == s["n_corrections"]
                and abs(r["max_displacement"] - s["max_displacement"]) < 1e-6)
        if not same:
            mismatches += 1
            print(f"  MISMATCH {r['object']} seed={r['seed']}: slipped {r['slipped']} vs {s['slipped']}, "
                  f"corrections {r['n_corrections']} vs {s['n_corrections']}, "
                  f"max_disp {r['max_displacement']:.6f} vs {s['max_displacement']:.6f}")
    print(f"Reproduction of {stored_path}: {checked - mismatches}/{checked} trials match")


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=2300)
    ap.add_argument("--variants", nargs="+", default=["default"], choices=list(VARIANTS))
    ap.add_argument("--n-trials-per-object", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--stored-default", default="../../Data/handtuned_modelA_2300step_results.jsonl")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)

    cls.load_model(args.checkpoint, args.in_channels, device="cpu")
    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    seeds = range(args.seed_start, args.seed_start + args.n_trials_per_object)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    with open(args.out_json, "a") as out:
        for variant in args.variants:
            stand_in = MujocoWithOptions(VARIANTS[variant])
            cls.mujoco = stand_in
            rows = []
            for obj in objects:
                for seed in seeds:
                    samples = []
                    r = cls.run_trial(args, obj, seed, args.max_steps, use_viewer=False, verbose=False,
                                      step_callback=recorder(stand_in, args.fingertips, samples))
                    model = stand_in.last_model
                    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
                    weight = float(model.body_mass[obj_id]) * float(abs(model.opt.gravity[2]))
                    row = {"variant": variant, **r, "object_weight": weight,
                           "full_load": full_load_summary(samples, weight, args.fingertips)}
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
            n = len(rows)
            held = lambda r, c: (not r["slipped"]) and r["formation_max_fall"] <= c
            print(f"{variant:<22} no rule {sum(not r['slipped'] for r in rows)}/{n}   "
                  f"10 cm {sum(held(r, .10) for r in rows)}/{n}   5 cm {sum(held(r, .05) for r in rows)}/{n}", flush=True)
            if variant == "default":
                compare_with_stored(rows, args.stored_default)
    cls.mujoco = mujoco


if __name__ == "__main__":
    main()
