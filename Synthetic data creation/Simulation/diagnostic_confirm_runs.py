"""
READ-ONLY DIAGNOSTIC (2026-11-04). Runs one unchanged controller under one set of contact
settings over one seed range. Changes no file other than its own output, and the settings
are applied to in-memory models only, the same way as diagnostic_contact_physics.py and
diagnostic_handtuned_physics.py, whose trial code it reuses. Measuring a setting here does
not adopt it: changing the benchmark's physics is the user's decision.

What it adds to those two scripts, for runs of 1,005 trials on a laptop:
  - Resumable by (object, seed), so a reboot or a memory failure loses at most one trial.
  - Compact rows, with no per-step samples: finger closure at a few points and the slip
    probability during the pull, plus the same full-load summary. About 1 MB per 1,005 trials.
  - Leaving out --checkpoint removes the TCN, for either controller. A policy then sees a
    slip probability of 0, and the hand-tuned controller never corrects its grip.

Run from this directory, with the usual scene arguments plus --controller, --variant,
--n-trials-per-object, --seed-start and --out-json, and --policy for a policy.
"""
import argparse
import json
import os
import statistics

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch loads, same as train_rl_policy.py

import mujoco
import torch
from stable_baselines3 import PPO

import closed_loop_sim as cls
from diagnostic_contact_physics import VARIANTS, PhysicsVariantEnv, full_load_summary
from diagnostic_contact_physics import run_trial as policy_trial
from diagnostic_handtuned_physics import MujocoWithOptions, recorder
from generate_trials import OBJECT_BODY_NAME

VARIANTS.update({
    # Is the noslip result sensitive to the iteration count, or to the friction cone?
    "noslip5": {"noslip_iterations": 5},
    "noslip50": {"noslip_iterations": 50},
    "elliptic_noslip20": {"cone": int(mujoco.mjtCone.mjCONE_ELLIPTIC), "noslip_iterations": 20},
})
CLOSURE_STEPS = (10, 20, 50, 160)  # RL steps; settle ends at 160
HAND_TUNED_MAX_STEPS = 2300  # RL's horizon, counted from the start of the trial
POLICY_MAX_STEPS = 1500  # counted from the end of settle, so the same 2300 in total


def policy_row(env, model, args, obj, seed):
    info, totals, samples = policy_trial(env, model, obj, seed)
    by_step = {s["rl_step"]: s for s in samples}
    pull = [s["slip_probability"] for s in samples if s["phase"] == 2]
    return {
        "variant": args.variant, "tcn": bool(args.checkpoint), "object": obj, "seed": seed,
        "held": not info["slipped"] and not info["dropped_during_formation"],
        "slipped": bool(info["slipped"]),
        "dropped_during_formation": bool(info["dropped_during_formation"]),
        "formation_max_fall": float(info["formation_max_fall"]),
        "max_displacement": float(info["max_displacement"]),
        "threshold": float(info["threshold"]),
        "object_weight": env._obj_weight,
        "full_load": full_load_summary(samples, env._obj_weight, args.fingertips),
        "closure": {str(k): by_step[k]["fraction"] for k in CLOSURE_STEPS if k in by_step},
        "final_fraction": samples[-1]["fraction"],
        "pull_probability_median": statistics.median(pull) if pull else None,
        "pull_probability_max": max(pull) if pull else None,
        **totals,
    }


def hand_row(stand_in, args, obj, seed):
    samples = []
    r = cls.run_trial(args, obj, seed, args.max_steps, use_viewer=False, verbose=False,
                      step_callback=recorder(stand_in, args.fingertips, samples))
    model = stand_in.last_model
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
    weight = float(model.body_mass[obj_id]) * float(abs(model.opt.gravity[2]))
    return {"variant": args.variant, "tcn": bool(args.checkpoint), **r,
            "held": (not r["slipped"]) and r["formation_max_fall"] <= 0.05,
            "object_weight": weight, "full_load": full_load_summary(samples, weight, args.fingertips)}


def read_done(path):
    """Finished rows by (object, seed). A crash mid-write can leave a torn last line,
    and the next row would land on it, so that line is removed first."""
    if not os.path.exists(path):
        return {}
    good, torn = [], False
    for line in open(path, encoding="utf-8"):
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            torn = True
    if torn:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r) + "\n" for r in good)
    return {(r["object"], int(r["seed"])): r for r in good}


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=None)
    ap.add_argument("--controller", required=True, choices=["policy", "hand"])
    ap.add_argument("--policy")
    ap.add_argument("--variant", default="default", choices=list(VARIANTS))
    ap.add_argument("--n-trials-per-object", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    if args.controller == "policy" and not args.policy:
        ap.error("--controller policy needs --policy")
    args.squeeze_intensity = 0.95
    args.max_steps = args.max_steps or (POLICY_MAX_STEPS if args.controller == "policy" else HAND_TUNED_MAX_STEPS)
    torch.set_num_threads(1)

    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    seeds = range(args.seed_start, args.seed_start + args.n_trials_per_object)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    done = read_done(args.out_json)

    if args.controller == "policy":
        model = PPO.load(args.policy, device="cpu")
        env = PhysicsVariantEnv(args, objects, VARIANTS[args.variant], episode_rng_seed=0,
                                terminate_on_formation_drop=False)
        make_row = lambda obj, seed: policy_row(env, model, args, obj, seed)
    else:
        if args.checkpoint:
            cls.load_model(args.checkpoint, args.in_channels, device="cpu")
        stand_in = MujocoWithOptions(VARIANTS[args.variant])
        cls.mujoco = stand_in  # this process only; see diagnostic_handtuned_physics.py
        make_row = lambda obj, seed: hand_row(stand_in, args, obj, seed)

    rows = []
    with open(args.out_json, "a", encoding="utf-8", newline="\n") as out:
        for obj in objects:
            for seed in seeds:
                row = done.get((obj, seed))
                if row is None:
                    row = make_row(obj, seed)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                rows.append(row)
                if len(rows) % 50 == 0:
                    print(f"{len(rows)} trials, held so far {sum(r['held'] for r in rows)}", flush=True)

    n = len(rows)
    h10 = sum((not r["slipped"]) and r["formation_max_fall"] <= 0.10 for r in rows)
    print(f"DONE {args.controller} {args.variant} tcn={bool(args.checkpoint)}: 5 cm {sum(r['held'] for r in rows)}/{n}   "
          f"10 cm {h10}/{n}   fell past 5 cm {sum(r['formation_max_fall'] > .05 for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
