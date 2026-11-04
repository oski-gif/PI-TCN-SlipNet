"""
READ-ONLY DIAGNOSTIC (2026-11-04). Changes nothing in the benchmark, the policies or
any stored result, and none of these grips is proposed as a controller here. Adopting
one would be a controller change, which is the user's decision.

Question: diagnostic_contact_physics.py shows the best RL policy behaving almost like
a fixed grip: thumb, index and ring close at full speed and stay closed on every
object, the pinky follows, and the middle finger mostly stays open. So how much of
its result does a fixed grip get with no policy, no tactile input and no TCN, and
which grip choices matter?

Each grip is a target closure, a closing speed and a start delay per finger, in the
order thumb, index, middle, ring, pinky, plus optional fixed values for joints the
benchmark's controllers leave at 0. It runs through the same SlipGraspEnv and is scored
the same way as evaluate_rl_policy.py --continue-past-formation-drop. With no
--checkpoint, predict_slip() returns 0.0 and nothing reads it, so the TCN plays no part.

Choosing the best of many grips on 150 dev trials inflates its score. Only a run on
seeds no grip was chosen on can say what a chosen grip really does.

Run from this directory, with the usual scene arguments plus --grips, --variant,
--n-trials-per-object, --seed-start and --out-json. Resumable by (grip, variant, object, seed).
"""
import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch loads, same as train_rl_policy.py

import mujoco
import numpy as np

import closed_loop_sim as cls
from diagnostic_contact_physics import VARIANTS, PhysicsVariantEnv
from rl_env import MAX_DELTA_PER_STEP


def grip(targets, speeds, delays=(0, 0, 0, 0, 0), pose=None):
    """Speed 1.0 is the fastest the action space allows, 0.08 closure per RL step."""
    return {"targets": list(targets), "speeds": list(speeds), "delays": list(delays), "pose": pose or {}}


FAST = [1, 1, 1, 1, 1]
S = [1, 1, 1, 0.8, 0.6]  # seed1_like's speeds, measured from seed 1's traces
NO_M = [1, 1, 0, 1, 1]
ALL = [1, 1, 1, 1, 1]
THUMB_C = {"right_t-cmc_actuator": 0.3}  # thumb tip about 6 mm from the object's center line, instead of 14
THUMB_C2 = {"right_t-cmc_actuator": 0.3, "right_t-abd_actuator": -0.2}  # about 3 mm
INWARD = {"right_i-abd_actuator": -0.4, "right_p-abd_actuator": 0.4}  # index and pinky turned toward the object

GRIPS = {
    # First set. hand_tuned_like is a control: closing to 0.95 over 100 RL steps is the hand-tuned ramp.
    "TIRP_fast": grip(NO_M, FAST),
    "seed1_like": grip(NO_M, S),
    "TIMRP_fast": grip(ALL, FAST),
    "TIR_fast": grip([1, 1, 0, 1, 0], FAST),
    "hand_tuned_like": grip([0.95] * 5, [0.119] * 5),
    # Do seeds 2 and 3 differ from seed 1 mainly in which fingers they close? Values read from their traces.
    "seed2_like": grip([1, 1, 0.3, 0.7, 1], [1, 1, 0.2, 0.15, 1]),
    "seed3_like": grip([1, 1, 0.1, 0.2, 1], [1, 1, 0.1, 0.1, 0.3]),
    "TIP_fast": grip([1, 1, 0, 0, 1], FAST),
    # Frozen joints. Does a centered thumb oppose the middle finger's push, and do inward fingers reach boxes?
    "S_thumb_c": grip(NO_M, S, pose=THUMB_C),
    "S_thumb_c2": grip(NO_M, S, pose=THUMB_C2),
    "S_inward": grip(NO_M, S, pose=INWARD),
    "S_thumb_c2_inward": grip(NO_M, S, pose={**THUMB_C2, **INWARD}),
    "M_thumb_c": grip(ALL, S, pose=THUMB_C),
    "M_thumb_c2": grip(ALL, S, pose=THUMB_C2),
    "M_inward": grip(ALL, S, pose=INWARD),
    "M_thumb_c2_inward": grip(ALL, S, pose={**THUMB_C2, **INWARD}),
    # Timing around seed1_like.
    "S_pinky30": grip(NO_M, [1, 1, 1, 0.8, 0.3]),
    "S_pinky45": grip(NO_M, [1, 1, 1, 0.8, 0.45]),
    "S_pinky80": grip(NO_M, [1, 1, 1, 0.8, 0.8]),
    "S_ring60": grip(NO_M, [1, 1, 1, 0.6, 0.6]),
    "S_ring100": grip(NO_M, [1, 1, 1, 1, 0.6]),
    "S_index80": grip(NO_M, [1, 0.8, 1, 0.8, 0.6]),
    "S_thumb80": grip(NO_M, [0.8, 1, 1, 0.8, 0.6]),
    # A middle finger that closes after the object is trapped, or only partway.
    "S_M_late30": grip(ALL, [1, 1, 0.3, 0.8, 0.6], delays=(0, 0, 30, 0, 0)),
    "S_M_late60_slow": grip(ALL, [1, 1, 0.1, 0.8, 0.6], delays=(0, 0, 60, 0, 0)),
    "S_M_partial": grip([1, 1, 0.5, 1, 1], [1, 1, 0.3, 0.8, 0.6], delays=(0, 0, 30, 0, 0)),
    # Softer: a lower target means less force at contact, since these are position servos.
    "S_soft90": grip([0.9, 0.9, 0, 0.9, 0.9], S),
    "S_soft80": grip([0.8, 0.8, 0, 0.8, 0.8], S),
    # Other finger subsets, at full speed.
    "TRP_fast": grip([1, 0, 0, 1, 1], FAST),
    "TI_fast": grip([1, 1, 0, 0, 0], FAST),
    "TR_fast": grip([1, 0, 0, 1, 0], FAST),
    "IRP_fast": grip([0, 1, 0, 1, 1], FAST),
    "TMR_fast": grip([1, 0, 1, 1, 0], FAST),
    "TIMR_fast": grip([1, 1, 1, 1, 0], FAST),
    "TMRP_fast": grip([1, 0, 1, 1, 1], FAST),
}


class GripEnv(PhysicsVariantEnv):
    """Adds fixed values for actuators the benchmark's controllers leave at 0."""

    def __init__(self, args, objects, physics, pose, **kwargs):
        self._pose = pose
        super().__init__(args, objects, physics, **kwargs)

    def _build(self, object_name, seed):
        super()._build(object_name, seed)
        for name, value in self._pose.items():
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                raise ValueError(f"Actuator {name} not found")
            self.data.ctrl[aid] = value


def grip_action(fraction, spec, rl_step):
    """Moves each finger toward its target at its speed, without overshooting."""
    gap = np.asarray(spec["targets"], dtype=float) - np.asarray(fraction, dtype=float)
    action = np.sign(gap) * np.minimum(np.asarray(spec["speeds"], dtype=float), np.abs(gap) / MAX_DELTA_PER_STEP)
    return np.where(rl_step >= np.asarray(spec["delays"]), action, 0.0)


def run_trial(env, spec, object_name, seed):
    env.reset(options={"object_name": object_name, "scene_seed": seed})
    totals = {"ep_len": 0, "ep_reward": 0.0, "formation_reward": 0.0, "phase2_reward": 0.0}
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(grip_action(env.fraction, spec, totals["ep_len"]))
        totals["ep_len"] += 1
        totals["ep_reward"] += float(reward)
        totals["phase2_reward" if info["phase"] == 2 else "formation_reward"] += float(reward)
    return info, totals


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--grips", nargs="+", default=list(GRIPS), choices=list(GRIPS))
    ap.add_argument("--variant", default="default", choices=list(VARIANTS))
    ap.add_argument("--n-trials-per-object", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    args.squeeze_intensity = 0.95
    if args.checkpoint:
        ap.error("this diagnostic runs without a TCN: leave out --checkpoint")

    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    seeds = range(args.seed_start, args.seed_start + args.n_trials_per_object)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    done = {}
    if os.path.exists(args.out_json):
        for line in open(args.out_json, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[(r["grip"], r.get("variant", "default"), r["object"], r["seed"])] = r

    with open(args.out_json, "a") as out:
        for name in args.grips:
            spec = GRIPS[name]
            env = GripEnv(args, objects, VARIANTS[args.variant], spec["pose"], episode_rng_seed=0,
                          terminate_on_formation_drop=False)
            rows = []
            for obj in objects:
                for seed in seeds:
                    key = (name, args.variant, obj, seed)
                    if key in done:
                        rows.append(done[key])
                        continue
                    info, totals = run_trial(env, spec, obj, seed)
                    row = {"grip": name, "variant": args.variant, **spec, "object": obj, "seed": seed,
                           "held": not info["slipped"] and not info["dropped_during_formation"],
                           "slipped": bool(info["slipped"]),
                           "dropped_during_formation": bool(info["dropped_during_formation"]),
                           "formation_max_fall": float(info["formation_max_fall"]),
                           "max_displacement": float(info["max_displacement"]),
                           "threshold": float(info["threshold"]), **totals}
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
            env.close()
            n = len(rows)
            held5 = sum(r["held"] for r in rows)
            held10 = sum((not r["slipped"]) and r["formation_max_fall"] <= 0.10 for r in rows)
            print(f"{name:<20} {args.variant:<20} 5 cm {held5}/{n} {100 * held5 / n:.1f}%   "
                  f"10 cm {held10}/{n} {100 * held10 / n:.1f}%   "
                  f"fell past 5 cm {sum(r['dropped_during_formation'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
