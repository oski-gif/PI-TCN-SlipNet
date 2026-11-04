"""
READ-ONLY DIAGNOSTIC (2026-11-04). Changes nothing in the benchmark, the policies
or any stored result. Physics variants are set on the in-memory model of each trial
and never reach a scene file, so nothing here affects any other run.

Question 1: when a held object moves under constant load, is it sliding against
fingertips that have stopped moving, while its contacts are still inside their
friction cones? Coulomb friction allows no sliding there, so that would be the soft
contact drift MuJoCo describes for its default settings (pyramidal cone, impratio 1,
no noslip pass), not a grip losing to the load.

Question 2: how much do the same policy's results move under stricter contact
settings? A sensitivity check only. The policy and the TCN were trained on the
default physics, and changing the benchmark's physics is a separate decision.

Also records what the policy does with each finger, because results.md only has
this for 8 objects at formation. The "default" variant must reproduce the stored
rows of Data/rl_seed1_dev_deterministic.jsonl, or nothing else here can be trusted.

Run from this directory, with the usual scene arguments plus --policy, --variants,
--n-trials-per-object, --seed-start and --out-json.
"""
import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch loads, same as train_rl_policy.py

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

import closed_loop_sim as cls
from rl_env import SlipGraspEnv

VARIANTS = {
    "default": {},
    "elliptic": {"cone": int(mujoco.mjtCone.mjCONE_ELLIPTIC)},
    "elliptic_impratio10": {"cone": int(mujoco.mjtCone.mjCONE_ELLIPTIC), "impratio": 10.0},
    "noslip20": {"noslip_iterations": 20},
}
SAMPLE_EVERY = 10  # RL steps; one RL step is 5 physics steps, or 0.01 s
SETTLE_END = 160  # RL step at which the pull starts
FULL_PULL = 260  # RL step at which the pull reaches its full value
SECONDS_PER_RL_STEP = 0.01


class PhysicsVariantEnv(SlipGraspEnv):
    """SlipGraspEnv with solver options overridden on each trial's compiled model."""

    def __init__(self, args, objects, physics, **kwargs):
        self._physics = physics
        super().__init__(args, objects, **kwargs)

    def _build(self, object_name, seed):
        super()._build(object_name, seed)
        for name, value in self._physics.items():
            setattr(self.model.opt, name, value)


def contact_summary(env):
    """Forces on the object from every active contact, split into what normal force
    and friction each carry upward, and how close friction is to its limit."""
    model, data = env.model, env.data
    force = np.zeros(6)
    up_normal = up_friction = capacity = tangential = worst = 0.0
    by_body = {}
    n_active = 0
    for i in range(data.ncon):
        con = data.contact[i]
        b1, b2 = int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])
        if env.obj_body_id not in (b1, b2) or con.efc_address < 0:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        fn, ft = float(force[0]), float(np.hypot(force[1], force[2]))
        if fn <= 0.0:
            continue
        n_active += 1
        mu = float(con.friction[0])
        capacity += mu * fn
        tangential += ft
        worst = max(worst, ft / (mu * fn))
        # The contact frame's rows are its axes in world coordinates. The computed
        # force acts on geom2, so it is reversed when the object is geom1.
        frame = np.asarray(con.frame).reshape(3, 3)
        sign = 1.0 if b2 == env.obj_body_id else -1.0
        up_normal += sign * fn * frame[0, 2]
        up_friction += sign * (force[1] * frame[1, 2] + force[2] * frame[2, 2])
        other = b1 if b2 == env.obj_body_id else b2
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)
        by_body[name] = by_body.get(name, 0.0) + fn
    return {"n_active": n_active, "capacity": capacity, "tangential": tangential,
            "worst_ratio": worst, "up_normal": up_normal, "up_friction": up_friction,
            "fn_by_body": {k: round(v, 3) for k, v in by_body.items()}}


def policy_mean(model, obs):
    """The policy's raw mean action, before clipping to [-1, 1]."""
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        return model.policy.get_distribution(obs_tensor).distribution.mean[0].numpy()


def take_sample(env, tip_ids, rl_step, info, raw_mean):
    data = env.data
    pos = data.xpos[env.obj_body_id]
    actuator_ids = [aid for pair in env.finger_actuator_ids for aid in pair]
    return {
        "rl_step": rl_step,
        "phase": info["phase"],
        "fraction": [round(f, 4) for f in env.fraction],
        "raw_mean_action": [round(float(a), 3) for a in raw_mean],
        "slip_probability": round(float(info["slip_probability"]), 4),
        "object_pos": [float(v) for v in pos],
        "displacement": None if env._ref_pos is None else float(np.linalg.norm(pos - env._ref_pos)),
        "tip_pos": [[float(v) for v in data.xpos[t]] for t in tip_ids],
        "actuator_force": [round(float(data.actuator_force[a]), 3) for a in actuator_ids],
        # The same readings the policy and the reward use, per fingertip.
        "tip_touch": [round(v, 4) for v in env._touch_values()],
        "tip_normal": [round(float(env._sensor_read(env.force_ids[n])[env.args.normal_axis]), 4)
                       for n in env.args.fingertips],
        "grip_force_sum": round(env._grip_force_sum(), 4),
        "contact": contact_summary(env),
    }


def run_trial(env, model, object_name, seed):
    obs, _ = env.reset(options={"object_name": object_name, "scene_seed": seed})
    tip_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in env.args.fingertips]
    samples, totals, info = [], {"ep_len": 0, "ep_reward": 0.0}, {}
    terminated = truncated = False
    while not (terminated or truncated):
        raw_mean = policy_mean(model, obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        totals["ep_len"] += 1
        totals["ep_reward"] += float(reward)
        if totals["ep_len"] % SAMPLE_EVERY == 0 or terminated or truncated:
            samples.append(take_sample(env, tip_ids, totals["ep_len"], info, raw_mean))
    return info, totals, samples


def full_load_summary(samples, object_weight, fingertips):
    """Motion and friction use from full pull to the end of the trial. None if the
    trial ended before enough of that window existed to measure a rate."""
    window = [s for s in samples if s["rl_step"] >= FULL_PULL and s["displacement"] is not None]
    if len(window) < 3:
        return None
    first, last = window[0], window[-1]
    seconds = (last["rl_step"] - first["rl_step"]) * SECONDS_PER_RL_STEP
    # Only fingertips pressing on the object at the start of the window.
    bodies = first["contact"]["fn_by_body"]
    touching = [i for i, name in enumerate(fingertips) if bodies.get(name, 0.0) > 0.0]
    object_move = np.array(last["object_pos"]) - np.array(first["object_pos"])
    tip_moves = [np.linalg.norm(np.array(last["tip_pos"][i]) - np.array(first["tip_pos"][i]))
                 for i in touching]
    capacity = np.mean([s["contact"]["capacity"] for s in window])
    tangential = np.mean([s["contact"]["tangential"] for s in window])
    up_friction = np.mean([s["contact"]["up_friction"] for s in window])
    up_normal = np.mean([s["contact"]["up_normal"] for s in window])
    return {
        "seconds": seconds,
        "object_speed_mm_s": 1000 * float(np.linalg.norm(object_move)) / seconds,
        "object_move_mm": 1000 * float(np.linalg.norm(object_move)),
        "tip_move_mm_median": 1000 * float(np.median(tip_moves)) if tip_moves else None,
        "friction_used": float(tangential / capacity) if capacity > 0 else None,
        "worst_contact_ratio": float(np.mean([s["contact"]["worst_ratio"] for s in window])),
        "n_contacts": float(np.mean([s["contact"]["n_active"] for s in window])),
        "up_friction": float(up_friction),
        "up_normal": float(up_normal),
        "load_expected": 2.5 * object_weight,
        "capacity_over_load": float(capacity / (2.5 * object_weight)),
        "bodies_touching": list(bodies),
        "n_tips_touching": len(touching),
    }


def compare_with_stored(rows, stored_path):
    stored = {}
    for line in open(stored_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            stored[(r["object"], int(r["seed"]))] = r
    mismatches = 0
    for r in rows:
        s = stored.get((r["object"], r["seed"]))
        if s is None:
            continue
        same = (r["held"] == s["held"] and r["slipped"] == s["slipped"] and r["ep_len"] == s["ep_len"]
                and abs(r["max_displacement"] - s["max_displacement"]) < 1e-9
                and abs(r["formation_max_fall"] - s["formation_max_fall"]) < 1e-9)
        if not same:
            mismatches += 1
            print(f"  MISMATCH {r['object']} seed={r['seed']}: held {r['held']} vs {s['held']}, "
                  f"ep_len {r['ep_len']} vs {s['ep_len']}, max_disp {r['max_displacement']:.6f} vs "
                  f"{s['max_displacement']:.6f}")
    print(f"Reproduction of {stored_path}: {len(rows) - mismatches}/{len(rows)} trials match")
    return mismatches


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--n-trials-per-object", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--stored-default", default="../../Data/rl_seed1_dev_deterministic.jsonl")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    args.squeeze_intensity = 0.95
    torch.set_num_threads(1)  # a second job on this machine shouldn't take more than one core

    model = PPO.load(args.policy, device="cpu")
    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    seeds = range(args.seed_start, args.seed_start + args.n_trials_per_object)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    with open(args.out_json, "a") as out:
        for variant in args.variants:
            env = PhysicsVariantEnv(args, objects, VARIANTS[variant], episode_rng_seed=0,
                                    terminate_on_formation_drop=False)
            rows = []
            for obj in objects:
                for seed in seeds:
                    info, totals, samples = run_trial(env, model, obj, seed)
                    weight = env._obj_weight
                    row = {"variant": variant, "object": obj, "seed": seed,
                           "held": not info["slipped"] and not info["dropped_during_formation"],
                           "slipped": bool(info["slipped"]),
                           "dropped_during_formation": bool(info["dropped_during_formation"]),
                           "formation_max_fall": float(info["formation_max_fall"]),
                           "max_displacement": float(info["max_displacement"]),
                           "threshold": float(info["threshold"]),
                           "object_weight": weight,
                           "full_load": full_load_summary(samples, weight, args.fingertips),
                           "samples": samples, **totals}
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    fl = row["full_load"]
                    note = "" if fl is None else (
                        f"  speed {fl['object_speed_mm_s']:6.2f} mm/s  tips moved {fl['tip_move_mm_median'] or 0:5.2f} mm  "
                        f"friction used {fl['friction_used'] or 0:.2f}  capacity/load {fl['capacity_over_load']:5.1f}")
                    print(f"{variant:<20} {obj:<20} seed={seed} held={row['held']!s:<5} slipped={row['slipped']!s:<5} "
                          f"fall={1000 * row['formation_max_fall']:6.1f} mm  max_disp={1000 * row['max_displacement']:6.1f} mm"
                          f"{note}", flush=True)
            env.close()
            print(f"== {variant}: held {sum(r['held'] for r in rows)}/{len(rows)}\n", flush=True)
            if variant == "default":
                compare_with_stored(rows, args.stored_default)


if __name__ == "__main__":
    main()
