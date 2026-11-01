"""
BOUNDED DIAGNOSTIC. Deeper investigation of WHY cube/rectangular_prism
(and flat_disk/l_block) drop during formation under the VERIFIED-GOOD
64.4% policy (Data/ppo_slip_policy_latest.zip, NOT the regressed
MIN_FINGER_FRACTION_COEF=3.0 fresh retrain) -- same per-step logging
methodology that directly confirmed actuator saturation
(diagnostic_actuator_saturation.py), now pointed at formation-phase
drops. Logs EVERY step (not sampled) through formation, tracking:
  - per-finger FIRST-CONTACT step (coordination: do fingers arrive
    together or does one lag far behind the others?)
  - closing speed: fraction's own rate of change per step
  - object velocity and exactly when "falling" starts relative to when
    fingers first contact -- distinguishes "already falling before any
    finger could help" from "fingers actively push it away"
  - compares directly against a successful object (sphere) run through
    the SAME policy, so differences are attributable to the object, not
    the policy being bad in general.
"""
import argparse

import numpy as np
from stable_baselines3 import PPO

import closed_loop_sim as cls
from rl_env import SlipGraspEnv

CONTACT_THRESHOLD = 0.01


def run_detailed_episode(env, model, object_name, scene_seed, max_formation_steps=600):
    obs, info = env.reset(options={"object_name": object_name, "scene_seed": scene_seed})
    spawn_z = float(env.data.xpos[env.obj_body_id][2])
    first_contact_step = {i: None for i in range(5)}
    rows = []
    terminated = truncated = False
    step_count = 0
    while not (terminated or truncated) and step_count < max_formation_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        step_count += 1
        if step_info["phase"] >= 2:
            break
        touches = env._touch_values()
        for i, t in enumerate(touches):
            if t > CONTACT_THRESHOLD and first_contact_step[i] is None:
                first_contact_step[i] = env._global_step
        obj_pos = env.data.xpos[env.obj_body_id].copy()
        obj_vel = env.data.qvel[env._dof_adr:env._dof_adr + 3].copy()
        rows.append({
            "global_step": env._global_step,
            "fraction": list(env.fraction),
            "touches": touches,
            "obj_z": float(obj_pos[2]),
            "obj_speed": float(np.linalg.norm(obj_vel)),
            "fall": spawn_z - float(obj_pos[2]),
        })
    return rows, first_contact_step, step_info, spawn_z


def summarize(object_name, seed, rows, first_contact_step, final_info):
    dropped = final_info.get("dropped_during_formation", False)
    print(f"\n=== {object_name} seed={seed}  dropped={dropped}  "
          f"n_formation_steps_logged={len(rows)} ===")
    print(f"  first-contact step per finger: {first_contact_step}")
    contacted = [s for s in first_contact_step.values() if s is not None]
    if len(contacted) >= 2:
        print(f"  contact spread (last-first among fingers that ever touched): "
              f"{max(contacted) - min(contacted)} steps")
    # Find the step where fall first exceeds 1cm, 3cm, 5cm (DROP_THRESHOLD).
    for cm_threshold in (0.01, 0.03, 0.05):
        hit = next((r["global_step"] for r in rows if r["fall"] >= cm_threshold), None)
        print(f"  fall first reaches {cm_threshold*100:.0f}cm at step: {hit}")
    # Object speed trend: print every 5th row plus the last 5.
    n = len(rows)
    idxs = sorted(set(list(range(0, n, max(1, n // 10))) + list(range(max(0, n - 5), n))))
    for i in idxs:
        r = rows[i]
        frac_str = " ".join(f"{f:.2f}" for f in r["fraction"])
        touch_str = " ".join(f"{t:.2f}" for t in r["touches"])
        print(f"    step={r['global_step']:4d}  frac=[{frac_str}]  touch=[{touch_str}]  "
              f"obj_speed={r['obj_speed']:.4f}  fall={r['fall']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n-trials-per-object", type=int, default=2)
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    model = PPO.load(args.policy, device="cpu")
    all_objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    env = SlipGraspEnv(args, all_objects, episode_rng_seed=0)

    targets = [("cube", 0), ("cube", 1), ("rectangular_prism", 0), ("rectangular_prism", 1),
              ("flat_disk", 1), ("l_block", 0),
              ("sphere", 0), ("cone", 0)]  # last two: successful-object comparison
    for obj_name, seed in targets:
        rows, first_contact_step, final_info, spawn_z = run_detailed_episode(env, model, obj_name, seed)
        summarize(obj_name, seed, rows, first_contact_step, final_info)

    env.close()


if __name__ == "__main__":
    main()
