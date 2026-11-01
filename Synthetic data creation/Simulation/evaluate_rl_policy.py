"""
Evaluates a trained PPO policy (from train_rl_policy.py) against the SAME
45-trial benchmark (15 objects x 3 seeds each) used for every hand-tuned
decide_action() closed-loop comparison in results.md -- for a direct,
apples-to-apples closed-loop success-rate number against the existing
40.0% baseline. Runs the policy DETERMINISTICALLY
(model.predict(obs, deterministic=True)), one object/seed at a time via
SlipGraspEnv's reset(options={"object_name", "scene_seed"}) override
(added alongside this script specifically for this purpose) -- single
process, no SubprocVecEnv/parallelism needed for evaluation.

"Held securely" here means NEITHER slipped during phase 2 (displacement
exceeded the object's own threshold) NOR dropped during formation (phase
0/1) -- both are real failure modes in this env and info["slipped"]
alone only captures the first one; checking it in isolation would
silently count a formation-phase drop as a success.

Usage:
    python evaluate_rl_policy.py \\
        --mjcf ... --fingertips ... --actuators ... --mesh-dir ... \\
        --checkpoint ../../src/training/model_a.pt --in-channels 22 \\
        --policy ../../Data/ppo_slip_policy_latest.zip
"""
import argparse

from stable_baselines3 import PPO

import closed_loop_sim as cls
from rl_env import SlipGraspEnv
from train_rl_policy import HOLDOUT_OBJECTS


def run_episode(env, model, object_name, scene_seed):
    obs, info = env.reset(options={"object_name": object_name, "scene_seed": scene_seed})
    terminated = truncated = False
    step_info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
    return step_info


def held_securely(info):
    return not info.get("slipped", False) and not info.get("dropped_during_formation", False)


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--policy", required=True, help="Path to a trained PPO .zip")
    ap.add_argument("--n-trials-per-object", type=int, default=3)
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    model = PPO.load(args.policy, device="cpu")
    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    env = SlipGraspEnv(args, objects, episode_rng_seed=0)

    results = []
    for obj_name in objects:
        for seed in range(args.n_trials_per_object):
            try:
                info = run_episode(env, model, obj_name, seed)
            except Exception as e:
                print(f"{obj_name:20s} seed={seed}  FAILED: {type(e).__name__}: {e}")
                continue
            ok = held_securely(info)
            results.append({"object": obj_name, "seed": seed, "held": ok, **info})
            print(f"{obj_name:20s} seed={seed}  held={ok}  slipped={info.get('slipped')}  "
                  f"dropped_during_formation={info.get('dropped_during_formation')}  "
                  f"max_disp={info.get('max_displacement', float('nan')):.4f}  "
                  f"threshold={info.get('threshold', float('nan')):.4f}")

    env.close()

    n_ok = len(results)
    n_held = sum(r["held"] for r in results)
    rate = 100 * n_held / n_ok if n_ok else 0.0
    print(f"\nOverall: {n_held}/{n_ok} held securely ({rate:.1f}%)")

    train_objects = [o for o in objects if o not in HOLDOUT_OBJECTS]
    for label, obj_set in [("TRAIN objects", train_objects), ("HOLDOUT objects", HOLDOUT_OBJECTS)]:
        subset = [r for r in results if r["object"] in obj_set]
        n_sub = len(subset)
        n_sub_held = sum(r["held"] for r in subset)
        sub_rate = 100 * n_sub_held / n_sub if n_sub else 0.0
        print(f"{label}: {n_sub_held}/{n_sub} held securely ({sub_rate:.1f}%)")

    print("\nPer-object breakdown:")
    for obj_name in objects:
        obj_results = [r for r in results if r["object"] == obj_name]
        if not obj_results:
            continue
        n_obj = len(obj_results)
        n_obj_held = sum(r["held"] for r in obj_results)
        print(f"  {obj_name:20s}  {n_obj_held}/{n_obj} held securely")


if __name__ == "__main__":
    main()
