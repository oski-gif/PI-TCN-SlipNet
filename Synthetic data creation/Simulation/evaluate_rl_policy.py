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

--vecnormalize (2026-11-01): if the policy was trained with VecNormalize
(train_rl_policy.py), its observations are normalized -- feeding it raw
ones here would silently hand it out-of-distribution inputs with no
error, just a wrong number at the end (the exact footgun this flag
exists to avoid). Only observation normalization is applied: this
harness never uses the reward value for anything (held_securely() reads
info, not reward), so reward normalization is irrelevant here and is
intentionally not touched. normalize_obs() is called directly on the
loaded VecNormalize object (it does not update its own statistics, by
its own docstring) rather than routing the real env through a VecEnv --
that would require forcing reset(options=...) through DummyVecEnv's
vectorized reset API, which does not support it.

Auto-detected if omitted (2026-11-02, independent code review): forgetting
--vecnormalize on a policy that needs it previously ran to completion
with no error and printed a confidently wrong number -- exactly the
"headline number" script where that's worst. Now reuses
train_rl_policy.py's find_sidecar() to look for a matching sidecar next
to --policy before falling back to none; a sidecar that exists nearby
but doesn't match --policy specifically still hard-errors (same
semantics as the training-side guard), rather than silently evaluating
unnormalized.

Usage:
    python evaluate_rl_policy.py \\
        --mjcf ... --fingertips ... --actuators ... --mesh-dir ... \\
        --checkpoint ../../src/training/model_a.pt --in-channels 22 \\
        --policy ../../Data/ppo_slip_policy_latest.zip
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import closed_loop_sim as cls
from rl_env import SlipGraspEnv
from train_rl_policy import HOLDOUT_OBJECTS, find_sidecar


def run_episode(env, model, object_name, scene_seed, vecnorm=None):
    obs, info = env.reset(options={"object_name": object_name, "scene_seed": scene_seed})
    terminated = truncated = False
    step_info = {}
    while not (terminated or truncated):
        policy_obs = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
        action, _ = model.predict(policy_obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
    return step_info


def held_securely(info):
    return not info.get("slipped", False) and not info.get("dropped_during_formation", False)


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--policy", required=True, help="Path to a trained PPO .zip")
    ap.add_argument("--vecnormalize", default=None,
                    help="Path to a saved VecNormalize stats .pkl. Auto-detected "
                         "next to --policy if omitted (same lookup used by "
                         "train_rl_policy.py's --resume-from) -- pass this "
                         "explicitly only to override that detection.")
    ap.add_argument("--n-trials-per-object", type=int, default=3)
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    model = PPO.load(args.policy, device="cpu")
    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    env = SlipGraspEnv(args, objects, episode_rng_seed=0)

    vecnorm_path = args.vecnormalize
    if vecnorm_path is None:
        vecnorm_path = find_sidecar(args.policy, "vecnormalize", "pkl")
        if vecnorm_path:
            print(f"Auto-detected VecNormalize sidecar: {vecnorm_path}")

    vecnorm = None
    if vecnorm_path:
        vecnorm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: env]))
        vecnorm.training = False
        print(f"Loaded VecNormalize stats from {vecnorm_path} -- "
              f"observations will be normalized to match training.")
    else:
        print("No VecNormalize sidecar found or given -- evaluating with raw "
              "observations (correct for policies trained without VecNormalize).")

    results = []
    for obj_name in objects:
        for seed in range(args.n_trials_per_object):
            try:
                info = run_episode(env, model, obj_name, seed, vecnorm=vecnorm)
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
