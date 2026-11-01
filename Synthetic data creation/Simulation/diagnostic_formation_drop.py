"""
BOUNDED DIAGNOSTIC. Investigates WHY the trained RL policy
(Data/ppo_slip_policy_latest.zip, 300224 steps) drops 4 specific objects
during FORMATION (phase 0/1, before the pull phase even starts) --
cube (3/3 drops), rectangular_prism (3/3), flat_disk (2/3), l_block (2/3)
-- per results.md's "THE real RL closed-loop number" entry (2026-10-31).
This is a different failure mode than the already-confirmed phase-2
actuator saturation (that diagnostic logged ctrl-vs-ctrlrange during
monitoring; this one logs per-finger closing fraction + contact during
FORMATION instead) and different from the mesh-object detection-miss
problem (which RL already fixed).

Runs the SlipGraspEnv directly (same deterministic policy.predict() loop
as evaluate_rl_policy.py), but reads the env's own public per-step state
(self.fraction, per-finger touch, object height) directly after each
step() call instead of needing a callback hook -- the eval loop IS the
instrumentation here, no changes to rl_env.py needed.
"""
import argparse

import numpy as np
from stable_baselines3 import PPO

import closed_loop_sim as cls
from rl_env import SlipGraspEnv

DROP_OBJECTS = ["cube", "rectangular_prism", "flat_disk", "l_block"]


def run_instrumented_episode(env, model, object_name, scene_seed):
    obs, info = env.reset(options={"object_name": object_name, "scene_seed": scene_seed})
    trace = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        obj_z = float(env.data.xpos[env.obj_body_id][2])
        touches = env._touch_values()
        n_touching = sum(1 for t in touches if t > 0.01)
        trace.append({
            "global_step": env._global_step,
            "phase": step_info["phase"],
            "fraction": list(env.fraction),
            "touches": touches,
            "n_touching": n_touching,
            "obj_z": obj_z,
            "dropped": step_info.get("dropped_during_formation", False),
        })
    return trace, step_info


def summarize(object_name, seed, trace, final_info):
    dropped = final_info.get("dropped_during_formation", False)
    print(f"\n=== {object_name} seed={seed}  dropped_during_formation={dropped} "
          f"(trace length={len(trace)} macro-steps) ===")
    if not trace:
        return
    spawn_z = trace[0]["obj_z"]
    # Sample every ~10th step during phase<2 (formation), plus the last
    # 5 steps before the episode ended, to see the lead-up without
    # printing hundreds of lines.
    formation = [t for t in trace if t["phase"] < 2]
    if not formation:
        print("  (no formation-phase steps recorded -- unexpected)")
        return
    sample_idxs = sorted(set(list(range(0, len(formation), max(1, len(formation) // 8)))
                             + list(range(max(0, len(formation) - 5), len(formation)))))
    for i in sample_idxs:
        t = formation[i]
        frac_str = " ".join(f"{f:.2f}" for f in t["fraction"])
        touch_str = " ".join(f"{v:.3f}" for v in t["touches"])
        fall = spawn_z - t["obj_z"]
        print(f"  step={t['global_step']:5d}  fractions=[{frac_str}]  "
              f"n_touching={t['n_touching']}/5  touches=[{touch_str}]  "
              f"fall_from_spawn={fall:.4f}")
    max_n_touching_ever = max(t["n_touching"] for t in formation)
    final_fractions = formation[-1]["fraction"]
    print(f"  max n_touching ever reached during formation: {max_n_touching_ever}/5")
    print(f"  final per-finger fractions before drop/phase-2: "
          f"{[f'{f:.2f}' for f in final_fractions]}  "
          f"(spread={max(final_fractions) - min(final_fractions):.2f})")


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n-trials-per-object", type=int, default=3)
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    model = PPO.load(args.policy, device="cpu")
    all_objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    env = SlipGraspEnv(args, all_objects, episode_rng_seed=0)

    for obj_name in DROP_OBJECTS:
        for seed in range(args.n_trials_per_object):
            trace, final_info = run_instrumented_episode(env, model, obj_name, seed)
            summarize(obj_name, seed, trace, final_info)

    env.close()


if __name__ == "__main__":
    main()
