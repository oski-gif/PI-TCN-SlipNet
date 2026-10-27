"""
BOUNDED SMOKE TEST. Confirms SlipGraspEnv (rl_env.py) runs end-to-end with
no crashes and produces sane obs/reward/info, before committing to a real
training run. Not a diagnostic of the trained policy -- there is no
trained policy yet, actions here are random.
"""
import time

import numpy as np

import closed_loop_sim as cls
from rl_env import SlipGraspEnv


def main():
    import argparse
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    env = SlipGraspEnv(args, objects, episode_rng_seed=0)

    print(f"observation_space={env.observation_space}  action_space={env.action_space}")

    for ep in range(3):
        t0 = time.time()
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs)), "non-finite values in observation!"
        total_reward = 0.0
        n_steps = 0
        rng = np.random.default_rng(ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
            obs, reward, terminated, truncated, step_info = env.step(action)
            assert np.all(np.isfinite(obs)), "non-finite values mid-episode!"
            assert np.isfinite(reward), "non-finite reward!"
            total_reward += reward
            n_steps += 1
        dt = time.time() - t0
        print(f"[ep {ep}] object={info['object']:20s} steps={n_steps:4d} "
              f"total_reward={total_reward:8.2f}  slipped={step_info['slipped']}  "
              f"max_disp={step_info['max_displacement']:.4f}  "
              f"threshold={step_info['threshold']:.4f}  wall_time={dt:.1f}s")

    env.close()
    print("\nSmoke test passed: env resets, steps, and closes with no crashes.")


if __name__ == "__main__":
    main()
