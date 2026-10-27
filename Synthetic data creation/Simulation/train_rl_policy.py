"""
Trains a PPO policy to replace decide_action()'s hand-tuned threshold rule,
using SlipGraspEnv (rl_env.py). Claude-written infrastructure, same split
as the rest of the closed-loop code. See rl_env.py's module docstring for
the environment design (action/observation space, reward, scope).

Train/holdout object split, so the eval afterward can honestly report
whether the learned policy generalizes to objects it never trained on,
not just memorized the training set -- same discipline as the slip-
detection model's own train/val/test split.

Usage (Windows needs the spawn-safe __main__ guard below for
SubprocVecEnv -- this file must be run as a script, not imported):
    python train_rl_policy.py \\
        --mjcf ../../third_party/orcahand_description/v2/scene_slip.xml \\
        --fingertips ... --actuators ... --mesh-dir ../../assets \\
        --out-dir ../../Data/closed_loop_scenes \\
        --checkpoint ../../src/training/model_a.pt --in-channels 22 \\
        --total-timesteps 300000 --n-envs 6
"""
import argparse
import multiprocessing as mp
import os

# Must happen before torch is imported ANYWHERE in this process (including
# transitively, via rl_env/closed_loop_sim below) -- torch reads this once
# at CUDA-runtime init, so setting it inside a subprocess's entrypoint
# function is too late (spawned children re-import every module, which
# imports torch, before that function body ever runs). The TCN is tiny and
# PPO's own policy here is CPU-trained anyway, so there's no GPU work worth
# the multi-process CUDA-context contention that caused the first training
# run to fail with a host MemoryError inside MuJoCo's own model loading.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import closed_loop_sim as cls
from rl_env import SlipGraspEnv

HOLDOUT_OBJECTS = ["cone", "cylindrical_prism", "plus_cross", "sphere"]


def make_env(args, object_names, episode_seed, build_lock):
    def _init():
        env = SlipGraspEnv(args, object_names, episode_rng_seed=episode_seed,
                           build_lock=build_lock)
        return Monitor(env)
    return _init


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--total-timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--save-path", default="ppo_slip_policy")
    ap.add_argument("--log-dir", default="rl_logs")
    ap.add_argument("--eval-freq", type=int, default=10_000,
                    help="Timesteps between holdout-object evaluations (per env).")
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    all_objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    train_objects = [o for o in all_objects if o not in HOLDOUT_OBJECTS]
    print(f"Train objects ({len(train_objects)}): {train_objects}")
    print(f"Holdout objects ({len(HOLDOUT_OBJECTS)}): {HOLDOUT_OBJECTS}")

    os.makedirs(args.log_dir, exist_ok=True)

    # Shared across every worker process (spawn context, matching the one
    # SubprocVecEnv uses on Windows) so only one of them ever compiles a
    # MuJoCo model at a time -- see rl_env.py's SlipGraspEnv docstring on
    # build_lock for why a fixed startup stagger wasn't reliable here.
    build_lock = mp.get_context("spawn").Lock()

    eval_env = DummyVecEnv(
        [make_env(args, HOLDOUT_OBJECTS, episode_seed=10_000, build_lock=build_lock)])

    train_env = SubprocVecEnv(
        [make_env(args, train_objects, episode_seed=i, build_lock=build_lock)
         for i in range(args.n_envs)])

    model = PPO(
        "MlpPolicy", train_env, verbose=1,
        tensorboard_log=args.log_dir,
        device="cpu",  # small MLP on a 105-dim obs -- CPU avoids GPU contention with the 6-8 subprocess TCN inferences
        n_steps=256, batch_size=256, learning_rate=3e-4,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.eval_freq, 5000), save_path=args.log_dir,
        name_prefix="ppo_slip_ckpt")
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=args.log_dir,
        log_path=args.log_dir, eval_freq=args.eval_freq,
        n_eval_episodes=10, deterministic=True)

    model.learn(total_timesteps=args.total_timesteps,
               callback=[checkpoint_cb, eval_cb])
    model.save(args.save_path)
    print(f"Saved final policy to {args.save_path}.zip")


if __name__ == "__main__":
    main()
