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

    # Resume a later segment (see --resume-from's help for why this is
    # needed -- a single process reliably fails after ~90-100 resets):
    python train_rl_policy.py \\
        --mjcf ... --fingertips ... --actuators ... --mesh-dir ... \\
        --checkpoint ../../src/training/model_a.pt --in-channels 22 \\
        --total-timesteps 300000 --n-envs 4 \\
        --resume-from ../../Data/rl_logs_production/ppo_slip_ckpt_80000_steps.zip
"""
import argparse
import faulthandler
import multiprocessing as mp
import os
import time

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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import closed_loop_sim as cls
from rl_env import SlipGraspEnv

HOLDOUT_OBJECTS = ["cone", "cylindrical_prism", "plus_cross", "sphere"]

# TEMP DIAGNOSTIC (2026-10-30) -- investigating the n-envs=6 run that died
# with BrokenPipeError/EOFError in the main process, with the actual dying
# worker's own root-cause exception never reaching the combined log. Two
# possibilities: (a) a hard native-level crash inside a compiled extension
# (MuJoCo's own C/C++ code) that no Python try/except can catch, or (b) a
# Windows spawn-mode stdout-redirection quirk hiding an ordinary Python
# exception. faulthandler.enable() catches (a) -- it installs a signal-level
# handler that dumps a traceback on segfault/access violation, which a plain
# except block cannot do. The print-and-flush checkpoints below narrow down
# WHICH stage a worker reached before dying even if neither faulthandler nor
# a normal exception fires at all. Not production code -- remove once this
# investigation concludes.
_DIAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")


def make_env(args, object_names, episode_seed, build_lock, worker_idx):
    def _init():
        diag_path = os.path.join(_DIAG_DIR, f"_worker_diag_{worker_idx}.log")
        diag_fh = open(diag_path, "a", buffering=1)  # line-buffered, not block-buffered
        faulthandler.enable(file=diag_fh)
        diag_fh.write(f"{time.strftime('%H:%M:%S')} [worker {worker_idx}] "
                      f"process started, faulthandler enabled\n")

        env = SlipGraspEnv(args, object_names, episode_rng_seed=episode_seed,
                           build_lock=build_lock, worker_idx=worker_idx)
        diag_fh.write(f"{time.strftime('%H:%M:%S')} [worker {worker_idx}] "
                      f"SlipGraspEnv constructed, wrapping in Monitor\n")
        return Monitor(env)
    return _init


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--total-timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--save-path", default="ppo_slip_policy")
    ap.add_argument("--log-dir", default="rl_logs")
    ap.add_argument("--checkpoint-freq", type=int, default=10_000,
                    help="CALLS (not timesteps -- SB3's CheckpointCallback "
                         "counts per-rollout-step calls, so actual timesteps "
                         "between saves is roughly this x n-envs) between "
                         "CheckpointCallback saves. Renamed from the old "
                         "--eval-freq 2026-10-31 when EvalCallback was removed "
                         "(see results.md) -- this now only controls "
                         "checkpoint frequency, not evaluation.")
    ap.add_argument("--resume-from", default=None,
                    help="Path to an existing PPO checkpoint (.zip) to resume from, "
                         "instead of starting a fresh policy -- needed because a "
                         "single long-lived worker process reliably fails after "
                         "~90-100 MjModel compiles (see results.md's "
                         "'Root-cause investigation' entry, 2026-10-30), so a full "
                         "300k-step run must be done as several shorter segments, "
                         "each a fresh process resuming from the previous segment's "
                         "last checkpoint. --total-timesteps is still the ABSOLUTE "
                         "target (e.g. 300000), not an additional amount on top of "
                         "the checkpoint -- remaining steps are computed from the "
                         "checkpoint's own restored num_timesteps, not parsed from "
                         "the filename, so this works even if a checkpoint is "
                         "renamed.")
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

    # No eval_env/EvalCallback here anymore -- removed 2026-10-30 after
    # it was implicated in two separate SIGSEGV crashes (results.md,
    # "Resumed-training segment 1" entries), both times stuck exactly at
    # "lock acquired, compiling model" on its very first in-process
    # reset. EvalCallback's DummyVecEnv runs IN this main process (not a
    # SubprocVecEnv subprocess), so a crash there takes down the whole
    # training run with no recovery -- unlike a dead SubprocVecEnv
    # worker, which at least surfaces as a catchable (if uninformative)
    # BrokenPipeError. It was only ever here for progress monitoring;
    # the real evaluation now happens after training, via
    # evaluate_rl_policy.py's full 45-trial closed-loop benchmark, which
    # is both more informative (actual held-securely rate, not reward)
    # and doesn't recompile a model repeatedly inside the training
    # process at all. CheckpointCallback (kept below) is the actual
    # progress safety net -- already proven reliable across every
    # incident so far, including both crashes above.
    train_env = SubprocVecEnv(
        [make_env(args, train_objects, episode_seed=i, build_lock=build_lock,
                  worker_idx=i)
         for i in range(args.n_envs)])

    if args.resume_from:
        # PPO.load() restores num_timesteps (and every original
        # hyperparameter -- n_steps/batch_size/learning_rate/etc) from the
        # checkpoint's own saved state, verified directly before wiring
        # this in (loading ppo_slip_ckpt_80000_steps.zip standalone showed
        # num_timesteps=80000, n_steps=256, batch_size=256, lr=0.0003 --
        # exactly what was saved) -- so this does NOT need to parse the
        # checkpoint filename or track progress in a separate sidecar file.
        model = PPO.load(args.resume_from, env=train_env, device="cpu")
        already_done = model.num_timesteps
        remaining = args.total_timesteps - already_done
        print(f"Resumed from {args.resume_from}: {already_done} steps already "
              f"completed, {remaining} remaining out of {args.total_timesteps} target.")
        if remaining <= 0:
            print("Already at or past --total-timesteps -- nothing to do.")
            return
    else:
        model = PPO(
            "MlpPolicy", train_env, verbose=1,
            tensorboard_log=args.log_dir,
            device="cpu",  # small MLP on a 105-dim obs -- CPU avoids GPU contention with the 6-8 subprocess TCN inferences
            n_steps=256, batch_size=256, learning_rate=3e-4,
        )
        remaining = args.total_timesteps

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq, 1), save_path=args.log_dir,
        name_prefix="ppo_slip_ckpt")

    # reset_num_timesteps=False on resume -- keeps counting up from the
    # loaded checkpoint's num_timesteps instead of restarting at 0, so
    # checkpoint filenames (which embed the model's absolute step count)
    # and the --total-timesteps budget both stay correct across segments.
    model.learn(total_timesteps=remaining, callback=[checkpoint_cb],
               reset_num_timesteps=(args.resume_from is None))
    model.save(args.save_path)
    print(f"Saved final policy to {args.save_path}.zip ({model.num_timesteps} total steps)")


if __name__ == "__main__":
    main()
