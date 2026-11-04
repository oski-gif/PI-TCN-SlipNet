"""
Trains a PPO policy to replace decide_action()'s hand-tuned threshold rule,
using SlipGraspEnv (rl_env.py). Claude-written infrastructure, same split
as the rest of the closed-loop code. See rl_env.py's module docstring for
the environment design (action/observation space, reward, scope).

Train/holdout object split, so the eval afterward can honestly report
whether the learned policy generalizes to objects it never trained on,
not just memorized the training set -- same discipline as the slip-
detection model's own train/val/test split.

VecNormalize (2026-11-01, results.md "Why are we plateaued at 64.4%"):
wraps the training SubprocVecEnv to normalize observations and rewards
using running mean/variance (a standard SB3 wrapper, opt-in, which PPO does
not apply on its own and this project had never used -- confirmed absent
via grep before adding it). Reward terms here span a >1000x raw magnitude range
(EFFORT_COEF=0.02 to ALPHA_DISPLACEMENT=50.0) with nothing previously
absorbing that scale; this changes reward/observation scale
rather than reward/observation CONTENT, unlike the
three experiments that already regressed performance this week. The
running stats are saved alongside every CheckpointCallback save
(save_vecnormalize=True) and reloaded on --resume-from via find_sidecar()
below, so a segmented run (crash-and-resume, same as every other run
this week) keeps one continuous set of statistics rather than restarting
them each segment -- getting this wrong would silently corrupt the run.

Reward-config guard (2026-11-02, results.md, independent code review):
the SAME silent-corruption risk exists for every plain module-global
reward constant in rl_env.py (MIN_FINGER_FRACTION_COEF already bit us
once this way -- left at 3.0 while resuming a checkpoint trained at
0.0). Generalized rather than special-cased: rl_env.get_reward_config()
snapshots every tunable constant, RewardConfigCallback saves it
alongside every checkpoint (same cadence as save_vecnormalize), and
--resume-from hard-errors on any mismatch between the checkpoint's
recorded config and rl_env.py's CURRENT constants, before any training
resumes. Both this and the VecNormalize sidecar are located via
find_sidecar() -- by checking actual file existence against both
naming shapes this project produces, not by re-deriving SB3's internal
checkpoint-naming scheme via string substitution (that approach missed
the final-save naming shape entirely: resuming from the FINAL saved
policy, e.g. ppo_slip_policy_vecnorm.zip -- exactly the workflow already
used to continue-train the 64.4% checkpoint -- silently fell through to
training on raw, unnormalized observations with no error at all; fixed
by checking existence of both shapes instead of assuming one). Both
checks run BEFORE SubprocVecEnv is constructed, so a missing/mismatched
sidecar aborts immediately instead of after a full worker-spawn +
MuJoCo-compile cycle.

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
import glob
import json
import multiprocessing as mp
import os
import re
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import closed_loop_sim as cls
from rl_env import SlipGraspEnv, get_reward_config

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


def find_sidecar(resume_path, kind, extension):
    """
    Finds a sidecar file of the given `kind` (e.g. "vecnormalize",
    "rewardconfig") matching a --resume-from checkpoint, by checking
    actual file existence rather than re-deriving SB3's internal
    checkpoint-naming scheme via string substitution (fragile if
    name_prefix or that scheme ever changes -- bit us once already,
    results.md). Handles both naming shapes this project produces:
      - CheckpointCallback's periodic saves: "<prefix>_<N>_steps.zip"
        paired with "<prefix>_<kind>_<N>_steps.<extension>"
      - This script's own final save: "<save_path>.zip" paired with
        "<save_path>_<kind>.<extension>"
    Returns the sidecar path if found. Returns None if NO sidecar of
    this kind exists anywhere in that directory -- a legacy checkpoint
    saved before this kind of sidecar existed, not an error. Raises
    FileNotFoundError if sidecars of this kind DO exist in that
    directory but none match this specific checkpoint -- a real
    mismatch, since silently proceeding without the matching one would
    silently lose whatever guarantee that sidecar kind provides, with
    no visible symptom.
    """
    resume_dir = os.path.dirname(resume_path)
    resume_stem = os.path.splitext(os.path.basename(resume_path))[0]

    candidate_final = os.path.join(resume_dir, f"{resume_stem}_{kind}.{extension}")
    if os.path.exists(candidate_final):
        return candidate_final

    # Scoped to THIS checkpoint's own lineage prefix, not the whole
    # directory -- every experiment's FINAL save lands directly in the
    # same flat Data/ folder (unlike periodic checkpoints, which each
    # get their own rl_logs_*/ subdirectory), so an unscoped glob here
    # false-positives on a completely unrelated experiment's sidecar
    # (hit this for real, 2026-11-02: seed3's lookup found seed1's
    # ppo_slip_policy_vecnorm_vecnormalize.pkl and wrongly concluded
    # seed3 was missing its own). For "<prefix>_<N>_steps", the lineage
    # is <prefix> (e.g. "ppo_slip_ckpt", shared across experiments but
    # already dir-isolated); for a final save, the lineage is the
    # resume_stem itself (e.g. "ppo_slip_policy_seed3"), which IS
    # unique per experiment even sharing one flat directory.
    steps_match = re.match(r"^(.*)_(\d+)_steps$", resume_stem)
    lineage_prefix = steps_match.group(1) if steps_match else resume_stem

    if steps_match:
        n = steps_match.group(2)
        matches = glob.glob(os.path.join(resume_dir, f"{lineage_prefix}*{kind}*_{n}_steps.{extension}"))
        if matches:
            return matches[0]

    any_nearby = glob.glob(os.path.join(resume_dir, f"{lineage_prefix}*{kind}*.{extension}"))
    if any_nearby:
        raise FileNotFoundError(
            f"--resume-from {resume_path} has no matching '{kind}' sidecar for "
            f"this specific checkpoint (checked {candidate_final!r} and a "
            f"{lineage_prefix}*_<N>_steps match), but this SAME LINEAGE "
            f"({lineage_prefix}) DOES have other '{kind}' files "
            f"({any_nearby[:3]}) -- this experiment records '{kind}', so "
            f"resuming without the exact matching one would silently lose "
            f"that guarantee mid-run. Stopping instead.")
    return None


class RewardConfigCallback(BaseCallback):
    """
    Saves rl_env.get_reward_config()'s snapshot alongside every
    CheckpointCallback save (same cadence, same N in the filename) so
    --resume-from can hard-error if a reward constant (e.g.
    MIN_FINGER_FRACTION_COEF) has changed since the checkpoint was
    trained, instead of silently continuing training under a different
    reward function than the checkpoint actually learned under (caught
    the hard way once already -- results.md).
    """
    def __init__(self, save_freq, save_path, name_prefix, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_rewardconfig_{self.num_timesteps}_steps.json")
            with open(path, "w") as f:
                json.dump(get_reward_config(), f, indent=2)
        return True


def main():
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=1500)
    ap.add_argument("--total-timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=256,
                    help="Rollout length per env, fresh runs only (a --resume-from "
                         "checkpoint keeps its own). 256 is the baseline; a held "
                         "trial lasts up to 460 env steps.")
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
    ap.add_argument("--use-vecnormalize", action="store_true",
                    help="Wrap training in SB3's VecNormalize (fresh runs only -- "
                         "a --resume-from checkpoint's own sidecar presence decides "
                         "for resumed runs, see find_sidecar()). Opt-in, default "
                         "off, so every orchestrator script that predates or isn't "
                         "deliberately testing VecNormalize (seed1, seed2/seedcheck, "
                         "seed3, production) keeps training on plain, unnormalized "
                         "observations -- the ORIGINAL baseline config these seeds "
                         "are being compared against. 2026-11-02: this flag was "
                         "added after catching, before launching seed3, that an "
                         "earlier version of this script made VecNormalize "
                         "unconditional for every fresh run, which would have "
                         "silently broken seed3's apples-to-apples comparison.")
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    all_objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    train_objects = [o for o in all_objects if o not in HOLDOUT_OBJECTS]
    print(f"Train objects ({len(train_objects)}): {train_objects}")
    print(f"Holdout objects ({len(HOLDOUT_OBJECTS)}): {HOLDOUT_OBJECTS}")

    os.makedirs(args.log_dir, exist_ok=True)

    # Resolve both sidecars (and hard-error on any mismatch) BEFORE paying
    # for SubprocVecEnv's worker-spawn + MuJoCo-compile cost -- a
    # missing/mismatched sidecar should abort immediately, not after a
    # full worker-startup cycle.
    vecnorm_sidecar = None
    if args.resume_from:
        vecnorm_sidecar = find_sidecar(args.resume_from, "vecnormalize", "pkl")

        rewardconfig_sidecar = find_sidecar(args.resume_from, "rewardconfig", "json")
        if rewardconfig_sidecar:
            with open(rewardconfig_sidecar) as f:
                recorded_config = json.load(f)
            current_config = get_reward_config()
            mismatches = {k: {"recorded": recorded_config[k], "current": current_config.get(k)}
                          for k in recorded_config
                          if recorded_config[k] != current_config.get(k)}
            if mismatches:
                raise ValueError(
                    f"--resume-from {args.resume_from}'s recorded reward config "
                    f"({rewardconfig_sidecar}) does not match rl_env.py's CURRENT "
                    f"constants -- continuing would silently train this checkpoint "
                    f"under a different reward function than it actually learned "
                    f"under. Mismatches: {mismatches}")
            print(f"Reward config matches recorded {rewardconfig_sidecar} -- OK.")

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
    raw_train_env = SubprocVecEnv(
        [make_env(args, train_objects, episode_seed=i, build_lock=build_lock,
                  worker_idx=i)
         for i in range(args.n_envs)])

    if args.resume_from:
        # vecnorm_sidecar was already resolved (and any mismatch already
        # hard-errored) above, before SubprocVecEnv was constructed.
        if vecnorm_sidecar:
            train_env = VecNormalize.load(vecnorm_sidecar, raw_train_env)
            train_env.training = True
            print(f"Resumed VecNormalize stats from {vecnorm_sidecar}.")
        else:
            train_env = raw_train_env

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
        if args.use_vecnormalize:
            # norm_obs + norm_reward both on, gamma=0.99 matching PPO's own
            # (left at SB3's default, confirmed via audit never overridden) --
            # see module docstring. Opt-in via --use-vecnormalize
            # (see its help) -- NOT the default, so every other experiment
            # keeps training on plain, unnormalized observations.
            train_env = VecNormalize(raw_train_env, norm_obs=True, norm_reward=True,
                                      clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        else:
            train_env = raw_train_env
        model = PPO(
            "MlpPolicy", train_env, verbose=1,
            tensorboard_log=args.log_dir,
            device="cpu",  # small MLP on a 105-dim obs -- CPU avoids GPU contention with the 6-8 subprocess TCN inferences
            n_steps=args.n_steps, batch_size=256, learning_rate=3e-4,
        )
        remaining = args.total_timesteps

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq, 1), save_path=args.log_dir,
        name_prefix="ppo_slip_ckpt", save_vecnormalize=True)
    rewardconfig_cb = RewardConfigCallback(
        save_freq=max(args.checkpoint_freq, 1), save_path=args.log_dir,
        name_prefix="ppo_slip_ckpt")

    # reset_num_timesteps=False on resume -- keeps counting up from the
    # loaded checkpoint's num_timesteps instead of restarting at 0, so
    # checkpoint filenames (which embed the model's absolute step count)
    # and the --total-timesteps budget both stay correct across segments.
    model.learn(total_timesteps=remaining, callback=[checkpoint_cb, rewardconfig_cb],
               reset_num_timesteps=(args.resume_from is None))
    model.save(args.save_path)
    print(f"Saved final policy to {args.save_path}.zip ({model.num_timesteps} total steps)")
    if isinstance(train_env, VecNormalize):
        final_vecnorm_path = f"{args.save_path}_vecnormalize.pkl"
        train_env.save(final_vecnorm_path)
        print(f"Saved final VecNormalize stats to {final_vecnorm_path}")
    final_rewardconfig_path = f"{args.save_path}_rewardconfig.json"
    with open(final_rewardconfig_path, "w") as f:
        json.dump(get_reward_config(), f, indent=2)
    print(f"Saved final reward config to {final_rewardconfig_path}")


if __name__ == "__main__":
    main()
