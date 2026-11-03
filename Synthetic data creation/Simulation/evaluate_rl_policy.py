"""
Evaluates a trained PPO policy (from train_rl_policy.py) over 15 objects x
N scene seeds, one object/seed at a time via SlipGraspEnv's
reset(options={"object_name", "scene_seed"}) override -- single process, no
SubprocVecEnv. Uses the policy mean by default; --stochastic samples actions
as training does.

Not directly comparable with batch_closed_loop_trials.py unless horizons are
matched: here --max-steps counts AFTER settle (1500 = 2300 total steps), there
it is the TOTAL including ramp and settle (use --max-steps 2300 to match). A
hand-tuned run also continues after a formation drop and only records it, so
score it as not slipped and not dropped to match this script. See results.md,
"Baseline comparability check".

--out-json is resumable: rows already written are kept and skipped, and a row
whose config differs is a hard error. After 3 consecutive failed trials the
process exits with status 3, because a broken MuJoCo process keeps failing, so
a fresh process can pick up where it stopped. Failed trials are reported, never
silently left out of the denominator.

"Held securely" here means NEITHER slipped during phase 2 (displacement
exceeded the object's own threshold) NOR dropped during formation (phase
0/1) -- both are real failure modes in this env and info["slipped"]
alone only captures the first one; checking it in isolation would
silently count a formation-phase drop as a success.

--continue-past-formation-drop (2026-11-03): by default a trial ends as soon as
the object has fallen 5 cm, so its failure says nothing about whether the grip
caught it. With this flag the trial runs to its normal end and records
formation_max_fall, so one run can be scored at any cutoff. "held" stays the
5 cm rule, so it should match a default run; the summary also prints the other
cutoffs. Evaluation only. Policies were never trained past a 5 cm fall, so what
they do there is untrained behavior. Logged rewards and ep_len aren't comparable
with default runs: there is no drop penalty, and a dropped trial can earn the
hold bonus. Default runs omit the fall, since it stops growing at the crossing.
See results.md, "Hand-tuned at 2300 steps".

--vecnormalize (2026-11-01): if the policy was trained with VecNormalize
(train_rl_policy.py), its observations are normalized -- feeding it raw
ones here would silently hand it out-of-distribution inputs with no
error, just a wrong number at the end (the exact footgun this flag
exists to avoid). Only observations are normalized: held_securely() reads
info, not reward, and logged rewards are always raw. --eval-path direct calls
normalize_obs() on the loaded object (it does not update statistics), while
--eval-path wrapped steps through the same wrapper stack as training.

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
import json
import os
import sys
import zlib

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import closed_loop_sim as cls
import rl_env
from rl_env import SlipGraspEnv, get_reward_config
from train_rl_policy import HOLDOUT_OBJECTS, find_sidecar

MAX_CONSECUTIVE_FAILURES = 3
REPORT_FALL_CUTOFFS = (0.05, 0.10, 0.15)  # meters, for --continue-past-formation-drop


def seed_action_noise(object_name, scene_seed):
    # crc32, not hash(): hash() varies per process, and stochastic runs must
    # draw the same noise for the same trial on every checkpoint.
    torch.manual_seed(zlib.crc32(f"{object_name}:{scene_seed}".encode()))


def new_totals():
    return {"ep_reward": 0.0, "ep_len": 0, "formation_reward": 0.0, "phase2_reward": 0.0}


def tally_reward(totals, reward, info):
    totals["ep_reward"] += reward
    totals["ep_len"] += 1
    totals["phase2_reward" if info["phase"] == 2 else "formation_reward"] += reward


def run_episode(env, model, object_name, scene_seed, vecnorm=None, deterministic=True):
    seed_action_noise(object_name, scene_seed)
    obs, info = env.reset(options={"object_name": object_name, "scene_seed": scene_seed})
    terminated = truncated = False
    step_info = {}
    totals = new_totals()
    while not (terminated or truncated):
        policy_obs = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
        action, _ = model.predict(policy_obs, deterministic=deterministic)
        obs, reward, terminated, truncated, step_info = env.step(action)
        tally_reward(totals, float(reward), step_info)
    return step_info, totals


def make_training_style_venv(env, vecnorm_path):
    # Same wrapper order as train_rl_policy.py (Monitor -> VecEnv -> VecNormalize),
    # with statistics frozen and rewards left raw.
    venv = DummyVecEnv([lambda: Monitor(env)])
    if vecnorm_path:
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    return venv


def run_episode_wrapped(venv, model, object_name, scene_seed, deterministic=True):
    seed_action_noise(object_name, scene_seed)
    venv.set_options({"object_name": object_name, "scene_seed": scene_seed})
    obs = venv.reset()
    totals = new_totals()
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)
        tally_reward(totals, float(rewards[0]), infos[0])
        if dones[0]:
            return infos[0], totals


def held_securely(info):
    return not info.get("slipped", False) and not info.get("dropped_during_formation", False)


def held_at_cutoff(record, cutoff):
    return not record["slipped"] and record["formation_max_fall"] <= cutoff


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
    ap.add_argument("--seed-start", type=int, default=0,
                    help="First scene seed per object. Dev set: 1000 with 10 trials "
                         "per object. Locked set: 0 with 67 trials per object. See "
                         "results.md, 'Pre-registered evaluation rules'.")
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions as training does, instead of using the "
                         "policy mean. Noise is seeded per trial, so it repeats "
                         "across checkpoints.")
    ap.add_argument("--eval-path", choices=["direct", "wrapped"], default="direct",
                    help="direct: step the env and normalize observations by hand. "
                         "wrapped: step through the same wrapper stack as training "
                         "(Monitor, DummyVecEnv, VecNormalize with frozen stats).")
    ap.add_argument("--continue-past-formation-drop", action="store_true",
                    help="Run each trial to its normal end instead of stopping at a "
                         "5 cm formation fall, and print held counts at other cutoffs.")
    ap.add_argument("--out-json", default=None,
                    help="Write one JSON line per trial, including this run's config.")
    args = ap.parse_args()
    args.squeeze_intensity = 0.95
    deterministic = not args.stochastic

    model = PPO.load(args.policy, device="cpu")
    objects = [o["name"] for o in cls.OBJECT_LIBRARY]
    env = SlipGraspEnv(args, objects, episode_rng_seed=0,
                       terminate_on_formation_drop=not args.continue_past_formation_drop)

    vecnorm_path = args.vecnormalize
    if vecnorm_path is None:
        vecnorm_path = find_sidecar(args.policy, "vecnormalize", "pkl")
        if vecnorm_path:
            print(f"Auto-detected VecNormalize sidecar: {vecnorm_path}")

    vecnorm = None
    venv = None
    if args.eval_path == "wrapped":
        venv = make_training_style_venv(env, vecnorm_path)
    elif vecnorm_path:
        vecnorm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: env]))
        vecnorm.training = False
    if vecnorm_path:
        print(f"Loaded VecNormalize stats from {vecnorm_path} -- "
              f"observations will be normalized to match training.")
    else:
        print("No VecNormalize sidecar found or given -- evaluating with raw "
              "observations (correct for policies trained without VecNormalize).")

    seeds = range(args.seed_start, args.seed_start + args.n_trials_per_object)
    # Round-tripped through JSON so tuples compare equal to rows read back on resume.
    run_config = json.loads(json.dumps({
        "policy": args.policy, "policy_sha256": cls.file_sha256(args.policy),
        "vecnormalize": vecnorm_path, "vecnormalize_sha256": cls.file_sha256(vecnorm_path),
        "deterministic": deterministic, "eval_path": args.eval_path,
        "continue_past_formation_drop": args.continue_past_formation_drop,
        "checkpoint": args.checkpoint, "checkpoint_sha256": cls.file_sha256(args.checkpoint),
        "in_channels": args.in_channels,
        "code_sha256": [cls.file_sha256(cls.__file__), cls.file_sha256(rl_env.__file__),
                        cls.file_sha256(__file__)],
        "seed_start": args.seed_start, "n_trials_per_object": args.n_trials_per_object,
        "mjcf": args.mjcf, "mjcf_sha256": cls.file_sha256(args.mjcf), "mesh_dir": args.mesh_dir,
        "fingertips": args.fingertips, "actuators": args.actuators,
        "normal_axis": args.normal_axis,
        "max_steps_after_settle": args.max_steps, "ramp_steps": args.ramp_steps,
        "settle_steps": args.settle_steps, "pull_ramp_steps": args.pull_ramp_steps,
        "total_steps": args.ramp_steps + args.settle_steps + args.max_steps,
        "full_pull_steps": args.max_steps - args.pull_ramp_steps,
        "pull_multiple": args.pull_multiple, "squeeze_intensity": args.squeeze_intensity,
        "object_pos": args.object_pos, "pad_friction_range": args.pad_friction_range,
        "orientation_tilt_deg_max": args.orientation_tilt_deg_max,
        "reward_config": get_reward_config()}))
    print(f"Eval config: {run_config}, seeds {seeds.start}-{seeds.stop - 1}")

    completed = {}
    if args.out_json and os.path.exists(args.out_json):
        for row in cls.read_rows_for_resume(args.out_json, run_config):
            if "error" not in row:
                completed[(row["object"], row["seed"])] = row
        print(f"Resuming: {len(completed)} completed trials found in {args.out_json}")

    out_file = open(args.out_json, "a") if args.out_json else None
    results = []
    failed = []
    consecutive_failures = 0
    for obj_name in objects:
        for seed in seeds:
            if (obj_name, seed) in completed:
                results.append(completed[(obj_name, seed)])
                continue
            try:
                if venv is not None:
                    info, totals = run_episode_wrapped(venv, model, obj_name, seed,
                                                       deterministic=deterministic)
                else:
                    info, totals = run_episode(env, model, obj_name, seed, vecnorm=vecnorm,
                                               deterministic=deterministic)
            except Exception as e:
                print(f"{obj_name:20s} seed={seed}  FAILED: {type(e).__name__}: {e}", flush=True)
                failed.append((obj_name, seed))
                if out_file:
                    out_file.write(json.dumps({**run_config, "object": obj_name, "seed": seed,
                                               "error": f"{type(e).__name__}: {e}"}) + "\n")
                    out_file.flush()
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"Stopping after {consecutive_failures} consecutive failures -- "
                          f"rerun the same command to resume in a fresh process.", flush=True)
                    if out_file:
                        out_file.close()
                    sys.exit(3)
                continue
            consecutive_failures = 0
            record = {"object": obj_name, "seed": seed, "held": bool(held_securely(info)),
                      "slipped": bool(info["slipped"]),
                      "dropped_during_formation": bool(info["dropped_during_formation"]),
                      "max_displacement": float(info["max_displacement"]),
                      "threshold": float(info["threshold"]), **totals}
            fall_note = ""
            if args.continue_past_formation_drop:
                record["formation_max_fall"] = float(info["formation_max_fall"])
                fall_note = f"  formation_max_fall={record['formation_max_fall']:.4f}"
            results.append(record)
            print(f"{obj_name:20s} seed={seed}  held={record['held']}  slipped={record['slipped']}  "
                  f"dropped_during_formation={record['dropped_during_formation']}  "
                  f"max_disp={record['max_displacement']:.4f}  "
                  f"threshold={record['threshold']:.4f}  "
                  f"ep_len={record['ep_len']}  ep_reward={record['ep_reward']:.1f}{fall_note}",
                  flush=True)
            if out_file:
                out_file.write(json.dumps({**run_config, **record}) + "\n")
                out_file.flush()

    if out_file:
        out_file.close()
    env.close()

    n_ok = len(results)
    n_expected = len(objects) * len(seeds)
    n_held = sum(r["held"] for r in results)
    rate = 100 * n_held / n_ok if n_ok else 0.0
    print(f"\nOverall: {n_held}/{n_ok} held securely ({rate:.1f}%)")
    if args.continue_past_formation_drop and n_ok:
        for cutoff in REPORT_FALL_CUTOFFS:
            n_at_cutoff = sum(held_at_cutoff(r, cutoff) for r in results)
            print(f"Held with formation fall <= {100 * cutoff:.0f} cm: {n_at_cutoff}/{n_ok} "
                  f"({100 * n_at_cutoff / n_ok:.1f}%)")
    if n_ok != n_expected:
        print(f"WARNING: INCOMPLETE -- {n_ok}/{n_expected} trials completed, "
              f"failed this run: {failed}. Rerun to retry them before citing this rate.")

    train_objects = [o for o in objects if o not in HOLDOUT_OBJECTS]
    for label, obj_set in [("TRAIN objects", train_objects), ("HOLDOUT objects", HOLDOUT_OBJECTS)]:
        subset = [r for r in results if r["object"] in obj_set]
        n_sub = len(subset)
        n_sub_held = sum(r["held"] for r in subset)
        sub_rate = 100 * n_sub_held / n_sub if n_sub else 0.0
        print(f"{label}: {n_sub_held}/{n_sub} held securely ({sub_rate:.1f}%)")
        if subset:
            print(f"  mean ep_len={sum(r['ep_len'] for r in subset) / n_sub:.0f}  "
                  f"ep_reward={sum(r['ep_reward'] for r in subset) / n_sub:.1f}  "
                  f"formation_reward={sum(r['formation_reward'] for r in subset) / n_sub:.1f}  "
                  f"phase2_reward={sum(r['phase2_reward'] for r in subset) / n_sub:.1f}")

    print("\nPer-object breakdown:")
    for obj_name in objects:
        obj_results = [r for r in results if r["object"] == obj_name]
        if not obj_results:
            continue
        n_obj = len(obj_results)
        n_obj_held = sum(r["held"] for r in obj_results)
        print(f"  {obj_name:20s}  {n_obj_held}/{n_obj} held securely")

    if n_ok != n_expected:
        sys.exit(3)


if __name__ == "__main__":
    main()
