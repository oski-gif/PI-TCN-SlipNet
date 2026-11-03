"""
Closed-loop grasp-and-pull demo for the OrcaHand, built on the exact same
scene construction already validated in generate_trials.py (real PLA-density
masses, causally-meaningful dual friction jitter, orientation/geometry
jitter, the gravcomp fix that stops the object free-falling during the
approach). This is the infrastructure half of the project -- Claude-written,
per the project's model-vs-closed-loop split.

THIS FILE STILL DOES NOT INCLUDE:
  - Any control policy -- decide_action() below is a stub that always
    returns "no change". This is intentionally left for you to write.

predict_slip() now loads and runs a real trained model when --checkpoint is
given (model_a.pt with --in-channels 22, or model_b.pt with --in-channels
23 or 25 depending on which checkpoint -- mu_combined/obj_mass/
grasp_force_at_settle all get computed automatically from this scene's own
actual values, never passed by hand). It returns a PROBABILITY in [0, 1]
now, not a hard 0/1 class. With no --checkpoint, it stays the original stub
that always returns 0.0, so this file still runs standalone.

The per-trial simulation logic lives in run_trial() (model loading is
separate, in load_model(), so a batch of many trials only loads the
checkpoint once). main() is now a thin wrapper that calls load_model() once
and run_trial() once, with a live viewer -- unchanged single-trial behavior
from before this refactor. batch_closed_loop_trials.py calls run_trial()
directly, many times, headless, to get real closed-loop statistics across
trials instead of watching one trial at a time.

Usage:
    python closed_loop_sim.py \\
        --mjcf third_party/orcahand_description/v2/scene_slip.xml \\
        --fingertips right_T-DP_b7429e50 right_I-FingerTipAssembly_ec49c16c \\
                     right_M-FingerTipAssembly_34afb748 right_M-FingerTipAssembly_424a8e75 \\
                     right_P-FingerTipAssembly_cd219176 \\
        --actuators right_t-mcp_actuator right_t-pip_actuator \\
                    right_i-mcp_actuator right_i-pip_actuator \\
                    right_m-mcp_actuator right_m-pip_actuator \\
                    right_r-mcp_actuator right_r-pip_actuator \\
                    right_p-mcp_actuator right_p-pip_actuator \\
        --mesh-dir assets \\
        --object cube
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "models"))
from tcn import TCN, LEAKAGE_FEATURES

from generate_trials import (
    OBJECT_LIBRARY, OBJECT_BODY_NAME, build_trial_xml, _require_id,
)


# Timesteps of feature history handed to predict_slip(). Must be >= the
# trained TCN's actual receptive field, not an arbitrary "feels like enough"
# number -- the original 50 here predated the real architecture. For
# hidden_channels=32, kernel_size=3, num_layers=7 (dilations 1,2,4,...,64),
# the causal receptive field is 1 + sum((kernel_size-1)*dilation) for all
# layers = 1 + 2*(1+2+4+8+16+32+64) = 255. 300 gives a safety margin.
WINDOW_SIZE = 300
FORMATION_DROP_THRESHOLD = 0.05  # meters fallen from spawn height in ramp/settle counted as a drop


def file_sha256(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows_for_resume(path, run_config):
    """Rows from a resumable results file. Raises if any row was written with a
    different config, before anything on disk is changed. A crash mid-write can
    leave a partial last line, which is dropped so the next append cannot glue
    onto it."""
    with open(path) as f:
        lines = f.readlines()
    truncated = bool(lines) and not lines[-1].endswith("\n")
    rows =[json.loads(line) for line in (lines[:-1] if truncated else lines)]
    for row in rows:
        mismatch = {k: (row.get(k), v) for k, v in run_config.items() if row.get(k) != v}
        if mismatch:
            raise ValueError(f"{path} was written with a different config, "
                             f"refusing to overwrite or mix results: {mismatch}")
    if truncated:
        print(f"Dropping truncated final row from {path}")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            f.writelines(lines[:-1])
        os.replace(tmp_path, path)
    return rows


# Set by load_model() -- the model itself is process-wide/load-once-per-batch
# state, which is fine (each SlipGraspEnv/run_trial() call in a given process
# shares the same loaded weights on purpose). keep_idx/extra_scalars used to
# live here too as module globals, but those are PER-TRIAL/PER-EPISODE state
# (friction/mass/grasp-force differ scene to scene) -- a module global for
# that silently corrupts results the moment two trials/envs are ever live in
# the same process at once (e.g. a future DummyVecEnv with n_envs>1). Now
# threaded through predict_slip() as explicit parameters instead; callers
# (run_trial() below, rl_env.py's SlipGraspEnv) own their own copies.
_model = None
_device = None


def build_feature_names(fingertips):
    """The SAME raw 27-column layout generate_trials.py records -- single
    source of truth, used by both run_trial() (to compute _keep_idx) and
    main()'s --training-dataset column-order check, so the two can never
    silently drift out of sync with each other."""
    names = []
    for name in fingertips:
        names += [f"{name}_touch", f"{name}_normal",
                 f"{name}_tangential", f"{name}_torque_mag"]
    names += ["obj_x", "obj_y", "obj_z",
             "displacement", "slip_speed", "pull_force", "phase"]
    return names


def load_model(checkpoint, in_channels, device=None):
    """Loads the trained model ONCE -- call this before run_trial(), not
    inside it, so a batch of many trials doesn't reload the checkpoint from
    disk every time.

    device: force "cpu" or "cuda" explicitly. Leave as None for the normal
    auto-detect behavior (single-process scripts) -- but RL training passes
    "cpu" explicitly, because torch.cuda.is_available() was found to return
    True even with CUDA_VISIBLE_DEVICES="" and torch.cuda.device_count()==0
    (a real torch inconsistency, not a typo) -- auto-detect would then try
    to move the model onto a CUDA device that doesn't actually exist, which
    sometimes silently no-ops and sometimes hangs the whole process, which
    is what caused every unexplained multi-minute hang tonight."""
    global _model, _device
    if device is not None:
        _device = torch.device(device)
    else:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = TCN(in_channels=in_channels, hidden_channels=32,
                kernel_size=3, num_layers=7)
    # Always deserialize onto CPU first, then move -- map_location=_device
    # (a torch.device object) does NOT reliably remap a GPU-saved checkpoint
    # when CUDA is unavailable in THIS process (hit this directly: RL training
    # subprocesses run with CUDA_VISIBLE_DEVICES="" to avoid multi-process
    # CUDA-context contention, and map_location=torch.device("cpu") still
    # raised "Attempting to deserialize object on CUDA device 0"). The
    # string "cpu" is the documented, reliable universal-remap form.
    _model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    _model = _model.to(_device)
    _model.eval()
    print(f"Loaded {checkpoint} ({in_channels}ch) on {_device}")


def predict_slip(feature_window, keep_idx, extra_scalars=None):
    """
    feature_window: np.ndarray, shape (WINDOW_SIZE, 27) -- the most recent
    raw per-timestep features, in the SAME column order generate_trials.py
    records: for each fingertip in order, [touch, normal, tangential,
    torque_mag], then [obj_x, obj_y, obj_z, displacement, slip_speed,
    pull_force, phase]. (displacement is always 0.0 here -- there's no
    latched settle-end reference in a live/ongoing loop the way there is in
    a fixed-length trial, so don't rely on that column as-is -- it's
    excluded from the model's input anyway, see LEAKAGE_FEATURES.)

    keep_idx: column indices to keep after dropping LEAKAGE_FEATURES, computed
    once by the caller from build_feature_names() (e.g. run_trial()'s local
    `keep_idx`, or SlipGraspEnv's self._keep_idx) -- NOT a module global,
    since it's per-trial/per-episode state, not process-wide state.

    extra_scalars: optional list of per-trial physics scalars (mu_combined,
    obj_mass, grasp_force_at_settle) for Model B's 23/25-channel variants,
    same per-trial-state reasoning as keep_idx. None for Model A (22ch).

    Returns: float in [0, 1] -- the model's predicted PROBABILITY of slip
    (softmax of class 1, "slip") for the MOST RECENT timestep in the window
    (the last row), which is "now" in a streaming/live context. This is a
    readout change only -- the model itself is unchanged, CrossEntropyLoss
    already trains it to produce a probability distribution internally;
    this just stops throwing that away via argmax. Falls back to always
    returning 0.0 if no model has been loaded (call load_model() first),
    so this file still runs standalone with no model wired in.
    """
    if _model is None:
        return 0.0

    x = feature_window[:, keep_idx]
    if extra_scalars is not None:
        extra_cols = np.tile(np.array(extra_scalars, dtype=x.dtype), (x.shape[0], 1))
        x = np.concatenate([x, extra_cols], axis=1)

    x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(_device)
    with torch.no_grad():
        outputs = _model(x_tensor)
        probs = torch.softmax(outputs, dim=1)
    return float(probs[0, 1, -1].item())


STEP_FRACTION = 0.35  # tune this -- how much of the remaining room to close per correction

def decide_action(slip_probability, driven_targets, feats=None):
    """feats: optional raw feature row (same 27-column layout as
    build_feature_names()/read_features()) for the step this decision is
    being made on -- added so a per-finger-localized policy variant can see
    each fingertip's own tangential/normal force without changing this
    function's job for the still-global policy below, which ignores it.
    Keyword-only in practice (always called as feats=... at the one call
    site in run_trial()) so existing monkeypatched diagnostic policies with
    the old 2-arg signature keep working unchanged."""
    if slip_probability <= 0.3:
        return {}
    actions = {}
    for aid, (lo, hi, target) in driven_targets.items():
        step = STEP_FRACTION * slip_probability * (hi - target)
        actions[aid] = target + step
    return actions


def build_scene(mjcf_path, mesh_dir, fingertip_names, object_name,
                object_pos, pad_friction_range, orientation_tilt_deg_max,
                seed, out_dir):
    obj_spec = next(o for o in OBJECT_LIBRARY if o["name"] == object_name)
    os.makedirs(out_dir, exist_ok=True)
    # Unique filename per (object, seed) so concurrent/sequential batch
    # trials never clobber each other's scene XML mid-run.
    scene_path = os.path.join(out_dir, f"closed_loop_scene_{object_name}_{seed}.xml")
    jitter_rng = random.Random(seed)
    actual = build_trial_xml(
        mjcf_path, scene_path, obj_spec, mesh_dir, jitter_rng,
        object_pos, fingertip_names, pad_friction_range,
        orientation_tilt_deg_max,
    )
    return scene_path, obj_spec, actual


def run_trial(args, object_name, seed, max_steps, use_viewer=False, verbose=True,
             step_callback=None):
    """Runs ONE full trial (ramp -> settle -> monitor-with-pull) and returns
    a result dict. Assumes load_model() was already called if you want real
    predictions -- otherwise predict_slip() stays a 0.0 stub and
    decide_action() never fires.

    step_callback: optional callable(step, phase, driven, data, disp), called
    every simulation step right after mj_step -- added for
    diagnostic_actuator_saturation.py's investigation (2026-10-30) into
    whether failed trials' displacement "runaway" corresponds to driven
    actuators pinned at their ctrlrange max, rather than inferring it
    circumstantially from n_corrections/max_slip_probability alone. `driven`
    is the live dict of {aid: [lo, hi, target]} -- `data.ctrl[aid]` is the
    actually-applied value, `hi` is the ctrlrange max. Default None (no-op),
    zero behavior/performance change for every other caller.

    Returns dict with: object, seed, slipped (bool, displacement exceeded
    the SAME per-object threshold definition generate_trials.py uses),
    max_displacement, threshold, max_slip_probability, final_slip_probability,
    n_corrections (how many times decide_action() actually adjusted the grip),
    dropped_during_formation (bool, the same fall test SlipGraspEnv.step() uses
    in ramp/settle) and formation_max_fall (meters). Recording these does not
    stop the trial or change control or `slipped`: unlike the RL env, a dropped
    trial runs on, so score it as not slipped AND not dropped to match RL.
    """
    feature_names = build_feature_names(args.fingertips)
    keep_idx = [i for i, n in enumerate(feature_names) if n not in LEAKAGE_FEATURES]

    scene_path, obj_spec, actual = build_scene(
        args.mjcf, args.mesh_dir, args.fingertips, object_name,
        args.object_pos, tuple(args.pad_friction_range),
        args.orientation_tilt_deg_max, seed, args.out_dir,
    )
    if verbose:
        print(f"Built scene for object='{object_name}' seed={seed}: {scene_path}")
        print(f"  friction={actual['friction']}  pos={actual['pos']}")

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    touch_ids = {n: _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                f"{n}_touch", "Touch sensor")
                 for n in args.fingertips}
    force_ids = {n: _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                f"{n}_force", "Force sensor")
                 for n in args.fingertips}
    torque_ids = {n: _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                 f"{n}_torque", "Torque sensor")
                  for n in args.fingertips}
    obj_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY,
                              OBJECT_BODY_NAME, "Object body")
    free_joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{OBJECT_BODY_NAME}_freejoint", "Free joint")
    dof_adr = model.jnt_dofadr[free_joint_id]

    driven = {}
    for a in args.actuators:
        aid = _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a, "Actuator")
        lo, hi = model.actuator_ctrlrange[aid]
        driven[aid] = [float(lo), float(hi), float(lo) + args.squeeze_intensity * float(hi - lo)]

    for aid in range(model.nu):
        if aid in driven:
            continue
        lo, hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(0.0, lo, hi))

    # Same free-fall fix as generate_trials.py: weightless during approach,
    # real gravity from settle onward.
    model.body_gravcomp[obj_body_id] = 1.0

    obj_mass = float(model.body_mass[obj_body_id])
    g = float(np.linalg.norm(model.opt.gravity))
    pull_force_max = args.pull_multiple * obj_mass * g

    if _model is not None and args.in_channels in (23, 25):
        # Same definition as PITCNDataset: max(object_friction_sliding,
        # mean of this trial's per-fingertip pad frictions) -- computed
        # from this scene's OWN actual jittered values, not hardcoded.
        pad_frictions = list(actual["pad_friction"].values())
        pad_friction_mean = float(np.mean(pad_frictions)) if pad_frictions else 0.0
        friction_sliding = float(actual["friction"].split()[0])
        mu_combined = max(friction_sliding, pad_friction_mean)
        if args.in_channels == 23:
            extra_scalars = [mu_combined]
        else:
            # grasp_force_at_settle isn't known yet -- appended at the
            # settle-end capture point below, before predict_slip() is
            # ever called (WINDOW_SIZE steps always take longer to
            # accumulate than ramp+settle does by default).
            extra_scalars = [mu_combined, obj_mass]
    else:
        extra_scalars = None

    tangent_axes = [i for i in range(3) if i != args.normal_axis]

    def sensor_read(sid):
        adr = model.sensor_adr[sid]
        dim = model.sensor_dim[sid]
        return data.sensordata[adr:adr + dim].copy()

    def read_features(phase, pull):
        row = []
        for name in args.fingertips:
            touch = float(sensor_read(touch_ids[name])[0])
            force_vec = sensor_read(force_ids[name])
            torque_vec = sensor_read(torque_ids[name])
            normal = float(force_vec[args.normal_axis])
            tangential = float(np.linalg.norm(
                [force_vec[tangent_axes[0]], force_vec[tangent_axes[1]]]))
            row += [touch, normal, tangential, float(np.linalg.norm(torque_vec))]
        obj_pos = data.xpos[obj_body_id]
        obj_linvel = data.qvel[dof_adr:dof_adr + 3]
        row += [obj_pos[0], obj_pos[1], obj_pos[2], 0.0,
               float(np.linalg.norm(obj_linvel)), pull, float(phase)]
        return np.array(row, dtype=np.float64)

    feature_history = []
    viewer_handle = mujoco.viewer.launch_passive(model, data) if use_viewer else None

    # A separate MjData, so finding the spawn height leaves `data` untouched:
    # the RL env's mj_forward at reset gives this same height at qpos0.
    spawn_probe = mujoco.MjData(model)
    mujoco.mj_kinematics(model, spawn_probe)
    spawn_z = float(spawn_probe.xpos[obj_body_id][2])
    formation_max_fall = 0.0

    ref_pos = None
    max_displacement = 0.0
    max_slip_probability = 0.0
    last_slip_probability = 0.0
    n_corrections = 0

    if verbose:
        print("Phase: ramp (closing grip)...")
    step = 0
    ramp_steps, settle_steps = args.ramp_steps, args.settle_steps
    monitor_start = ramp_steps + settle_steps
    announced_monitor = False

    try:
        while step < max_steps:
            if step == ramp_steps:
                model.body_gravcomp[obj_body_id] = 0.0
                if verbose:
                    print("Phase: settle...")

            if step < ramp_steps:
                phase = 0
                frac = (step + 1) / max(ramp_steps, 1)
                pull = 0.0
            elif step < monitor_start:
                phase = 1
                frac = 1.0
                pull = 0.0
            else:
                phase = 2
                frac = 1.0
                if not announced_monitor:
                    if verbose:
                        print("Phase: monitoring (pull engaged)...")
                    announced_monitor = True
                pull_frac = min((step - monitor_start + 1) / max(args.pull_ramp_steps, 1), 1.0)
                pull = pull_frac * pull_force_max

            for aid, (lo, hi, target) in driven.items():
                data.ctrl[aid] = lo + frac * (target - lo)

            data.xfrc_applied[obj_body_id, :3] = [0.0, 0.0, -pull]

            mujoco.mj_step(model, data)

            if phase < 2:
                formation_max_fall = max(formation_max_fall,
                                         spawn_z - float(data.xpos[obj_body_id][2]))

            if step == monitor_start - 1:
                # Reference position for displacement, latched at the end of
                # settle -- same timing/reasoning as generate_trials.py's
                # FIX 7 (don't score normal grasp acquisition as slip).
                ref_pos = data.xpos[obj_body_id].copy()
                if extra_scalars is not None and len(extra_scalars) == 2:
                    grasp_force_at_settle = sum(
                        float(sensor_read(force_ids[name])[args.normal_axis])
                        for name in args.fingertips
                    )
                    extra_scalars.append(grasp_force_at_settle)
                    if verbose:
                        print(f"  captured grasp_force_at_settle={grasp_force_at_settle:.4f}N")

            if ref_pos is not None:
                disp = float(np.linalg.norm(data.xpos[obj_body_id] - ref_pos))
                max_displacement = max(max_displacement, disp)
            else:
                disp = 0.0

            if step_callback is not None:
                step_callback(step, phase, driven, data, disp)

            feats = read_features(phase, pull)
            feature_history.append(feats)
            if len(feature_history) > WINDOW_SIZE:
                feature_history.pop(0)

            if phase == 2 and len(feature_history) == WINDOW_SIZE and step % 5 == 0:
                window = np.stack(feature_history)
                predicted = predict_slip(window, keep_idx, extra_scalars)
                last_slip_probability = predicted
                max_slip_probability = max(max_slip_probability, predicted)
                actions = decide_action(predicted, driven, feats=feats)
                if actions:
                    n_corrections += 1
                for aid, new_target in actions.items():
                    driven[aid][2] = new_target
                if verbose:
                    sample_aid = next(iter(driven))
                    print(f"  step {step}  slip_probability={predicted:.4f}  "
                          f"obj_z={data.xpos[obj_body_id][2]:.4f}  pull={pull:.3f}N  "
                          f"n_actuators_adjusted={len(actions)}  "
                          f"sample_target={driven[sample_aid][2]:.4f}")

            if viewer_handle is not None:
                if not viewer_handle.is_running():
                    break
                viewer_handle.sync()
                time.sleep(max(0.0, dt - 0.0005))

            step += 1
    except KeyboardInterrupt:
        if verbose:
            print("\nStopped by user.")
    finally:
        if viewer_handle is not None:
            viewer_handle.close()
        if os.path.exists(scene_path):
            os.remove(scene_path)

    # Same threshold DEFINITION generate_trials.py uses for its own
    # trial_slip_label -- 15% of this object's own char_length, not one
    # flat meters value for every geometry.
    effective_threshold = 0.15 * obj_spec["char_length"]
    slipped = max_displacement > effective_threshold

    return {
        "object": object_name,
        "seed": seed,
        "slipped": slipped,
        "max_displacement": max_displacement,
        "threshold": effective_threshold,
        "max_slip_probability": max_slip_probability,
        "final_slip_probability": last_slip_probability,
        "n_corrections": n_corrections,
        "dropped_during_formation": bool(formation_max_fall > FORMATION_DROP_THRESHOLD),
        "formation_max_fall": formation_max_fall,
        "steps_completed": step,
    }


def add_common_args(ap, max_steps_default):
    """Scene-construction/model/control args shared identically between
    closed_loop_sim.py (one trial, live viewer) and batch_closed_loop_trials.py
    (many trials, headless) -- single source of truth, so the two entry
    points can't silently drift apart on what should be the same defaults
    (this already happened once: ramp/settle-steps had different, undocumented
    defaults in each file before this was factored out)."""
    ap.add_argument("--mjcf", required=True,
                    help="Sensor-instrumented scene (output of instrument_scene.py)")
    ap.add_argument("--fingertips", nargs="+", required=True)
    ap.add_argument("--actuators", nargs="+", required=True,
                    help="FLEXION actuator names only, same convention as generate_trials.py")
    ap.add_argument("--mesh-dir", default="assets")
    ap.add_argument("--object-pos", nargs=3, type=float,
                    default=[0.085, 0.04, 0.245], metavar=("X", "Y", "Z"))
    ap.add_argument("--pad-friction-range", nargs=2, type=float,
                    default=[0.35, 1.2], metavar=("MIN", "MAX"))
    ap.add_argument("--orientation-tilt-deg-max", type=float, default=10.0)
    ap.add_argument("--squeeze-intensity", type=float, default=0.8,
                    help="Fixed grip strength (0-1 of actuator ctrlrange).")
    ap.add_argument("--pull-multiple", type=float, default=1.5,
                    help="Downward pull applied during monitoring, as a "
                         "multiple of the object's own weight.")
    ap.add_argument("--ramp-steps", type=int, default=500)
    ap.add_argument("--settle-steps", type=int, default=300)
    ap.add_argument("--pull-ramp-steps", type=int, default=500,
                    help="Steps over which the pull ramps in smoothly once "
                         "monitoring starts, instead of hitting as an impulse.")
    ap.add_argument("--normal-axis", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--out-dir", default="Data/closed_loop_scenes")
    ap.add_argument("--max-steps", type=int, default=max_steps_default,
                    help="Hard step budget per trial.")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to a trained model_a.pt/model_b.pt. If omitted, "
                         "predict_slip() stays a stub that always returns 0.0, "
                         "same as before -- this file still runs with no "
                         "model wired in.")
    ap.add_argument("--in-channels", type=int, default=22, choices=[22, 23, 25],
                    help="22 for Model A (TCN, model_a.pt). 23 or 25 for "
                         "Model B (PI-TCN) variants -- 22 tactile/force "
                         "features plus mu_combined (23ch), or plus "
                         "mu_combined + obj_mass + grasp_force_at_settle "
                         "(25ch, the current model_b.pt). All extra scalars "
                         "are computed automatically from this scene's own "
                         "actual values, never passed by hand.")
    return ap


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, max_steps_default=5000)
    ap.add_argument("--object", default="cube",
                    choices=[o["name"] for o in OBJECT_LIBRARY])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-viewer", action="store_true",
                    help="Run headless (no live MuJoCo viewer window).")
    ap.add_argument("--training-dataset", default=None,
                    help="Path to the .npz produced by export_data.py that "
                         "your model was trained on. If given, this loop's "
                         "own feature column order is checked against that "
                         "file's saved feature_names before running. "
                         "IMPORTANT: --fingertips order here determines the "
                         "column order predict_slip() receives -- if it "
                         "doesn't match training order (e.g. two fingertips "
                         "swapped), sensor identities resolve fine but "
                         "predict_slip() silently gets, say, the middle "
                         "finger's data where it expects the index finger's, "
                         "with no error. This flag is how you catch that.")
    args = ap.parse_args()

    if args.training_dataset:
        feature_names = build_feature_names(args.fingertips)
        ref = np.load(args.training_dataset, allow_pickle=True)["feature_names"].tolist()
        if ref != feature_names:
            raise ValueError(
                "Feature column order mismatch between this closed-loop run "
                f"and {args.training_dataset}!\n"
                f"  this run:  {feature_names}\n"
                f"  training:  {ref}\n"
                "Your --fingertips order (or normal-axis, which changes "
                "what 'normal'/'tangential' mean) must exactly match how "
                "the training data was generated -- otherwise predict_slip() "
                "receives columns in the wrong order with no error.")
        print("Feature column order verified against training dataset.")

    if args.checkpoint:
        load_model(args.checkpoint, args.in_channels)
    else:
        print("No --checkpoint given -- predict_slip() stays a stub (always returns 0.0).")

    result = run_trial(args, args.object, args.seed, args.max_steps,
                       use_viewer=not args.no_viewer, verbose=True)
    print(f"\nResult: slipped={result['slipped']}  "
          f"max_displacement={result['max_displacement']:.4f}  "
          f"threshold={result['threshold']:.4f}  "
          f"max_slip_probability={result['max_slip_probability']:.4f}  "
          f"n_corrections={result['n_corrections']}")


if __name__ == "__main__":
    main()
