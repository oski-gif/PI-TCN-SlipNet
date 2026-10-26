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

# Set by load_model(), read by predict_slip(). _extra_scalars is also reset
# per-trial inside run_trial() (friction/mass/grasp-force are scene-specific,
# unlike the model weights, which only need loading once per batch).
_model = None
_device = None
_keep_idx = None
_extra_scalars = None  # None for 22ch. List for 23/25ch, see run_trial().


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


def load_model(checkpoint, in_channels):
    """Loads the trained model ONCE -- call this before run_trial(), not
    inside it, so a batch of many trials doesn't reload the checkpoint from
    disk every time."""
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = TCN(in_channels=in_channels, hidden_channels=32,
                kernel_size=3, num_layers=7).to(_device)
    _model.load_state_dict(torch.load(checkpoint, map_location=_device))
    _model.eval()
    print(f"Loaded {checkpoint} ({in_channels}ch) on {_device}")


def predict_slip(feature_window):
    """
    feature_window: np.ndarray, shape (WINDOW_SIZE, 27) -- the most recent
    raw per-timestep features, in the SAME column order generate_trials.py
    records: for each fingertip in order, [touch, normal, tangential,
    torque_mag], then [obj_x, obj_y, obj_z, displacement, slip_speed,
    pull_force, phase]. (displacement is always 0.0 here -- there's no
    latched settle-end reference in a live/ongoing loop the way there is in
    a fixed-length trial, so don't rely on that column as-is -- it's
    excluded from the model's input anyway, see LEAKAGE_FEATURES.)

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

    x = feature_window[:, _keep_idx]
    if _extra_scalars is not None:
        extra_cols = np.tile(np.array(_extra_scalars, dtype=x.dtype), (x.shape[0], 1))
        x = np.concatenate([x, extra_cols], axis=1)

    x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(_device)
    with torch.no_grad():
        outputs = _model(x_tensor)
        probs = torch.softmax(outputs, dim=1)
    return float(probs[0, 1, -1].item())


STEP_FRACTION = 0.35  # tune this -- how much of the remaining room to close per correction

def decide_action(slip_probability, driven_targets):
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


def run_trial(args, object_name, seed, max_steps, use_viewer=False, verbose=True):
    """Runs ONE full trial (ramp -> settle -> monitor-with-pull) and returns
    a result dict. Assumes load_model() was already called if you want real
    predictions -- otherwise predict_slip() stays a 0.0 stub and
    decide_action() never fires.

    Returns dict with: object, seed, slipped (bool, displacement exceeded
    the SAME per-object threshold definition generate_trials.py uses),
    max_displacement, threshold, max_slip_probability, final_slip_probability,
    n_corrections (how many times decide_action() actually adjusted the grip).
    """
    global _keep_idx, _extra_scalars

    feature_names = build_feature_names(args.fingertips)
    _keep_idx = [i for i, n in enumerate(feature_names) if n not in LEAKAGE_FEATURES]

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
            _extra_scalars = [mu_combined]
        else:
            # grasp_force_at_settle isn't known yet -- appended at the
            # settle-end capture point below, before predict_slip() is
            # ever called (WINDOW_SIZE steps always take longer to
            # accumulate than ramp+settle does by default).
            _extra_scalars = [mu_combined, obj_mass]
    else:
        _extra_scalars = None

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

            if step == monitor_start - 1:
                # Reference position for displacement, latched at the end of
                # settle -- same timing/reasoning as generate_trials.py's
                # FIX 7 (don't score normal grasp acquisition as slip).
                ref_pos = data.xpos[obj_body_id].copy()
                if _extra_scalars is not None and len(_extra_scalars) == 2:
                    grasp_force_at_settle = sum(
                        float(sensor_read(force_ids[name])[args.normal_axis])
                        for name in args.fingertips
                    )
                    _extra_scalars.append(grasp_force_at_settle)
                    if verbose:
                        print(f"  captured grasp_force_at_settle={grasp_force_at_settle:.4f}N")

            if ref_pos is not None:
                disp = float(np.linalg.norm(data.xpos[obj_body_id] - ref_pos))
                max_displacement = max(max_displacement, disp)

            feats = read_features(phase, pull)
            feature_history.append(feats)
            if len(feature_history) > WINDOW_SIZE:
                feature_history.pop(0)

            if phase == 2 and len(feature_history) == WINDOW_SIZE and step % 5 == 0:
                window = np.stack(feature_history)
                predicted = predict_slip(window)
                last_slip_probability = predicted
                max_slip_probability = max(max_slip_probability, predicted)
                actions = decide_action(predicted, driven)
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
