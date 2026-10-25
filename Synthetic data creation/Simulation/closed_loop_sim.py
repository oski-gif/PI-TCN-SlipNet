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

What it DOES do: build one real trial scene, close the grip, hold, apply a
downward pull, and run an ongoing monitoring loop with a live MuJoCo viewer
so you can watch the whole thing happen, with predict_slip() now wired to a
real model and decide_action() still marked as the one integration point
left for you to write.

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

# Set once in main() after the model/scene are built, read by predict_slip().
# Keeping these as module globals rather than changing predict_slip()'s
# documented (feature_window) -> prediction signature, since that contract
# is what the file's docstring already promises callers.
_model = None
_device = None
_keep_idx = None
# None for a 22-channel model (Model A). A list for a 23- or 25-channel
# model (Model B variants), matching PITCNDataset's exact column order:
# [mu_combined] for 23ch, [mu_combined, obj_mass, grasp_force_at_settle]
# for 25ch. grasp_force_at_settle isn't known until the settle phase ends
# (see the capture point in main()'s loop below), so for the 25ch case
# this list starts with only 2 entries and gets its 3rd appended there --
# predict_slip() is never called before that capture happens in practice,
# since WINDOW_SIZE (300) steps always take longer to accumulate than
# ramp_steps + settle_steps does by default.
_extra_scalars = None


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
    returning 0.0 if no --checkpoint was given (same stub behavior as
    before, so this file still runs with no model wired in).
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

STEP_FRACTION = 0.15  # tune this -- how much of the remaining room to close per correction

def decide_action(slip_probability, driven_targets):
    if slip_probability <= 0.5:
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
    scene_path = os.path.join(out_dir, "closed_loop_scene.xml")
    jitter_rng = random.Random(seed)
    actual = build_trial_xml(
        mjcf_path, scene_path, obj_spec, mesh_dir, jitter_rng,
        object_pos, fingertip_names, pad_friction_range,
        orientation_tilt_deg_max,
    )
    return scene_path, obj_spec, actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True,
                    help="Sensor-instrumented scene (output of instrument_scene.py)")
    ap.add_argument("--fingertips", nargs="+", required=True)
    ap.add_argument("--actuators", nargs="+", required=True,
                    help="FLEXION actuator names only, same convention as generate_trials.py")
    ap.add_argument("--mesh-dir", default="assets")
    ap.add_argument("--object", default="cube",
                    choices=[o["name"] for o in OBJECT_LIBRARY])
    ap.add_argument("--object-pos", nargs=3, type=float,
                    default=[0.085, 0.04, 0.245], metavar=("X", "Y", "Z"))
    ap.add_argument("--pad-friction-range", nargs=2, type=float,
                    default=[0.35, 1.2], metavar=("MIN", "MAX"))
    ap.add_argument("--orientation-tilt-deg-max", type=float, default=10.0)
    ap.add_argument("--squeeze-intensity", type=float, default=0.8,
                    help="Fixed grip strength (0-1 of actuator ctrlrange). "
                         "This demo isn't randomizing trials -- it's one "
                         "scene to watch, so this is a single fixed value.")
    ap.add_argument("--pull-multiple", type=float, default=1.5,
                    help="Downward pull applied during monitoring, as a "
                         "multiple of the object's own weight.")
    ap.add_argument("--ramp-steps", type=int, default=500)
    ap.add_argument("--settle-steps", type=int, default=300)
    ap.add_argument("--pull-ramp-steps", type=int, default=500,
                    help="Steps over which the pull ramps in smoothly once "
                         "monitoring starts, instead of hitting as an impulse.")
    ap.add_argument("--normal-axis", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="Data/closed_loop_scenes")
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
    ap.add_argument("--checkpoint", default=None,
                    help="Path to a trained model_a.pt/model_b.pt. If omitted, "
                         "predict_slip() stays a stub that always returns 0, "
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
    args = ap.parse_args()

    feature_names = []
    for name in args.fingertips:
        feature_names += [f"{name}_touch", f"{name}_normal",
                          f"{name}_tangential", f"{name}_torque_mag"]
    feature_names += ["obj_x", "obj_y", "obj_z",
                      "displacement", "slip_speed", "pull_force", "phase"]

    if args.training_dataset:
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

    scene_path, obj_spec, actual = build_scene(
        args.mjcf, args.mesh_dir, args.fingertips, args.object,
        args.object_pos, tuple(args.pad_friction_range),
        args.orientation_tilt_deg_max, args.seed, args.out_dir,
    )
    print(f"Built scene for object='{args.object}': {scene_path}")
    print(f"  friction={actual['friction']}  pos={actual['pos']}")

    global _model, _device, _keep_idx, _extra_scalars
    _keep_idx = [i for i, n in enumerate(feature_names) if n not in LEAKAGE_FEATURES]

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

    if args.checkpoint:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = TCN(in_channels=args.in_channels, hidden_channels=32,
                     kernel_size=3, num_layers=7).to(_device)
        _model.load_state_dict(torch.load(args.checkpoint, map_location=_device))
        _model.eval()
        if args.in_channels in (23, 25):
            # Same definition as PITCNDataset: max(object_friction_sliding,
            # mean of this trial's per-fingertip pad frictions) -- computed
            # from this scene's OWN actual jittered values, not hardcoded.
            pad_frictions = list(actual["pad_friction"].values())
            pad_friction_mean = float(np.mean(pad_frictions)) if pad_frictions else 0.0
            friction_sliding = float(actual["friction"].split()[0])
            mu_combined = max(friction_sliding, pad_friction_mean)
            if args.in_channels == 23:
                _extra_scalars = [mu_combined]
                print(f"Loaded {args.checkpoint} (PI-TCN, 23ch) on {_device} -- mu_combined={mu_combined:.5f}")
            else:
                # grasp_force_at_settle isn't known yet -- this list starts
                # with 2 entries and gets its 3rd appended at the settle-end
                # capture point below, before predict_slip() is ever called
                # (see the comment on _extra_scalars' declaration).
                _extra_scalars = [mu_combined, obj_mass]
                print(f"Loaded {args.checkpoint} (PI-TCN, 25ch) on {_device} -- "
                      f"mu_combined={mu_combined:.5f}  obj_mass={obj_mass:.5f}  "
                      f"(grasp_force_at_settle captured at end of settle phase)")
        else:
            print(f"Loaded {args.checkpoint} (TCN, 22ch) on {_device}")
    else:
        print("No --checkpoint given -- predict_slip() stays a stub (always returns 0.0).")

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
        # displacement (column index -4 below) has no meaning here -- see
        # predict_slip()'s docstring.
        row += [obj_pos[0], obj_pos[1], obj_pos[2], 0.0,
               float(np.linalg.norm(obj_linvel)), pull, float(phase)]
        return np.array(row, dtype=np.float64)

    feature_history = []

    viewer_handle = None
    if not args.no_viewer:
        viewer_handle = mujoco.viewer.launch_passive(model, data)

    print("Phase: ramp (closing grip)...")
    step = 0
    ramp_steps, settle_steps = args.ramp_steps, args.settle_steps
    monitor_start = ramp_steps + settle_steps
    announced_monitor = False

    try:
        while True:
            if step == ramp_steps:
                model.body_gravcomp[obj_body_id] = 0.0
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
                    print("Phase: monitoring (pull engaged)...")
                    announced_monitor = True
                pull_frac = min((step - monitor_start + 1) / max(args.pull_ramp_steps, 1), 1.0)
                pull = pull_frac * pull_force_max

            for aid, (lo, hi, target) in driven.items():
                data.ctrl[aid] = lo + frac * (target - lo)

            data.xfrc_applied[obj_body_id, :3] = [0.0, 0.0, -pull]

            mujoco.mj_step(model, data)

            # Same capture point and definition as generate_trials.py: sum of
            # RAW normal-force sensor readings across all fingertips, at the
            # very last step of settle, before any pull is applied. No noise
            # to strip here -- unlike generate_trials.py, this file never
            # adds synthetic sensor noise, so these readings are already the
            # "raw" value. Only runs for the 25-channel case (_extra_scalars
            # has exactly 2 entries -- mu_combined, obj_mass -- until this
            # fires and appends the 3rd).
            if step == monitor_start - 1 and _extra_scalars is not None and len(_extra_scalars) == 2:
                grasp_force_at_settle = sum(
                    float(sensor_read(force_ids[name])[args.normal_axis])
                    for name in args.fingertips
                )
                _extra_scalars.append(grasp_force_at_settle)
                print(f"  captured grasp_force_at_settle={grasp_force_at_settle:.4f}N")

            feats = read_features(phase, pull)
            feature_history.append(feats)
            if len(feature_history) > WINDOW_SIZE:
                feature_history.pop(0)

            if phase == 2 and len(feature_history) == WINDOW_SIZE and step % 25 == 0:
                window = np.stack(feature_history)
                predicted = predict_slip(window)
                actions = decide_action(predicted, driven)
                for aid, new_target in actions.items():
                    driven[aid][2] = new_target
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
        print("\nStopped by user.")
    finally:
        if viewer_handle is not None:
            viewer_handle.close()


if __name__ == "__main__":
    main()
