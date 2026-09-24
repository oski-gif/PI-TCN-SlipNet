"""
Closed-loop grasp-and-pull demo for the OrcaHand, built on the exact same
scene construction already validated in generate_trials.py (real PLA-density
masses, causally-meaningful dual friction jitter, orientation/geometry
jitter, the gravcomp fix that stops the object free-falling during the
approach). This is the infrastructure half of the project -- Claude-written,
per the project's model-vs-closed-loop split.

THIS FILE DELIBERATELY DOES NOT INCLUDE:
  - A trained PI-TCN/TCN making real predictions -- predict_slip() below is
    a stub that always returns "no slip". Wire your trained model in there.
  - Any control policy -- decide_action() below is a stub that always
    returns "no change". This is intentionally left for you to write.

What it DOES do: build one real trial scene, close the grip, hold, apply a
downward pull, and run an ongoing monitoring loop with a live MuJoCo viewer
so you can watch the whole thing happen, with the two integration points
above clearly marked and already wired into the loop -- so plugging in a
trained model or a real policy later is a matter of filling in those two
functions, not restructuring anything.

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
import time

import numpy as np
import mujoco
import mujoco.viewer

from generate_trials import (
    OBJECT_LIBRARY, OBJECT_BODY_NAME, build_trial_xml, _require_id,
)


# Timesteps of feature history handed to predict_slip() -- matches the kind
# of rolling window a real TCN would consume (it needs temporal context, not
# a single instant).
WINDOW_SIZE = 50


def predict_slip(feature_window):
    """
    STUB -- wire in your trained model here.

    feature_window: np.ndarray, shape (WINDOW_SIZE, 27) -- the most recent
    raw per-timestep features, in the SAME column order generate_trials.py
    records: for each fingertip in order, [touch, normal, tangential,
    torque_mag], then [obj_x, obj_y, obj_z, displacement, slip_speed,
    pull_force, phase]. (displacement is always 0.0 here -- there's no
    latched settle-end reference in a live/ongoing loop the way there is in
    a fixed-length trial, so don't rely on that column as-is.)

    Must return: 0 (no slip) or 1 (slip). Currently always returns 0.
    """
    return 0


def decide_action(predicted_class, driven_targets):
    """
    STUB -- intentionally left empty. Write your control policy here.

    predicted_class: whatever predict_slip() just returned.
    driven_targets: dict {actuator_id: [lo, current_target]} for the driven
    (flexion) actuators, so a policy can compute a new target relative to
    the current one (e.g. squeeze harder) rather than from scratch.

    Must return: dict {actuator_id: new_target} for any actuators to change
    this step. Return {} (the default) to change nothing.
    """
    return {}


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
        driven[aid] = [float(lo), float(lo) + args.squeeze_intensity * float(hi - lo)]

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

            for aid, (lo, target) in driven.items():
                data.ctrl[aid] = lo + frac * (target - lo)

            data.xfrc_applied[obj_body_id, :3] = [0.0, 0.0, -pull]

            mujoco.mj_step(model, data)

            feats = read_features(phase, pull)
            feature_history.append(feats)
            if len(feature_history) > WINDOW_SIZE:
                feature_history.pop(0)

            if phase == 2 and len(feature_history) == WINDOW_SIZE and step % 25 == 0:
                window = np.stack(feature_history)
                predicted = predict_slip(window)
                actions = decide_action(predicted, driven)
                for aid, new_target in actions.items():
                    driven[aid][1] = new_target
                print(f"  step {step}  predicted_slip={predicted}  "
                      f"obj_z={data.xpos[obj_body_id][2]:.4f}  pull={pull:.3f}N")

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
