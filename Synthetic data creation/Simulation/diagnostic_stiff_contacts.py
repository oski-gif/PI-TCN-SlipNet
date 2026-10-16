"""
BOUNDED SIDE-EXPERIMENT, not part of the production pipeline.

Question: does today's finding (naive Coulomb-ratio/torque trend signals
don't discriminate real slip onset from stable grasping -- tested 8 ways,
all ~1.0-1.25x) hold up if contacts are made much stiffer (shorter solref
timeconst) than MuJoCo's default regularized contacts? If a stiffer-contact
batch shows a much stronger ratio, that's real evidence the simulator's
default contact softness is masking a genuine signal, worth a proper fix
in the March window. If it's still ~1x, that's evidence the issue isn't
primarily contact stiffness.

This does NOT modify generate_trials.py at all -- it reuses build_trial_xml
and run_single_trial completely unmodified, and only POST-PROCESSES the
scene XML build_trial_xml already produces, adding a solref attribute to
the same pad/object geoms that already get their friction set. Output goes
to its own directory, never touching Data/raw_trials_batch*.
"""
import argparse
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch

from generate_trials import (
    OBJECT_LIBRARY, OBJECT_BODY_NAME, build_trial_xml, run_single_trial,
    build_object_assignments, _find_body,
)

DEFAULT_SOLREF = "0.002 1"  # ~10x stiffer timeconst than MuJoCo's default (0.02, 1)


def stiffen_contacts(xml_path, fingertip_names, solref):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    n_patched = 0
    for tip_name in fingertip_names:
        body = _find_body(root, tip_name)
        if body is None:
            continue
        for geom in body.findall("geom"):
            if geom.get("contype") == "0":
                continue
            geom.set("solref", solref)
            n_patched += 1
    for geom in root.iter("geom"):
        if geom.get("name") == f"{OBJECT_BODY_NAME}_geom":
            geom.set("solref", solref)
            n_patched += 1
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    return n_patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--fingertips", nargs="+", required=True)
    ap.add_argument("--actuators", nargs="+", required=True)
    ap.add_argument("--mesh-dir", default="assets")
    ap.add_argument("--object-pos", nargs=3, type=float, default=[0.085, 0.04, 0.245])
    ap.add_argument("--pad-friction-range", nargs=2, type=float, default=[0.35, 1.2])
    ap.add_argument("--orientation-tilt-deg-max", type=float, default=10.0)
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--normal-axis", type=int, default=2)
    ap.add_argument("--sensor-noise-std", type=float, default=0.05)
    ap.add_argument("--settle-steps", type=int, default=300)
    ap.add_argument("--hold-steps", type=int, default=1000)
    ap.add_argument("--displacement-threshold-fraction", type=float, default=0.15)
    ap.add_argument("--solref", default=DEFAULT_SOLREF)
    ap.add_argument("--out-dir", default="Data/_diagnostic_stiff_contacts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    object_assignments = build_object_assignments(args.n_trials, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_tmp_scenes")
    os.makedirs(tmp_dir, exist_ok=True)

    trials = []
    for trial_idx in range(args.n_trials):
        obj_spec = object_assignments[trial_idx]
        squeeze_intensity = float(rng.uniform(0.5, 1.0))
        pull_multiple = float(rng.uniform(0.5, 3.0))
        ramp_steps = int(rng.integers(400, 601))
        effective_threshold = args.displacement_threshold_fraction * obj_spec["char_length"]

        jitter_rng = random.Random(args.seed * 100003 + trial_idx)
        noise_seed = args.seed * 500009 + trial_idx

        trial_xml = os.path.join(tmp_dir, f"trial_{trial_idx:05d}.xml")
        obj_actual = build_trial_xml(args.mjcf, trial_xml, obj_spec, args.mesh_dir, jitter_rng,
                                     args.object_pos, args.fingertips,
                                     tuple(args.pad_friction_range), args.orientation_tilt_deg_max)
        n_patched = stiffen_contacts(trial_xml, args.fingertips, args.solref)
        if trial_idx == 0:
            print(f"Patched {n_patched} geoms with solref={args.solref} (1 object + up to {len(args.fingertips)} pads)")

        result = run_single_trial(
            trial_xml, args.fingertips, args.actuators,
            squeeze_intensity, pull_multiple, args.normal_axis,
            ramp_steps, args.settle_steps, args.hold_steps,
            effective_threshold, args.sensor_noise_std, noise_seed,
            verbose_neutral=(trial_idx == 0),
        )
        if not result["ever_contacted"]:
            continue
        # object_friction_sliding/pad_friction live on build_trial_xml's return
        # value, not run_single_trial's -- merge them in here.
        result["object_friction_sliding"] = float(obj_actual["friction"].split()[0])
        result["pad_friction"] = obj_actual["pad_friction"]
        trials.append(result)
        if (trial_idx + 1) % 10 == 0:
            print(f"  {trial_idx+1}/{args.n_trials} trials done")

    print(f"\n{len(trials)} usable (contacted) trials out of {args.n_trials}")

    # Same discrimination test as before: sum tangential/normal across all 5
    # fingertips, trend (derivative) of slack = mu_combined*normal - tangential,
    # compare the WINDOW before real slip onset vs. a matched stable window.
    feature_names = trials[0]["feature_names"]
    normal_idx = [i for i, n in enumerate(feature_names) if n.endswith("_normal")]
    tangential_idx = [i for i, n in enumerate(feature_names) if n.endswith("_tangential")]

    WINDOW = 20
    pre_onset_vals, stable_vals = [], []
    for r in trials:
        fm = torch.from_numpy(r["feature_matrix"]).float()
        ys = torch.from_numpy(r["slip_label_series"]).long()
        mu = max(r["object_friction_sliding"], float(np.mean(list(r["pad_friction"].values()))))
        normal = fm[:, normal_idx].sum(dim=1)
        tangential = fm[:, tangential_idx].sum(dim=1)
        slack = mu * normal - tangential
        deriv = torch.zeros_like(slack)
        deriv[1:] = slack[1:] - slack[:-1]
        penalty = torch.clamp(torch.relu(-deriv), max=5)

        if r["trial_slip_label"] == 1:
            onset = (ys == 1).nonzero()
            if len(onset) == 0:
                continue
            onset = onset[0].item()
            if onset < WINDOW:
                continue
            pre_onset_vals.append(penalty[onset-WINDOW:onset])
        else:
            L = len(penalty)
            if L > 2 * WINDOW:
                mid = L // 2
                stable_vals.append(penalty[mid-WINDOW//2:mid+WINDOW//2])

    if not pre_onset_vals or not stable_vals:
        print("Not enough slip/stable trials in this small batch to compare -- "
              "re-run with more trials or a different seed.")
        return

    pre_onset_vals = torch.cat(pre_onset_vals)
    stable_vals = torch.cat(stable_vals)
    print(f"\nn pre-onset timesteps: {len(pre_onset_vals)} | n stable timesteps: {len(stable_vals)}")
    print(f"pre-onset mean={pre_onset_vals.mean():.4f} | stable mean={stable_vals.mean():.4f} "
          f"| ratio={pre_onset_vals.mean()/max(stable_vals.mean().item(),1e-6):.2f}x")


if __name__ == "__main__":
    main()
