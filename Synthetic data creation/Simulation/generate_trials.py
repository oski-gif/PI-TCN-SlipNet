"""
Generate randomized grasp trials for PI-TCN training data, across 15 objects.

PATCHED VERSION. Changes vs. the original are marked with "FIX n:" comments,
numbered to match the review:

  1. mj_name2id results are validated -- a -1 no longer silently indexes the
     last sensor/body and produces plausible-looking garbage.
  2. Squeeze intensity interpolates across the FULL ctrlrange (lo..hi) and the
     ramp starts from lo, not from a hardcoded ctrl=0.
  3. Compound sub-geom positions are scaled by the same factor as sizes, so the
     object scales instead of distorting.
  4. Balancing uses a seeded local RNG, so --seed reproduces the balanced split.
  5. Objects that end up with an empty (object, label) bucket are dropped from
     the balanced set entirely, so "equal per object AND 50/50" stays true.
  6. Normal/tangential force are decomposed from the force sensor's own
     site-frame components (--normal-axis), instead of subtracting a touch
     scalar from a force-vector magnitude.
  7. The displacement reference is latched at the END of a settle phase, not at
     first contact, so grasp acquisition is not scored as slip.
  8. A perturbation phase applies a ramped downward pull (a randomized multiple
     of the object's own weight) so slip has a real decision boundary instead
     of depending only on gravity vs. a soft kp=2 position actuator.
  9. Actuators not passed to --actuators are explicitly commanded to a clamped
     neutral value and reported, instead of silently sitting at ctrl=0.

Plus: assetdir is handled alongside meshdir/texturedir; mesh scale jitter is
uniform in x/y with a separate factor in z; slip velocity is read from the free
joint's DOFs rather than finite-differenced from a distance magnitude; the full
object spec is recorded in the manifest; --keep-scenes preserves trial XMLs.

NOTE: generate_meshes.py must emit WATERTIGHT meshes -- MuJoCo needs a closed
manifold to compute inertia from the `mass` attribute.

NOTE: --normal-axis assumes the fingertip sensor sites are oriented with that
local axis pointing out of the contact surface. Verify this in
instrument_scene.py before trusting the normal/tangential split.

NOTE: the object spawns with gravcomp=1 (weightless) and stays that way for
the whole ramp phase, so it doesn't free-fall to the scene's floor plane
while the fingers are still closing around it -- at dt=0.002 an unsupported
object only takes ~100-150 steps to hit the floor, far less than the default
ramp duration. Real gravity is switched back on for run_single_trial at the
first settle step, so the settle-stabilization + hold-phase pull that
actually gets measured for slip runs under ordinary, unmodified physics.
--object-pos must still put the object where --fingertips actually close
around it (measured directly via forward kinematics for the default OrcaHand
scene) -- gravcomp only stops it falling away, it doesn't pull it into reach.

REALISM UPGRADE (this pass): OBJECT_LIBRARY masses are now computed from each
shape's actual geometry x a PLA density (1240 kg/m^3, solid infill -- if you
print at partial infill, real mass will be lower than what's recorded here),
instead of a flat 0.05 kg placeholder for every object regardless of size.
Object sliding friction is set to a literature-typical PLA-vs-rubber value
(~0.4) -- see the pad-friction note below for why this number alone doesn't
fully determine the outcome.

IMPORTANT PHYSICS NOTE ON FRICTION: MuJoCo combines two contacting geoms'
friction coefficients by taking the ELEMENT-WISE MAXIMUM, not an average
(confirmed empirically: forcing the object's sliding friction to 0.01, 0.4,
and 1.0 all produced an identical effective contact friction of exactly 1.0,
matching the fingertip pad's fixed default of "1 0.005 0.0001" -- only at 2.0
did the object's own value actually take effect). With a realistic PLA
friction (~0.3-0.5) and a FIXED pad friction of 1.0, the object's friction
would always be silently overridden and have zero causal effect on slip,
while still looking like a real, varying recorded column. To make it
genuinely causal, the fingertip pad's own friction is now ALSO randomized per
trial per finger (--pad-friction-range, default 0.35-1.2, representing
realistic pad-condition variation from worn/oily to fresh dry silicone) so
its range overlaps the object's jittered friction -- sometimes the pad is the
limiting factor, sometimes the object's own (lower) friction is. Both are
recorded per trial (object_friction_sliding, pad_friction per fingertip).

Object geometry is now jittered for primitives too (previously only mesh
scale and compound sub-geom size varied; box/sphere/cylinder/capsule/
ellipsoid primitives had a completely fixed size every trial).

Object orientation is randomized per trial: full random yaw (0-360 deg about
the vertical axis) plus a small random tilt (+/- --orientation-tilt-deg-max,
default 10 deg, roll and pitch) -- previously every single trial spawned the
object in an identical, fixed orientation, so asymmetric objects (l_block,
dumbbell, plus_cross, rectangular_prism, ...) were only ever presented to the
hand from one relative angle.

Sensor readings (touch/normal/tangential/torque_mag) now get independent
Gaussian noise added per channel per step (--sensor-noise-std, default 0.05,
same units as the underlying signal -- an approximation, not a per-sensor
calibrated noise model) so the recorded features resemble real, imperfect
tactile sensors rather than noiseless ground truth. Contact detection
(ever_contacted, first_contact_step) is computed from the PRE-noise signal,
so injected noise can't spuriously flip contact detection near zero.

Approach speed is randomized per trial by drawing the ramp duration from
--ramp-steps-range (default 400-600 steps) instead of a fixed 500, and the
resulting mean actuator angular velocity during the ramp is recorded
(approach_velocity_rad_s) -- previously every trial closed the grip at
exactly the same speed.

New recorded fields (manifest row + .npz), beyond what already existed:
object_friction_sliding/torsional/rolling, pad_friction (per fingertip) +
pad_friction_mean/min, object_kind, object_size_jitter, n_fingers_contacted,
first_contact_step, grasp_force_at_settle (measured grip force, distinct from
the merely-commanded squeeze_intensity), object_yaw_deg/roll_deg/pitch_deg,
sensor_noise_std, ramp_steps (now variable)/ramp_duration_s/
approach_velocity_rad_s, initial_object_pos/initial_object_quat.

Usage:
    python generate_trials.py \\
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
        --n-trials 750 \\
        --out-dir data/raw_trials \\
        --force-range 0.5 1.0 \\
        --pull-range 0.5 3.0 \\
        --displacement-threshold-fraction 0.15
"""
import argparse
import json
import math
import os
import random
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import mujoco


# 15 objects. "char_length" is each object's own characteristic size (m),
# used to make the slip threshold geometry-normalized: effective_threshold
# = --displacement-threshold-fraction * char_length.
#
# "mass" is computed as (solid volume from the geometry below) x PLA density
# (1240 kg/m^3) -- see the module docstring's REALISM UPGRADE note. This
# assumes 100% infill; a real FDM print at partial infill will weigh less.
# "friction" sliding component is a literature-typical PLA-vs-rubber value
# (~0.4); see the module docstring's IMPORTANT PHYSICS NOTE for why the
# fingertip pad's own (also randomized) friction matters just as much.
#
# primitive: "size" follows MuJoCo's own convention per geom type
#   box: half-extents "x y z" | sphere: "radius" | cylinder/capsule: "radius half-length"
#   ellipsoid: semi-axes "x y z"
# mesh: "mesh_key" must match a file generate_meshes.py produces;
#   "scale" = "rx ry rz" applied to the unit mesh (rx=ry~=base size, rz~=height)
# compound: "geoms" is a list of {type, size, pos}; "masses" is a parallel
#   list of per-geom mass, each computed the same way (volume x PLA density)
PLA_DENSITY_KG_M3 = 1240.0
OBJECT_FRICTION = "0.4 0.005 0.0001"

OBJECT_LIBRARY = [
    # box, half-extents (0.02,0.02,0.02) -> 0.04^3 m^3 * 1240 = 0.0794 kg
    {"name": "cube", "kind": "primitive", "type": "box",
     "size": "0.02 0.02 0.02", "mass": 0.0794, "friction": OBJECT_FRICTION,
     "char_length": 0.04},

    # sphere, r=0.02 -> 4/3*pi*r^3 * 1240 = 0.0415 kg
    {"name": "sphere", "kind": "primitive", "type": "sphere",
     "size": "0.02", "mass": 0.0415, "friction": OBJECT_FRICTION,
     "char_length": 0.04},

    # 32-sided pyramid (cone approximation), base r=0.02, height 0.04
    # -> (1/3)*base_area*height * 1240 = 0.0206 kg
    {"name": "cone", "kind": "mesh", "mesh_key": "cone",
     "scale": "0.02 0.02 0.04", "mass": 0.0206, "friction": OBJECT_FRICTION,
     "char_length": 0.04},

    # box, half-extents (0.015,0.025,0.035) -> 0.03*0.05*0.07 * 1240 = 0.1302 kg
    {"name": "rectangular_prism", "kind": "primitive", "type": "box",
     "size": "0.015 0.025 0.035", "mass": 0.1302, "friction": OBJECT_FRICTION,
     "char_length": 0.07},

    # cylinder, r=0.02, half-len=0.03 -> pi*r^2*(2*halflen) * 1240 = 0.0935 kg
    {"name": "cylindrical_prism", "kind": "primitive", "type": "cylinder",
     "size": "0.02 0.03", "mass": 0.0935, "friction": OBJECT_FRICTION,
     "char_length": 0.06},

    # capsule, r=0.015, half-len=0.025 -> (cylinder + one sphere from caps) * 1240 = 0.0614 kg
    {"name": "capsule", "kind": "primitive", "type": "capsule",
     "size": "0.015 0.025", "mass": 0.0614, "friction": OBJECT_FRICTION,
     "char_length": 0.08},

    # ellipsoid, semi-axes (0.015,0.02,0.03) -> 4/3*pi*a*b*c * 1240 = 0.0467 kg
    {"name": "ellipsoid", "kind": "primitive", "type": "ellipsoid",
     "size": "0.015 0.02 0.03", "mass": 0.0467, "friction": OBJECT_FRICTION,
     "char_length": 0.06},

    # cylinder, r=0.035, half-len=0.008 -> pi*r^2*(2*halflen) * 1240 = 0.0764 kg
    {"name": "flat_disk", "kind": "primitive", "type": "cylinder",
     "size": "0.035 0.008", "mass": 0.0764, "friction": OBJECT_FRICTION,
     "char_length": 0.07},

    # cylinder, r=0.008, half-len=0.045 -> pi*r^2*(2*halflen) * 1240 = 0.0224 kg
    {"name": "rod", "kind": "primitive", "type": "cylinder",
     "size": "0.008 0.045", "mass": 0.0224, "friction": OBJECT_FRICTION,
     "char_length": 0.09},

    # two boxes; each is a 0.06x0.02x0.02 solid -> 2.4e-5 m^3 * 1240 = 0.0298 kg
    # each (both arms have the same volume by construction). NOTE: the two
    # sub-boxes are placed without checking for overlap, so real mass would be
    # very slightly less than the sum of independently-computed volumes --
    # negligible here relative to overall placeholder-grade precision.
    {"name": "l_block", "kind": "compound", "friction": OBJECT_FRICTION,
     "char_length": 0.08,
     "geoms": [
         {"type": "box", "size": "0.03 0.01 0.01", "pos": "0 0 0"},
         {"type": "box", "size": "0.01 0.01 0.03", "pos": "-0.02 0 0.02"},
     ],
     "masses": [0.0298, 0.0298]},

    # two spheres (r=0.015 -> 0.0175 kg each) + connecting cylinder
    # (r=0.005, half-len=0.03 -> 0.0058 kg)
    {"name": "dumbbell", "kind": "compound", "friction": OBJECT_FRICTION,
     "char_length": 0.09,
     "geoms": [
         {"type": "sphere", "size": "0.015", "pos": "0 0 0.03"},
         {"type": "sphere", "size": "0.015", "pos": "0 0 -0.03"},
         {"type": "cylinder", "size": "0.005 0.03", "pos": "0 0 0"},
     ],
     "masses": [0.0175, 0.0175, 0.0058]},

    # three boxes (each 0.06x0.016x0.016 -> 0.019 kg), one per arm. NOTE: all
    # three overlap at the shared center cube (~4e-6 m^3, <10% of the total)
    # -- real mass would be very slightly less; not corrected for here.
    {"name": "plus_cross", "kind": "compound", "friction": OBJECT_FRICTION,
     "char_length": 0.06,
     "geoms": [
         {"type": "box", "size": "0.03 0.008 0.008", "pos": "0 0 0"},
         {"type": "box", "size": "0.008 0.03 0.008", "pos": "0 0 0"},
         {"type": "box", "size": "0.008 0.008 0.03", "pos": "0 0 0"},
     ],
     "masses": [0.0190, 0.0190, 0.0190]},

    # 4-sided pyramid, base r=0.02, height 0.04 -> (1/3)*base_area*height * 1240 = 0.0132 kg
    {"name": "square_pyramid", "kind": "mesh", "mesh_key": "square_pyramid",
     "scale": "0.02 0.02 0.04", "mass": 0.0132, "friction": OBJECT_FRICTION,
     "char_length": 0.04},

    # 3-sided prism, base r=0.02, height 0.05 -> base_area*height * 1240 = 0.0322 kg
    {"name": "triangular_prism", "kind": "mesh", "mesh_key": "triangular_prism",
     "scale": "0.02 0.02 0.05", "mass": 0.0322, "friction": OBJECT_FRICTION,
     "char_length": 0.05},

    # 6-sided prism, base r=0.018, height 0.04 -> base_area*height * 1240 = 0.0418 kg
    {"name": "hexagonal_prism", "kind": "mesh", "mesh_key": "hexagonal_prism",
     "scale": "0.018 0.018 0.04", "mass": 0.0418, "friction": OBJECT_FRICTION,
     "char_length": 0.04},
]

OBJECT_BODY_NAME = "slip_test_object"


# ---------------------------------------------------------------------------
# scene construction
# ---------------------------------------------------------------------------

def _find_body(root, name):
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def build_trial_xml(base_xml_path, out_xml_path, obj_spec, mesh_dir, jitter_rng,
                    object_pos, fingertip_names, pad_friction_range,
                    orientation_tilt_deg_max):
    """Swap in one object variant (primitive/mesh/compound) with randomized
    jitter on mass/friction/size/position/orientation, into an already
    sensor-instrumented scene. Also jitters each fingertip pad's own contact
    friction (see module docstring's IMPORTANT PHYSICS NOTE on why). Returns
    a dict of what was actually used."""
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> in {base_xml_path}")

    # The trial XML gets written into a different directory than
    # base_xml_path (a per-trial temp folder), but any relative asset paths
    # inside it were only ever correct relative to base_xml_path's own
    # directory. Make the asset dirs absolute so they keep resolving no matter
    # where this generated copy lives -- if a relative dir already existed,
    # join it onto base_dir rather than discarding it.
    base_dir = os.path.dirname(os.path.abspath(base_xml_path))
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)

    # assetdir is MuJoCo's shorthand that sets meshdir AND texturedir at once.
    # If the base scene used it, rewriting only meshdir/texturedir would leave
    # a stale relative assetdir in place, so fix it too.
    for dir_attr in ("assetdir", "meshdir", "texturedir"):
        existing = compiler.get(dir_attr)
        if existing and not os.path.isabs(existing):
            compiler.set(dir_attr, os.path.join(base_dir, existing))
        # else: absent or already absolute

    # Only set the explicit dirs if neither they nor assetdir provide a base.
    if not compiler.get("assetdir"):
        for dir_attr in ("meshdir", "texturedir"):
            if not compiler.get(dir_attr):
                compiler.set(dir_attr, base_dir)

    for body in list(worldbody):
        if body.get("name") == OBJECT_BODY_NAME:
            worldbody.remove(body)

    pos_jitter = [jitter_rng.uniform(-0.005, 0.005) for _ in range(2)]
    pos_str = (f"{object_pos[0] + pos_jitter[0]:.4f} "
               f"{object_pos[1] + pos_jitter[1]:.4f} "
               f"{object_pos[2]:.4f}")

    friction_vals = [float(f) * jitter_rng.uniform(0.85, 1.15)
                     for f in obj_spec["friction"].split()]
    friction_str = " ".join(f"{v:.5f}" for v in friction_vals)

    # Random orientation: full yaw (any angle about the vertical axis) plus a
    # small tilt, so asymmetric objects aren't always presented to the hand
    # from the same relative angle. See module docstring.
    yaw = jitter_rng.uniform(0.0, 2 * math.pi)
    tilt_max_rad = math.radians(orientation_tilt_deg_max)
    roll = jitter_rng.uniform(-tilt_max_rad, tilt_max_rad)
    pitch = jitter_rng.uniform(-tilt_max_rad, tilt_max_rad)
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([roll, pitch, yaw]), "xyz")
    quat_str = " ".join(f"{v:.6f}" for v in quat)

    obj_body = ET.SubElement(worldbody, "body")
    obj_body.set("name", OBJECT_BODY_NAME)
    obj_body.set("pos", pos_str)
    obj_body.set("quat", quat_str)
    # Spawn weightless (FIX: prevents the object free-falling to the floor
    # during the ramp phase, before the fingers have closed around it --
    # run_single_trial switches real gravity back on at the start of the
    # settle phase). See NOTE below on --object-pos for the other half of
    # this fix.
    obj_body.set("gravcomp", "1")

    joint = ET.SubElement(obj_body, "joint")
    joint.set("type", "free")
    joint.set("name", f"{OBJECT_BODY_NAME}_freejoint")

    kind = obj_spec["kind"]
    actual = {
        "friction": friction_str,
        "pos": pos_str,
        "kind": kind,
        "orientation_quat": quat.tolist(),
        "orientation_yaw_deg": math.degrees(yaw),
        "orientation_roll_deg": math.degrees(roll),
        "orientation_pitch_deg": math.degrees(pitch),
    }

    if kind == "primitive":
        # Geometry jitter, previously only applied to mesh scale / compound
        # sub-geom size -- primitives had a completely fixed size every trial.
        size_jitter = jitter_rng.uniform(0.9, 1.1)
        mass = obj_spec["mass"] * jitter_rng.uniform(0.9, 1.1)
        sizes = [float(s) * size_jitter for s in obj_spec["size"].split()]
        size_str = " ".join(f"{s:.5f}" for s in sizes)

        geom = ET.SubElement(obj_body, "geom")
        geom.set("name", f"{OBJECT_BODY_NAME}_geom")
        geom.set("type", obj_spec["type"])
        geom.set("size", size_str)
        geom.set("mass", f"{mass:.5f}")
        geom.set("friction", friction_str)
        geom.set("rgba", "0.2 0.6 0.9 1")
        actual["mass"] = mass
        actual["size"] = size_str
        actual["size_jitter"] = size_jitter

    elif kind == "mesh":
        mass = obj_spec["mass"] * jitter_rng.uniform(0.9, 1.1)
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")

        base_scale = [float(s) for s in obj_spec["scale"].split()]
        # Jitter x and y with ONE factor and z with another, so the cross
        # section stays circular/regular and only the aspect ratio varies.
        # (Independent per-axis jitter turned cones into elliptical cones.)
        xy_jitter = jitter_rng.uniform(0.9, 1.1)
        z_jitter = jitter_rng.uniform(0.9, 1.1)
        jittered_scale = [base_scale[0] * xy_jitter,
                          base_scale[1] * xy_jitter,
                          base_scale[2] * z_jitter]
        scale_str = " ".join(f"{s:.5f}" for s in jittered_scale)

        mesh_file = os.path.abspath(
            os.path.join(mesh_dir, f"{obj_spec['mesh_key']}.obj"))
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(
                f"Mesh file not found: {mesh_file}. Run generate_meshes.py "
                f"--out-dir {mesh_dir} first.")

        mesh_name = f"{obj_spec['mesh_key']}_trial"
        mesh_asset = ET.SubElement(asset, "mesh")
        mesh_asset.set("name", mesh_name)
        mesh_asset.set("file", mesh_file)
        mesh_asset.set("scale", scale_str)

        geom = ET.SubElement(obj_body, "geom")
        geom.set("name", f"{OBJECT_BODY_NAME}_geom")
        geom.set("type", "mesh")
        geom.set("mesh", mesh_name)
        geom.set("mass", f"{mass:.5f}")
        geom.set("friction", friction_str)
        geom.set("rgba", "0.2 0.6 0.9 1")
        actual["mass"] = mass
        actual["scale"] = scale_str
        actual["xy_jitter"] = xy_jitter
        actual["z_jitter"] = z_jitter
        actual["size_jitter"] = xy_jitter  # flat alias, see main()'s manifest row

    elif kind == "compound":
        size_jitter = jitter_rng.uniform(0.9, 1.1)
        sub_actual = []
        for i, (sub_geom, sub_mass) in enumerate(
                zip(obj_spec["geoms"], obj_spec["masses"])):
            geom = ET.SubElement(obj_body, "geom")
            geom.set("name", f"{OBJECT_BODY_NAME}_geom_{i}")
            geom.set("type", sub_geom["type"])

            sizes = [float(s) * size_jitter for s in sub_geom["size"].split()]
            geom.set("size", " ".join(f"{s:.5f}" for s in sizes))

            # FIX 3: scale the offsets by the SAME factor as the sizes.
            # Previously sizes grew while positions stayed fixed, which
            # changed the object's proportions (and could make the dumbbell's
            # spheres overlap the bar or float off it).
            positions = [float(p) * size_jitter for p in sub_geom["pos"].split()]
            geom.set("pos", " ".join(f"{p:.5f}" for p in positions))

            m = sub_mass * jitter_rng.uniform(0.9, 1.1)
            geom.set("mass", f"{m:.5f}")
            geom.set("friction", friction_str)
            geom.set("rgba", "0.2 0.6 0.9 1")
            sub_actual.append({"size": geom.get("size"),
                               "pos": geom.get("pos"),
                               "mass": m})
        actual["size_jitter"] = size_jitter
        actual["geoms"] = sub_actual

    else:
        raise ValueError(f"Unknown object kind: {kind}")

    # Fingertip pad friction jitter -- see module docstring's IMPORTANT
    # PHYSICS NOTE. Only touches the COLLIDABLE geom(s) on each fingertip
    # body (contype != 0); the visual-only "skin" geom is left alone.
    pad_friction_actual = {}
    for tip_name in fingertip_names:
        body = _find_body(root, tip_name)
        if body is None:
            # FIX: used to silently `continue` with zero output. A typo'd or
            # stale --fingertips name would leave that fingertip's friction
            # un-randomized for the entire run with no signal anything was
            # wrong -- instrument_scene.py already warns for this same
            # situation, this now matches.
            print(f"WARNING: fingertip body '{tip_name}' not found in scene -- "
                  f"its pad friction was NOT randomized for this trial.")
            continue
        pad_mu = jitter_rng.uniform(pad_friction_range[0], pad_friction_range[1])
        n_applied = 0
        for geom in body.findall("geom"):
            if geom.get("contype") == "0":
                continue
            geom.set("friction", f"{pad_mu:.5f} 0.005 0.0001")
            n_applied += 1
        if n_applied == 0:
            # FIX: used to record pad_mu into pad_friction_actual regardless
            # of whether any geom actually matched the contype filter --
            # claiming in the manifest that a randomized friction took
            # physical effect when it silently didn't (e.g. all of this
            # body's geoms rely on a <default> class instead of an explicit
            # contype attribute).
            print(f"WARNING: fingertip '{tip_name}' has no collidable geom "
                  f"(contype != 0) -- its pad friction was NOT applied for "
                  f"this trial.")
            continue
        pad_friction_actual[tip_name] = pad_mu
    actual["pad_friction"] = pad_friction_actual

    tree.write(out_xml_path, encoding="UTF-8", xml_declaration=True)
    return actual


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def _require_id(model, obj_type, name, what):
    """FIX 1: mj_name2id returns -1 on a miss, and -1 is a VALID numpy index
    (the last element). Without this check a naming mismatch between
    instrument_scene.py and this script silently reads the wrong sensor or
    body for every single trial and the dataset looks fine."""
    oid = mujoco.mj_name2id(model, obj_type, name)
    if oid == -1:
        raise ValueError(
            f"{what} '{name}' not found in the model. Check that "
            f"instrument_scene.py emitted it with exactly this name.")
    return oid


def run_single_trial(mjcf_path, fingertip_names, actuator_names,
                     squeeze_intensity, pull_multiple, normal_axis,
                     ramp_steps, settle_steps, hold_steps,
                     displacement_threshold, sensor_noise_std, noise_seed,
                     verbose_neutral=False):
    """
    Returns a fixed-order feature matrix (T, F), per-timestep slip labels, and
    continuous displacement/slip-velocity traces.

    Phases:
      ramp   -- close the grip from ctrlrange lo toward the squeeze target
      settle -- hold the grip, let the grasp stabilize; the displacement
                reference is latched at the END of this phase (FIX 7)
      hold   -- ramp a downward pull of `pull_multiple` x the object's own
                weight onto the object and see whether the grasp holds (FIX 8)

    displacement_threshold is the GEOMETRY-NORMALIZED absolute value
    (fraction * this object's char_length), computed by the caller.

    sensor_noise_std adds independent Gaussian noise (same units as the
    underlying signal) to touch/normal/tangential/torque_mag AFTER contact
    detection is computed from the clean signal, so noise can't spuriously
    flip ever_contacted near zero.
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    noise_rng = np.random.default_rng(noise_seed)

    # FIX 1: every lookup is validated.
    touch_ids = {n: _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                f"{n}_touch", "Touch sensor")
                 for n in fingertip_names}
    force_ids = {n: _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                f"{n}_force", "Force sensor")
                 for n in fingertip_names}
    torque_ids = {n: _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                 f"{n}_torque", "Torque sensor")
                  for n in fingertip_names}

    obj_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY,
                              OBJECT_BODY_NAME, "Object body")
    free_joint_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{OBJECT_BODY_NAME}_freejoint", "Free joint")
    # The object is spawned with gravcomp=1 (see build_trial_xml) so it
    # doesn't free-fall to the floor while the fingers are still closing
    # around it during the ramp phase -- the ramp is a kinematic approach,
    # not part of what's being measured. Real gravity is switched back on
    # below at the start of the settle phase, so the ENTIRE window that
    # matters for slip (settle's stabilization + the hold-phase pull) runs
    # under unmodified physics, exactly as before this fix.
    model.body_gravcomp[obj_body_id] = 1.0
    dof_adr = model.jnt_dofadr[free_joint_id]

    # FIX 2: interpolate across the whole ctrlrange instead of assuming the
    # lower bound is 0. These are <position> actuators, so ctrl is a target
    # joint angle -- 0 is not "relaxed", it is a commanded pose.
    driven = {}
    for a in actuator_names:
        aid = _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a, "Actuator")
        lo, hi = model.actuator_ctrlrange[aid]
        driven[aid] = (float(lo), float(lo) + squeeze_intensity * float(hi - lo))

    # Mean actuator angular velocity during the ramp -- a real, physically
    # meaningful "approach speed" number now that ramp_steps varies per trial.
    approach_velocity_rad_s = float(np.mean(
        [abs(target - lo) / (ramp_steps * dt) for lo, target in driven.values()]
    )) if driven else 0.0

    # FIX 9: every other actuator is commanded explicitly to a clamped neutral
    # value rather than being left at ctrl=0, which for a position actuator is
    # itself a pose (and may have been splaying the abduction joints).
    neutral_ids = []
    for aid in range(model.nu):
        if aid in driven:
            continue
        lo, hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(0.0, lo, hi))
        neutral_ids.append(aid)
    if verbose_neutral and neutral_ids:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in neutral_ids]
        print(f"  [info] {len(neutral_ids)} actuator(s) held at clamped "
              f"neutral for the whole trial: {names}")

    # Object weight, for scaling the perturbation pull.
    obj_mass = float(model.body_mass[obj_body_id])
    g = float(np.linalg.norm(model.opt.gravity))
    pull_force_max = pull_multiple * obj_mass * g

    def sensor_read(sid):
        adr = model.sensor_adr[sid]
        dim = model.sensor_dim[sid]
        return data.sensordata[adr:adr + dim].copy()

    tangent_axes = [i for i in range(3) if i != normal_axis]

    feature_names = []
    for name in fingertip_names:
        feature_names += [f"{name}_touch", f"{name}_normal",
                          f"{name}_tangential", f"{name}_torque_mag"]
    feature_names += ["obj_x", "obj_y", "obj_z",
                      "displacement", "slip_speed", "pull_force", "phase"]

    total_steps = ramp_steps + settle_steps + hold_steps
    n_features = len(feature_names)
    feature_matrix = np.zeros((total_steps, n_features), dtype=np.float64)
    slip_label_series = np.zeros(total_steps, dtype=np.int8)

    ref_pos = None
    max_displacement = 0.0
    ever_contacted = False
    ever_contacted_per_finger = {n: False for n in fingertip_names}
    first_contact_step = -1
    grasp_force_at_settle = 0.0

    for step in range(total_steps):
        if step == ramp_steps:
            model.body_gravcomp[obj_body_id] = 0.0

        if step < ramp_steps:
            phase = 0
            frac = (step + 1) / max(ramp_steps, 1)
            pull = 0.0
        elif step < ramp_steps + settle_steps:
            phase = 1
            frac = 1.0
            pull = 0.0
        else:
            phase = 2
            frac = 1.0
            # Ramp the pull in over the first half of the hold phase so the
            # load is applied smoothly rather than as an impulse.
            pull_frac = min((step - ramp_steps - settle_steps + 1)
                            / max(hold_steps * 0.5, 1), 1.0)
            pull = pull_frac * pull_force_max

        for aid, (lo, target) in driven.items():
            data.ctrl[aid] = lo + frac * (target - lo)

        # FIX 8: applied at the body CoM in world coordinates, straight down.
        data.xfrc_applied[obj_body_id, :3] = [0.0, 0.0, -pull]

        mujoco.mj_step(model, data)

        obj_pos = data.xpos[obj_body_id].copy()
        obj_linvel = data.qvel[dof_adr:dof_adr + 3].copy()

        col = 0
        step_normal_sum = 0.0
        step_normal_raw_sum = 0.0
        for name in fingertip_names:
            touch_raw = float(sensor_read(touch_ids[name])[0])
            force_vec = sensor_read(force_ids[name])
            torque_vec = sensor_read(torque_ids[name])

            # FIX 6: the force sensor value is already expressed in its site's
            # frame, so the normal/tangential split is a component split along
            # that frame's axes -- not (||F|| - touch_scalar), which mixed two
            # different quantities and could go negative.
            normal_raw = float(force_vec[normal_axis])
            tangential_raw = float(np.linalg.norm(
                [force_vec[tangent_axes[0]], force_vec[tangent_axes[1]]]))
            torque_mag_raw = float(np.linalg.norm(torque_vec))

            # Contact detection uses the CLEAN signal, before noise (below)
            # is layered on for the recorded features.
            if touch_raw > 1e-3:
                ever_contacted = True
                ever_contacted_per_finger[name] = True
                if first_contact_step == -1:
                    first_contact_step = step

            touch = max(0.0, touch_raw + noise_rng.normal(0.0, sensor_noise_std))
            normal = normal_raw + noise_rng.normal(0.0, sensor_noise_std)
            tangential = max(0.0, tangential_raw + noise_rng.normal(0.0, sensor_noise_std))
            torque_mag = max(0.0, torque_mag_raw + noise_rng.normal(0.0, sensor_noise_std))
            step_normal_sum += normal
            step_normal_raw_sum += normal_raw

            feature_matrix[step, col] = touch; col += 1
            feature_matrix[step, col] = normal; col += 1
            feature_matrix[step, col] = tangential; col += 1
            feature_matrix[step, col] = torque_mag; col += 1

        # FIX 7: latch the reference at the end of settle, once the grasp has
        # stabilized. Latching at first contact scored the object rolling into
        # the fingers -- normal grasp acquisition -- as slip. Grip force is
        # measured at this same instant: the actual grasp force achieved,
        # distinct from the merely-commanded squeeze_intensity.
        if step == ramp_steps + settle_steps - 1:
            ref_pos = obj_pos.copy()
            # FIX: was step_normal_sum (the noise-injected value also used
            # for the recorded feature columns). ever_contacted/
            # first_contact_step are deliberately computed from the clean
            # pre-noise signal above; this "actual grasp force achieved"
            # metadata field should be equally clean, not degraded by the
            # same synthetic sensor noise as the training features.
            grasp_force_at_settle = step_normal_raw_sum

        if ref_pos is not None:
            displacement = float(np.linalg.norm(obj_pos - ref_pos))
            max_displacement = max(max_displacement, displacement)
            slip_label_series[step] = int(displacement > displacement_threshold)
        else:
            displacement = 0.0

        # Slip speed straight from the free joint's linear DOFs -- a real
        # velocity, not the derivative of a distance magnitude (which could
        # never be negative and ignored direction).
        slip_speed = float(np.linalg.norm(obj_linvel))

        feature_matrix[step, col] = obj_pos[0]; col += 1
        feature_matrix[step, col] = obj_pos[1]; col += 1
        feature_matrix[step, col] = obj_pos[2]; col += 1
        feature_matrix[step, col] = displacement; col += 1
        feature_matrix[step, col] = slip_speed; col += 1
        feature_matrix[step, col] = pull; col += 1
        feature_matrix[step, col] = float(phase); col += 1

    trial_slip_label = int(max_displacement > displacement_threshold)
    slip_indices = np.nonzero(slip_label_series)[0]
    onset_step = int(slip_indices[0]) if len(slip_indices) > 0 else -1

    return {
        "feature_matrix": feature_matrix,
        "feature_names": feature_names,
        "slip_label_series": slip_label_series,
        "trial_slip_label": trial_slip_label,
        "onset_step": onset_step,
        "max_displacement": max_displacement,
        "ever_contacted": bool(ever_contacted),
        "ever_contacted_per_finger": {n: bool(v) for n, v in ever_contacted_per_finger.items()},
        "n_fingers_contacted": int(sum(ever_contacted_per_finger.values())),
        "first_contact_step": int(first_contact_step),
        "grasp_force_at_settle": float(grasp_force_at_settle),
        "approach_velocity_rad_s": approach_velocity_rad_s,
        "ramp_duration_s": float(ramp_steps * dt),
        "obj_mass": obj_mass,
        "pull_force_max": pull_force_max,
    }


# ---------------------------------------------------------------------------
# trial planning + balancing
# ---------------------------------------------------------------------------

def build_object_assignments(n_trials, seed):
    """Equal counts per object, order still randomized (blocked shuffle:
    repeat a shuffled full set of objects, truncate to n_trials)."""
    rng = random.Random(seed)
    reps = (n_trials + len(OBJECT_LIBRARY) - 1) // len(OBJECT_LIBRARY)
    assignments = []
    for _ in range(reps):
        block = OBJECT_LIBRARY.copy()
        rng.shuffle(block)
        assignments.extend(block)
    return assignments[:n_trials]


def balance_manifest(manifest, seed):
    """Stratified undersampling to equal-per-object AND 50/50 slip/stable.

    FIX 4: uses a seeded local RNG so --seed reproduces the split.
    FIX 5: an object missing either label is DROPPED entirely. The old version
    took min() over non-empty buckets only, so such an object contributed
    n_per_bucket rows while every other object contributed 2*n_per_bucket --
    the output was neither equal-per-object nor 50/50, but was still described
    as both.
    """
    rng = random.Random(seed)

    buckets = {}
    for obj in OBJECT_LIBRARY:
        for label in (0, 1):
            buckets[(obj["name"], label)] = [
                m for m in manifest
                if m["object_name"] == obj["name"] and m["slip_label"] == label]

    usable, dropped = [], []
    for obj in OBJECT_LIBRARY:
        n_stable = len(buckets[(obj["name"], 0)])
        n_slip = len(buckets[(obj["name"], 1)])
        if n_stable == 0 or n_slip == 0:
            dropped.append((obj["name"], n_stable, n_slip))
        else:
            usable.append(obj["name"])

    if not usable:
        return [], 0, usable, dropped

    n_per_bucket = min(min(len(buckets[(name, 0)]), len(buckets[(name, 1)]))
                       for name in usable)

    balanced = []
    for name in usable:
        for label in (0, 1):
            trials = list(buckets[(name, label)])
            rng.shuffle(trials)
            balanced.extend(trials[:n_per_bucket])
    rng.shuffle(balanced)

    return balanced, n_per_bucket, usable, dropped


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True,
                    help="Sensor-instrumented scene (output of instrument_scene.py)")
    ap.add_argument("--fingertips", nargs="+", required=True)
    ap.add_argument("--actuators", nargs="+", required=True,
                    help="FLEXION actuator names only (mcp + pip per finger). "
                         "Every other actuator is commanded to a clamped "
                         "neutral value for the whole trial.")
    ap.add_argument("--mesh-dir", default="assets",
                    help="Directory containing meshes from generate_meshes.py")
    ap.add_argument("--object-pos", nargs=3, type=float,
                    default=[0.085, 0.04, 0.245], metavar=("X", "Y", "Z"),
                    help="Object spawn position (world coords), before the "
                         "small per-trial xy jitter. Default is the measured "
                         "centroid of the --fingertips sites across a "
                         "squeeze-intensity sweep (0.5-1.0) for the "
                         "right OrcaHand in scene_slip.xml -- if you swap "
                         "hands/fingertips/scene, remeasure this.")
    ap.add_argument("--pad-friction-range", nargs=2, type=float,
                    default=[0.35, 1.2], metavar=("MIN", "MAX"),
                    help="Per-fingertip, per-trial random sliding friction for "
                         "the pad's own collision geom, representing realistic "
                         "pad-condition variation (worn/oily to fresh dry "
                         "silicone). See module docstring's IMPORTANT PHYSICS "
                         "NOTE on why this needs to overlap the object's own "
                         "friction range to make the latter causally relevant "
                         "under MuJoCo's max-of-the-two contact rule.")
    ap.add_argument("--orientation-tilt-deg-max", type=float, default=10.0,
                    help="Max +/- roll/pitch tilt (degrees) applied on top of "
                         "a full random yaw (0-360 deg) for the object's "
                         "spawn orientation.")
    ap.add_argument("--sensor-noise-std", type=float, default=0.05,
                    help="Std of independent Gaussian noise added to each of "
                         "touch/normal/tangential/torque_mag, per fingertip "
                         "per step (same units as the underlying signal -- an "
                         "approximation, not a calibrated per-sensor noise "
                         "model). Pass 0 to keep noiseless ground truth.")
    ap.add_argument("--ramp-steps-range", nargs=2, type=int, default=[400, 600],
                    metavar=("MIN", "MAX"),
                    help="Steps to close the grip, drawn per trial from this "
                         "range so approach speed varies -- previously a "
                         "fixed 500 for every trial.")
    ap.add_argument("--n-trials", type=int, default=750,
                    help="Total trials, split evenly across the 15 objects "
                         "(750 / 15 = 50 per object)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force-range", nargs=2, type=float, default=[0.5, 1.0],
                    metavar=("MIN", "MAX"),
                    help="Min/max squeeze intensity (0-1), interpolated across "
                         "each actuator's full ctrlrange.")
    ap.add_argument("--pull-range", nargs=2, type=float, default=[0.5, 3.0],
                    metavar=("MIN", "MAX"),
                    help="Min/max downward pull applied during the hold phase, "
                         "as a multiple of the object's own weight. This is "
                         "what creates the slip/stable decision boundary -- "
                         "without it, slip depends only on gravity vs. a soft "
                         "kp=2 position actuator and you get a degenerate split.")
    ap.add_argument("--normal-axis", type=int, default=2, choices=[0, 1, 2],
                    help="Which axis of the fingertip force sensor's SITE frame "
                         "points out of the contact surface (0=x, 1=y, 2=z). "
                         "Verify this against instrument_scene.py -- the "
                         "normal/tangential split depends on it.")
    ap.add_argument("--settle-steps", type=int, default=300,
                    help="Steps to let the grasp stabilize before the "
                         "displacement reference is latched.")
    ap.add_argument("--hold-steps", type=int, default=1000,
                    help="Steps of loaded hold, during which the pull ramps in.")
    ap.add_argument("--displacement-threshold-fraction", type=float, default=0.15,
                    help="Slip threshold as a FRACTION of each object's own "
                         "char_length (e.g. 0.15 = 15%% of that object's size "
                         "counted as slip), rather than one flat meters value "
                         "for every geometry. Placeholder -- tune against a "
                         "few visually-checked trials.")
    ap.add_argument("--keep-scenes", action="store_true",
                    help="Keep the per-trial XML scenes instead of deleting "
                         "them, so a trial can be replayed exactly.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for obj in OBJECT_LIBRARY:
        if obj["kind"] == "mesh":
            mesh_file = os.path.join(args.mesh_dir, f"{obj['mesh_key']}.obj")
            if not os.path.exists(mesh_file):
                raise FileNotFoundError(
                    f"Mesh file not found: {mesh_file}. Run first:\n"
                    f"    python generate_meshes.py --out-dir {args.mesh_dir}")

    rng = np.random.default_rng(args.seed)
    object_assignments = build_object_assignments(args.n_trials, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_tmp_scenes")
    os.makedirs(tmp_dir, exist_ok=True)

    manifest = []
    no_contact_trials = []

    for trial_idx in range(args.n_trials):
        obj_spec = object_assignments[trial_idx]
        squeeze_intensity = float(rng.uniform(args.force_range[0],
                                              args.force_range[1]))
        pull_multiple = float(rng.uniform(args.pull_range[0],
                                          args.pull_range[1]))
        ramp_steps = int(rng.integers(args.ramp_steps_range[0],
                                      args.ramp_steps_range[1] + 1))
        effective_threshold = (args.displacement_threshold_fraction
                               * obj_spec["char_length"])

        jitter_rng = random.Random(args.seed * 100003 + trial_idx)
        noise_seed = args.seed * 500009 + trial_idx

        trial_xml = os.path.join(tmp_dir, f"trial_{trial_idx:05d}.xml")
        obj_actual = build_trial_xml(args.mjcf, trial_xml, obj_spec,
                                     args.mesh_dir, jitter_rng,
                                     args.object_pos, args.fingertips,
                                     tuple(args.pad_friction_range),
                                     args.orientation_tilt_deg_max)

        result = run_single_trial(
            trial_xml, args.fingertips, args.actuators,
            squeeze_intensity, pull_multiple, args.normal_axis,
            ramp_steps, args.settle_steps, args.hold_steps,
            effective_threshold, args.sensor_noise_std, noise_seed,
            verbose_neutral=(trial_idx == 0),
        )
        slip_label = result["trial_slip_label"]
        max_disp = result["max_displacement"]

        if not result["ever_contacted"]:
            no_contact_trials.append(trial_idx)

        friction_sliding, friction_torsional, friction_rolling = (
            float(v) for v in obj_actual["friction"].split())
        pad_frictions = list(obj_actual["pad_friction"].values())
        pad_friction_mean = float(np.mean(pad_frictions)) if pad_frictions else 0.0
        pad_friction_min = float(np.min(pad_frictions)) if pad_frictions else 0.0

        trial_data_path = os.path.join(args.out_dir, f"trial_{trial_idx:05d}.npz")
        np.savez_compressed(
            trial_data_path,
            feature_matrix=result["feature_matrix"],
            feature_names=np.array(result["feature_names"]),
            slip_label_series=result["slip_label_series"],
            trial_slip_label=slip_label,
            onset_step=result["onset_step"],
            max_displacement=max_disp,
            effective_threshold=effective_threshold,
            object_name=obj_spec["name"],
            object_kind=obj_spec["kind"],
            object_size_jitter=obj_actual.get("size_jitter", 1.0),
            squeeze_intensity=squeeze_intensity,
            pull_multiple=pull_multiple,
            pull_force_max=result["pull_force_max"],
            obj_mass=result["obj_mass"],
            object_friction_sliding=friction_sliding,
            object_friction_torsional=friction_torsional,
            object_friction_rolling=friction_rolling,
            pad_friction_mean=pad_friction_mean,
            pad_friction_min=pad_friction_min,
            object_yaw_deg=obj_actual["orientation_yaw_deg"],
            object_roll_deg=obj_actual["orientation_roll_deg"],
            object_pitch_deg=obj_actual["orientation_pitch_deg"],
            initial_object_pos=np.array(
                [float(v) for v in obj_actual["pos"].split()]),
            initial_object_quat=np.array(obj_actual["orientation_quat"]),
            ever_contacted=result["ever_contacted"],
            n_fingers_contacted=result["n_fingers_contacted"],
            first_contact_step=result["first_contact_step"],
            grasp_force_at_settle=result["grasp_force_at_settle"],
            sensor_noise_std=args.sensor_noise_std,
            ramp_steps=ramp_steps,
            ramp_duration_s=result["ramp_duration_s"],
            approach_velocity_rad_s=result["approach_velocity_rad_s"],
            settle_steps=args.settle_steps,
            hold_steps=args.hold_steps,
        )

        manifest.append({
            "trial_id": trial_idx,
            "object_name": obj_spec["name"],
            "object_kind": obj_spec["kind"],
            "object_spec": obj_spec,          # full spec, for exact replay
            "object_actual": obj_actual,
            "object_size_jitter": obj_actual.get("size_jitter", 1.0),
            "squeeze_intensity": squeeze_intensity,
            "pull_multiple": pull_multiple,
            "pull_force_max": result["pull_force_max"],
            "obj_mass": result["obj_mass"],
            "object_friction_sliding": friction_sliding,
            "object_friction_torsional": friction_torsional,
            "object_friction_rolling": friction_rolling,
            "pad_friction_mean": pad_friction_mean,
            "pad_friction_min": pad_friction_min,
            "object_yaw_deg": obj_actual["orientation_yaw_deg"],
            "object_roll_deg": obj_actual["orientation_roll_deg"],
            "object_pitch_deg": obj_actual["orientation_pitch_deg"],
            "ever_contacted": result["ever_contacted"],
            "n_fingers_contacted": result["n_fingers_contacted"],
            "first_contact_step": result["first_contact_step"],
            "grasp_force_at_settle": result["grasp_force_at_settle"],
            "sensor_noise_std": args.sensor_noise_std,
            "ramp_steps": ramp_steps,
            "ramp_duration_s": result["ramp_duration_s"],
            "approach_velocity_rad_s": result["approach_velocity_rad_s"],
            "slip_label": slip_label,
            "onset_step": result["onset_step"],
            "max_displacement": max_disp,
            "effective_threshold": effective_threshold,
            "data_path": trial_data_path,
        })

        if trial_idx % 25 == 0:
            print(f"trial {trial_idx}/{args.n_trials}  "
                  f"obj={obj_spec['name']}  squeeze={squeeze_intensity:.3f}  "
                  f"pull={pull_multiple:.2f}xW  slip={slip_label}  "
                  f"grasp_force={result['grasp_force_at_settle']:.2f}N  "
                  f"n_fingers={result['n_fingers_contacted']}  "
                  f"onset_step={result['onset_step']}  "
                  f"max_disp={max_disp:.4f}  thr={effective_threshold:.4f}")

    if not args.keep_scenes:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"\nPer-trial scenes kept in: {tmp_dir}")

    if no_contact_trials:
        print(f"\nWARNING: {len(no_contact_trials)} trial(s) never registered "
              f"contact on any fingertip. The object may be spawning outside "
              f"the hand's reach, or the grip may not be closing. "
              f"First few: {no_contact_trials[:10]}")

    print("\nRaw counts per object:")
    for obj in OBJECT_LIBRARY:
        rows = [m for m in manifest if m["object_name"] == obj["name"]]
        slip_count = sum(1 for m in rows if m["slip_label"] == 1)
        print(f"  {obj['name']}: {len(rows)} trials "
              f"({slip_count} slip, {len(rows) - slip_count} stable)")

    balanced, n_per_bucket, usable, dropped = balance_manifest(manifest, args.seed)

    if dropped:
        print("\nWARNING: these objects had an EMPTY slip or stable bucket and "
              "were dropped from the balanced set:")
        for name, n_stable, n_slip in dropped:
            print(f"  {name}: {n_stable} stable, {n_slip} slip")
        print("Widen --pull-range or --force-range, adjust "
              "--displacement-threshold-fraction, or run more trials, then "
              "regenerate. Do not train on a partial object set.")

    manifest_path = os.path.join(args.out_dir, "manifest_full.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if not balanced:
        print("\nNo object had both labels represented -- nothing to balance. "
              "Wrote the full manifest only.")
        print(f"Full manifest: {manifest_path}")
        return

    balanced_manifest_path = os.path.join(args.out_dir, "manifest_balanced.json")
    with open(balanced_manifest_path, "w") as f:
        json.dump(balanced, f, indent=2)

    print(f"\nBalanced dataset: {n_per_bucket} trials per (object, slip_label) "
          f"bucket x {len(usable)} objects x 2 labels = {len(balanced)} trials")
    print(f"Full manifest:     {manifest_path}")
    print(f"Balanced manifest: {balanced_manifest_path}  <- use this for PI-TCN training")


if __name__ == "__main__":
    main()
