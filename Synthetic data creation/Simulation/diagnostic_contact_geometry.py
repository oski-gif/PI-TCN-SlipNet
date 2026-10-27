"""
BOUNDED DIAGNOSTIC. Investigates WHY mesh-kind objects (cone, square_pyramid,
triangular_prism, hexagonal_prism) fail the closed-loop hold far more often
than primitive/compound objects, even at the same char_length/threshold.
Two competing hypotheses:
  (a) simulation artifact -- MuJoCo's mesh collision gives fewer/less stable
      contact points against the fingertip geoms than analytic primitives do
  (b) genuine geometry -- angular/pointed shapes (cone apex, pyramid edges)
      give the 5-fingered hand less surface to cradle against, regardless
      of simulation fidelity
This script builds the SAME scene build_trial_xml/build_scene would, steps
through ramp+settle exactly like run_trial() does, then inspects
data.ncon / data.contact at the end of settle (grip fully closed, object
only under gravity, no pull yet) for cone (mesh, fails) vs sphere
(primitive, succeeds) -- same char_length=0.04, same seed. Read-only
inspection of mjData; never writes to closed_loop_sim.py.
"""
import mujoco
import numpy as np
import closed_loop_sim as cls


def inspect_object(args, object_name, seed):
    scene_path, obj_spec, actual = cls.build_scene(
        args.mjcf, args.mesh_dir, args.fingertips, object_name,
        args.object_pos, tuple(args.pad_friction_range),
        args.orientation_tilt_deg_max, seed, args.out_dir,
    )
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    obj_body_id = cls._require_id(model, mujoco.mjtObj.mjOBJ_BODY,
                                   cls.OBJECT_BODY_NAME, "Object body")
    driven = {}
    for a in args.actuators:
        aid = cls._require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a, "Actuator")
        lo, hi = model.actuator_ctrlrange[aid]
        driven[aid] = [float(lo), float(hi), float(lo) + args.squeeze_intensity * float(hi - lo)]
    for aid in range(model.nu):
        if aid not in driven:
            lo, hi = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = float(np.clip(0.0, lo, hi))

    model.body_gravcomp[obj_body_id] = 1.0
    ramp_steps, settle_steps = args.ramp_steps, args.settle_steps

    for step in range(ramp_steps + settle_steps):
        if step == ramp_steps:
            model.body_gravcomp[obj_body_id] = 0.0
        if step < ramp_steps:
            frac = (step + 1) / max(ramp_steps, 1)
        else:
            frac = 1.0
        for aid, (lo, hi, target) in driven.items():
            data.ctrl[aid] = lo + frac * (target - lo)
        mujoco.mj_step(model, data)

    obj_geom_ids = {g for g in range(model.ngeom) if model.geom_bodyid[g] == obj_body_id}
    print(f"\n=== {object_name} (seed={seed}) ===")
    print(f"ncon={data.ncon}")
    n_obj_contacts = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in obj_geom_ids or c.geom2 in obj_geom_ids:
            n_obj_contacts += 1
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            print(f"  contact {i}: {g1} <-> {g2}  dist={c.dist:.5f}  pos={np.round(c.pos, 4)}")
    print(f"-> {n_obj_contacts} contacts involving the object (of {data.ncon} total)")
    return n_obj_contacts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    cls.add_common_args(ap, max_steps_default=2500)
    args = ap.parse_args()
    args.squeeze_intensity = 0.95

    for obj_name in ("sphere", "cone", "cube", "hexagonal_prism"):
        inspect_object(args, obj_name, seed=0)


if __name__ == "__main__":
    main()
