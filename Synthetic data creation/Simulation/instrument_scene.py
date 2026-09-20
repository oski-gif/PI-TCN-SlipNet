"""
Instrument an OrcaHand MJCF scene for slip measurement.

Adds, per fingertip body you specify:
  - a <site> at the body origin (sensors attach to sites, not bodies directly)
  - a <touch> sensor (contact normal force magnitude)
  - a <force> and <torque> sensor (full contact wrench, for tangential/normal
    force ratio -- this is what you'll use for a Coulomb-friction slip
    criterion: slip begins when tangential force / normal force exceeds mu)

Also adds a free-floating graspable object (box, edit size/mass as needed)
positioned above the palm so you have something to actually grasp and slip.

This edits the XML in place (via ElementTree) rather than hardcoding fingertip
names, since I haven't been able to fetch the real OrcaHand MJCF myself to
confirm exact naming -- run inspect_model.py first and pass in the real names.

Usage:
    python instrument_scene.py \\
        --in third_party/orcahand_description/v2/scene_combined.xml \\
        --out third_party/orcahand_description/v2/scene_slip.xml \\
        --fingertips thumb_tip index_tip middle_tip ring_tip pinky_tip
"""
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import mujoco


# Margin added to each half-extent of the touch sensor's box zone, so contacts
# right at the mesh surface (solver margin, numerical slop) aren't missed by
# an exact-fit box.
TOUCH_ZONE_MARGIN = 0.001


def collision_mesh_aabb_in_body_frame(model, body_id):
    """FIX: MuJoCo's <touch> sensor only counts contacts whose contact point
    falls inside the volume of the sensor's site -- it is a spatial zone, not
    just an attachment point. The original site was a 2mm sphere at the
    body's local ORIGIN (0,0,0), which for these fingertip bodies is not on
    the mesh surface at all (confirmed: for right_T-DP the collision mesh
    spans roughly z in [-0.008, 0.0305] in its own body frame -- the origin
    sits near one end, nowhere near the pad). The result was a touch sensor
    that read exactly 0.0 even while the force/torque sensors on the SAME
    site correctly reported real contact forces of several newtons, because
    those aren't zone-gated the same way.

    This computes the union bounding box (in the body's own frame) of every
    COLLIDABLE geom (contype != 0) on the body, straight from the mesh
    vertex data, so the sensor site can be sized/positioned to actually
    enclose the fingertip's contact surface instead of guessing a fixed
    site radius.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    found = False
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id or model.geom_contype[g] == 0:
            continue
        found = True
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[g]
            vadr = model.mesh_vertadr[mesh_id]
            vnum = model.mesh_vertnum[mesh_id]
            verts = model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            pts = verts @ rot.reshape(3, 3).T + model.geom_pos[g]
        else:
            # Non-mesh collision geom: approximate with its own local
            # extent (geom_rbound is the bounding-sphere radius) centered
            # at its offset from the body origin.
            r = float(model.geom_rbound[g])
            pts = model.geom_pos[g] + np.array(
                [[-r, -r, -r], [r, r, r]])
        lo = np.minimum(lo, pts.min(axis=0))
        hi = np.maximum(hi, pts.max(axis=0))

    if not found:
        return None
    return lo, hi


def add_sensor_site(body_elem, site_name, compiled_model, body_name):
    """Add a site sized to actually enclose the fingertip's collision
    geometry, so the <touch> sensor's zone overlaps the real contact
    surface (see collision_mesh_aabb_in_body_frame)."""
    existing = body_elem.find(f"./site[@name='{site_name}']")
    if existing is not None:
        return

    body_id = mujoco.mj_name2id(compiled_model, mujoco.mjtObj.mjOBJ_BODY,
                                body_name)
    aabb = collision_mesh_aabb_in_body_frame(compiled_model, body_id)

    site = ET.SubElement(body_elem, "site")
    site.set("name", site_name)
    site.set("rgba", "1 0 0 0.5")

    if aabb is None:
        # No collidable geom found on this body -- fall back to the old
        # small marker rather than silently sizing a box from nothing.
        print(f"  WARNING: no collision geom found on body '{body_name}' -- "
              f"falling back to a 2mm marker site. The touch sensor on this "
              f"site will likely never register real contact.")
        site.set("size", "0.002")
        return

    lo, hi = aabb
    center = (lo + hi) / 2
    halfext = (hi - lo) / 2 + TOUCH_ZONE_MARGIN
    site.set("type", "box")
    site.set("pos", " ".join(f"{v:.5f}" for v in center))
    site.set("size", " ".join(f"{v:.5f}" for v in halfext))


def find_body_by_name(root, name):
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--fingertips", nargs="+", required=True,
                     help="Exact body names for fingertip bodies "
                          "(get these from inspect_model.py)")
    ap.add_argument("--object-size", default="0.02 0.02 0.02",
                     help="Half-extents of the graspable box (m)")
    ap.add_argument("--object-mass", type=float, default=0.05,
                     help="Mass of the graspable object (kg)")
    ap.add_argument("--object-friction", default="0.5 0.005 0.0001",
                     help="Sliding/torsional/rolling friction of the object")
    ap.add_argument("--object-pos", default="0 0 0.15",
                     help="Initial position of the object, in palm frame-ish coords")
    args = ap.parse_args()

    tree = ET.parse(args.in_path)
    root = tree.getroot()

    # Compiled read-only, purely to get real mesh/geom geometry (vertex
    # positions, bounding boxes) for sizing each touch sensor's zone -- the
    # actual output file is still produced by editing the ElementTree above.
    compiled_model = mujoco.MjModel.from_xml_path(args.in_path)

    sensor_elem = root.find("sensor")
    if sensor_elem is None:
        sensor_elem = ET.SubElement(root, "sensor")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("No <worldbody> found -- is this a valid MJCF?")

    missing = []
    for tip_name in args.fingertips:
        body = find_body_by_name(root, tip_name)
        if body is None:
            missing.append(tip_name)
            continue

        site_name = f"{tip_name}_sensor_site"
        add_sensor_site(body, site_name, compiled_model, tip_name)

        touch = ET.SubElement(sensor_elem, "touch")
        touch.set("name", f"{tip_name}_touch")
        touch.set("site", site_name)

        force = ET.SubElement(sensor_elem, "force")
        force.set("name", f"{tip_name}_force")
        force.set("site", site_name)

        torque = ET.SubElement(sensor_elem, "torque")
        torque.set("name", f"{tip_name}_torque")
        torque.set("site", site_name)

    if missing:
        print("WARNING: these fingertip names were not found in the model "
              f"and were skipped: {missing}")
        print("Run inspect_model.py on your scene file to get exact names.")

    # Add a free-floating graspable object.
    obj_body = ET.SubElement(worldbody, "body")
    obj_body.set("name", "slip_test_object")
    obj_body.set("pos", args.object_pos)

    joint = ET.SubElement(obj_body, "joint")
    joint.set("type", "free")
    joint.set("name", "slip_test_object_freejoint")

    geom = ET.SubElement(obj_body, "geom")
    geom.set("type", "box")
    geom.set("name", "slip_test_object_geom")
    geom.set("size", args.object_size)
    geom.set("mass", str(args.object_mass))
    geom.set("friction", args.object_friction)
    geom.set("rgba", "0.2 0.6 0.9 1")

    tree.write(args.out_path, encoding="UTF-8", xml_declaration=True)
    print(f"Wrote instrumented scene to {args.out_path}")
    if missing:
        print("NOTE: some sensors were skipped -- see warning above.")


if __name__ == "__main__":
    main()