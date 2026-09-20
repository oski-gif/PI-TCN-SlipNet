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


def add_sensor_site(body_elem, site_name):
    """Add a small site at the body's local origin for sensor attachment."""
    existing = body_elem.find(f"./site[@name='{site_name}']")
    if existing is not None:
        return
    site = ET.SubElement(body_elem, "site")
    site.set("name", site_name)
    site.set("size", "0.002")
    site.set("rgba", "1 0 0 0.5")


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
        add_sensor_site(body, site_name)

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
