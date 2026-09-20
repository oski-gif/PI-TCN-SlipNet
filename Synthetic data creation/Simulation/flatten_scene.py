"""
Flatten an MJCF file that uses <include> tags into one fully self-contained
XML file, using MuJoCo's own compiler.

Why this is needed: instrument_scene.py and generate_trials.py edit the
model with Python's plain xml.etree.ElementTree, which does NOT understand
MuJoCo's <include> tag (that's a MuJoCo-compiler-specific mechanism, not
standard XML). If scene_combined.xml pulls in the actual hand geometry from
a separate included file (common in robot description repos), ElementTree
never sees those bodies -- which is why instrument_scene.py reported your
fingertip names as "not found" even though inspect_model.py (which uses
mujoco.MjModel, and DOES resolve includes) found them fine.

This script loads the model through MuJoCo (resolving all includes) and
saves it back out as one flat file, which the ElementTree-based scripts can
then work with correctly.

IMPORTANT: save the output into the SAME DIRECTORY as the input file (the
default below does this), since mesh/texture paths inside the file are
relative to that directory -- moving the flattened file elsewhere can break
asset loading.

Usage:
    python flatten_scene.py \\
        --in third_party/orcahand_description/v2/scene_combined.xml \\
        --out third_party/orcahand_description/v2/scene_flat.xml
"""
import argparse
import mujoco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.in_path)
    mujoco.mj_saveLastXML(args.out_path, model)
    print(f"Flattened {args.in_path} -> {args.out_path}")
    print("Use this flattened file as the --in for instrument_scene.py from now on.")


if __name__ == "__main__":
    main()