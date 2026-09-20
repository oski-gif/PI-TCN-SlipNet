"""
Inspect an MJCF model and print every body, geom, joint, and actuator name.
"""
import sys
import mujoco


def main(mjcf_path: str):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    print(">>> RUNNING UPDATED SCRIPT v2 <<<")

    print(f"Loaded model: {mjcf_path}")
    print(f"  nbody={model.nbody}  ngeom={model.ngeom}  njnt={model.njnt}  nu={model.nu}\n")

    print("=== Bodies ===")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            print(f"  [{i:3d}] {name}")

    print("\n=== Geoms ===")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name:
            print(f"  [{i:3d}] {name}")

    print("\n=== Joints ===")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            print(f"  [{i:3d}] {name}")

    print("\n=== Actuators ===")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        ctrlrange = model.actuator_ctrlrange[i]
        if name:
            print(f"  [{i:3d}] {name}   ctrlrange={ctrlrange}")
    if model.nu == 0:
        print("  (no actuators found -- model may be position/torque-driven "
              "via a different mechanism, check the repo's control docs)")

    keywords = ("tip", "distal", "fingertip", "phalanx3", "dp")
    guesses = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and any(k in name.lower() for k in keywords):
            guesses.append(name)

    if guesses:
        print("\n=== Likely fingertip bodies (heuristic guess, verify visually) ===")
        for g in guesses:
            print(f"  {g}")
    else:
        print("\nNo obvious fingertip-name matches found -- inspect the list "
              "above manually and pick the distal-most body per finger.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])