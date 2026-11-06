"""Is Data/benchmark_scenes/scene_slip_elliptic.xml the same thing as setting the option in memory?

Not yet run to completion: the Nov 4 run was cut off by the crash. Run it before training on that file.

Fresh-seed and dev-set runs set cone and impratio on the compiled model. Training can't do that
without a code change, so it uses a scene file instead. This builds each object through the real
scene builder from both scene files, compiles them the way rl_env.py does, and compares:
  1. the solver options against the base scene, field by field
  2. the compiled models, byte for byte
  3. 800 physics steps with the fingers closing, qpos and qvel bit for bit
It also runs the base scene under default settings, to show the comparison can tell them apart.

Needs only mujoco and numpy, so it's light enough to run beside other jobs. Writes temporary scene
files to --out-dir and changes nothing else.

Run from "Synthetic data creation/Simulation":
    ../../venv/Scripts/python.exe ../../src/analysis/nov3_drop_analysis/elliptic_scene_check.py --out-dir <folder>
"""
import argparse
import os
import random
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.getcwd())
import generate_trials as gt  # noqa: E402

BASE = "../../third_party/orcahand_description/v2/scene_slip.xml"
NEW = "../../Data/benchmark_scenes/scene_slip_elliptic.xml"
MESH_DIR = "../../assets"
FINGERTIPS = ["right_T-DP_b7429e50", "right_I-FingerTipAssembly_ec49c16c",
              "right_M-FingerTipAssembly_34afb748", "right_M-FingerTipAssembly_424a8e75",
              "right_P-FingerTipAssembly_cd219176"]
OBJECT_POS = [0.085, 0.04, 0.245]
PAD_RANGE = (0.35, 1.2)
TILT = 10.0
STEPS = 800


def compiled(base, spec, seed, out_dir, tag):
    path = os.path.join(out_dir, f"{tag}_{spec['name']}_{seed}.xml")
    gt.build_trial_xml(base, path, spec, MESH_DIR, random.Random(seed), OBJECT_POS, FINGERTIPS,
                       PAD_RANGE, TILT)
    with open(path, "r", encoding="utf-8") as f:
        return mujoco.MjModel.from_xml_string(f.read())  # as rl_env._compile_model does


def model_bytes(model):
    buf = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return buf.tobytes()


def trajectory(model):
    data = mujoco.MjData(model)
    hi = model.actuator_ctrlrange[:, 1] if model.nu else np.zeros(0)
    trace = []
    for step in range(STEPS):
        data.ctrl[:] = hi * min(1.0, step / 400.0)
        mujoco.mj_step(model, data)
        if step % 100 == 99:
            trace.append(data.qpos.tobytes() + data.qvel.tobytes())
    return b"".join(trace), int(data.warning.number.sum())


def option_fields(opt):
    names = [n for n in dir(opt) if not n.startswith("_")]
    out = {}
    for n in names:
        v = getattr(opt, n)
        if callable(v):
            continue
        out[n] = np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 2000])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    n = bytes_same = traj_same = differs_from_default = warnings = 0
    bad_options = []
    for spec in gt.OBJECT_LIBRARY:
        for seed in args.seeds:
            default_model = compiled(BASE, spec, seed, args.out_dir, "base")
            in_memory = compiled(BASE, spec, seed, args.out_dir, "base")
            in_memory.opt.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
            in_memory.opt.impratio = 10.0
            scene_file = compiled(NEW, spec, seed, args.out_dir, "new")

            a, b = option_fields(default_model.opt), option_fields(scene_file.opt)
            changed = sorted(k for k in a if a[k] != b[k])
            if changed != ["cone", "impratio"] or scene_file.opt.impratio != 10.0 or \
                    scene_file.opt.cone != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
                bad_options.append((spec["name"], seed, changed))

            n += 1
            bytes_same += model_bytes(in_memory) == model_bytes(scene_file)
            t_mem, w_mem = trajectory(in_memory)
            t_new, w_new = trajectory(scene_file)
            t_def, _ = trajectory(default_model)
            traj_same += t_mem == t_new
            differs_from_default += t_new != t_def
            warnings += w_mem + w_new

    print(f"scenes compared: {n} ({len(gt.OBJECT_LIBRARY)} objects x seeds {args.seeds})")
    print(f"solver options: only cone and impratio differ from the base scene on "
          f"{n - len(bad_options)} of {n}")
    print(f"compiled models byte-identical to the in-memory setting: {bytes_same} of {n}")
    print(f"{STEPS}-step qpos and qvel bit-identical to the in-memory setting: {traj_same} of {n}")
    print(f"trajectory differs from default settings (the check can tell them apart): "
          f"{differs_from_default} of {n}")
    print(f"solver warnings: {warnings}")
    for item in bad_options[:5]:
        print("  unexpected options:", item)
    ok = (not bad_options) and traj_same == n and warnings == 0 and differs_from_default > 0
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
