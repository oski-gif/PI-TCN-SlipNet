"""Read-only: is force axis 2 of each fingertip sensor really the contact normal?
generate_trials.py says to verify this and uses --normal-axis 2 everywhere. Run from
"Synthetic data creation/Simulation". Part 1 needs no simulation of a policy: it closes the
hand on a few objects with a fixed grip and reads each fingertip contact's normal in the
sensor's own frame. Part 2 reads the stored samples of diagnostic_contact_physics.py."""
import json
import os
import sys
from argparse import Namespace

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import mujoco

import closed_loop_sim as cls
from rl_env import SlipGraspEnv

TIPS = ["right_T-DP_b7429e50", "right_I-FingerTipAssembly_ec49c16c", "right_M-FingerTipAssembly_34afb748",
        "right_M-FingerTipAssembly_424a8e75", "right_P-FingerTipAssembly_cd219176"]
pairs = "t i m r p".split()
ACTUATORS = [f"right_{f}-{j}_actuator" for f in pairs for j in ("mcp", "pip")]
args = Namespace(mjcf="../../third_party/orcahand_description/v2/scene_slip.xml", mesh_dir="../../assets",
                 fingertips=TIPS, actuators=ACTUATORS, object_pos=[0.085, 0.04, 0.245],
                 pad_friction_range=[0.35, 1.2], orientation_tilt_deg_max=10.0, squeeze_intensity=0.95,
                 pull_multiple=1.5, ramp_steps=500, settle_steps=300, pull_ramp_steps=500, normal_axis=2,
                 out_dir="../../Data/diag_contact_physics/scenes", max_steps=1500, checkpoint=None, in_channels=22)
env = SlipGraspEnv(args, [o["name"] for o in cls.OBJECT_LIBRARY], episode_rng_seed=0, terminate_on_formation_drop=False)

print("Part 1: fingertip contact normals in the sensor site's frame, |x|, |y|, |z| of the unit normal")
totals = np.zeros(3)
weights = 0.0
for obj in ("sphere", "capsule", "cube", "cone", "ellipsoid", "rod"):
    env.reset(options={"object_name": obj, "scene_seed": 1000})
    for _ in range(100):  # 500 physics steps: weightless closing only
        _, _, term, trunc, _ = env.step(np.array([1, 1, -1, 1, 1], dtype=float))
    m, d = env.model, env.data
    force = np.zeros(6)
    for name in TIPS:
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{name}_sensor_site")
        rot = d.site_xmat[site].reshape(3, 3)
        per_axis, fn_sum = np.zeros(3), 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            if con.efc_address < 0 or body not in (m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]):
                continue
            mujoco.mj_contactForce(m, d, i, force)
            normal_world = np.asarray(con.frame).reshape(3, 3)[0]
            normal_site = rot.T @ normal_world
            per_axis += abs(normal_site) * force[0]
            fn_sum += force[0]
        if fn_sum > 0.05:
            totals += per_axis
            weights += fn_sum
            adr = m.sensor_adr[env.force_ids[name]]
            f_vec = d.sensordata[adr:adr + 3]
            touch = env._sensor_read(env.touch_ids[name])[0]
            print(f"  {obj:<10} {name.split('_')[1][:12]:<12} {name[-4:]}  contact force {fn_sum:6.2f} N  "
                  f"normal direction |x,y,z| = {np.round(per_axis / fn_sum, 2)}  "
                  f"touch sensor {touch:6.2f}  force sensor x,y,z {np.round(f_vec, 2)}")
print(f"  force-weighted over all of these: |x,y,z| = {np.round(totals / weights, 3)}")
print("  If axis 2 were the contact normal, the last number would be near 1 and the first two near 0.")

if len(sys.argv) > 1:
    print("\nPart 2: stored samples, fingertips with touch above 0.5, default physics")
    ratios = []
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        if r.get("variant") != "default":
            continue
        for s in r["samples"]:
            ratios += [abs(n) / t for t, n in zip(s["tip_touch"], s["tip_normal"]) if t > 0.5]
    ratios = np.array(ratios)
    print(f"  samples {len(ratios)}, |normal channel| / touch sensor: median {np.median(ratios):.3f}, "
          f"10th-90th percentile {np.percentile(ratios, 10):.3f}-{np.percentile(ratios, 90):.3f}")
    print("  Near 1 would mean the channel measures the pressing force. Well below 1 means it mostly doesn't.")
env.close()
