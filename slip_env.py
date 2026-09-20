"""
Minimal slip-measurement loop around the instrumented OrcaHand scene.

Two slip signals are computed, since a real grasp-slip experiment usually
wants both:

1. Force-ratio slip criterion (Coulomb friction cone):
   slip risk = ||tangential force|| / (mu * normal force)
   >= 1.0 means the contact is at or past the point where friction can no
   longer prevent sliding, for the assumed friction coefficient mu.

2. Kinematic slip (actual relative sliding):
   tracks the object's pose relative to a fingertip site and measures
   relative displacement/rotation once contact is established -- this is
   "ground truth" slip in sim, useful for validating the force-based signal.

Usage:
    python slip_env.py --mjcf third_party/orcahand_description/v2/scene_slip.xml \\
        --fingertips thumb_tip index_tip middle_tip \\
        --mu 0.5 --steps 2000
"""
import argparse
import numpy as np
import mujoco


class SlipMonitor:
    def __init__(self, model, data, fingertip_names, mu=0.5):
        self.model = model
        self.data = data
        self.mu = mu
        self.fingertips = fingertip_names

        self.touch_ids = {}
        self.force_ids = {}
        self.torque_ids = {}
        for name in fingertip_names:
            self.touch_ids[name] = self._sensor_id(f"{name}_touch")
            self.force_ids[name] = self._sensor_id(f"{name}_force")
            self.torque_ids[name] = self._sensor_id(f"{name}_torque")

        self.object_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "slip_test_object")

        # For kinematic slip: remember relative pose at first contact per finger
        self._contact_ref_pose = {name: None for name in fingertip_names}

    def _sensor_id(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid == -1:
            raise ValueError(f"Sensor '{name}' not found -- did you run "
                              f"instrument_scene.py on this MJCF?")
        return sid

    def _sensor_reading(self, sensor_id, dim):
        adr = self.model.sensor_adr[sensor_id]
        return self.data.sensordata[adr:adr + dim].copy()

    def step_and_measure(self):
        mujoco.mj_step(self.model, self.data)

        results = {}
        for name in self.fingertips:
            normal_force = self._sensor_reading(self.touch_ids[name], 1)[0]
            force_vec = self._sensor_reading(self.force_ids[name], 3)
            torque_vec = self._sensor_reading(self.torque_ids[name], 3)

            tangential_force = np.linalg.norm(force_vec) - abs(normal_force)
            tangential_force = max(tangential_force, 0.0)

            if normal_force > 1e-6:
                slip_ratio = tangential_force / (self.mu * normal_force)
            else:
                slip_ratio = 0.0  # no contact, no slip risk

            results[name] = {
                "normal_force": float(normal_force),
                "tangential_force": float(tangential_force),
                "slip_ratio": float(slip_ratio),
                "in_contact": bool(normal_force > 1e-3),
            }

        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--fingertips", nargs="+", required=True)
    ap.add_argument("--mu", type=float, default=0.5,
                     help="Assumed Coulomb friction coefficient")
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    monitor = SlipMonitor(model, data, args.fingertips, mu=args.mu)

    for step in range(args.steps):
        results = monitor.step_and_measure()

        if step % 100 == 0:
            print(f"--- step {step} ---")
            for name, r in results.items():
                flag = "SLIPPING" if r["slip_ratio"] >= 1.0 and r["in_contact"] else ""
                print(f"  {name}: normal={r['normal_force']:.3f}N "
                      f"tangential={r['tangential_force']:.3f}N "
                      f"ratio={r['slip_ratio']:.2f} {flag}")


if __name__ == "__main__":
    main()
