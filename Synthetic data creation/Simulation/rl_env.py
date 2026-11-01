"""
Gymnasium environment wrapping the existing closed-loop grasp simulation,
so an RL policy can replace BOTH decide_action()'s hand-tuned threshold
rule AND the scripted ramp/settle grasp-formation schedule. Claude-written
infrastructure -- same model-vs-closed-loop split as closed_loop_sim.py and
batch_closed_loop_trials.py. Reuses the exact scene construction, sensor
wiring, and threshold/slipped definition those files use (build_scene,
OBJECT_LIBRARY, _require_id, the 0.15*char_length threshold).

SCOPE (v2, expanded from the v1 "correction-only" version after the
squeeze_intensity sweep / fixed-step / hysteresis diagnostics all plateaued
at ~40%): the hand-tuned policy and v1 of this env both only corrected a
grip that had ALREADY formed via a fixed, scripted squeeze schedule
identical for every object. The per-object failure pattern (several
objects 0/3 regardless of correction policy) looks like it's at least
partly a "one fixed approach doesn't fit every shape" problem, which no
amount of post-grasp correction can fix. This version hands the policy the
WHOLE episode -- formation, settle, AND pull -- so it can learn
object-adaptive grasp formation, not just reactive correction.

Episode structure (same total step budget as the hand-tuned pipeline, now
RL-controlled instead of scripted for the first two phases):
  phase 0 (0..ramp_steps):            hand starts OPEN, object weightless
                                       (gravcomp=1), RL closes the fingers
  phase 1 (ramp_steps..+settle_steps): real gravity re-enabled, RL keeps
                                       adjusting the grip
  phase 2 (settle_end..+max_steps):   pull engages and ramps up, same as
                                       the v1 env's only phase

Action: 5 continuous values in [-1, 1], one per finger -- a PERSISTENT
delta on that finger's ctrlrange fraction (ratcheting, like
decide_action()'s `target + step`), active across all three phases. The
hand starts fully open (fraction=0 for every finger); there is no scripted
closing schedule anymore.

Observation: a short window of [per-fingertip touch/normal/tangential/
torque_mag, pull_force, phase, predicted slip probability] -- no raw
object position/velocity/displacement (matches LEAKAGE_FEATURES'
discipline: no external object tracking, only tactile sensing + the
model's own inference + the two scalars (pull_force, phase) the
slip-detection model itself was already trained on).

Reward: phase 0/1 (formation) rewards fraction-of-fingertips-in-contact
and penalizes object speed (discourage flinging the object around) and a
large one-time penalty if the object is dropped outright (judged by a
large fall from its spawn height) -- there's no settled reference position
yet, so there's nothing displacement-based to reward here. Phase 2 reuses
the v1 env's reward exactly (dense displacement penalty, slip penalty,
hold bonus). All of these are first-pass constants (CONTACT_COEF,
INSTABILITY_COEF, DROP_THRESHOLD, ...) -- expect to retune them once real
training curves are in hand, same as any new reward design.

GRIP_FORCE_COEF (added after a design review, before any real training run
existed to confirm or refute it): without it, phase 0/1 only rewards
TOUCHING (contact_frac, a binary per-finger threshold) with no incentive to
actually squeeze -- the only signal that a grip needs real force behind it
arrives via the phase-2 slip/displacement penalty, ~100+ macro-steps later.
That's a classic delayed-credit-assignment gap: a policy could satisfy the
formation reward by lightly touching every finger without building a grip
that has any chance of surviving the pull, and only find out it was wrong
once it's too late in the episode to fix. The fix is a dense per-step
reward on summed fingertip normal force (same quantity as
grasp_force_at_settle elsewhere in this codebase), normalized by the
object's own weight (so it means something physical -- "gripping with N
multiples of this object's weight" -- rather than an arbitrary Newton
scale that would mean something different for a 0.02kg vs 0.13kg object)
and capped at GRIP_FORCE_CAP_MULTIPLE so the policy isn't rewarded for
squeezing harder forever once it already has a comfortable margin.
Checked empirically via smoke_test_rl_env.py before settling on these
constants: an early GRIP_FORCE_COEF=0.5/cap=3.0 pass made per-episode
total reward jump 5-40x (from the low-tens-negative range the v2 reward
previously produced, under random actions, to the hundreds) because any
real finger contact on an object this light clears several multiples of
its own weight almost immediately, saturating the term on nearly every
substep -- which would have made this term dominate contact_frac's
(max 1.0/step) and instability's contribution outright, defeating the
point of adding a secondary signal rather than replacing the existing
one. GRIP_FORCE_COEF=0.15/cap=2.0 keeps its max per-substep contribution
(0.3) below CONTACT_COEF's (1.0) while still being nonzero and present
every step a finger is pressing -- a deliberate, documented first pass,
not a tuned result from a real training curve (there isn't one yet).

One RL step = ACTION_REPEAT physics steps (action held constant across
them). A macro-step whose 5 physics substeps straddle a phase boundary
(at most 2 such macro-steps per episode, out of ~450+) is scored using
only its LAST substep's phase -- a known, negligible edge case, not worth
the bookkeeping to split it exactly.

MIN_FINGER_FRACTION_COEF (added 2026-10-31, after the first full 300k-step
training run and its real evaluation): `diagnostic_formation_drop.py` found
the trained policy had learned to NEVER close one specific finger (the
middle finger, actuator pair index 2) AT ALL -- fraction locked at exactly
0.0 -- on every single object tested, including ones it holds perfectly
(sphere, cone, capsule, dumbbell), not just the 4 it drops during formation
(cube, rectangular_prism, flat_disk, l_block). A universal 4-finger
strategy is enough margin for curved/pointed/thin objects but not for
flat/boxy/asymmetric ones, which is exactly the failure pattern observed.
Root cause, inferred from the reward terms above rather than proven: both
`CONTACT_COEF*contact_frac` and `GRIP_FORCE_COEF*grip_force_frac` are
AGGREGATE/MEAN-style terms across all 5 fingers -- a policy can max out
`grip_force_frac` (it saturates at GRIP_FORCE_CAP_MULTIPLE) using only 4
fingers' force, and `contact_frac` only loses 0.2/step (1/5 of
CONTACT_COEF) for leaving one finger idle, not enough to reliably escape
the local optimum of "4 fingers is already working, stop there" once
training settles into it. Neither term can express "finger N specifically
is idle" -- only "what fraction/how much force in aggregate."
MIN_FINGER_FRACTION_COEF fixes this directly rather than by reweighting
the existing aggregate terms (a less targeted fix the same investigation
considered and rejected -- raising CONTACT_COEF's weight is still just a
steeper version of the same mean-based incentive, which doesn't change
WHICH 4-of-5 fingers the policy locks onto, it just makes locking onto any
4 slightly more/less rewarding than before): it rewards `min(self.fraction)`
-- the WORST-engaged finger's own commanded ctrlrange fraction, not an
average. This is a maximin-style incentive: no amount of over-closing the
other 4 fingers can compensate for leaving a 5th one idle, since the term
is exactly 0 as long as ANY single finger stays at 0, regardless of what
the rest are doing. Deliberately rewards the COMMANDED fraction, not
measured touch -- the finding was that the policy never even attempted to
close that finger, not that it tried and failed to make contact, so the
fix targets the actual observed symptom (no attempt) rather than a
stricter, measured-contact version that could penalize objects where a
given finger's contact is genuinely geometry-dependent and not always
achievable regardless of effort.

MIN_FINGER_FRACTION_COEF, escalated 0.5 -> 3.0 (2026-10-31, same day):
the 0.5 first pass was sanity-checked on a 15,000-step CONTINUATION from
the already-converged 300k-step checkpoint -- the middle finger's
fraction did not move AT ALL (still exactly 0.0 on every object checked,
unchanged from before the fix). At 0.5, this term's max per-substep
contribution (0.5) was smaller than CONTACT_COEF's (1.0) -- not dominant
enough to outweigh 300k steps of reinforcement behind the existing
4-finger habit, if that's really why it didn't move (a continuation
fighting an entrenched policy is a weaker test than training from
scratch, so the 0.5 result doesn't by itself prove the term is too weak
in general, only that it wasn't strong enough to win against that
specific, heavily-trained incumbent). 3.0 makes this term's max
contribution (3.0) the clearly dominant term in the formation-phase
reward -- 3x CONTACT_COEF's own max and 10x GRIP_FORCE_COEF's -- a
deliberately aggressive escalation rather than a small nudge, on the
theory that if a gentle version can't move an entrenched habit, the fix
being tested is whether a strong enough one can, not whether a slightly
stronger one can. Not yet re-validated against a real training curve at
this new magnitude -- next step after this escalation is itself a fresh
full retrain regardless of what a short continuation check shows (a
continuation was already shown to be a weak/ambiguous test either way),
so unlike GRIP_FORCE_COEF's original tuning, this value is not expected
to be re-adjusted further before that full run -- see results.md for the
decision trail.
"""
import os
import random
import time

import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

import closed_loop_sim as cls
from generate_trials import OBJECT_LIBRARY, OBJECT_BODY_NAME, _require_id
from tcn import LEAKAGE_FEATURES

# TEMP DIAGNOSTIC (2026-10-30) -- see train_rl_policy.py's matching comment.
# Opens, writes, and closes the file on every call (no held-open handle,
# no buffering) specifically so the LAST checkpoint before a hard process
# death is never lost -- a buffered/batched write could still be sitting in
# memory, never flushed to disk, at the exact moment a worker dies. Remove
# once this investigation concludes.
_DIAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")


def _diag_log(worker_idx, msg):
    if worker_idx is None:
        return
    try:
        path = os.path.join(_DIAG_DIR, f"_worker_diag_{worker_idx}.log")
        with open(path, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} [worker {worker_idx}] {msg}\n")
    except OSError:
        pass


ACTION_REPEAT = 5
OBS_WINDOW = 5
MAX_DELTA_PER_STEP = 0.08  # max change in ctrlrange fraction per RL step, per finger

# Phase 2 (pull/monitor) reward -- unchanged from the v1 (correction-only) env.
ALPHA_DISPLACEMENT = 50.0
EFFORT_COEF = 0.02
SLIP_PENALTY = 20.0
HOLD_BONUS = 20.0

# Phase 0/1 (formation/settle) reward -- new in v2, first-pass constants.
CONTACT_TOUCH_THRESHOLD = 0.01
CONTACT_COEF = 1.0
INSTABILITY_COEF = 0.1
DROP_THRESHOLD = 0.05  # meters fallen from spawn height counted as a drop
DROP_PENALTY = 15.0
GRIP_FORCE_COEF = 0.15  # see module docstring's GRIP_FORCE_COEF note
GRIP_FORCE_CAP_MULTIPLE = 2.0  # saturate once grip exceeds 2x the object's own weight
MIN_FINGER_FRACTION_COEF = 3.0  # see module docstring's MIN_FINGER_FRACTION_COEF note


class SlipGraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, args, object_names, device="cpu", force_cpu_model=True,
                episode_rng_seed=None, build_lock=None, worker_idx=None):
        super().__init__()
        self.args = args
        self._worker_idx = worker_idx  # TEMP DIAGNOSTIC, see _diag_log above
        _diag_log(worker_idx, "SlipGraspEnv.__init__ starting")
        # Several workers calling mujoco.MjModel.from_xml_path() at the
        # same instant reliably crashes with "MemoryError: bad allocation"
        # / "engine error: Could not allocate memory" -- looks like
        # contention inside MuJoCo's own one-time model-compilation step,
        # not a real memory shortage (plenty of free RAM every time this
        # was checked). A fixed startup stagger (sleep proportional to
        # worker index) was tried first and wasn't reliable -- SubprocVecEnv
        # calls reset() on every worker at once regardless of how staggered
        # their startup was, and even a ~5s offset between workers 0 and 7
        # still collided. A real cross-process lock, passed in from the
        # training script and acquired around from_xml_path() specifically,
        # fixes this regardless of timing instead of guessing a safe delay.
        self._build_lock = build_lock
        self.object_names = list(object_names)
        self.n_fingers = len(args.actuators) // 2
        assert len(args.actuators) == 2 * self.n_fingers, (
            "actuators must be given as consecutive (mcp, pip) pairs per finger")
        self.finger_pairs = [args.actuators[2 * i:2 * i + 2] for i in range(self.n_fingers)]

        if cls._model is None and args.checkpoint:
            _diag_log(worker_idx, "about to load_model (TCN checkpoint)")
            cls.load_model(args.checkpoint, args.in_channels,
                           device="cpu" if force_cpu_model else None)
            _diag_log(worker_idx, "load_model complete")

        tactile_dim = 4 * len(args.fingertips)
        self._tactile_dim = tactile_dim
        self._row_dim = tactile_dim + 3  # + pull_force, phase, slip_probability
        obs_dim = OBS_WINDOW * self._row_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_fingers,), dtype=np.float32)

        self.model = None
        self.data = None
        self._episode_rng = random.Random(episode_rng_seed)
        _diag_log(worker_idx, "SlipGraspEnv.__init__ complete")

    def _compile_model(self, scene_path):
        """mujoco.MjModel.from_xml_path() raising "ValueError: Error:
        engine error: Could not allocate memory" turned out to be
        deterministic per scene file AND not fixed by retrying with fresh
        random (object, seed) pairs either -- ruling out both a resource
        race and a specific bad scene. All three failures happened at
        roughly the same WALL-CLOCK time (~100-150s into an 8-worker run)
        across otherwise-different runs, which points at something
        external and time-based. Both Windows Defender (MsMpEng) and
        Malwarebytes are active on this machine and real-time-scan newly
        written files -- 8 workers rapidly writing+reading fresh scene XML
        files into the same shared out_dir is exactly the pattern that
        triggers real-time AV scanning, and a file briefly locked for
        scanning mid-read could plausibly surface as a generic MuJoCo
        allocation error. Loading from an in-memory XML string instead of
        re-reading the path removes that disk round-trip entirely rather
        than trying to out-wait or out-guess the AV engine."""
        _diag_log(self._worker_idx, f"_compile_model: read scene file {scene_path}")
        with open(scene_path, "r", encoding="utf-8") as f:
            xml_string = f.read()
        if self._build_lock is not None:
            _diag_log(self._worker_idx, "about to acquire build_lock")
            with self._build_lock:
                _diag_log(self._worker_idx, "lock acquired, compiling model")
                model = mujoco.MjModel.from_xml_string(xml_string)
            _diag_log(self._worker_idx, "model compiled, lock released")
            return model
        _diag_log(self._worker_idx, "no build_lock, compiling model directly")
        model = mujoco.MjModel.from_xml_string(xml_string)
        _diag_log(self._worker_idx, "model compiled")
        return model

    def _build(self, object_name, seed):
        args = self.args
        _diag_log(self._worker_idx, f"_build: about to build_scene XML for {object_name}")
        scene_path, obj_spec, actual = cls.build_scene(
            args.mjcf, args.mesh_dir, args.fingertips, object_name,
            args.object_pos, tuple(args.pad_friction_range),
            args.orientation_tilt_deg_max, seed, args.out_dir,
        )
        _diag_log(self._worker_idx, f"_build: scene XML written to {scene_path}")
        self.obj_spec = obj_spec
        model = self._compile_model(scene_path)
        _diag_log(self._worker_idx, "_build: MjData(model) next")
        data = mujoco.MjData(model)
        _diag_log(self._worker_idx, "_build: MjData constructed, looking up sensor/actuator IDs")

        self.touch_ids = {n: _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                          f"{n}_touch", "Touch sensor")
                          for n in args.fingertips}
        self.force_ids = {n: _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                          f"{n}_force", "Force sensor")
                           for n in args.fingertips}
        self.torque_ids = {n: _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                                           f"{n}_torque", "Torque sensor")
                            for n in args.fingertips}
        self.obj_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY,
                                        OBJECT_BODY_NAME, "Object body")

        driven = {}
        finger_actuator_ids = []
        for pair in self.finger_pairs:
            ids = []
            for a in pair:
                aid = _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a, "Actuator")
                lo, hi = model.actuator_ctrlrange[aid]
                driven[aid] = [float(lo), float(hi)]
                ids.append(aid)
            finger_actuator_ids.append(ids)
        self.finger_actuator_ids = finger_actuator_ids
        for aid in range(model.nu):
            if aid not in driven:
                lo, hi = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = float(np.clip(0.0, lo, hi))

        # Weightless during formation (phase 0), same as generate_trials.py's
        # approach-phase fix -- re-enabled once formation's step budget ends.
        model.body_gravcomp[self.obj_body_id] = 1.0

        if os.path.exists(scene_path):
            os.remove(scene_path)

        self.model, self.data, self.driven = model, data, driven

        obj_mass = float(model.body_mass[self.obj_body_id])
        g = float(np.linalg.norm(model.opt.gravity))
        self.pull_force_max = args.pull_multiple * obj_mass * g
        self._obj_weight = obj_mass * g

        # Instance state, not module globals on `cls` -- these are per-
        # episode (friction/mass differ scene to scene), and a module global
        # here would silently corrupt results the moment two SlipGraspEnv
        # instances are ever live in the same process (predict_slip() takes
        # these as explicit parameters now, see closed_loop_sim.py).
        feature_names = cls.build_feature_names(args.fingertips)
        self._keep_idx = [i for i, n in enumerate(feature_names) if n not in LEAKAGE_FEATURES]
        if cls._model is not None and args.in_channels in (23, 25):
            pad_frictions = list(actual["pad_friction"].values())
            pad_friction_mean = float(np.mean(pad_frictions)) if pad_frictions else 0.0
            friction_sliding = float(actual["friction"].split()[0])
            mu_combined = max(friction_sliding, pad_friction_mean)
            self._extra_scalars = ([mu_combined] if args.in_channels == 23
                                    else [mu_combined, obj_mass])
        else:
            self._extra_scalars = None

        self.tangent_axes = [i for i in range(3) if i != args.normal_axis]
        _diag_log(self._worker_idx, "_build: complete")

    def _sensor_read(self, sid):
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return self.data.sensordata[adr:adr + dim].copy()

    def _read_raw_features(self, phase, pull):
        args = self.args
        row = []
        for name in args.fingertips:
            touch = float(self._sensor_read(self.touch_ids[name])[0])
            force_vec = self._sensor_read(self.force_ids[name])
            torque_vec = self._sensor_read(self.torque_ids[name])
            normal = float(force_vec[args.normal_axis])
            tangential = float(np.linalg.norm(
                [force_vec[self.tangent_axes[0]], force_vec[self.tangent_axes[1]]]))
            row += [touch, normal, tangential, float(np.linalg.norm(torque_vec))]
        obj_pos = self.data.xpos[self.obj_body_id]
        obj_linvel = self.data.qvel[self._dof_adr:self._dof_adr + 3]
        row += [obj_pos[0], obj_pos[1], obj_pos[2], 0.0,
               float(np.linalg.norm(obj_linvel)), pull, float(phase)]
        return np.array(row, dtype=np.float64)

    def _apply_fractions(self):
        for ids, frac in zip(self.finger_actuator_ids, self.fraction):
            for aid in ids:
                lo, hi = self.driven[aid]
                self.data.ctrl[aid] = lo + frac * (hi - lo)

    def _touch_values(self):
        return [float(self._sensor_read(self.touch_ids[n])[0]) for n in self.args.fingertips]

    def _grip_force_sum(self):
        """Summed fingertip normal force -- same quantity as
        grasp_force_at_settle elsewhere, used here as a dense per-step
        formation-phase reward signal instead of a one-time settle-end
        measurement. See GRIP_FORCE_COEF in the module docstring."""
        return sum(float(self._sensor_read(self.force_ids[n])[self.args.normal_axis])
                   for n in self.args.fingertips)

    def _observe_step(self, phase, pull):
        """Advances the tactile/slip-probability history by one physics
        step and returns the predicted slip probability for it."""
        raw = self._read_raw_features(phase, pull)
        self._full_history.append(raw)
        if len(self._full_history) > cls.WINDOW_SIZE:
            self._full_history.pop(0)
        tactile = raw[:self._tactile_dim]
        if len(self._full_history) == cls.WINDOW_SIZE:
            window = np.stack(self._full_history)
            prob = cls.predict_slip(window, self._keep_idx, self._extra_scalars)
        else:
            prob = 0.0
        row = np.concatenate([tactile, [pull, float(phase), prob]]).astype(np.float32)
        self._obs_buffer.append(row)
        if len(self._obs_buffer) > OBS_WINDOW:
            self._obs_buffer.pop(0)
        return prob

    def _obs_from_buffer(self):
        rows = list(self._obs_buffer)
        while len(rows) < OBS_WINDOW:
            rows.insert(0, np.zeros(self._row_dim, dtype=np.float32))
        return np.concatenate(rows[-OBS_WINDOW:]).astype(np.float32)

    def reset(self, seed=None, options=None):
        _diag_log(self._worker_idx, "reset() starting")
        super().reset(seed=seed)
        # Standard Gymnasium reset-time configuration, wired up for a
        # specific, reproducible (object, seed) pair -- needed for an
        # apples-to-apples closed-loop success-rate evaluation of a
        # trained policy against the same 15-objects x 3-seeds benchmark
        # used for every hand-tuned-policy comparison in results.md,
        # instead of the random draw from self._episode_rng used during
        # training (unaffected when options is None, the normal case).
        forced_object = options.get("object_name") if options else None
        forced_seed = options.get("scene_seed") if options else None
        last_exc = None
        for attempt in range(5):
            object_name = forced_object or self._episode_rng.choice(self.object_names)
            scene_seed = (forced_seed if forced_seed is not None
                          else self._episode_rng.randint(0, 2**31 - 1))
            _diag_log(self._worker_idx,
                     f"reset attempt {attempt}: object={object_name} seed={scene_seed}, "
                     f"about to call _build")
            try:
                self._build(object_name, scene_seed)
                _diag_log(self._worker_idx, f"_build succeeded on attempt {attempt}")
                break
            except (ValueError, OSError) as e:
                # ValueError: MuJoCo's own compile failure (see
                # _compile_model's docstring). OSError: the scene file read
                # racing with real-time antivirus scanning of a just-
                # written file (same suspected cause, different failure
                # mode). Draw a fresh (object, seed) pair and try again
                # rather than let one bad sample/race kill the whole run.
                _diag_log(self._worker_idx,
                         f"_build FAILED on attempt {attempt}: {type(e).__name__}: {e}")
                last_exc = e
        else:
            _diag_log(self._worker_idx, "all 5 reset attempts failed, re-raising")
            raise last_exc

        free_joint_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{OBJECT_BODY_NAME}_freejoint", "Free joint")
        self._dof_adr = self.model.jnt_dofadr[free_joint_id]

        self._full_history = []
        self._obs_buffer = []
        self.fraction = [0.0] * self.n_fingers  # hand starts open
        self._apply_fractions()
        mujoco.mj_forward(self.model, self.data)

        args = self.args
        self._ramp_steps = args.ramp_steps
        self._settle_end = args.ramp_steps + args.settle_steps
        self._global_step = 0
        self._ref_pos = None
        self._max_displacement = 0.0
        self._effective_threshold = 0.15 * self.obj_spec["char_length"]
        self._spawn_pos = self.data.xpos[self.obj_body_id].copy()

        self._observe_step(phase=0, pull=0.0)

        obs = self._obs_from_buffer()
        info = {"object": object_name, "seed": scene_seed}
        _diag_log(self._worker_idx, "reset() complete")
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for i in range(self.n_fingers):
            self.fraction[i] = float(np.clip(
                self.fraction[i] + action[i] * MAX_DELTA_PER_STEP, 0.0, 1.0))
        self._apply_fractions()

        args = self.args
        last_prob, last_phase = 0.0, 0
        disp = 0.0
        contact_reward_acc, instability_acc, grip_force_acc, min_finger_acc = 0.0, 0.0, 0.0, 0.0
        terminated, dropped = False, False

        for _ in range(ACTION_REPEAT):
            step_idx = self._global_step
            if step_idx < self._ramp_steps:
                phase = 0
            elif step_idx < self._settle_end:
                phase = 1
            else:
                phase = 2

            if step_idx == self._ramp_steps:
                self.model.body_gravcomp[self.obj_body_id] = 0.0

            if phase < 2:
                pull = 0.0
                self.data.xfrc_applied[self.obj_body_id, :3] = [0.0, 0.0, 0.0]
            else:
                monitor_step = step_idx - self._settle_end
                pull_frac = min((monitor_step + 1) / max(args.pull_ramp_steps, 1), 1.0)
                pull = pull_frac * self.pull_force_max
                self.data.xfrc_applied[self.obj_body_id, :3] = [0.0, 0.0, -pull]

            mujoco.mj_step(self.model, self.data)
            self._global_step += 1

            if step_idx == self._settle_end - 1:
                self._ref_pos = self.data.xpos[self.obj_body_id].copy()
                if self._extra_scalars is not None and len(self._extra_scalars) == 2:
                    grasp_force_at_settle = sum(
                        float(self._sensor_read(self.force_ids[name])[args.normal_axis])
                        for name in args.fingertips
                    )
                    self._extra_scalars.append(grasp_force_at_settle)

            last_prob = self._observe_step(phase, pull)
            last_phase = phase

            if phase < 2:
                contact_frac = sum(1.0 for t in self._touch_values()
                                   if t > CONTACT_TOUCH_THRESHOLD) / len(args.fingertips)
                obj_speed = float(np.linalg.norm(
                    self.data.qvel[self._dof_adr:self._dof_adr + 3]))
                grip_force_frac = min(self._grip_force_sum() / max(self._obj_weight, 1e-8),
                                      GRIP_FORCE_CAP_MULTIPLE)
                min_finger_frac = min(self.fraction)
                contact_reward_acc += CONTACT_COEF * contact_frac
                instability_acc += INSTABILITY_COEF * obj_speed
                grip_force_acc += GRIP_FORCE_COEF * grip_force_frac
                min_finger_acc += MIN_FINGER_FRACTION_COEF * min_finger_frac

                fall = self._spawn_pos[2] - self.data.xpos[self.obj_body_id][2]
                if fall > DROP_THRESHOLD:
                    terminated, dropped = True, True
                    break
            else:
                disp = float(np.linalg.norm(
                    self.data.xpos[self.obj_body_id] - self._ref_pos))
                self._max_displacement = max(self._max_displacement, disp)

        slipped = False
        if last_phase < 2:
            reward = (contact_reward_acc + grip_force_acc + min_finger_acc - instability_acc
                     - EFFORT_COEF * float(np.mean(np.abs(action))))
            if dropped:
                reward -= DROP_PENALTY
        else:
            slipped = self._max_displacement > self._effective_threshold
            terminated = terminated or slipped
            reward = (-ALPHA_DISPLACEMENT * disp
                     - EFFORT_COEF * float(np.mean(np.abs(action))))
            if slipped:
                reward -= SLIP_PENALTY

        truncated = (not terminated) and (
            self._global_step >= self._settle_end + args.max_steps)
        if truncated:
            reward += HOLD_BONUS

        obs = self._obs_from_buffer()
        info = {
            "slipped": slipped,
            "dropped_during_formation": dropped,
            "max_displacement": self._max_displacement,
            "threshold": self._effective_threshold,
            "slip_probability": last_prob,
            "phase": last_phase,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self.model = None
        self.data = None
