"""Headless hop evaluation battery — turn a trained policy into a DECISION.

Answers the S5 decision rule with numbers instead of impressions:

    >=25 mm net displacement per hop, landing upright, over >=3 consecutive hops
        -> forward-hop track; size the course cells to the measured distance.
    <=8 mm (the open-loop baseline), or upright landings <50%
        -> learning added nothing; the hop is vertical-only, and hopscotch
           becomes hop-in-place-into-cells or the stepping pivot.

Upstream's CLAUDE.md: "Measure before theorizing. When a run 'fails', run a
headless eval of the actual checkpoint BEFORE changing rewards" — past failures
in this project turned out to be early checkpoints and mis-set success criteria.
It also warns the other way: "sim metrics can pass while the video fails the
human eye", so read this next to the run's video, not instead of it.

WHY THIS IS CPU-ONLY, AND WHY THAT MATTERS
------------------------------------------
It drives the exported ONNX through plain CPU MuJoCo, exactly as
scripts/infer_policy.py does for deployment rehearsal, rather than
reconstructing the mjlab env. mjlab is MuJoCo Warp and needs CUDA, which this
project's dev machine does not have — an mjlab-based eval would cost $0.80/hr
per iteration on HF Jobs. This runs locally, free, in seconds, so judging a
checkpoint is never the expensive step.

The cost of that choice, stated plainly: this is the DEPLOYMENT path, not the
training env. No domain randomization, no backlash, no observation noise, no
command delay.

**AND — read this before believing a verdict — A DIFFERENT ACTUATOR.** Training
drives these joints with the BAM voltage model (kp_fw=200, back-EMF,
load-dependent friction, force limit +/-1.07 Nm). This harness drives plain
MJCF position servos at --kp. A policy optimised against BAM dynamics does NOT
reproduce faithfully under a position servo, and the gap shows up exactly where
a hop lives: fast joint motion, where back-EMF dominates.

Measured on the first real policy (run s5-forward-hop, 2026-09-05): training
logged forward travel at ~95% of the 0.4 m/s cap and the video plainly showed
forward hopping, while this harness reported a median of -2.2 mm/hop. The
harness is the outlier. Treat its FORWARD numbers as a lower bound of unknown
tightness until the actuator gap is closed, and prefer wandb + the video for
"did it learn to travel".

What this harness IS reliable for, because they do not depend on actuator
fidelity: hop COUNT, consecutive-hop streaks, landing tilt distribution, and
fall rate. Those are geometry and contact, not torque.

THE TWIST SLOT IS A PHASE CLOCK, NOT A VELOCITY COMMAND (S5.3)
--------------------------------------------------------------
As of S5.3 the forward hop env replaces the twist command with a
``GroundPickPhaseCommand``: the first three obs command slots carry
``[cos(2*pi*phase), sin(2*pi*phase), 0]`` on a 1.6 s cycle, exactly as
ground_pick and sit_stand already do. The 61D contract is untouched; the
SEMANTICS of three slots changed.

This harness used to write zeros there, which was correct for a velocity
command (all-zero is the idle state) and is CATASTROPHIC for a phase carrier:
``(cos, sin) = (0, 0)`` is not on the unit circle at all, so the policy is fed
a state that does not exist anywhere in its training distribution. It does not
error — it produces a confident, wrong BAD-POLICY verdict, which is the same
failure shape as the projected-gravity default documented below.

So the clock is driven here too (``--phase-period``, default 1.6 s). Pass
``--no-phase`` for policies whose twist slot really is a velocity command: the
hop-in-place baseline ``Mjlab-Hop-Flat-MicroDuck``, and any forward policy
trained before S5.3 (runs s5-forward-hop, s51-forward-hop, s52-head-rhythm).
The mode is printed in the header — check it against the policy you are
judging, because neither mode can detect that it is the wrong one.

Driving the clock also makes the RHYTHM measurable, which is the open question
S5.2 and S5.3 were both aimed at: the report adds hops per cycle (1.0 is the
intended cadence) and how tightly takeoff locks to a consistent phase.

THE METRIC TRAPS (both cost real money in this project)
-------------------------------------------------------
1. **Contact-loss is not flight.** A duck falling over loses both foot contacts
   and logs excellent "air time" — 384 of the S1 probe's apparent successes
   were topples. Every flight interval here is gated on tilt AND trunk rise.
2. **Per-foot air time is not simultaneous flight.** An ordinary walking policy
   reports 125-300 ms per foot. Flight is `n_contact == 0`, measured from the
   contact geoms directly.

Usage:
    uv run python scripts/hopscotch/hop_eval.py policy.onnx
    uv run python scripts/hopscotch/hop_eval.py policy.onnx --episodes 40 --json out.json
    uv run python scripts/hopscotch/hop_eval.py policy.onnx --hop-cmd 0.06
    uv run python scripts/hopscotch/hop_eval.py policy.onnx --no-phase   # pre-S5.3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

SCENE = Path("src/mjlab_microduck/robot/microduck/scene_walk.xml")

# Matches HOME_FRAME in microduck_constants.py / DEFAULT_POSE in infer_policy.py.
# Actions are offsets from this, and joint observations are relative to it.
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,      # left leg
    0.3491, 0.3491, 0.0, 0.0,                     # neck/head
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,         # right leg
], dtype=np.float32)

DECIMATION = 4        # 50 Hz control on a 0.005 s physics step
ACTION_SCALE = 1.0

# Flight gates — MUST match microduck_hop_env_cfg.py, or this measures a
# different quantity than the reward paid for.
FLIGHT_MIN_HEIGHT = 0.10
FLIGHT_MAX_TILT_DEG = 30.0
FLIGHT_MIN_S = 0.02

# Hop cycle length, s. MUST match HOP_PERIOD in microduck_hop_env_cfg.py — the
# policy learned one hop per cycle at this period, and driving the clock at any
# other rate asks it to hop to a beat it never heard. Duplicated rather than
# imported to keep this script torch/warp-free (the infer_policy.py idiom);
# tests/test_hop_eval_phase.py fails if the two drift apart.
PHASE_PERIOD_S = 1.6

# S5 decision rule (architecture doc). The pass mark is scaled off the measured
# open-loop forward baseline, NOT asserted — see docs/s1-flight-probe.md.
OPEN_LOOP_FORWARD_M = 0.008
S5_PASS_FORWARD_M = 0.025
S5_MIN_UPRIGHT_RATE = 0.50
S5_MIN_CONSECUTIVE = 3


@dataclass
class Hop:
    """One simultaneous-flight interval that survived the topple screen."""
    flight_s: float
    forward_m: float          # trunk +x travel across the flight interval
    rise_m: float             # peak trunk z above its pre-takeoff height
    land_tilt_deg: float
    landed_upright: bool
    takeoff_phase: float      # hop-cycle phase at liftoff; -1.0 when --no-phase
    yaw_change_deg: float     # heading turned between takeoff and touchdown


@dataclass
class Episode:
    hops: list = field(default_factory=list)
    max_consecutive: int = 0   # longest run of hops with no fall between them
    fell: bool = False
    steps: int = 0
    # "Is it hopping or scooting?" — horizontal path length, split by whether
    # both feet were off the floor. Measured the same way in both states, so
    # the RATIO survives the position-servo caveat better than the absolute
    # distances do.
    path_m: float = 0.0
    air_path_m: float = 0.0
    # Net heading change over the episode. Nothing constrains yaw since S5.3
    # deleted track_angular_velocity with the phase carrier, so a policy can
    # bank displacement while turning — which reads on video as hopping in a
    # circle.
    yaw_drift_deg: float = 0.0
    # Head posture, averaged over the whole episode rather than at touchdown:
    # "head up ALL the time" is the stated requirement, and the run's
    # head_pose_bias penalty says the DC droop is what is failing.
    head_err_deg_sum: float = 0.0
    head_pitch_deg_sum: float = 0.0
    # Per-joint signed error, [neck_pitch, head_pitch, head_yaw, head_roll].
    # The mean |error| alone cannot say WHICH joint is off, and "the head is
    # down" and "the head is turned" want different fixes.
    head_joint_deg_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64))


def _tilt_deg(quat: np.ndarray) -> float:
    """Angle between the trunk's local +z and world +z, from a (w,x,y,z) quat."""
    up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def _yaw_deg(quat: np.ndarray) -> float:
    """Trunk heading about world +z, from a (w,x,y,z) quat."""
    w, x, y, z = (float(v) for v in quat[:4])
    return math.degrees(math.atan2(2.0 * (w * z + x * y),
                                   1.0 - 2.0 * (y * y + z * z)))


def _wrap_deg(a: float) -> float:
    """Signed shortest angular difference, degrees -> (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


class _Runner:
    """Drives one ONNX policy through CPU MuJoCo and records hop events."""

    def __init__(self, model, data, session, hop_cmd: float,
                 use_projected_gravity: bool, phase_period: float):
        self.model, self.data, self.session = model, data, session
        self.hop_cmd = hop_cmd
        self.use_projected_gravity = use_projected_gravity
        # 0.0 disables the clock and leaves the twist slots at zero — the idle
        # velocity command, correct for pre-S5.3 policies only.
        self.phase_period = phase_period
        self.dt = model.opt.timestep * DECIMATION
        self.phase = 0.0
        self.input_name = session.get_inputs()[0].name
        self.obs_dim = session.get_inputs()[0].shape[-1]

        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
        self.right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
        if min(self.trunk, self.left, self.right) < 0:
            raise RuntimeError("trunk_base or foot collision geoms not found in scene")

        self.gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.accel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")

        # Servo joints in ctrl order, skipping the free joint and any passive_*.
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(model.njnt)]
        servo = [i for i, n in enumerate(names)
                 if n and not n.startswith("passive_")
                 and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        self.qpos_idx = [model.jnt_qposadr[i] for i in servo]
        self.qvel_idx = [model.jnt_dofadr[i] for i in servo]
        self.last_action = np.zeros(len(servo), dtype=np.float32)

    # -- observation -----------------------------------------------------------

    def _sensor(self, sid: int) -> np.ndarray:
        adr = self.model.sensor_adr[sid]
        return self.data.sensordata[adr:adr + 3].copy().astype(np.float32)

    def _gravity_term(self) -> np.ndarray:
        quat = self.data.xquat[self.trunk].copy().astype(np.float32)
        if self.use_projected_gravity or self.accel < 0:
            return _quat_rotate_inverse(quat, np.array([0, 0, -1], dtype=np.float32))
        raw = -self._sensor(self.accel)
        mag = float(np.linalg.norm(raw))
        if mag > 0.1:
            return raw / mag
        return _quat_rotate_inverse(quat, np.array([0, 0, -1], dtype=np.float32))

    def observe(self) -> np.ndarray:
        """61D: 48 proprioception + [twist(3), head_pose(4), body_pose(6)].

        Hop intent lives in body_pose[2] per docs/command-block.md. The head
        slots stay zero, which is the deployment idle state.

        The twist slots carry the hop-cycle phase, NOT a velocity command — see
        the module docstring. Same cos/sin encoding as
        ``GroundPickPhaseCommand.compute``, so what the policy reads here is
        bit-for-bit what it read in training.
        """
        cmd = np.zeros(13, dtype=np.float32)
        if self.phase_period > 0.0:
            cmd[0] = math.cos(2.0 * math.pi * self.phase)
            cmd[1] = math.sin(2.0 * math.pi * self.phase)
        cmd[9] = self.hop_cmd      # 3 twist + 4 head + body[2] -> index 9
        obs = np.concatenate([
            self._sensor(self.gyro) if self.gyro >= 0 else np.zeros(3, np.float32),
            self._gravity_term(),
            self.data.qpos[self.qpos_idx].astype(np.float32) - DEFAULT_POSE,
            self.data.qvel[self.qvel_idx].astype(np.float32),
            self.last_action,
            cmd,
        ]).astype(np.float32)
        if obs.shape[0] != self.obs_dim:
            raise SystemExit(
                f"obs mismatch: built {obs.shape[0]}D, policy wants {self.obs_dim}D. "
                "The 61D contract changed, or this ONNX is from another family."
            )
        return obs

    def head_error_deg(self) -> tuple:
        """(mean |error| over the 4 neck/head joints, signed pitch error), deg.

        Same quantity head_pose_tracking / head_pose_bias_penalty price in
        training: joint position minus HOME, against an all-zero commanded head
        delta. Joints 5-8 in ctrl order are neck_pitch, head_pitch, head_yaw,
        head_roll (AGENTS.md joint layout).
        """
        err = np.degrees(self.data.qpos[self.qpos_idx][5:9] - DEFAULT_POSE[5:9])
        return float(np.abs(err).mean()), float(err[:2].mean()), err

    def n_feet_down(self) -> int:
        down = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for g in (c.geom1, c.geom2):
                if g == self.left:
                    down.add("L")
                elif g == self.right:
                    down.add("R")
        return len(down)

    # -- rollout ---------------------------------------------------------------

    def episode(self, stand_qpos, stand_ctrl, seed: int, settle_s: float,
                duration_s: float, noise: float) -> Episode:
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = stand_qpos
        # Small pose noise so the battery reports robustness, not one trajectory.
        for a in self.qpos_idx:
            d.qpos[a] += float(rng.normal(0.0, noise))
        d.ctrl[:] = stand_ctrl
        mujoco.mj_forward(m, d)
        self.last_action[:] = 0.0

        dt = self.dt
        for _ in range(int(settle_s / dt)):
            self._control_step(stand_ctrl_only=True)

        # Training randomizes the phase per env on reset (randomize_phase
        # defaults True), so every phase is in distribution; drawing from the
        # episode's seeded rng keeps the battery reproducible while still
        # sampling the whole cycle across episodes. The clock starts when the
        # policy takes over, not during the open-loop settle.
        self.phase = float(rng.random()) if self.phase_period > 0.0 else 0.0

        ep = Episode()
        cur_flight = 0.0
        x_takeoff = float(d.qpos[0])
        z_pre = float(d.qpos[2])
        apex = z_pre
        phase_takeoff = -1.0
        yaw_at_takeoff = 0.0
        max_tilt_in_flight = 0.0
        streak = 0
        xy_prev = d.qpos[0:2].copy()
        yaw_prev = _yaw_deg(d.xquat[self.trunk])

        for _ in range(int(duration_s / dt)):
            self._control_step()
            ep.steps += 1

            n_down = self.n_feet_down()
            z = float(d.qpos[2])
            tilt = _tilt_deg(d.xquat[self.trunk])

            xy = d.qpos[0:2].copy()
            step_path = float(np.linalg.norm(xy - xy_prev))
            xy_prev = xy
            ep.path_m += step_path
            if n_down == 0:
                ep.air_path_m += step_path
            yaw = _yaw_deg(d.xquat[self.trunk])
            ep.yaw_drift_deg += _wrap_deg(yaw - yaw_prev)
            yaw_prev = yaw
            head_abs, head_pitch, head_joints = self.head_error_deg()
            ep.head_err_deg_sum += head_abs
            ep.head_pitch_deg_sum += head_pitch
            ep.head_joint_deg_sum += head_joints

            if n_down == 0:
                if cur_flight == 0.0:
                    x_takeoff, z_pre, apex = float(d.qpos[0]), z, z
                    max_tilt_in_flight = tilt
                    phase_takeoff = self.phase if self.phase_period > 0.0 else -1.0
                    yaw_at_takeoff = ep.yaw_drift_deg
                cur_flight += dt
                apex = max(apex, z)
                max_tilt_in_flight = max(max_tilt_in_flight, tilt)
            elif cur_flight > 0.0:
                # Touchdown. Screen the interval: a topple also loses contact.
                genuine = (
                    cur_flight >= FLIGHT_MIN_S
                    and max_tilt_in_flight < FLIGHT_MAX_TILT_DEG
                    and apex > FLIGHT_MIN_HEIGHT
                )
                if genuine:
                    upright = tilt < FLIGHT_MAX_TILT_DEG
                    ep.hops.append(Hop(
                        flight_s=cur_flight,
                        forward_m=float(d.qpos[0]) - x_takeoff,
                        rise_m=apex - z_pre,
                        land_tilt_deg=tilt,
                        landed_upright=upright,
                        takeoff_phase=phase_takeoff,
                        yaw_change_deg=ep.yaw_drift_deg - yaw_at_takeoff,
                    ))
                    streak = streak + 1 if upright else 0
                    ep.max_consecutive = max(ep.max_consecutive, streak)
                else:
                    streak = 0
                cur_flight = 0.0

            if tilt > 70.0:            # unambiguously down; stop the clock
                ep.fell = True
                break

        return ep

    def _control_step(self, stand_ctrl_only: bool = False) -> None:
        if not stand_ctrl_only:
            # Advance before observing, matching the manager order in training
            # (commands compute, then observations are gathered).
            if self.phase_period > 0.0:
                self.phase = (self.phase + self.dt / self.phase_period) % 1.0
            obs = self.observe()[None, :]
            action = self.session.run(None, {self.input_name: obs})[0][0]
            self.last_action = action.astype(np.float32)
            self.data.ctrl[:] = DEFAULT_POSE + self.last_action * ACTION_SCALE
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path, help="exported policy (scripts/export.py)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--duration", type=float, default=6.0, help="seconds per episode")
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=0.02, help="init joint noise (rad)")
    ap.add_argument("--hop-cmd", type=float, default=0.06,
                    help="body_pose[2] hop intent, m (HOP_CMD_MAX default)")
    # MUST match USE_PROJECTED_GRAVITY in microduck_velocity_env_cfg.py (True as
    # of pin 1e79c29). Getting this wrong feeds the policy a gravity signal it
    # never trained on, and it fails in a way that looks like a BAD POLICY
    # rather than a bad harness: during development the mismatch produced
    # "hops" with 0.0 mm apex rise, negative displacement and 100% falls, while
    # training metrics said the policy was hopping forward at 95% of the
    # velocity cap. Trust the impossible-looking number, not the verdict.
    ap.add_argument("--raw-accel", dest="projected_gravity", action="store_false",
                    default=True,
                    help="policy was trained on raw accelerometer, not projected gravity")
    # MUST match the twist-slot SEMANTICS the policy was trained with — see the
    # module docstring. Zeros in a phase carrier are off the unit circle
    # entirely, a state no training step ever produced, and the harness cannot
    # detect the mismatch; it just returns a confident wrong verdict.
    ap.add_argument("--phase-period", type=float, default=PHASE_PERIOD_S,
                    help="hop-cycle period driven into the twist slots, s "
                         "(HOP_PERIOD default)")
    ap.add_argument("--no-phase", dest="phase_period", action="store_const",
                    const=0.0,
                    help="twist slots are a VELOCITY command, not a phase clock "
                         "— for Mjlab-Hop-Flat-MicroDuck and any pre-S5.3 policy")
    ap.add_argument("--kp", type=float, default=20.0,
                    help="position gain override; the MJCF ships kp~0.5 placeholders "
                         "that CANNOT hold STAND (see docs/s1-flight-probe.md)")
    ap.add_argument("--kd", type=float, default=0.3, help="damping gain override")
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"no such policy: {args.onnx}")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    # THE TRAP (docs/s1-flight-probe.md, "a third trap, in the model itself"):
    # the MJCF's kp=0.386-0.55 position gains are PLACEHOLDERS — training drives
    # these joints with the BAM voltage model (kp_fw=200) instead, and
    # XmlPositionActuatorCfg is commented out in microduck_constants.py. Left
    # stock, the robot topples in ~0.6 s untouched, so EVERY policy would score
    # 100% fallen and this battery would rank noise. The probe overdrives the
    # servo for the same reason: it makes the binding constraint the forcerange
    # torque clamp (+/-0.96 Nm), which is the real XL330 hardware limit.
    model.actuator_gainprm[:, 0] = args.kp
    model.actuator_biasprm[:, 1] = -args.kp
    model.actuator_biasprm[:, 2] = -args.kd
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    stand_qpos, stand_ctrl = model.key_qpos[key].copy(), model.key_ctrl[key].copy()

    # Sanity check FIRST: if holding STAND does not settle upright, every number
    # below is noise. CLAUDE.md calls this out directly, and it must check TILT,
    # not just height — "a settle test that only records z reports fallen states
    # as resting fine".
    mujoco.mj_resetData(model, data)
    data.qpos[:] = stand_qpos
    data.ctrl[:] = stand_ctrl
    mujoco.mj_forward(model, data)
    for _ in range(int(1.0 / model.opt.timestep)):
        data.ctrl[:] = stand_ctrl
        mujoco.mj_step(model, data)
    settle_tilt = _tilt_deg(data.xquat[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")])
    if settle_tilt > 10.0:
        raise SystemExit(
            f"open-loop STAND does not hold: tilt {settle_tilt:.1f} deg after 1 s "
            f"at kp={args.kp}. Every measurement would be noise. Raise --kp."
        )

    session = ort.InferenceSession(str(args.onnx))
    runner = _Runner(model, data, session, args.hop_cmd, args.projected_gravity,
                     args.phase_period)

    print(f"policy {args.onnx.name}  obs {runner.obs_dim}D  "
          f"hop_cmd={args.hop_cmd:.3f}  {args.episodes} episodes x {args.duration:g}s")
    print("twist slots: " + (
        f"PHASE CLOCK at {args.phase_period:g} s (S5.3 and later)"
        if args.phase_period > 0.0
        else "zero VELOCITY command (--no-phase; pre-S5.3 policies only)")
        + " — a mismatch here produces a wrong verdict, not an error")
    print(f"kp={args.kp:g} kd={args.kd:g} (MJCF placeholders overridden); "
          f"STAND settles at {settle_tilt:.1f} deg")
    print("NOTE: deployment path — no DR, backlash, obs noise or command delay, "
          "so these numbers are OPTIMISTIC vs the training distribution.\n")

    eps = [runner.episode(stand_qpos, stand_ctrl, seed, args.settle,
                          args.duration, args.noise)
           for seed in range(args.episodes)]

    hops = [h for e in eps for h in e.hops]
    fell = sum(e.fell for e in eps)
    consec = max((e.max_consecutive for e in eps), default=0)

    steps = sum(e.steps for e in eps)
    policy_s = steps * runner.dt
    path = sum(e.path_m for e in eps)
    air_path = sum(e.air_path_m for e in eps)
    air_share = air_path / path if path > 1e-9 else 0.0
    yaw_per_s = (sum(abs(e.yaw_drift_deg) for e in eps) / policy_s
                 if policy_s > 0 else 0.0)
    head_err = sum(e.head_err_deg_sum for e in eps) / max(steps, 1)
    head_pitch = sum(e.head_pitch_deg_sum for e in eps) / max(steps, 1)

    print(f"{'episodes':<28}{len(eps)}")
    print(f"{'  ended fallen':<28}{fell}  ({fell / max(len(eps),1):.0%})")
    print(f"{'genuine hops':<28}{len(hops)}")
    print(f"{'  per episode':<28}{len(hops) / max(len(eps),1):.2f}")
    print(f"{'  best consecutive':<28}{consec}")
    if args.phase_period > 0.0:
        cycles = policy_s / args.phase_period
        print(f"{'  per hop cycle':<28}{len(hops) / max(cycles, 1e-9):.2f}"
              "   (1.00 = the intended cadence)")

    # "Hopping or scooting?" — the share of horizontal travel that happened
    # with both feet off the floor. A hop moves the robot through the AIR;
    # ground travel between hops is scooting, however it is dressed up.
    print(f"\n{'travel in the air':<28}{air_share:.0%} of "
          f"{path*1e3:.0f} mm total path")
    print(f"{'heading drift':<28}{yaw_per_s:+.1f} deg/s "
          f"(straight-ahead hopping is ~0; nothing prices yaw since S5.3)")
    head_joints = sum(e.head_joint_deg_sum for e in eps) / max(steps, 1)
    print(f"{'head error (whole run)':<28}{head_err:.1f} deg mean |err|, "
          f"pitch {head_pitch:+.1f} deg from HOME")
    print(f"{'  by joint, signed':<28}" + "  ".join(
        f"{n} {v:+.1f}" for n, v in
        zip(("neck_pitch", "head_pitch", "head_yaw", "head_roll"), head_joints)))

    if not hops:
        print("\nNo genuine simultaneous flight. Either the policy never leaves "
              "the ground,\nor every contact-loss was a topple (both are S5 FAIL).")
        verdict, fwd_med, upright_rate = "FAIL — no flight", 0.0, 0.0
    else:
        fwd = [h.forward_m for h in hops]
        fly = [h.flight_s for h in hops]
        rise = [h.rise_m for h in hops]
        tilt = [h.land_tilt_deg for h in hops]
        upright_rate = sum(h.landed_upright for h in hops) / len(hops)
        fwd_med = statistics.median(fwd)

        def line(label, xs, scale, unit):
            print(f"{label:<28}median {statistics.median(xs)*scale:7.1f} {unit}   "
                  f"p90 {sorted(xs)[int(0.9*(len(xs)-1))]*scale:7.1f}   "
                  f"max {max(xs)*scale:7.1f}")

        print()
        line("forward per hop", fwd, 1e3, "mm")
        line("flight duration", fly, 1e3, "ms")
        line("apex rise", rise, 1e3, "mm")
        line("landing tilt", tilt, 1.0, "deg")
        line("turn per hop", [abs(h.yaw_change_deg) for h in hops], 1.0, "deg")
        print(f"{'upright landings':<28}{upright_rate:.0%}")

        if args.phase_period > 0.0:
            # Mean resultant length of the takeoff phases: 1.0 = every hop
            # leaves the ground at the same point in the cycle (a rhythm),
            # ~0 = takeoffs are scattered across it (a bounce that ignores the
            # clock). Circular statistics, because phase wraps at 1.0.
            ph = np.array([h.takeoff_phase for h in hops])
            lock = float(abs(np.mean(np.exp(2j * np.pi * ph))))
            print(f"{'takeoff phase lock':<28}{lock:.2f}   "
                  f"(1.00 = every hop on the same beat, 0 = ignores the clock)")

        print("\n--- S5 decision rule ---")
        print(f"open-loop baseline (free):  {OPEN_LOOP_FORWARD_M*1e3:.0f} mm/hop")
        print(f"measured (this policy):     {fwd_med*1e3:.1f} mm/hop, "
              f"{upright_rate:.0%} upright, {consec} consecutive")
        print("CAVEAT: forward travel here is measured under a POSITION SERVO, "
              "not BAM.\nCross-check Episode_Reward/forward_flight_progress and "
              "the video before acting on a forward FAIL (see module docstring).")
        if (fwd_med >= S5_PASS_FORWARD_M and upright_rate >= S5_MIN_UPRIGHT_RATE
                and consec >= S5_MIN_CONSECUTIVE):
            verdict = ("PASS -> forward-hop track. Size the course cells to the "
                       "measured distance, then proceed to commanded distance (E3).")
        elif fwd_med <= OPEN_LOOP_FORWARD_M:
            verdict = ("FAIL -> no better than open-loop; learning added nothing "
                       "forward. Hop is vertical-only: hop-in-place-into-cells, "
                       "or the stepping pivot.")
        elif upright_rate < S5_MIN_UPRIGHT_RATE:
            verdict = (f"FAIL -> travels {fwd_med*1e3:.0f} mm but only "
                       f"{upright_rate:.0%} of landings are upright. It is falling "
                       "forward, not hopping forward.")
        else:
            verdict = ("PARTIAL -> beats open-loop but misses the bar. Judge from "
                       "the video and the profiles above before changing rewards; "
                       "this is what mid-tuning looks like.")
        print(verdict)

    print("\nRead this next to the run's video — CLAUDE.md: sim metrics can pass "
          "while the video fails the human eye.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "policy": str(args.onnx),
            "episodes": len(eps),
            "fell": fell,
            "hops": [asdict(h) for h in hops],
            "median_forward_m": fwd_med,
            "upright_rate": upright_rate,
            "max_consecutive": consec,
            "phase_period_s": args.phase_period,
            "air_travel_share": air_share,
            "path_m": path,
            "heading_drift_deg_per_s": yaw_per_s,
            "head_err_deg": head_err,
            "head_pitch_deg": head_pitch,
            "verdict": verdict,
        }, indent=2))
        print(f"report -> {args.json}")


if __name__ == "__main__":
    main()
