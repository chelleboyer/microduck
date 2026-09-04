"""S1 physics probe: can Microduck break ground contact at all?

Answers the blocking spike in the hopscotch architecture doc WITHOUT training,
on CPU, for free. Upstream's CLAUDE.md: "verify physics assumptions in sim
BEFORE training - this is the single biggest time-saver."

The test is open-loop: crouch, then extend both legs as fast as the actuators
allow, and measure whether both feet leave the floor simultaneously. A grid
search over crouch depth, crouch duration and extension speed reports the best
flight phase achievable.

WHAT THIS IS AND IS NOT
-----------------------
The MJCF enforces the real XL330 torque ceiling (forcerange +/-0.96 Nm), so
this is not a fantasy-torque test. But it runs the plain MJCF position
actuators, NOT the BAM voltage model used in training - so it does NOT model
back-EMF, which cuts available torque exactly when joints move fast, which is
exactly what a hop needs. It also has no backlash and no domain randomization.

=> This probe is OPTIMISTIC. It is a NECESSARY-condition test:
     no flight here            -> strong evidence flight is unreachable
     flight here               -> flight is not ruled out; the GPU spike (MD-3)
                                  still has to show PPO can find it under BAM

Usage:
    uv run python scripts/hopscotch/flight_probe.py            # grid search
    uv run python scripts/hopscotch/flight_probe.py --csv out.csv   # + best trace
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path("src/mjlab_microduck/robot/microduck/scene_walk.xml")
STAND_KEY = "STAND"

# Actuator indices (upstream CLAUDE.md invariant: 0-4 left leg, 5-8 neck/head,
# 9-13 right leg). Left and right are sign-mirrored in the STAND keyframe.
L_HIP_PITCH, L_KNEE, L_ANKLE = 2, 3, 4
R_HIP_PITCH, R_KNEE, R_ANKLE = 11, 12, 13

_LEGS = {
    "left": ("left_hip_pitch", "left_knee", "left_ankle",
             "ankle_left", (L_HIP_PITCH, L_KNEE, L_ANKLE)),
    "right": ("right_hip_pitch", "right_knee", "right_ankle",
              "ankle_right", (R_HIP_PITCH, R_KNEE, R_ANKLE)),
}


def leg_extension_pattern(model: mujoco.MjModel, stand_qpos: np.ndarray) -> dict[int, float]:
    """Derive the flat-foot leg extension direction from the kinematics.

    Do NOT hand-guess this. The obvious planar pattern (hip +1, knee -2,
    ankle +1) is wrong for this model: foot pitch goes as (-hip + knee - ankle),
    so that pattern rotates the foot ~228 deg/rad and topples the robot within
    0.1 rad, quasi-statically, before any hop is attempted.

    With the trunk held fixed, we want the joint direction that moves the foot
    DOWN (i.e. pushes the trunk up) while keeping foot pitch and fore-aft
    position unchanged. That is the null space of the [dfoot_x; dpitch] rows of
    the leg Jacobian. Returns {actuator_index: coefficient}, normalised so the
    knee coefficient is 1.0, oriented so positive s EXTENDS.
    """
    data = mujoco.MjData(model)
    pattern: dict[int, float] = {}
    eps = 1e-3

    for jnames_hip, jname_knee, jname_ankle, bodyname, act_idx in _LEGS.values():
        jnames = (jnames_hip, jname_knee, jname_ankle)
        adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
               for j in jnames]
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bodyname)

        def foot_state(delta: np.ndarray) -> tuple[float, float, float]:
            q = stand_qpos.copy()
            for a, dv in zip(adr, delta):
                q[a] += dv
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            p = data.xpos[body]
            R = data.xmat[body].reshape(3, 3)
            pitch = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
            return float(p[0]), float(p[2]), pitch

        x0, z0, pitch0 = foot_state(np.zeros(3))
        jac = np.zeros((3, 3))  # rows: dfoot_x, dfoot_z, dpitch
        for c in range(3):
            dv = np.zeros(3)
            dv[c] = eps
            x1, z1, pitch1 = foot_state(dv)
            jac[:, c] = ((x1 - x0) / eps, (z1 - z0) / eps, (pitch1 - pitch0) / eps)

        # Keep foot flat (dpitch=0) and fore-aft fixed (dfoot_x=0).
        _, _, vt = np.linalg.svd(jac[[0, 2], :])
        direction = vt[-1]
        direction = direction / abs(direction[1])          # normalise on the knee
        if jac[1] @ direction > 0:                          # foot must go DOWN
            direction = -direction
        for idx, coeff in zip(act_idx, direction):
            pattern[idx] = float(coeff)

    return pattern


@dataclass
class Result:
    s_crouch: float
    s_extend: float
    t_crouch: float
    t_extend: float
    flight_s: float          # longest contiguous both-feet-off interval
    apex_rise_m: float       # peak trunk z above its settled standing height
    tilt_at_land_deg: float  # trunk tilt from vertical when contact resumes
    max_tilt_deg: float      # worst tilt at any point after settling
    landed_upright: bool

    @property
    def is_hop(self) -> bool:
        """A hop, not a topple.

        Feet leaving the floor is NOT sufficient: a duck falling over also
        loses both contacts, with a long "flight" time and a great-looking
        air-time number. Upstream's CLAUDE.md is explicit that height-only
        (or contact-only) checks miss exactly this. Require that the trunk
        actually rose AND stayed upright the whole time.
        """
        return (
            self.flight_s > 0.0
            and self.max_tilt_deg < 30.0
            and self.apex_rise_m > 0.002
        )


def _foot_geoms(model: mujoco.MjModel) -> tuple[int, int]:
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    if left < 0 or right < 0:
        raise RuntimeError("foot collision geoms not found in scene")
    return left, right


def _n_feet_down(data: mujoco.MjData, left: int, right: int) -> int:
    down = set()
    for i in range(data.ncon):
        c = data.contact[i]
        for g in (c.geom1, c.geom2):
            if g == left:
                down.add("L")
            elif g == right:
                down.add("R")
    return len(down)


def _tilt_deg(data: mujoco.MjData) -> float:
    """Angle between the trunk's local +z and world +z."""
    # Free-joint quaternion is qpos[3:7] (w, x, y, z).
    w, x, y, z = data.qpos[3:7]
    # Third column of the rotation matrix = body z-axis in world frame.
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def _ctrl_at(stand: np.ndarray, s: float, pattern: dict[int, float]) -> np.ndarray:
    ctrl = stand.copy()
    for idx, k in pattern.items():
        ctrl[idx] += k * s
    return ctrl


def simulate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    stand_ctrl: np.ndarray,
    stand_qpos: np.ndarray,
    pattern: dict[int, float],
    s_crouch: float,
    s_extend: float,
    t_crouch: float,
    t_extend: float,
    t_settle: float = 0.6,
    t_observe: float = 0.9,
    trace: list | None = None,
) -> Result:
    left, right = _foot_geoms(model)
    dt = model.opt.timestep

    mujoco.mj_resetData(model, data)
    data.qpos[:] = stand_qpos
    data.ctrl[:] = stand_ctrl
    mujoco.mj_forward(model, data)

    # Settle so the probe measures a hop, not the transient from spawning.
    for _ in range(int(t_settle / dt)):
        data.ctrl[:] = stand_ctrl
        mujoco.mj_step(model, data)
    z0 = float(data.qpos[2])

    total = t_crouch + t_extend + t_observe
    n_steps = int(total / dt)

    best_flight = 0.0
    cur_flight = 0.0
    apex = z0
    tilt_at_land = 0.0
    max_tilt = 0.0
    saw_flight = False

    for i in range(n_steps):
        t = i * dt
        if t < t_crouch:                      # ramp down into the crouch
            s = s_crouch * (t / t_crouch)
        elif t < t_crouch + t_extend:         # drive up through extension
            u = (t - t_crouch) / t_extend
            s = s_crouch + (s_extend - s_crouch) * u
        else:                                 # hold extended, observe
            s = s_extend

        data.ctrl[:] = _ctrl_at(stand_ctrl, s, pattern)
        mujoco.mj_step(model, data)

        n_down = _n_feet_down(data, left, right)
        z = float(data.qpos[2])
        apex = max(apex, z)
        max_tilt = max(max_tilt, _tilt_deg(data))

        if n_down == 0:
            cur_flight += dt
            saw_flight = True
        else:
            if cur_flight > best_flight:
                best_flight = cur_flight
                tilt_at_land = _tilt_deg(data)
            cur_flight = 0.0

        if trace is not None:
            trace.append((t, s, n_down, z, _tilt_deg(data)))

    if cur_flight > best_flight:              # still airborne at the end
        best_flight = cur_flight
        tilt_at_land = _tilt_deg(data)

    return Result(
        s_crouch=s_crouch,
        s_extend=s_extend,
        t_crouch=t_crouch,
        t_extend=t_extend,
        flight_s=best_flight,
        apex_rise_m=apex - z0,
        tilt_at_land_deg=tilt_at_land if saw_flight else 0.0,
        max_tilt_deg=max_tilt,
        landed_upright=saw_flight and tilt_at_land < 30.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, help="write the best run's per-step trace here")
    ap.add_argument("--top", type=int, default=10, help="how many results to print")
    ap.add_argument("--kp", type=float, default=20.0, help="probe position gain (Nm/rad)")
    ap.add_argument("--kd", type=float, default=0.3, help="probe damping gain")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    # The MJCF ships kp=0.55 / kd=0, which cannot even hold STAND (the robot
    # topples in 0.6 s). That gain is a placeholder - XmlPositionActuatorCfg is
    # commented out in microduck_constants.py and training drives these joints
    # with the BAM voltage model (kp_fw=200) instead.
    #
    # We deliberately overdrive the servo so the BINDING constraint is the
    # forcerange torque clamp (+/-0.96 Nm), which is real XL330 hardware. That
    # makes a negative result decisive: if the duck cannot leave the floor when
    # the only limit is its true torque ceiling, no control law can make it.
    model.actuator_gainprm[:, 0] = args.kp
    model.actuator_biasprm[:, 1] = -args.kp
    model.actuator_biasprm[:, 2] = -args.kd
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, STAND_KEY)
    stand_ctrl = model.key_ctrl[key].copy()
    stand_qpos = model.key_qpos[key].copy()

    # Sign of s is not known a priori (which way the leg folds), so sweep both.
    crouches = [round(-v, 3) for v in np.arange(0.15, 1.45, 0.15)]
    extends = [0.0, 0.1, 0.2, 0.3, 0.4]
    t_crouches = [0.15, 0.25, 0.40]
    t_extends = [0.03, 0.04, 0.06, 0.09, 0.14]

    # Sanity check first: if simply holding STAND does not settle upright, every
    # number below is meaningless. Upstream's CLAUDE.md calls this out directly.
    pattern = leg_extension_pattern(model, stand_qpos)
    print("derived flat-foot extension direction (+s = extend):")
    for idx in sorted(pattern):
        print("   act %2d %-16s %+.3f" % (
            idx, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx), pattern[idx]))
    print()

    baseline = simulate(model, data, stand_ctrl, stand_qpos, pattern,
                        s_crouch=0.0, s_extend=0.0, t_crouch=0.15, t_extend=0.04)

    results: list[Result] = []
    for sc in crouches:
        for se in extends:
            for tc in t_crouches:
                for te in t_extends:
                    results.append(
                        simulate(model, data, stand_ctrl, stand_qpos, pattern,
                                 sc, se, tc, te)
                    )

    print(f"swept {len(results)} open-loop countermovements on {SCENE.name}")
    print(f"torque ceiling {model.actuator_forcerange[0][1]:.2f} Nm/joint, "
          f"mass {model.body_mass.sum():.3f} kg, dt {model.opt.timestep}")
    print(f"baseline: holding STAND settles at tilt {baseline.max_tilt_deg:.1f} deg "
          f"(must be small, or every result below is noise)\n")

    hops = sorted([r for r in results if r.is_hop],
                  key=lambda r: (r.flight_s, r.apex_rise_m), reverse=True)
    topples = [r for r in results if not r.is_hop and r.flight_s > 0.0]
    print(f"{len(hops)} genuine hops, {len(topples)} contact-loss-by-toppling "
          f"(excluded), {len(results) - len(hops) - len(topples)} never left the floor\n")

    hdr = (f"{'flight_ms':>10} {'rise_mm':>8} {'maxtilt':>8} {'landtilt':>9}   params")
    print(hdr)
    print("-" * len(hdr))
    for r in (hops or topples)[: args.top]:
        print(f"{r.flight_s * 1e3:10.1f} {r.apex_rise_m * 1e3:8.1f} "
              f"{r.max_tilt_deg:8.1f} {r.tilt_at_land_deg:9.1f}   "
              f"crouch={r.s_crouch:+.2f}/{r.t_crouch:.2f}s "
              f"extend={r.s_extend:+.2f}/{r.t_extend:.2f}s")
    if not hops:
        print("(no genuine hops - showing the toppling runs for diagnosis)")

    best = hops[0] if hops else None
    print("\n--- S1 decision rule (architecture doc) ---")
    if best is None:
        flight_ms = 0.0
        verdict = ("FAIL: nothing left the ground upright. Strong evidence for the "
                   "stepping-hopscotch pivot.")
    elif best.flight_s >= 0.080 and best.landed_upright:
        flight_ms = best.flight_s * 1e3
        verdict = "PASS (>=80 ms, upright): flight not ruled out -> proceed to MD-3"
    elif best.flight_s < 0.030:
        flight_ms = best.flight_s * 1e3
        verdict = "FAIL (<30 ms): strong evidence for the stepping-hopscotch pivot"
    else:
        flight_ms = best.flight_s * 1e3
        verdict = "AMBIGUOUS (30-80 ms): needs the GPU spike to resolve"
    print(f"best UPRIGHT simultaneous flight: {flight_ms:.1f} ms   {verdict}")
    print("NOTE: no BAM back-EMF, no backlash, no DR -> this is an OPTIMISTIC bound.")
    print("NOTE: open-loop only. PPO may find a countermovement this sweep cannot.")

    if args.csv:
        trace: list = []
        simulate(model, data, stand_ctrl, stand_qpos, pattern,
                 best.s_crouch, best.s_extend, best.t_crouch, best.t_extend, trace=trace)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8") as fh:
            fh.write("t,s,n_feet_down,trunk_z,tilt_deg\n")
            for row in trace:
                fh.write("%.4f,%.4f,%d,%.5f,%.2f\n" % row)
        print(f"\nbest-run trace -> {args.csv} ({len(trace)} steps)")


if __name__ == "__main__":
    main()
