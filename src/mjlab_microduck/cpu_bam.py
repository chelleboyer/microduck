"""Drive a microduck MJCF with the BAM actuator model in plain CPU MuJoCo.

WHY THIS EXISTS
---------------
Training drives every servo with the BAM M6 voltage model — a control law with
back-EMF, load-dependent friction, a firmware PWM limit and a voltage that sags
under load. Every CPU-side tool in this repo (hop_eval.py, and any local video
recorder) previously drove plain MJCF POSITION servos instead, because the
scene's actuators are position actuators and mjlab is the only thing that knew
about BAM. That gap is not cosmetic: it shows up exactly where a hop lives —
fast joint motion, where back-EMF dominates.

The measured cost of it (2026-09-05, policy s5-forward-hop): the position-servo
harness reported a median of **-2.2 mm/hop** while training logged forward
travel at ~95% of its velocity cap and the video plainly showed the duck hopping
forward. The harness was the outlier, and its FORWARD verdict had to be
disclaimed in its own docstring.

This module closes that gap by reusing BAM's own CPU integration
(``bam.mujoco.MujocoController``) rather than reimplementing the control law —
the same ``bam`` package mjlab drives during training, so there is one
implementation of the physics and no second copy to drift.

WHAT IT REPRODUCES, AND WHAT IT STILL DOES NOT
----------------------------------------------
Reproduced: the BAM voltage control law, back-EMF, the firmware P-gain
(``kp_fw``), load-dependent friction and damping rewritten every physics step,
voltage sag under load with its floor, the torque ceiling MuJoCo enforces as
``forcerange``, and the rotor's extra inertia as joint armature.

NOT reproduced, and deliberately so — this is still the DEPLOYMENT path, not the
training distribution: no domain randomization (a single nominal voltage instead
of a per-env sample), no backlash, no observation noise, no command delay. Those
make the numbers OPTIMISTIC relative to training, which is the honest direction
for a deployment rehearsal to err in.

The actuator parameters are IMPORTED from microduck_constants rather than
copied, so a change to the training actuator cannot silently leave this behind.
"""

from __future__ import annotations

import numpy as np

try:  # bam is a hard dependency of this project; fail loudly, not mysteriously.
    import mujoco
    from bam.model import _resolve_json_path, load_model
    from bam.mujoco import MujocoController
except ImportError as e:  # pragma: no cover - environment problem, not logic
    raise SystemExit(
        f"BAM CPU drive needs `bam` and `mujoco` installed ({e}). Run `uv sync`."
    ) from e

from mjlab_microduck.robot.microduck_constants import _BAM_ACTUATOR_KWARGS


def _nominal(range_key: str, default: float) -> float:
    """Midpoint of a training DR range — the honest 'no DR' operating point.

    Training samples these per env; a deterministic rehearsal has to pick one
    value, and the centre of the distribution is the only defensible choice.
    Deriving it from the cfg (rather than hardcoding 7.35) means a retuned DR
    range moves this with it.
    """
    rng = _BAM_ACTUATOR_KWARGS.get(range_key)
    return default if rng is None else 0.5 * (float(min(rng)) + float(max(rng)))


NOMINAL_VIN = _nominal("vin_range", 7.4)
NOMINAL_VIN_DROP_GAIN = _nominal("vin_drop_gain_range", 0.1)
VIN_MIN = float(_BAM_ACTUATOR_KWARGS.get("vin_min", 6.0))
KP_FW = float(_BAM_ACTUATOR_KWARGS["kp_fw"])
MOTOR_NAME = _BAM_ACTUATOR_KWARGS["motor_name"]
MOTOR_MODEL = _BAM_ACTUATOR_KWARGS["model"]


class BamDrive:
    """BAM-driven servos on a compiled CPU ``MjModel`` / ``MjData`` pair.

    Usage mirrors a position servo, so callers change one line:

        drive = BamDrive(model, data)      # converts the actuators in place
        drive.reset(data.qpos)
        drive.set_targets(joint_angle_targets)
        for _ in range(decimation):
            drive.update()                 # BEFORE the step, every step
            mujoco.mj_step(model, data)

    ``update()`` must run at the PHYSICS rate, not the control rate: the
    firmware loop and the friction rewrite are per-step quantities, and calling
    it once per control step would evaluate the voltage law at 50 Hz instead of
    200 Hz and quietly change the dynamics.
    """

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        vin: float = NOMINAL_VIN,
        vin_drop_gain: float = NOMINAL_VIN_DROP_GAIN,
        vin_min: float = VIN_MIN,
        kp_fw: float = KP_FW,
    ) -> None:
        self.model, self.data = model, data
        self.vin = float(vin)

        bam_model = load_model(_resolve_json_path(None, MOTOR_NAME, MOTOR_MODEL))
        bam_model.actuator.vin = self.vin
        bam_model.actuator.kp = float(kp_fw)
        self.bam_model = bam_model

        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)
        ]
        if any(n is None for n in names):
            raise RuntimeError("unnamed actuator in the scene; BAM needs names")
        self.actuator_names: list[str] = list(names)  # ctrl order

        # Match mjlab's edit_spec: torque actuators, clamped at the voltage
        # ceiling, with the rotor inertia as armature and MuJoCo's own damping /
        # frictionloss zeroed because BAM rewrites them every step.
        self.force_limit = self._force_limit(bam_model)
        self._to_motor_actuators()

        # MujocoController.__init__ sets the armature and then calls
        # mj_setConst, which RESETS data.qpos/qvel to the model's qpos0. A
        # caller that has already posed the robot (e.g. loaded the STAND
        # keyframe) would silently lose it, and the failure looks like physics:
        # the duck collapses because it is being driven from an all-zero pose
        # toward a stand target. Snapshot and restore so construction order
        # cannot matter.
        qpos, qvel, time = data.qpos.copy(), data.qvel.copy(), data.time
        self.controller = MujocoController(
            bam_model, self.actuator_names, model, data,
            vin_drop_gain=float(vin_drop_gain), vin_min=float(vin_min),
        )
        data.qpos[:], data.qvel[:], data.time = qpos, qvel, time
        mujoco.mj_forward(model, data)
        self.controller.last_ts = data.time
        for dof in self.controller.dof_indexes:
            model.dof_damping[dof] = 0.0
            model.dof_frictionloss[dof] = 0.0

    def _force_limit(self, bam_model) -> float:
        """Torque ceiling, computed the way mjlab computes ``forcerange``.

        mjlab uses the TOP of the voltage DR range so the clamp is a safe
        ceiling for every env; matching that keeps the binding constraint here
        identical to the one the policy was trained against.
        """
        rng = _BAM_ACTUATOR_KWARGS.get("vin_range")
        vin_for_limit = max(rng) if rng else self.vin
        return float(vin_for_limit * bam_model.kt.value / bam_model.R.value)

    def _to_motor_actuators(self) -> None:
        m = self.model
        for i in range(m.nu):
            m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_gainprm[i, :] = 0.0
            m.actuator_gainprm[i, 0] = 1.0
            m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
            m.actuator_biasprm[i, :] = 0.0
            m.actuator_gear[i, 0] = 1.0
            m.actuator_forcelimited[i] = 1
            m.actuator_forcerange[i] = (-self.force_limit, self.force_limit)
            # ctrl is a TORQUE now; the position actuator's radian ctrlrange
            # would clamp it to nonsense.
            m.actuator_ctrllimited[i] = 0

    # -- runtime ---------------------------------------------------------------

    def reset(self, qpos: np.ndarray) -> None:
        """Clear the firmware target and voltage-sag state after mj_resetData."""
        self.controller.reset(np.asarray(qpos))

    def set_targets(self, q_target: np.ndarray) -> None:
        """Set firmware position targets, in ctrl (actuator) order, radians."""
        self.controller.q_target[:] = np.asarray(q_target, dtype=float)

    def update(self) -> None:
        """Compute torques and rewrite friction/damping. Call before mj_step."""
        self.controller.update()

    def describe(self) -> str:
        return (
            f"BAM {MOTOR_NAME}/{MOTOR_MODEL} kp_fw={self.bam_model.actuator.kp:g} "
            f"vin={self.vin:.2f}V (nominal, no DR) "
            f"force_limit=+/-{self.force_limit:.2f} Nm"
        )
