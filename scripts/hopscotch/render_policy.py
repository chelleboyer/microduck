"""Render a trained ONNX policy to an mp4 — locally, on CPU, at any resolution.

WHY THIS EXISTS
---------------
Until now the only way to SEE a Microduck policy move was a training job's
``--video`` clips. That is a poor instrument for the one thing this project
decides everything by:

* it needs CUDA, and the dev machine has none, so `uv run play` cannot run here;
* the framing is whatever was passed at SUBMIT time and cannot be changed
  afterwards — mjlab's defaults are 320x240 at camera distance 3.0, which
  renders a 25 cm robot as a few pixels of an empty floor. Every clip of the
  original bunny hop (run s5-forward-hop) is stuck like that;
* re-rendering an old policy means resubmitting a job at that era's commit,
  because a checkpoint only means what its own env meant.

This renders any exported ONNX offscreen, free, in seconds, with the camera
tracking the robot — as many takes and angles as you like.

**IT DRIVES BAM BY DEFAULT**, the same voltage/back-EMF actuator model training
uses (``bam.mujoco.MujocoController``, exactly as scripts/infer_policy.py sets
it up). That is the whole point of not simply reusing hop_eval.py's rollout:
that harness drives plain MJCF position servos, and the gap shows up precisely
where a hop lives — fast joint motion, where back-EMF dominates. It reported a
median of −2.2 mm/hop on a policy training logged at ~95% of its velocity cap
and the video plainly showed hopping forward. A video rendered through position
servos would be a picture of a robot we never trained.

``--no-bam`` falls back to position servos to reproduce hop_eval.py's dynamics
when you want to see what that harness is actually looking at.

THE TWIST SLOTS MEAN DIFFERENT THINGS IN DIFFERENT POLICIES
-----------------------------------------------------------
Same trap hop_eval.py fell into, so the same flag. Policies from S5.3 onward
carry a PHASE CLOCK in command slots 0-2 (``[cos 2πφ, sin 2πφ, 0]``, 1.6 s);
earlier ones carry a velocity command, where all-zero is the idle state. Feeding
a phase carrier zeros is off the unit circle entirely — a state that occurs
nowhere in training — and it does not error, it just renders a policy behaving
like something it isn't. The mode is printed in the header. Use ``--no-phase``
for the hop-in-place baseline and anything before S5.3 (s5-forward-hop,
s51-forward-hop, s52-head-rhythm, and the community policies).

Usage:
    # the original bunny hop, 720p, tracking camera, real time
    uv run python scripts/hopscotch/render_policy.py logs/dl/policy.onnx \
        --no-phase --out logs/bunnyhop/bunnyhop-hires.mp4

    # the current recipe, slow motion (fps below 50 slows it down)
    uv run python scripts/hopscotch/render_policy.py policy.onnx --fps 15

    # a side-on view, further back, 10 seconds
    uv run python scripts/hopscotch/render_policy.py policy.onnx \
        --azimuth 180 --distance 1.2 --seconds 10
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[2]

# Framing that makes a 25 cm robot legible. Derived from the flags the good
# training clips were shot with (--env.viewer.distance 0.9 --elevation -12),
# pulled slightly closer because this camera TRACKS the robot instead of
# watching a fixed point of floor.
DEFAULT_DISTANCE = 0.8
DEFAULT_ELEVATION = -12.0
DEFAULT_AZIMUTH = 90.0
# 50 Hz is the control rate, so one frame per control step at fps=50 is real
# time. Lower fps is slow motion, which is how you actually see a 32 ms flight.
CONTROL_HZ = 50


def _load(name: str, relpath: str):
    """Import a sibling script as a module (the tests' idiom)."""
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses need this before exec
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path, help="exported policy (scripts/export.py)")
    ap.add_argument("--out", type=Path, default=Path("logs/render/policy.mp4"))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--settle", type=float, default=0.5,
                    help="open-loop hold at STAND before the policy takes over")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.02, help="init joint noise (rad)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=CONTROL_HZ,
                    help=f"{CONTROL_HZ} = real time; lower = slow motion")
    ap.add_argument("--distance", type=float, default=DEFAULT_DISTANCE)
    ap.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION)
    ap.add_argument("--azimuth", type=float, default=DEFAULT_AZIMUTH)
    ap.add_argument("--scene", type=Path, default=None,
                    help="scene XML; defaults to hop_eval.py's walk scene")
    # Command semantics — see the module docstring.
    ap.add_argument("--phase-period", type=float, default=None,
                    help="hop-cycle period driven into the twist slots, s "
                         "(defaults to the env's HOP_PERIOD)")
    ap.add_argument("--no-phase", dest="phase_period", action="store_const", const=0.0,
                    help="twist slots are a VELOCITY command (pre-S5.3 policies)")
    ap.add_argument("--hop-cmd", type=float, default=0.06, help="body_pose[2], m")
    ap.add_argument("--raw-accel", dest="projected_gravity", action="store_false",
                    default=True, help="policy trained on raw accelerometer")
    # Actuation.
    ap.add_argument("--no-bam", dest="bam", action="store_false", default=True,
                    help="drive MJCF position servos instead of BAM — reproduces "
                         "hop_eval.py's dynamics, NOT training's")
    ap.add_argument("--vin", type=float, default=7.4, help="battery volts (BAM)")
    ap.add_argument("--vin-drop-gain", type=float, default=0.1,
                    help="load-dependent voltage sag (BAM)")
    ap.add_argument("--kp", type=float, default=20.0, help="position gain (--no-bam)")
    ap.add_argument("--kd", type=float, default=0.3, help="damping (--no-bam)")
    args = ap.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"no such policy: {args.onnx}")

    he = _load("hop_eval", "scripts/hopscotch/hop_eval.py")
    if args.phase_period is None:
        args.phase_period = he.PHASE_PERIOD_S
    scene = args.scene or (REPO / he.SCENE)

    # ── model + actuation ────────────────────────────────────────────────────
    bam_ctrl = None
    if args.bam:
        ip = _load("infer_policy", "scripts/infer_policy.py")
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, args.vin, ip.BAM_MAX_CURRENT)
        model, data, bam_ctrl, _names = ip.load_mujoco_with_bam(
            str(scene), bam_model, 0.005, args.vin_drop_gain, ip.BAM_VIN_MIN
        )
    else:
        model = mujoco.MjModel.from_xml_path(str(scene))
        # The MJCF's kp~0.5 gains are placeholders that cannot hold STAND — see
        # hop_eval.py and docs/s1-flight-probe.md.
        model.actuator_gainprm[:, 0] = args.kp
        model.actuator_biasprm[:, 1] = -args.kp
        model.actuator_biasprm[:, 2] = -args.kd
        data = mujoco.MjData(model)

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    stand_qpos = model.key_qpos[key].copy()
    stand_ctrl = model.key_ctrl[key].copy()

    session = ort.InferenceSession(str(args.onnx))
    runner = he._Runner(model, data, session, args.hop_cmd,
                        args.projected_gravity, args.phase_period)

    print(f"policy {args.onnx.name}  obs {runner.obs_dim}D  ->  {args.out}")
    print("actuators: " + ("BAM m6 (as trained)" if args.bam else
                           f"position servos kp={args.kp:g} (hop_eval dynamics, NOT training's)"))
    print("twist slots: " + (f"PHASE CLOCK at {args.phase_period:g} s (S5.3+)"
                             if args.phase_period > 0 else
                             "zero VELOCITY command (--no-phase, pre-S5.3)"))
    print(f"{args.width}x{args.height} @ {args.fps} fps"
          + ("" if args.fps == CONTROL_HZ else
             f"  ({CONTROL_HZ/args.fps:.1f}x slow motion)"))

    # ── init ─────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = stand_qpos
    for a in runner.qpos_idx:
        data.qpos[a] += float(rng.normal(0.0, args.noise))
    data.ctrl[:] = stand_ctrl
    mujoco.mj_forward(model, data)
    runner.last_action[:] = 0.0
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)

    dt = runner.dt
    trunk = runner.trunk

    def drive(target=None):
        """One control step. ``target`` None = hold the stand pose open-loop."""
        if target is not None:
            if bam_ctrl is not None:
                bam_ctrl.q_target[:] = target
            else:
                data.ctrl[:] = target
        for _ in range(he.DECIMATION):
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)

    settle_target = (stand_qpos[runner.qpos_idx].copy()
                     if bam_ctrl is not None else stand_ctrl)
    for _ in range(int(args.settle / dt)):
        drive(settle_target)

    runner.phase = float(rng.random()) if args.phase_period > 0 else 0.0

    # ── render ───────────────────────────────────────────────────────────────
    # The scenes declare a 640x480 offscreen framebuffer, and mujoco.Renderer
    # refuses anything larger. Raising it on the loaded model is what lets this
    # script render at resolutions the training clips could never reach.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)

    cam = mujoco.MjvCamera()
    cam.distance = args.distance
    cam.elevation = args.elevation
    cam.azimuth = args.azimuth

    frames = []
    fell_at = None
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        for i in range(int(args.seconds / dt)):
            if args.phase_period > 0:
                runner.phase = (runner.phase + dt / args.phase_period) % 1.0
            obs = runner.observe()[None, :]
            action = session.run(None, {runner.input_name: obs})[0][0]
            runner.last_action = action.astype(np.float32)
            drive(he.DEFAULT_POSE + runner.last_action * he.ACTION_SCALE)

            # Track the robot — the single biggest legibility win over mjlab's
            # fixed camera, which watches a point of floor the duck hops out of.
            if fell_at is None and he._tilt_deg(data.xquat[trunk]) > 60.0:
                fell_at = i * dt

            cam.lookat[:] = data.xpos[trunk]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
    finally:
        renderer.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio  # arrives with mjlab; not a declared dep
    imageio.mimwrite(str(args.out), frames, fps=args.fps, macro_block_size=1)

    tilt = he._tilt_deg(data.xquat[trunk])
    print(f"\n{len(frames)} frames -> {args.out}")
    print(f"ended {'UPRIGHT' if tilt < 30 else f'FALLEN (tilt {tilt:.0f} deg)'}"
          + (f", first went over at t={fell_at:.2f}s of {args.seconds:g}s"
             if fell_at is not None else ""))


if __name__ == "__main__":
    main()
