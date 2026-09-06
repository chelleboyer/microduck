# The 13D command block — semantics, and where the hop goes

> Closes the architecture doc's largest open question ("Exact command-block semantics") and MD-3's
> first acceptance criterion. Read from source at pin `1e79c29`, 2026-09-04.
> Source: `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:640-745`.

Actor observation is **61D = 48 proprioception + 13D command**, laid out in this order and shared by
every policy so the runtime can hot-swap them. Never delete a slot.

## `twist` (3) — indices 0–2

`[lin_vel_x, lin_vel_y, ang_vel_z]`, velocity env ranges `(-0.4, 0.4)`, `(-0.3, 0.3)`, `(-1.0, 1.0)`.

Two non-obvious behaviours:

- **All-zero twist means "stand"**, not "no command". Sampled explicitly via `rel_standing_envs`,
  ramped by curriculum `0.02 → 0.25`. Upstream's playbook warns that feeding all-zeros at runtime
  looks like "the policy ignores the button".
- **Turn-in-place is its own bucket**, `TURN_IN_PLACE_FRACTION = 0.15`, because under uniform sampling
  it emerged as ~2% of experience and never trained.

**⚠ These three slots do not always carry a velocity.** Several envs replace the twist command with a
cyclic **phase carrier**, `[cos(2πφ), sin(2πφ), 0]` (`GroundPickPhaseCommand`): ground_pick,
sit_stand, roller_crouch, spin, and — since S5.3 — the forward hop, at a 1.6 s period. The 61D width is
unchanged, so ONNX export and the runtime need no changes; only the **meaning** of indices 0–2 differs
per policy.

Two consequences, both learned the expensive way:

- **Any off-policy driver must write what the policy was trained to read.** Zeros are the idle velocity
  command, but in a phase carrier `(cos, sin) = (0, 0)` is not on the unit circle at all — a state that
  occurs nowhere in training. `hop_eval.py` did exactly this and returned a confident BAD-POLICY
  verdict rather than an error (fixed 2026-09-06; `scripts/infer_policy.py` is unverified and likely
  still affected).
- **Deleting the velocity-tracking rewards is mandatory when you install a phase carrier** — they would
  pay the policy for matching `cos(2πt)` as a target speed. But note what else goes with them:
  `track_angular_velocity` is the only term constraining heading, and the forward hop lost its heading
  hold this way without noticing.

## `head_pose` (4) — indices 3–6

Deltas from HOME in joint order `neck_pitch, head_pitch, head_yaw, head_roll`. Primary reward in the
velocity env (`head_pose_tracking`, weight 2.0, std 0.5). Mechanical caps: ±1.10 rad neck/head pitch,
±1.40 head_yaw, ±0.31 head_roll.

## `body_pose` (6) — indices 7–12

**`[x, y, z, roll, pitch, yaw]`, a delta from nominal standing, with `nominal_height = 0.095` m.**

> **`nominal_height` does not match the model.** Measured 2026-09-04 on `scene_walk.xml`: the STAND
> keyframe authors trunk z = **120.0 mm**, settling to **116.7 mm** under load. The velocity env passes
> `nominal_height = 0.095` (`mdp.py:5323`, wired at `microduck_velocity_env_cfg.py:722`) and
> `body_pose_tracking_locomotion` defaults to `0.105` (`mdp.py:5408`) — so the velocity reference is
> ~22 mm low. Both measure the same quantity: `root_link_pos_w[:, 2] − env_origins[:, 2]`.
>
> Harmless *today* only because `body_pose_tracking` runs at weight 0 in the velocity env, so nothing
> optimizes against it — which is exactly how the constant survived. Anyone raising that weight
> (the standup env already does, via its own `STAND_Z`) inherits the error: a commanded `z = 0`
> would ask the robot to stand 22 mm shorter than it does. Re-measure before trusting it.

In the velocity env this whole block is **carried at reward weight 0**, with deliberately tiny ranges
(±5 mm on x/y/z, ±0.05 rad on roll/pitch/yaw) purely to keep the obs slot and its input neurons alive.
The standup env raises the weight and widens the ranges.

**So the entire 6D block is free in a velocity-derived hop env — not just one component.**

## Where the hop command goes

**Use `body_pose[2]` (z) as hop intent / amplitude. Keep `body_pose_tracking` at weight 0.**

The slot is already sampled non-zero from step 0, which satisfies the dead-neuron invariant with no
extra work.

### Do NOT raise `body_pose_tracking` to reward commanded hop height

`body_pose_tracking_6d` is a *pose-holding* reward (`z_std = 0.02`). Command `z = +0.05` and reward
tracking it, and the optimal policy is to **extend the legs and hold a tall stance** — strictly easier
than ballistic flight, and it scores better. The reward must come from the simultaneous-flight term
(`n_contact == 0`), not from tracking z.

This failure mode is documented in-repo, at `microduck_velocity_env_cfg.py:729-737`: tightening a pose
tolerance (`fine_std=0.1`) made the policy stop walking entirely by iteration 300 —
`air_time 1.01 → 0.02`, peak foot height `15 mm → 2 mm`, entropy collapsed `10.9 → 1.9`. Upstream's
note: *"Standing still scored higher, so it stood still."*

### Sampling

Give hop commands an **explicit bucket fraction**, modelled on `TURN_IN_PLACE_FRACTION`, rather than
sampling z uniformly. Uniform sampling is what buried turn-in-place at ~2% and left it untrained.

## Related

- Flight metric and physics baseline: [`s1-flight-probe.md`](./s1-flight-probe.md)
- `single_support_reward` (`tasks/mdp.py:4700`) already computes `n_contact` and enumerates the
  `0 blades down (flight/hop)` case — the detection primitive for the new reward exists.
