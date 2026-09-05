# Feature: S5 — forward hop (E1) + landing quality

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Extend the existing hop spike environment so it can answer the **S5** question: *can Microduck hop
forward and land upright?* Three new reward terms in `mdp.py` (forward progress while airborne, landing
quality, landing impact), a `forward: bool` variant of the hop env factory, a new registered task id and
its backlash twin, and an extension to the CPU flight probe that measures the open-loop forward-distance
baseline before any GPU money is spent.

This is a **spike env**. Its deliverable is a trustworthy decision, not a polished gait. "Done" is not
"the duck hops forward well"; it is "we know whether it can, with data good enough to bet Phase 1 on."

## User Story

As the developer of the Microduck Hopscotch project
I want an environment that rewards forward flight and upright landings, with a measured no-learning baseline
So that I can decide whether hopscotch is *jumping* between cells or *stepping* into them, before committing Phase 1

## Problem Statement

The brief's Success #1 is *"Microduck intentionally hops forward and lands upright."* Neither half is
currently answered or even measurable:

1. **Forward is unrewarded.** The hop env (`microduck_hop_env_cfg.py`) deliberately hops *in place* —
   twist ranges crushed to ±0.05, `rel_standing_envs = 0.5`. Nothing pays for travelling.
2. **Landing is unmeasured.** Every existing gate (`simultaneous_flight`, `bilateral_foot_clearance`)
   screens the robot *during* flight on tilt and trunk height. A forward hop that reliably face-plants
   scores exactly as well as one that sticks the landing.
3. **A live reward conflict blocks E1.** See Solution Statement — this is the most important finding in
   the plan.

## Solution Statement

Add three reward terms and one env variant, all inside the existing 61D contract, plus a probe extension.

**`forward_flight_progress` (E1)** — pay capped forward body-frame velocity **multiplied by the flight
gate**. Zero while any foot is down, so it cannot be farmed by walking or running; capped, so a forward
dive earns no more than a controlled hop.

**`hop_landing_quality`** — upright-Gaussian × height-Gaussian × a **stateless "just landed from a real
flight" latch**, mirroring `roulade_landing_sharp`'s composite shape with the roll gate swapped out.

**`hop_landing_impact_penalty`** — self-negating (returns ≤ 0, takes a **positive** weight per the
microduck convention) on downward velocity at touchdown beyond a free allowance.

**THE CRITICAL FIX — `track_linear_velocity` 2.0 → 0.3.** The hop env retunes six inherited terms but
leaves velocity tracking at the walker's `weight = 2.0, std = sqrt(0.1) ≈ 0.316`
(`microduck_velocity_env_cfg.py:342-343`). With the hop env's near-zero commanded velocity that term pays
≈2.0/step for standing still and ≈0.4/step at 0.4 m/s — so **a forward hop costs ~1.6 reward/step**,
against an E1 term that the existing `test_flight_is_the_dominant_reward` invariant caps below
`simultaneous_flight`'s 5.0. Without this change E1 is out-massed before it starts. This is CLAUDE.md's
*"compare reward mass, not weights"* rule biting exactly as documented. Tracking is kept non-zero (0.3)
so it still discourages aimless drift.

## Out of Scope / Non-Goals

- **Not included: commanded hop distance (E3).** Deferred to Phase 1 by the architecture doc. E1 is
  un-commanded on purpose; `simultaneous_flight` keeps `command_name=None`.
- **Not included: course geometry, cells, or any exteroception.** S6, deferred behind S5.
- **Not included: consecutive-hop rhythm rewards or a hop curriculum.** Shaped by S5's result.
- **Not included: raising `body_pose_tracking` above 0.** `test_body_pose_tracking_stays_at_zero` exists
  precisely because this is the most tempting wrong change. Do not touch it.
- **Not changing: the existing `Mjlab-Hop-Flat-MicroDuck` task.** It stays exactly as-is as the
  hop-in-place A/B baseline. All new behavior is behind `forward=True`.
- **Not changing: BAM, DR, backlash twins, or the 61D observation layout.** Sim-only scope keeps these
  (architecture doc, "Scope" decision).
- **Not included: hardware, ONNX deployment, or `infer_policy.py` rehearsal.** S4, deferred.

## Feature Metadata

**Feature Type**: New Capability (spike environment)
**Estimated Complexity**: Medium
**Primary Systems Affected**: `tasks/mdp.py` (rewards), `tasks/microduck_hop_env_cfg.py` (env cfg),
`tasks/__init__.py` (registration), `scripts/hopscotch/flight_probe.py` (measurement), `tests/`
**Dependencies**: None new. torch, mujoco, mjlab 1.3.0 — all already pinned.

## Related Work

**Implements**: Spike **S5** in [`microduck-hopscotch-architecture.md`](../../microduck-hopscotch-architecture.md)
· **Epic**: [`microduck-hopscotch-project-brief.md`](../../microduck-hopscotch-project-brief.md)

**Back-references**:

- `microduck-hopscotch-architecture.md` — inherits: sim-only scope, hardware deferred, 61D contract kept,
  E1-now/E3-later encoding, course-as-free-variable, 2-3 run budget. **Do not reopen these.**
- `docs/hopscotch-rules.md` — fork working conventions and the penalty-sign rule.
- `docs/s1-flight-probe.md` — the two metric traps and the 34 ms open-loop baseline.
- `docs/prior-art-hop.md` — where the 0.035 m clearance target came from.
- `docs/command-block.md` — why `body_pose_tracking` stays at weight 0.

**Forward-references**: (none yet — Phase 1 gets sliced after S5 reports)

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

- `src/mjlab_microduck/tasks/mdp.py` (lines 7191-7358) - Why: the hopscotch reward section. `simultaneous_flight`
  and `bilateral_foot_clearance` — the two terms the new ones must compose with, and the section comment
  explaining the metric traps. **New terms go directly below these, in the same section.**
- `src/mjlab_microduck/tasks/mdp.py` (lines 7054-7085) - Why: `roulade_landing_sharp` — the exact
  composite shape to mirror for `hop_landing_quality` (upright Gaussian × height Gaussian × gate).
- `src/mjlab_microduck/tasks/mdp.py` (lines 3912-3928) - Why: the only in-repo use of
  `contact_sensor.data.last_air_time` — confirms the field exists on `feet_ground_contact` and is indexed
  `[:, :2]` for left/right foot.
- `src/mjlab_microduck/tasks/mdp.py` (line 1504, 4729) - Why: `sensor.data.current_contact_time` usage —
  the other half of the landing latch.
- `src/mjlab_microduck/tasks/mdp.py` (lines 1269, 4697, 6014) - Why: `root_link_lin_vel_b[:, 0]` — the
  established body-frame forward-velocity accessor for E1.
- `src/mjlab_microduck/tasks/microduck_hop_env_cfg.py` (whole file, 270 lines) - Why: the factory being
  extended. Its docstring enumerates all seven deliberate deviations from the walker — read it before
  changing any weight.
- `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` (lines 333-355) - Why: the inherited reward
  weights, including `track_linear_velocity` 2.0 / std sqrt(0.1) — the term this plan retunes.
- `src/mjlab_microduck/tasks/__init__.py` (lines 232-240, 260-286) - Why: the exact registration call
  shape and the `_BACKLASH_TASKS` table row format.
- `tests/test_simultaneous_flight.py` (whole file) - Why: **the test pattern to mirror.** Hand-rolled
  duck-typed `_Env` / `_Scene` / `_Asset` fakes, pure torch, no mjlab env construction, runs on CPU in ms.
- `tests/test_bilateral_clearance.py` (whole file) - Why: the same pattern with a `site_pos_w` fake and a
  `_FeetCfg` stub.
- `tests/test_hop_cfg.py` (whole file) - Why: the cfg-invariant assertions the new variant must not break,
  especially `test_flight_is_the_dominant_reward` and `test_every_penalty_term_has_a_sign_that_can_only_log_negative`.
- `scripts/hopscotch/flight_probe.py` (lines 56-108, 176-258) - Why: `leg_extension_pattern`'s Jacobian
  derivation (and its warning against hand-guessing patterns), and `simulate()`'s per-step loop where
  forward-displacement tracking must be added.

### New Files to Create

- `tests/test_forward_flight_progress.py` - Unit tests for the E1 reward term
- `tests/test_hop_landing.py` - Unit tests for `hop_landing_quality` and `hop_landing_impact_penalty`
- `tests/test_hop_forward_cfg.py` - Cfg invariants for the `forward=True` variant

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

The authoritative references for this work are **in-repo**, not external. There is no public mjlab API
reference with stable anchors; do not go hunting for one.

- `CLAUDE.md` (repo root) — sections **"Reward design"** and **"Sim2real footguns"**
  - Why: the penalty sign convention, the no-jackpot rule, and "compare reward mass, not weights" — all
    three are load-bearing for this plan.
- `docs/s1-flight-probe.md`
  - Why: the two metric traps (contact-loss is not flight; per-foot air time is not simultaneous flight)
    and the 34 ms open-loop baseline the probe extension parallels.
- `microduck-hopscotch-architecture.md` — sections **"Key decisions"** and **"S5"**
  - Why: the inherited decisions and the exact decision rule the run will be judged against.
- [mjlab](https://github.com/mujocolab/mjlab) — for `ContactSensor` field semantics if the in-repo usages
  are ambiguous. Read the source, not a doc site.

### Patterns to Follow

**Reward function signature** — every term takes `env` first, then tuned scalars with defaults, then
`asset_cfg` last:

```python
def simultaneous_flight(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    min_flight_s: float = 0.02,
    ...
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
```

**NaN discipline** — every tensor read from sim state is sanitized at the point of read, with the
"unsafe" fallback value chosen so it *fails the gate*:

```python
cos_tilt = torch.nan_to_num(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=-1.0)
z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
```

Note `nan=-1.0` for `cos_tilt` (a NaN orientation must read as "fallen"), `nan=0.0` for heights. A NaN
reaching the reward sum kills the whole run via rsl_rl's `check_nan`.

**Height is always terrain-relative** — never raw world z:

```python
asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
```

**Tilt via the quaternion, no trig in the hot path** — `cos(tilt) = R22 = 1 - 2(qx² + qy²)`.

**The settle guard** — every gated positive term excludes freshly-reset envs, because a robot spawned
clear of the floor passes every physical gate on step 0:

```python
settled = env.episode_length_buf > settle_steps
```

**Penalty sign convention (CLAUDE.md, and enforced by a test)** — two styles coexist:
- mjlab-base cost functions return ≥ 0 → **negative** weight (e.g. `roulade_overspeed_penalty`).
- microduck `*_penalty` / `*_l1` functions **self-negate** (return ≤ 0) → **positive** weight.

`tests/test_hop_cfg.py::test_every_penalty_term_has_a_sign_that_can_only_log_negative` asserts that any
**reward-term key** ending in `_penalty` or `_l1` has `weight >= 0`. Since this plan names its term
`hop_landing_impact_penalty`, **the function must return ≤ 0 and the cfg must use a positive weight.**
Getting this backwards double-negates into a reward for slamming into the ground.

**Cfg factory pattern** — `ENABLE_*` toggles and tuned constants at module top, factory takes
`(play: bool, rough: bool)`, mutations applied to the inherited cfg dict:

```python
cfg.rewards["air_time"].weight = 0.5
cfg.rewards["new_term"] = RewardTermCfg(func=microduck_mdp.fn, weight=2.0, params={...})
```

**Test pattern** — duck-typed fakes, no mjlab env. Copy the `_SensorData` / `_Sensor` / `_AssetData` /
`_Asset` / `_Terrain` / `_Scene` / `_Env` ladder verbatim from `tests/test_simultaneous_flight.py:26-68`
and extend it. Every test name states the behavior it pins; every test comment states the failure mode it
blocks. Follow that — it is the house style and it is why these tests are worth reading.

---

## IMPLEMENTATION PLAN

### Phase 0: Video plumbing — make the run watchable

**Depends on:** nothing. **Independent of:** Phases 1-3. Do it first anyway; it is small and it gates AC #12.

**Why this is in the plan and not a side quest.** CLAUDE.md, under reading a run: *"Sim metrics can pass
while the video fails the human eye — watch the video AND check which geom/axis touches."* This plan's
Level 4 acceptance was originally five wandb numbers and no video, which contradicts that instruction —
and the pipeline currently cannot produce a video even if asked, because `scripts/hf/uploader.py` ships
only `model_*.pt` and `params/*`. Videos written inside an HF job die with the container.

There is no local alternative: `warp` reports `cuda available: False` on the dev machine, so `uv run train`
and `uv run play` cannot run here at all. Remote video is the **only** way to see this policy move.

**Tasks:**

- Teach the uploader to ship videos
- Enable video on the smoke and training runs, with an interval that actually produces clips

### Phase 1: Measure the open-loop forward baseline

**Independent of:** Phases 2-3 (different file, no shared code). Can run in parallel.
**Feeds:** Phase 3's constants — `FORWARD_VEL_CAP` and the S5 decision threshold are grounded by this.

Extend the CPU probe to measure how far an open-loop countermovement travels, giving E1 a no-learning
baseline exactly as the 34 ms figure did for flight. Free, on CPU, and the last probe caught two metric
traps that would each have cost a GPU run.

**Tasks:**

- Derive a rearward-push direction from the leg Jacobian (companion to the existing extension direction)
- Sweep a lean/push blend and record net forward displacement across the flight phase
- Report a forward-distance baseline and re-state the S5 decision rule against it

### Phase 2: The three reward terms

**Depends on:** nothing (pure functions + unit tests).

**Tasks:**

- Extract the shared trunk gate from the two existing hop terms (refactor, guarded by 27 existing tests)
- Implement `forward_flight_progress`, `hop_landing_quality`, `hop_landing_impact_penalty`
- Unit-test each against its exploit

### Phase 3: The `forward=True` env variant and registration

**Depends on:** Phase 2 (the reward functions must exist to be referenced).

**Tasks:**

- Add `forward: bool` to the hop factory, wiring the three terms and the `track_linear_velocity` fix
- Add `MicroduckHopForwardRlCfg` with a distinct `experiment_name`
- Register `Mjlab-HopForward-Flat-MicroDuck` and its backlash twin
- Cfg-invariant tests for the new variant

### Phase 4: Validation

**Depends on:** Phases 2-3.

**Tasks:**

- Full CPU suite green, registry lists both new tasks
- Remote 64-env / 5-iteration smoke test on HF Jobs (**no local GPU — this cannot run on this machine**)
- Confirm every penalty logs ≤ 0 and both new positive terms read non-zero under a random policy

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### UPDATE `scripts/hf/uploader.py`

- **IMPLEMENT**: Add video files to the watched set. The glob block is `uploader.py:37-40`; add one line
  alongside the existing `model_*.pt` / `params/*` globs:
  ```python
  files += [p for p in root.glob("**/videos/**/*.mp4")]
  ```
  mjlab writes to `<log_dir>/videos/train/`, and `CKPT_ROOT` defaults to `logs/rsl_rl`, so the existing
  `rel = f.relative_to(root)` path-in-repo logic already mirrors run dirs correctly — no other change.
- **PATTERN**: `scripts/hf/uploader.py:37-55` — the existing glob-then-dedupe-by-mtime loop.
- **IMPORTS**: none new
- **GOTCHA**: **Skip files still being written.** The loop dedupes on `st_mtime` and re-uploads on change,
  so a video caught mid-write uploads a truncated file and then re-uploads it — wasted commits and a
  corrupt artifact in between. Guard it: only upload a video whose mtime is older than a few seconds
  (`time.time() - mtime > 5`). Checkpoints don't need this because they're written atomically enough at a
  60 s poll; a video is written progressively over its 200 frames.
- **GOTCHA**: Update the module docstring — it currently says *"Watches `logs/rsl_rl/**/model_*.pt`"*,
  which becomes false.
- **VALIDATE**: `CKPT_ROOT=<tmp> CKPT_REPO=<user>/scratch CKPT_ONE_SHOT=1 uv run python scripts/hf/uploader.py`
  against a temp dir containing a dummy `videos/train/x.mp4` — confirm it is picked up and pathed correctly
- **SATISFIES**: AC #12

### UPDATE the run commands to record video

- **IMPLEMENT**: Add the video flags to the Level 4 smoke test and to whatever launches the real run.
  Confirmed spellings (verified against `train --help` on 2026-09-04) — these are **top-level flags, not
  `--agent.*`**:
  ```
  --video True --video-length 200 --video-interval <N>
  ```
- **PATTERN**: `src/mjlab_microduck/train_cli.py` forwards argv untouched to `mjlab.scripts.train`, so
  every mjlab flag is already available. Nothing to add to our CLI.
- **GOTCHA**: **`--video-interval` defaults to 2000 and the trigger is `step % interval == 0`, so a
  5-iteration smoke test (≈120 env steps) records ZERO videos.** Enable video, run the smoke test, get
  nothing back, and it looks like the plumbing is broken when it is working correctly. Use
  `--video-interval 24` (one clip per iteration) for the smoke test.
- **GOTCHA**: For the real run, pick the interval so you get roughly 8-12 clips across training — that
  progression (early flail → mid → late) is the actual diagnostic value. At 1000-1500 iterations ×
  `NUM_STEPS_PER_ENV = 24` that is ~24k-36k steps, so `--video-interval 3000` is about right.
- **GOTCHA**: On Windows, prefix with `PYTHONIOENCODING=utf-8`. Without it the local process dies on
  non-ASCII output — confirmed to break `train --help` outright, not just the HF log streamer.
- **VALIDATE**: after the smoke test, confirm `.mp4` files land in the Hub checkpoint repo
- **SATISFIES**: AC #12

### UPDATE `scripts/hopscotch/flight_probe.py`

- **IMPLEMENT**: Add `leg_push_pattern(model, stand_qpos)` — the companion to `leg_extension_pattern`,
  returning the joint direction that drives the foot **backward** (dfoot_x < 0) while keeping foot pitch
  flat (dpitch = 0). Same Jacobian/SVD machinery, different null-space rows: solve the `[dfoot_z; dpitch]`
  rows instead of `[dfoot_x; dpitch]`, and orient so positive coefficient pushes the foot rearward
  (which drives the trunk forward).
- **PATTERN**: `scripts/hopscotch/flight_probe.py:56-108` — mirror `leg_extension_pattern` exactly,
  including the normalise-on-the-knee step and the sign-orientation check.
- **IMPORTS**: none new (`mujoco`, `numpy`, `math` already imported)
- **GOTCHA**: The file's docstring warns explicitly that hand-guessing leg patterns is wrong for this
  model — *"foot pitch goes as (−hip + knee − ankle), so [the intuitive] pattern rotates the foot
  ~228°/rad and topples the robot at ±0.1 rad quasi-statically."* **Derive it, do not guess it.**
- **VALIDATE**: `uv run python scripts/hopscotch/flight_probe.py --top 3` (still runs, still prints a
  derived extension pattern, baseline STAND settle tilt still small)
- **SATISFIES**: AC #9

### UPDATE `scripts/hopscotch/flight_probe.py`

- **IMPLEMENT**: Extend `Result` with `forward_m: float` (net trunk x displacement from takeoff to
  landing) and `s_push: float`. In `simulate()`, record `data.qpos[0]` at the step flight begins
  (`n_down` transitions to 0) and at the step it ends, and store the difference for the *best* flight
  interval. Add an `s_push` parameter blending `leg_push_pattern` into the control alongside `s_extend`,
  and extend the grid sweep with a small `pushes` list.
- **PATTERN**: `scripts/hopscotch/flight_probe.py:214-258` — the existing per-step loop already tracks
  `cur_flight` / `best_flight` and captures `tilt_at_land` on the transition. Hook forward displacement
  into the same transition logic so it is measured over the *same* interval.
- **IMPORTS**: none new
- **GOTCHA**: Measure displacement over the **flight interval only**, not the whole episode — a duck that
  walks forward then hops in place would otherwise report a large "forward hop". Mirror how
  `tilt_at_land` is captured exactly at the transition rather than at the end.
- **GOTCHA**: `is_hop` must keep gating on `max_tilt_deg < 30.0` and `apex_rise_m > 0.002`. A forward
  *topple* travels a long way — do not let forward distance bypass the topple screen.
- **VALIDATE**: `uv run python scripts/hopscotch/flight_probe.py --top 5` prints a `fwd_mm` column and a
  forward baseline line
- **SATISFIES**: AC #9, AC #10

### REFACTOR `src/mjlab_microduck/tasks/mdp.py`

- **IMPLEMENT**: Extract the trunk gate duplicated by `simultaneous_flight` and
  `bilateral_foot_clearance` into a private helper:
  ```python
  def _hop_trunk_gate(
      env: ManagerBasedRlEnv,
      min_height: float,
      max_tilt_deg: float,
      settle_steps: int,
      asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:   # bool (num_envs,) — upright AND risen AND settled
  ```
  Rewrite both existing terms to call it. Behavior must be bit-identical.
- **PATTERN**: `mdp.py:7265-7282` and `mdp.py:7335-7357` — the two copies being merged.
- **IMPORTS**: none new
- **GOTCHA**: Keep the asymmetric `nan_to_num` fallbacks — `nan=-1.0` for `cos_tilt`, `nan=0.0` for `z`.
  Collapsing both to 0.0 would make a NaN orientation read as *upright*.
- **GOTCHA**: This refactor is safe **only** because 27 existing tests cover these two functions. Run
  them before and after; any diff is a bug in the refactor, not an improvement.
- **VALIDATE**: `uv run --with pytest pytest tests/test_simultaneous_flight.py tests/test_bilateral_clearance.py -q`
  (all pass, unchanged)
- **SATISFIES**: AC #8

### ADD `forward_flight_progress` to `src/mjlab_microduck/tasks/mdp.py`

- **IMPLEMENT**:
  ```python
  def forward_flight_progress(
      env: ManagerBasedRlEnv,
      sensor_name: str,
      vel_cap: float = 0.4,
      min_flight_s: float = 0.02,
      max_flight_s: float = 0.30,
      min_height: float = 0.10,
      max_tilt_deg: float = 30.0,
      settle_steps: int = 2,
      asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
  ```
  Returns `clamp(v_fwd, 0, vel_cap) / vel_cap` × the airborne window × `_hop_trunk_gate(...)`, in [0, 1].
  Airborne window is `min_flight_s < min(current_air_time) <= max_flight_s`, identical to
  `simultaneous_flight`. Forward velocity is `root_link_lin_vel_b[:, 0]`, `nan_to_num(..., nan=0.0)`.
  Docstring must state: positive weight; why it is gated on flight (walking cannot farm it); why it is
  capped (a forward dive must not out-earn a controlled hop); and that backward velocity pays zero rather
  than negative.
- **PATTERN**: `mdp.py:7214-7291` (`simultaneous_flight`) for structure and gate reuse;
  `mdp.py:4697` / `mdp.py:1269` for the `root_link_lin_vel_b[:, 0]` accessor.
- **IMPORTS**: none new
- **GOTCHA**: **Body frame, not world frame.** `root_link_lin_vel_b[:, 0]` is forward-relative-to-heading,
  which is what "hop forward" means. `root_link_lin_vel_w[:, 0]` is world +x and would reward a duck that
  drifts sideways after turning.
- **GOTCHA**: Clamp at `min=0.0` **before** dividing by `vel_cap`. An unclamped negative would produce a
  negative reward on a positive-weight term, breaking the sign invariant.
- **GOTCHA**: This term pays nothing until flight exists — that is intended, not a bug. Do **not** add a
  ground-phase ramp; `bilateral_foot_clearance` is already the dense ramp toward flight.
- **VALIDATE**: `uv run --with pytest pytest tests/test_forward_flight_progress.py -q`
- **SATISFIES**: AC #1

### CREATE `tests/test_forward_flight_progress.py`

- **IMPLEMENT**: Mirror the fake-env ladder from `tests/test_simultaneous_flight.py`, extended with
  `root_link_lin_vel_b`. Tests, each pinning one failure mode:
  - `test_pays_for_forward_velocity_while_airborne` — 0.4 m/s airborne upright → 1.0
  - `test_walking_forward_pays_nothing` — **the central case**: full forward velocity, one foot planted
    (`[[0.20, 0.0]]`) → 0.0
  - `test_backward_flight_pays_zero_not_negative` — v_fwd = −0.4 → exactly 0.0
  - `test_velocity_is_capped` — 2.0 m/s scores the same as `vel_cap`, no jackpot
  - `test_partial_velocity_gives_partial_credit` — half `vel_cap` → 0.5
  - `test_toppling_forward_pays_nothing` — 50° pitch, fast forward → 0.0
  - `test_sitting_and_sliding_pays_nothing` — trunk z = 0.05 → 0.0
  - `test_contact_flicker_pays_nothing` — air time 0.005 → 0.0
  - `test_freshly_reset_env_banks_nothing` — `step=1` → 0.0
  - `test_never_negative` / `test_nan_velocity_does_not_pay`
  - `test_batched_envs_are_independent`
- **PATTERN**: `tests/test_simultaneous_flight.py:26-77` verbatim for the fakes and `_reward` helper.
- **IMPORTS**: `import torch`, `import math`, `from mjlab_microduck.tasks.mdp import forward_flight_progress`
- **GOTCHA**: `_AssetData` must gain a `root_link_lin_vel_b` field; the existing fake only has
  `root_link_pos_w` and `root_link_quat_w`.
- **VALIDATE**: `uv run --with pytest pytest tests/test_forward_flight_progress.py -q`
- **SATISFIES**: AC #1, AC #6

### ADD `hop_landing_quality` to `src/mjlab_microduck/tasks/mdp.py`

- **IMPLEMENT**:
  ```python
  def hop_landing_quality(
      env: ManagerBasedRlEnv,
      sensor_name: str,
      target_height: float = 0.1167,
      height_std: float = 0.02,
      upright_std: float = 0.3,
      min_flight_s: float = 0.02,
      landing_window_s: float = 0.15,
      settle_steps: int = 2,
      asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
  ```
  `upright_gaussian × height_gaussian × just_landed`, in [0, 1], positive weight. The **stateless latch**:
  ```python
  landed = (last_air.min(dim=1).values >= min_flight_s) \
         & (contact_t.min(dim=1).values > 0.0) \
         & (contact_t.min(dim=1).values <= landing_window_s)
  ```
  where `last_air = sensor.data.last_air_time[:, :2]` and `contact_t = sensor.data.current_contact_time[:, :2]`.
- **PATTERN**: `mdp.py:7054-7085` (`roulade_landing_sharp`) for the two-Gaussian composite;
  `mdp.py:3916` for `last_air_time[:, :2]`; `mdp.py:1504` for `current_contact_time`.
- **IMPORTS**: none new
- **GOTCHA**: **`target_height` is 0.1167 (measured), NOT 0.095.** The velocity env's
  `nominal_height = 0.095` is a known ~22 mm error, documented in `docs/command-block.md`; it survives
  only because `body_pose_tracking` runs at weight 0. Do not propagate it here.
- **GOTCHA**: Verify the sensor's zero semantics before trusting the latch — this plan assumes
  `current_contact_time == 0` while a foot is airborne (mirroring `current_air_time == 0` while planted).
  If that assumption is wrong, `min(contact_t) > 0` is not "both feet down" and the latch silently never
  fires. Check against `mdp.py:1504` and `mdp.py:4729` usage, or read the `ContactSensor` source.
- **GOTCHA**: `min` across feet on `last_air_time`, not `max` — the shorter of the two air phases is the
  simultaneous part. `max` would let an ordinary stride qualify. This is the same reduction bug
  `test_min_across_feet_not_max` guards in the flight term.
- **GOTCHA**: The `min_flight_s` condition is what stops this becoming a general "stand upright" reward
  that double-pays alongside the inherited `upright` term (already at weight 2.0).
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_landing.py -q`
- **SATISFIES**: AC #2

### ADD `hop_landing_impact_penalty` to `src/mjlab_microduck/tasks/mdp.py`

- **IMPLEMENT**:
  ```python
  def hop_landing_impact_penalty(
      env: ManagerBasedRlEnv,
      sensor_name: str,
      free_speed: float = 0.5,
      min_flight_s: float = 0.02,
      impact_window_s: float = 0.04,
      settle_steps: int = 2,
      asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
  ```
  **Self-negating — returns ≤ 0, takes a POSITIVE weight.**
  `-(clamp(-vz - free_speed, min=0.0)) * just_landed_now`, where `vz = root_link_lin_vel_w[:, 2]` and
  `just_landed_now` uses the same latch as `hop_landing_quality` but with the tight `impact_window_s`
  (~2 control steps at 50 Hz) so it samples touchdown rather than the whole settling phase.
- **PATTERN**: `mdp.py:7136-7152` (`roulade_overspeed_penalty`) for the excess-above-threshold shape —
  **but note that function returns positive and takes a negative weight**; this one must do the opposite.
- **IMPORTS**: none new
- **GOTCHA**: **The sign is the whole ballgame.** Because the reward-term key ends in `_penalty`,
  `test_every_penalty_term_has_a_sign_that_can_only_log_negative` requires `weight >= 0`, so the function
  MUST return ≤ 0. A positive return with a positive weight pays the policy to slam into the ground.
  Add an explicit `test_never_positive` and verify `Episode_Reward/hop_landing_impact_penalty ≤ 0` in the
  smoke test.
- **GOTCHA**: `free_speed = 0.5` m/s is derived from the probe's ~11 mm apex
  (`v = sqrt(2·9.81·0.011) ≈ 0.46 m/s`) — i.e. a *currently achievable* hop pays nothing. Re-derive it if
  Phase 1 measures a higher apex, or the term will tax every hop the policy can actually perform.
- **GOTCHA**: World-frame `vz` here (`root_link_lin_vel_w[:, 2]`), not body-frame — impact is against the
  ground, which is a world-frame fact. This differs deliberately from E1's body-frame forward velocity.
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_landing.py -q`
- **SATISFIES**: AC #3

### CREATE `tests/test_hop_landing.py`

- **IMPLEMENT**: Fake-env ladder extended with `last_air_time`, `current_contact_time` and
  `root_link_lin_vel_w`. For `hop_landing_quality`:
  - `test_pays_on_upright_landing_after_real_flight` → near 1.0
  - `test_landing_after_no_flight_pays_nothing` — `last_air_time` below `min_flight_s`; **the central
    case**, this is what stops it double-paying for merely standing
  - `test_tilted_landing_scores_low` / `test_crouched_landing_scores_low`
  - `test_pays_nothing_while_still_airborne` — `current_contact_time` = 0
  - `test_pays_nothing_long_after_landing` — beyond `landing_window_s`
  - `test_min_across_feet_not_max` — one foot's long air phase must not qualify a stride
  - `test_freshly_reset_env_banks_nothing`, `test_never_negative`, `test_nan_does_not_pay`

  For `hop_landing_impact_penalty`:
  - `test_never_positive` — **the sign invariant**, across soft/hard/no-landing cases
  - `test_soft_landing_is_free` — `|vz| < free_speed` → exactly 0.0
  - `test_hard_landing_is_penalized` → strictly negative, linear in excess
  - `test_no_penalty_without_a_real_flight` and `test_no_penalty_outside_the_impact_window`
  - `test_nan_does_not_pay`
- **PATTERN**: `tests/test_bilateral_clearance.py` for the fakes-with-extra-fields shape.
- **IMPORTS**: `torch`, `math`, both functions from `mjlab_microduck.tasks.mdp`
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_landing.py -q`
- **SATISFIES**: AC #2, AC #3, AC #6

### UPDATE `src/mjlab_microduck/tasks/microduck_hop_env_cfg.py`

- **IMPLEMENT**: Add module-level constants below the existing block:
  ```python
  FORWARD_VEL_CAP = 0.4          # m/s; re-derive against the Phase 1 probe baseline
  LANDING_TARGET_Z = 0.1167      # MEASURED walk-model settle height — not 0.095
  LANDING_WINDOW_S = 0.15
  IMPACT_FREE_SPEED = 0.5        # m/s; ~free-fall from the probe's 11 mm apex
  IMPACT_WINDOW_S = 0.04
  HOP_TRACK_LIN_VEL_WEIGHT = 0.3 # down from the walker's 2.0 — see docstring
  ```
  Change the factory signature to `make_microduck_hop_env_cfg(play=False, rough=False, forward=False)`.
  Under `if forward:` add the three reward terms (`forward_flight_progress` weight **1.5**,
  `hop_landing_quality` weight **1.0**, `hop_landing_impact_penalty` weight **+0.5**) and set
  `cfg.rewards["track_linear_velocity"].weight = HOP_TRACK_LIN_VEL_WEIGHT`.
- **PATTERN**: `microduck_hop_env_cfg.py:160-205` — the existing `cfg.rewards[...] = RewardTermCfg(...)`
  block and the retuned-inherited-terms block.
- **IMPORTS**: none new (`microduck_mdp`, `RewardTermCfg` already imported)
- **GOTCHA**: **Extend the module docstring** with deviations 8-10 (forward progress, landing quality, the
  velocity-tracking fix) in the same numbered style. That docstring is how the next reader learns why each
  weight is what it is — the file's whole convention. Include the arithmetic: tracking at 2.0/std 0.316
  costs a 0.4 m/s hop ~1.6 reward/step.
- **GOTCHA**: All three new terms must stay **below** `simultaneous_flight`'s 5.0, or
  `test_flight_is_the_dominant_reward` fails. That test is an invariant, not an obstacle — flight is the
  goal, forward is a modifier on it.
- **GOTCHA**: Do NOT widen the twist ranges. That is encoding E2, explicitly rejected in the architecture
  doc because velocity tracking has a strictly easier solution than hopping — walking.
- **GOTCHA**: Leave `simultaneous_flight`'s `command_name=None`. E1 is un-commanded by decision;
  `test_flight_reward_is_ungated_by_command_for_the_spike` asserts it.
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_cfg.py -q` (the **baseline** variant must be
  completely unaffected — `forward=False` is the default)
- **SATISFIES**: AC #4, AC #5

### ADD `MicroduckHopForwardRlCfg` to `src/mjlab_microduck/tasks/microduck_hop_env_cfg.py`

- **IMPLEMENT**: Copy `MicroduckHopRlCfg` with `experiment_name="hop_forward"` and
  `run_name="hop_forward"`. Keep `symmetry_cfg=None`.
- **PATTERN**: `microduck_hop_env_cfg.py:230-270`
- **GOTCHA**: A shared `experiment_name` would overwrite the hop baseline's logs in
  `logs/<experiment_name>/` and collide in wandb — which destroys the A/B this variant exists for.
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_forward_cfg.py -q`
- **SATISFIES**: AC #5

### UPDATE `src/mjlab_microduck/tasks/__init__.py`

- **IMPLEMENT**: Import `MicroduckHopForwardRlCfg` alongside `MicroduckHopRlCfg`. Register
  `Mjlab-HopForward-Flat-MicroDuck` with `make_microduck_hop_env_cfg(forward=True)` /
  `make_microduck_hop_env_cfg(play=True, forward=True)`. Add the backlash row:
  ```python
  ("Mjlab-HopForward-Flat-Backlash-MicroDuck", make_microduck_hop_env_cfg,
   {"forward": True}, MicroduckHopForwardRlCfg, _BL_WALK),
  ```
- **PATTERN**: `tasks/__init__.py:232-240` for the registration call; `:264` for the backlash row format
  (note it already passes `{"rough": True}`-style kwargs, so `{"forward": True}` fits the existing shape
  with no changes to the loop at `:279-286`).
- **IMPORTS**: extend the existing `from .microduck_hop_env_cfg import (...)` block
- **GOTCHA**: `_BL_WALK`, not `_BL_ALLCOL` — the hop family mirrors Velocity's walk model so backlash
  A/Bs stay unconfounded. The existing comment at `:263` says exactly this; keep the new row next to it.
- **VALIDATE**: `uv run list-envs` shows both `Mjlab-HopForward-Flat-MicroDuck` and
  `Mjlab-HopForward-Flat-Backlash-MicroDuck`
- **SATISFIES**: AC #5

### CREATE `tests/test_hop_forward_cfg.py`

- **IMPLEMENT**: Cfg invariants for `make_microduck_hop_env_cfg(forward=True)`:
  - `test_forward_terms_absent_from_the_baseline_variant` — **guards the A/B**: `forward=False` has none
    of the three new terms and keeps `track_linear_velocity` at the walker's weight
  - `test_flight_still_dominates_every_new_term`
  - `test_velocity_tracking_is_demoted` — strictly less than the velocity env's, and still > 0
  - `test_landing_target_height_is_the_measured_one` — asserts ≈0.1167 and explicitly **not** 0.095
  - `test_impact_penalty_takes_a_positive_weight` — the sign convention
  - `test_every_penalty_term_has_a_sign_that_can_only_log_negative` — same check as the baseline
  - `test_twist_ranges_are_unchanged_from_the_baseline` — pins that we did not drift into E2
  - `test_body_pose_tracking_still_zero`, `test_flight_still_ungated_by_command`
  - `test_runner_cfg_experiment_name_is_distinct`
  - `test_tasks_are_registered` — both new ids in `list_tasks()`
  - `test_play_variant_builds`
- **PATTERN**: `tests/test_hop_cfg.py` verbatim in style — one assertion per documented failure mode,
  with the comment naming the failure.
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_forward_cfg.py -q`
- **SATISFIES**: AC #5, AC #6, AC #7

### UPDATE docs

- **IMPLEMENT**: Record the Phase 1 forward baseline in `docs/s1-flight-probe.md` (new section, mirroring
  the existing result table), and update `microduck-hopscotch-architecture.md`'s S5 block to replace the
  placeholder 5 cm threshold with the measured-and-reasoned figure.
- **PATTERN**: `docs/s1-flight-probe.md` — result table, "why it is trustworthy as a bound", "why it is
  optimistic".
- **GOTCHA**: The architecture doc flags 5 cm as *"asserted, not derived."* If Phase 1 shows the open-loop
  forward reach is far from 5 cm, change the threshold and say why — do not quietly keep a number the
  measurement contradicts.
- **VALIDATE**: manual read
- **SATISFIES**: AC #10

### VALIDATE end-to-end

- **IMPLEMENT**: Run the full CPU suite, then submit the remote smoke test.
- **GOTCHA**: **There is no local GPU.** `uv run train` cannot run on this machine; the smoke test must go
  through `--hf-jobs`. Budget for scheduling latency — measured at 46 / 0 / 12 minutes across three
  submissions, which dominates the ~5 minutes of actual compute.
- **GOTCHA**: On Windows the HF log streamer dies on non-ASCII output unless `PYTHONIOENCODING=utf-8` is
  set. The job itself is unaffected, but the local command reports a false failure — a known, unfixed
  issue recorded in `docs/hopscotch-rules.md`.
- **VALIDATE**: see VALIDATION COMMANDS below
- **SATISFIES**: AC #7, AC #11

---

## TESTING STRATEGY

### Unit Tests

`pytest`, CPU-only, no GPU and no mjlab env construction. Reward functions are tested against hand-rolled
duck-typed fakes (`tests/test_simultaneous_flight.py:26-68` is the canonical ladder), which is why the
199-test suite runs in ~12 s. Every new reward term gets its own file. **Each test pins one documented
failure mode and says so in a comment** — that is the house style and the reason these tests are readable.

### Integration Tests

Cfg-invariant tests that build the real env config and assert on weights, params, gates and registration.
These catch the errors that actually cost money (sign inversions, weight drift, a term silently absent
from the variant) and still run on CPU.

The genuine integration test is the **remote 64-env / 5-iteration smoke test** — CLAUDE.md reports it
catches ~95% of config errors for cents.

### Edge Cases

- Walking forward at full speed with one foot planted → E1 pays exactly 0
- Backward flight → 0, never negative
- Forward topple (fast, tilted, feet off) → all three terms reject it
- Landing with no preceding flight → landing quality pays 0 (does not double-pay with `upright`)
- Still airborne / long after touchdown → landing latch closed
- Freshly reset env at step 0-2 → every gated term banks nothing
- NaN in air time, velocity, orientation or foot height → 0, never NaN into the reward sum
- Soft landing within `free_speed` → impact penalty exactly 0, not a small negative

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Syntax & Style

```bash
uv run --with ruff ruff check src/ tests/ scripts/
```

`ruff` is configured in `pyproject.toml` (`[tool.ruff]`) but is **not** a declared dependency, so a bare
`uv run ruff` fails with "program not found". Note the repo has 7 pre-existing findings in
`scripts/hopscotch/flight_probe.py` alone — compare against `git stash` rather than expecting a clean run.

### Level 2: Unit Tests

```bash
uv run --with pytest pytest tests/test_forward_flight_progress.py tests/test_hop_landing.py -q
uv run --with pytest pytest tests/test_simultaneous_flight.py tests/test_bilateral_clearance.py -q   # refactor safety net
```

### Level 3: Integration Tests

```bash
uv run --with pytest pytest tests/ -q          # expect 199 existing + new, all green, ~15 s
uv run list-envs | grep HopForward             # both base and Backlash ids present
uv run python scripts/hopscotch/flight_probe.py --top 5
```

### Level 4: Manual Validation

**No local GPU — this stage is remote.**

```bash
PYTHONIOENCODING=utf-8 uv run train Mjlab-HopForward-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5 \
    --video True --video-length 200 --video-interval 24 --hf-jobs
```

Then in wandb, confirm on the smoke run:

1. It builds and steps NaN-free for 5 iterations.
2. Actor obs is 61D (unchanged).
3. Every reward term computes — 21 terms expected (18 existing + 3 new).
4. **Every `Episode_Reward/<penalty>` is ≤ 0**, `hop_landing_impact_penalty` included. CLAUDE.md calls
   this check infallible for sign inversions.
5. `Episode_Reward/forward_flight_progress` and `Episode_Reward/hop_landing_quality` read **non-zero**
   under the random initial policy — if either is identically 0, its gate is unreachable and the term is
   silently dead. (This is exactly how the two existing hop terms were validated.)
6. **`.mp4` files land in the Hub checkpoint repo, and you watch one.** Metrics-only acceptance is
   explicitly insufficient here — CLAUDE.md: *"Sim metrics can pass while the video fails the human eye —
   watch the video AND check which geom/axis touches."* On a smoke run the policy is random, so the bar is
   only "the plumbing delivers watchable footage"; on a real run this is where you judge whether the hop
   is a hop or a stumble the numbers happened to like.

### Level 5: Additional Validation (Optional)

```bash
uv run scripts/export.py Mjlab-HopForward-Flat-MicroDuck --wandb-run-path <...>   # ONNX still exports
```

---

## ACCEPTANCE CRITERIA

- [ ] AC #1 — `forward_flight_progress` pays capped forward body-frame velocity only while genuinely
      airborne, upright and risen; pays exactly 0 for walking, backward flight and topples
- [ ] AC #2 — `hop_landing_quality` pays only after a genuine simultaneous flight, scoring upright and
      height at touchdown; does not double-pay for standing
- [ ] AC #3 — `hop_landing_impact_penalty` returns ≤ 0, takes a positive weight, and is free below
      `free_speed`
- [ ] AC #4 — `track_linear_velocity` demoted to ~0.3 in the forward variant, with the reward-mass
      arithmetic recorded in the module docstring
- [ ] AC #5 — `Mjlab-HopForward-Flat-MicroDuck` and `-Backlash-` twin registered, on the walk model, with
      a distinct `experiment_name`
- [ ] AC #6 — every new term has unit tests covering its exploit, its NaN path and its sign
- [ ] AC #7 — full CPU suite green (199 existing + new), zero regressions; `Mjlab-Hop-Flat-MicroDuck`
      byte-identical in behavior
- [ ] AC #8 — the `_hop_trunk_gate` refactor leaves the two existing terms bit-identical
- [ ] AC #9 — the flight probe measures and reports open-loop forward displacement, still screening topples
- [ ] AC #10 — the S5 decision threshold is grounded in the measured baseline, not the placeholder 5 cm,
      and the docs say which
- [ ] AC #11 — remote smoke test passes all six checks in Level 4
- [ ] AC #12 — training video is recorded remotely, uploaded to the Hub checkpoint repo, and watchable;
      the uploader skips partially-written files

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes
- [ ] No linting errors
- [ ] Remote smoke test confirms the feature works
- [ ] Acceptance criteria all met
- [ ] Architecture doc + `docs/s1-flight-probe.md` updated with the measured baseline

---

## OPEN QUESTIONS / ASSUMPTIONS

**Assumed — `current_contact_time == 0` while a foot is airborne**, mirroring `current_air_time == 0`
while planted. The landing latch depends on this: if false, `min(contact_t) > 0` is not "both feet down"
and the latch never fires (or always fires). **Verify against the `ContactSensor` source before trusting
`hop_landing_quality`'s tests** — the unit tests use fakes and would happily pass against a wrong
assumption. Cheapest check: assert the term reads non-zero in the smoke test (Level 4, check 5).

**Assumed — `vel_cap = 0.4` m/s.** Chosen to match the velocity env's walking command ceiling. Phase 1's
probe should either confirm or replace it. Too low caps a good hop; too high leaves the term nearly flat
across the achievable range and kills its gradient.

**Assumed — `free_speed = 0.5` m/s** for the impact penalty, from free-fall off the probe's 11 mm apex.
If a trained policy reaches a much higher apex, this taxes every landing it can physically make. Re-derive
against Phase 1.

**Assumed — weights 1.5 / 1.0 / +0.5.** These are starting points, not tuned values. CLAUDE.md's
expectation of 2-5 reward-hacking iterations applies; the architecture doc budgets 2-3 runs for exactly
this. Judge from a headless eval of the checkpoint before changing them — past "failures" in this project
turned out to be early checkpoints and mis-set success criteria.

**Open — should `hop_landing_quality` also gate on forward displacement?** As specified it pays for any
upright landing after any flight, including a purely vertical hop. That is deliberate for the spike (the
two questions stay separable), but if runs show the policy earning landing quality from vertical hops
while ignoring E1, coupling them is the obvious next lever. Do not pre-emptively couple them.

**Open — no `foot_slip` / `foot_clearance` retune.** These inherited terms carry `command_threshold`
gates tied to the twist command, which stays near zero here. Whether they interfere with a forward hop is
unmeasured. Flagged, not changed — changing untested things is how spikes get confounded.

## NOTES (open canvas)

### Why the velocity-tracking fix is the highest-value line in this plan

The arithmetic, spelled out, because it is the thing most likely to be quietly reverted by someone
tidying up:

| | reward/step |
|---|---|
| `track_linear_velocity` at v = 0 (standing still) | ≈ 2.0 |
| `track_linear_velocity` at v = 0.4 m/s | 2.0 × exp(−(0.4/0.316)²) ≈ 0.41 |
| **cost of hopping forward at 0.4 m/s** | **≈ 1.6** |
| `forward_flight_progress` at weight 1.5, fully airborne | ≤ 1.5 |

E1 loses, *and* it only pays during the small fraction of steps that are airborne while the tracking
penalty applies on every step. Demoting tracking to 0.3 drops the opposition to ≈0.24/step. This is
CLAUDE.md's *"the same action_rate weight is 4× weaker under a 4×-larger positive task stack"* lesson in
the opposite direction.

### Why E1 is on from step 0 while `action_rate_l2` is curriculum'd in

CLAUDE.md's rule is that an **attempt-tax** active during exploration makes "do nothing" win. That applies
to *penalties*. `forward_flight_progress` is a positive term gated on a skill: it pays 0 until flight
exists and can never make doing nothing better than trying. No curriculum needed — adding one would just
delay the shaping until after the policy has already committed to vertical hops.

### Alternatives rejected

- **E2 (forward velocity via `twist.lin_vel_x`)** — rejected in the architecture doc with receipts. Walking
  is the cheaper solution to a velocity command and the base env is tuned to find it.
- **Stateful landing latch** (mirroring `_update_roulade_accum`) — rejected. `simultaneous_flight`'s
  docstring argues the stateless case directly: *"a reward that accumulates its own flight counter would
  have to be reset-aware."* The sensor already carries the state we need.
- **`body_impact_cost` for landing softness** — unusable. It needs a trunk/head contact sensor, and the
  hop family inherits `robot_walk.xml`, which per the README has *"stripped trunk/head contacts."*
- **Folding softness into the landing composite** — rejected in planning. A product collapses on any single
  deficient factor, which is desirable at a goal state but here would make a slightly-hard landing erase
  an otherwise good one, and removes the ability to weight accuracy and softness independently.

### Sequencing risk

Phase 1 (probe) feeds constants into Phase 3 (cfg), but does not block Phases 2-3 structurally — the
functions can be written and unit-tested against the defaults, then the constants updated. If you want the
shortest wall-clock, run Phase 1 while writing Phase 2. Do **not** submit the GPU smoke test before Phase 1
reports, though: the whole point of the probe is to avoid judging a run against an ungrounded threshold.

## AMENDMENTS

<!-- newest at the bottom -->

- 2026-09-04 — **Added Phase 0 (video plumbing) and AC #12.** The plan's Level 4 acceptance was five
  wandb numbers and no video, which contradicts CLAUDE.md's explicit instruction to watch the video
  because metrics can pass while the video fails. Investigation while answering "how do I see the duck"
  found three facts that made this a hole rather than a nice-to-have: (1) `warp` reports
  `cuda available: False` on the dev machine, so nothing renders locally and remote video is the only
  option; (2) `scripts/hf/uploader.py` globs only `model_*.pt` and `params/*`, so videos written in the
  job are destroyed with the container; (3) mjlab already supports `--video` / `--video-length` /
  `--video-interval` and `train_cli.py` forwards argv untouched, so no CLI work is needed. Also corrected
  the Level 1 lint command (`ruff` is configured in `pyproject.toml` but is not a declared dependency, so
  a bare `uv run ruff` fails).
