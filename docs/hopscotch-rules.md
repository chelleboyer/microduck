# Microduck Hopscotch

Teach Pollen Robotics' Microduck to hop, then to hopscotch — trained in MuJoCo simulation on Hugging
Face Jobs. **Sim-only** as of session 3; hardware is deferred, not dropped (see Scope).

> **Start here (fresh session).** Read this file, then
> [`../microduck-hopscotch-architecture.md`](../microduck-hopscotch-architecture.md) (decisions) and
> [`hopscotch-routine.md`](./hopscotch-routine.md) (what we're ultimately building). Current state and
> the next action are in **Session 5 — START HERE** below. Everything after that section is standing
> context; everything in it is live.

## Scope (session 3, 2026-09-04)

**Sim-only.** Getting hopscotch working in simulation is what matters right now. Physical deployment is
**deferred, not dropped** — BAM, domain randomization and the backlash twins stay on, because they cost
nothing to keep (already wired; the one known working Microduck hop was trained *with* them) and dropping
them is a one-way door. What the scope change actually unlocks is that two hardware-justified switches are
now ours to flip if we want them: mjlab's native `height_scan`, and promoting `base_lin_vel` from
critic-only to the actor. Neither is flipped yet — see S6 in the architecture doc.

## Session 5 — START HERE (state as of 2026-09-06)

### The requirement, in the user's words

Watching the S5.3 run's video: *"he's kind of hopping, but more scooting himself around in a circle"*,
and the target is **"he needs to hop straight ahead, pause and then repeat, all with his head up."**

Four properties, and the honest status of each after four forward runs:

| Requirement | Status | Why |
|---|---|---|
| **Hop** (leaves the ground) | partial | Genuine flight exists, but it is shallow and most travel is on the ground |
| **Straight ahead** | **failing** | Heading drifts **+107 deg/s** — a full circle every ~3.4 s. Nothing has priced yaw since S5.3 |
| **Pause, then repeat** | **failing** | **8.7 hops per 1.6 s cycle**; the phase clock is being ignored. Third attempt, third failure |
| **Head up** | **failing** | `head_pose_bias` is *diverging* under weight 3.0; harness measures 21.3° mean head error |

### The runs so far (all `Mjlab-HopForward-Flat-MicroDuck`, 4096 envs)

| Increment | Run | wandb | Hub repo | Result |
|---|---|---|---|---|
| S5 | `s5-forward-hop`, 1500 it | — | `chelleboyer/s5-forward-hop` | Forward bunny hop, confirmed on video. Airborne 52% of its life |
| S5.1 | 1500 it | `5b77zi33` | `chelleboyer/s51-forward-hop` | Displacement-at-landing replaces air time |
| S5.2 | ~700 it (crashed at 692) + rerun | `8j6xu1nk` | `chelleboyer/s52-head-rhythm` | Head to walker strength; pause-scaled hop. **Rhythm failed measurably** |
| **S5.3** | **2500 it, COMPLETED** | **`tbs1k86h`** | **`chelleboyer/s53-phase-hop`** | Phase cycle + crouch + apex. **Crouch works; rhythm and heading do not** |

`chelleboyer/s53-phase-hop` holds 11 checkpoints, 21 videos and `exported/policy.onnx` — the in-job
auto-export is confirmed working end to end. Local copies: `logs/s53/`.

### What last night's run (S5.3) actually shows

`Episode_Reward` logs the **weighted** per-step mean, so divide by the weight to recover behaviour:

| Term | End | Raw | Read |
|---|---|---|---|
| `hop_displacement` | 4.04 | 0.40 | Dominant term; the iter-300 handover took cleanly |
| `hop_crouch` | 0.49 | 0.16 of a **0.20 ceiling** | **The crouch works** — ~80% of the crouch window scores. S5.3's clearest win |
| `hop_apex_rise` | 0.45 | peaked **0.61 @ iter 909**, then −25% | Height is being traded away for distance |
| `simultaneous_flight` | 0.067 | airborne **10% → 27%** | Bounce creeping back (S5 was 52%) |
| `hop_landing_quality` | 0.062 | 0.031 | Weakest term in the stack, three runs running |
| `hop_settle` | 0.057 | 0.038 | The pause is not happening |
| `head_pose_bias` | **−0.85** | −0.28/step | **Worsening** — was −0.43 at iter 909, where the weight already pinned at 3.0 |

Everything else: every penalty ≤ 0 ✓, episode length 910–960 (falls are rare under BAM), mean reward
**plateaued from ~iter 1400** — the last 1000 iterations bought nothing measurable, so the next run
should be ~1500 iters, not 2500.

### What the eval battery says (now that it can see this policy)

`uv run python scripts/hopscotch/hop_eval.py logs/s53/policy.onnx --episodes 20 --duration 8`,
20 episodes × 8 s, phase clock driven at 1.6 s:

```
genuine hops        61  (3.05/episode, best consecutive 3)
  per hop cycle     8.71     <- intended cadence is 1.00
takeoff phase lock  0.23     <- 1.00 = every hop on the same beat
travel in the air   34% of 3560 mm total path
heading drift       +107.5 deg/s
head error          21.3 deg mean |err|, pitch +3.1 deg from HOME
landing tilt        median 8.5 deg, 98% upright
forward per hop     median 4.2 mm    (NOT TRUSTWORTHY — see the caveat below)
apex rise           median 0.1 mm    (NOT TRUSTWORTHY — same reason)
```

**The three numbers that matter are the ones the actuator gap cannot touch**, and all three corroborate
the video: the clock is ignored (8.71 hops/cycle, lock 0.23), two thirds of the travel happens with a
foot on the floor (**scooting**, not hopping), and the heading turns one way at a steady ~107 deg/s
(**the circle**). The sign is consistently positive, i.e. a systematic left/right asymmetry, not noise.

### The diagnosis — three named causes, each with a fix

**1. Nothing prices yaw, and the circle is free.** S5.3 replaced the twist command with a phase carrier
and, correctly, deleted the three terms that read it as a velocity — including
**`track_angular_velocity`** (`microduck_hop_env_cfg.py:534`). Nothing replaced it, so heading is
completely unconstrained. Worse, `hop_displacement` measures travel **along the takeoff heading**, which
was designed so a turn-and-drift scores nothing *within* a hop — but it also means turning *between*
hops is free, and a hop in a new direction is paid in full. Turning is a strictly cheaper way to keep
earning than hopping straight. Fixes to weigh: `heading_hold_reward` (`mdp.py:4874`, already used by the
roller/swizzle envs), a yaw-rate penalty, and/or measuring displacement along a **fixed episode heading**
rather than the per-hop takeoff heading.

**2. The phase clock is advisory, not binding.** `hop_crouch_by_phase` pays for being low inside phase
0.10–0.30, and that worked — but `hop_displacement`, `hop_apex_rise` and `simultaneous_flight` pay at
**any** phase, so the policy takes the crouch money and then hops 8–9 times per cycle regardless. The
rhythm was never a requirement, only a suggestion. Upstream PR #28's ladder does bind it: crouch window,
then a **launch window** (phase 0.28–0.38) that pays vertical velocity. Fix: gate the hop payments on a
takeoff inside a launch window, and/or pay for feet-down outside it.

**3. Head droop is not responding to weight.** This is the third instrument tried (touchdown factor →
walker-strength bias → both), and the term is now *diverging* under a fixed weight of 3.0. Something is
paying the policy more to keep the head down than the penalty costs — plausibly the apex/displacement
terms using the 280 g head as a countermovement, which deviation 3 deliberately allowed. **Do not simply
raise the weight again**; measure what the head is doing during the cycle first
(`microduck_velocity_env_cfg.py:729-737` records that tightening `head_pose_tracking`'s std made the
policy stop moving entirely).

A fourth, lower-confidence observation: **apex and displacement appear to compete.** Apex peaked at iter
909 and decayed 25% while displacement kept climbing. If they are genuinely trading, one of them needs a
floor rather than a weight.

### S5.4 — BUILT 2026-09-06, not yet run

All three diagnoses above are now implemented in `Mjlab-HopForward-Flat-MicroDuck`, locked by
`tests/test_hop_cadence.py` (368 CPU tests green, ~18 s):

- **`heading_hold` (1.5)** — restores a heading constraint to an env that has had none since the phase
  carrier displaced `track_angular_velocity`. Reuses the existing `heading_hold_reward`, which prices
  the yaw *angle* against the spawn heading so the policy can steer back; its docstring records why a
  yaw-*rate* penalty is the wrong tool.
- **The clock becomes binding for the terms that PAY.** `_hop_cadence_factor` pays the first genuine hop
  of each cycle in full and the extras at `repeat_pay` (0.0), with a gentle taper toward a launch window
  at phase 0.28–0.38 (floor 0.5, so an off-beat hop is still worth half). The shape matters: it does
  **not** scale down the hop that counts — that is exactly how S5.2's pause requirement silenced the
  term — it removes the reward for the extras and leaves the first one whole.
- **Repricing: `hop_displacement` 10 → 25, `hop_apex_rise` 6 → 15.** Rate-limiting a payment without
  repricing it is an attempt tax, and CLAUDE.md is explicit that those make "do nothing" win. S5.3
  collected on 8.7 hops per cycle; paying one hop across one 0.15 s window in a 1.6 s cycle is a ~9%
  duty, a ~10x mass cut against penalties that do not shrink. **If the next run stops hopping, these two
  are the dials — not the cadence.**
- **The head is deliberately untouched**, so the run is a clean read on the hypothesis below.

Also fixed in passing: the latch used to apply `min_ground_s` *inside* the step-guarded update, so its
value depended on which reward term the manager happened to evaluate first — it worked only because
`hop_displacement` was registered before `hop_settle`. The latch now banks raw quantities and every
scaling is applied by the term that pays. A test pins the ordering-independence.

**The head hypothesis this run tests.** The per-joint breakdown explains why the aggregate error looked
survivable: `neck_pitch −20.4°` and `head_pitch +28.1°` are a **counter-fold** that nearly cancels in
the mean, and `head_yaw` sits at **+21.1°** — cranked to one side, *in the same direction* as the
+97 deg/s body drift. The hypothesis is that the head is being used to steer, and that taking the circle
away takes the crank with it. If `head_yaw` error collapses next run, the head problem was a heading
problem. If it does not, the head needs its own instrument — and a fourth weight increase still isn't it.

### What to watch on the next run

- **`Episode_Reward/hop_displacement` will DROP sharply at first, and that is the mechanism working**,
  not a failure: it is paid on ~1 hop per cycle instead of ~8.7. Judge it by whether it *recovers* as
  hops get bigger, not by its level against S5.3.
- **`Episode_Reward/heading_hold` should climb toward ~1.5** (its weight); it is earnable from step 0 by
  standing still, so a value stuck near 0.7 means the policy is still circling.
- **`hop_settle` should finally become non-zero.** It has read ~0 for three runs because a scooting duck
  never holds still; if it stays at 0 while heading_hold rises, the pause is a separate problem.
- Every penalty ≤ 0, and episode length not collapsing (the repricing is the risk here).

### What is owed before the next run

- **The 64-env / 5-iteration smoke test**, always. This machine has no CUDA, so nothing is ever proven
  to build or step locally.
- Any reward change lands with a cfg test first (`tests/test_hop_*.py`), CPU, free.
- Target ~1500 iterations, not 2500 — the run plateaus by 1400.

```bash
# SMOKE TEST — cents, ~minutes. PYTHONIOENCODING is mandatory on Windows:
# without it the log streamer (and even `train --help`) dies on non-ASCII.
PYTHONIOENCODING=utf-8 uv run train Mjlab-HopForward-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5 --hf-jobs

# THEN the real run. NOTE `--video True`, not a bare `--video`: mjlab parses
# train args with tyro, which wants an explicit value and exits 2 without one.
PYTHONIOENCODING=utf-8 uv run train Mjlab-HopForward-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 1500 --hf-jobs --video True
```

## The eval battery — what to trust, and what it cannot tell you

`scripts/hopscotch/hop_eval.py` drives the exported ONNX through CPU MuJoCo. It is free, local and
instant, and it has now been wrong in two different ways, both of which returned a **confident bad
verdict rather than an error**:

- **Fixed 2026-09-06 — the twist slots.** S5.3 made them a phase carrier
  (`[cos 2πφ, sin 2πφ, 0]`, 1.6 s); the harness was still writing zeros, which is off the unit circle
  entirely and a state no training step ever produced. It now drives the clock (`--phase-period`,
  default 1.6 s) and `--no-phase` selects the old velocity-command semantics for the hop-in-place
  baseline and any pre-S5.3 policy. **The mode is printed in the header — check it**, because neither
  mode can detect that it is the wrong one. `tests/test_hop_eval_phase.py` fails if the period or the
  flight gates drift from the cfg.
- **Fixed earlier — projected gravity.** It defaulted to raw accelerometer while training uses
  `USE_PROJECTED_GRAVITY = True`. The only tell was a physically impossible 0.0 mm apex rise.
  **Trust the impossible number, not the verdict.**

**Still open — the actuator gap.** It drives plain MJCF **position servos**; training drives **BAM**
(voltage model, back-EMF). On the S5 policy it reported −2.2 mm/hop while training logged ~95% of the
velocity cap and the video plainly showed forward hopping.

- **Do not trust:** forward travel per hop, apex rise, and (new evidence) **fall rate** — the S5.3 run
  held 910–960-step episodes under BAM while the harness ended 100% of episodes fallen. That
  contradiction is the actuator gap talking, not the policy.
- **Do trust:** hop count, hops per cycle, takeoff phase lock, heading drift, the air-vs-ground share
  of travel (both halves measured identically, so the ratio survives), and landing tilt. Geometry and
  contact, not torque.

Closing that gap — or accepting it permanently — is still an open task.

## Tooling (all CPU, all free)

- `scripts/hopscotch/hop_eval.py` — headless eval battery; per-hop displacement, flight duration, apex
  rise, landing tilt, upright rate, consecutive streaks, plus (2026-09-06) hops per cycle, takeoff
  phase lock, heading drift, air-vs-ground travel share and whole-run head error.
- `scripts/hopscotch/training_montage.py` — stitches a run's clips into one labelled progression video.
- `scripts/hopscotch/flight_probe.py --view` — watch the best open-loop hop in slow motion.
- `scripts/export.py` works on CPU, and the **in-job auto-export works**, so a finished run leaves
  `exported/policy.onnx` in its own Hub repo.

## Historical: sessions 2–4

**Session 2 (2026-09-04).** Remote pipeline proven end to end. HF Jobs namespace `chelleboyer`
(personal; Pro active — the org `context-course` is NOT Enterprise and would fail); wandb entity
`chelleboyer-road-ranger`. `Mjlab-Hop-Flat-MicroDuck` registered (+ `-Backlash-` twin) with
`simultaneous_flight` and `bilateral_foot_clearance`. Prior art found
([`prior-art-hop.md`](./prior-art-hop.md)) — a community Microduck hop, which is why **S1 was closed
without spending**.

**Session 3 (2026-09-05).** S5 ran and the duck hopped forward. It also exposed the reward shape as
wrong: `simultaneous_flight` pays per airborne step, so air time WAS the objective and the policy spent
52% of its life airborne. Video works end to end (`osmesa` + `libosmesa6`, gated behind `_wants_video()`;
GLFW fails for want of Xlib, and EGL breaks `import mujoco` outright). The S5 threshold was **measured,
not asserted**: open-loop forward travel is 8.0 mm/hop, so the old 5 cm placeholder was ~6× beyond
physics; pass ≥25 mm, fail ≤8 mm.

**Session 4 (2026-09-05/06).** Three increments, all shipped and all trained:

- **S5.1** — `hop_displacement` (takeoff→touchdown along the takeoff heading, paid across the landing
  window, capped at 10 cm) becomes the main term, handed the lead by a curriculum at iter 300 rather
  than swapped in. `forward_flight_progress` cap 0.4 → 0.8 m/s (it had saturated at ~95%), weight
  1.5 → 0.5. Head priced at touchdown.
- **S5.2** — head up *always* via `head_pose_bias` at the walker's full 1/2/3 strength, plus a
  pause-scaled hop and a `hop_settle` reward. **The rhythm half failed measurably**: `hop_settle`
  earned ~0.005/1.5, the policy was airborne 53% of its life, and `hop_displacement` collapsed 7×
  because a 0.5 s pause demanded from step 0 is a requirement the current behaviour could not meet, so
  the term went silent.
- **S5.3** — the phase cycle (1.6 s `GroundPickPhaseCommand` in the twist slot), `hop_crouch_by_phase`
  and `hop_apex_rise`, adopting the structure both independent working hops on this robot use. Through
  S5, S5.1 and S5.2 **nothing in the stack had ever paid for the trunk going UP**.

Also merged: 44 upstream commits, pin `1e79c29` → `29e887e` (upstream's default branch is `develop`,
not `main` — see [`upstream-pin.md`](./upstream-pin.md)).

## Known-stale / loose ends

- [`tickets/microduck-hopscotch-phase-0.md`](./tickets/microduck-hopscotch-phase-0.md) frames MD-3
  around the **closed** S1 question. Phase 1 is still unsliced — slice it against
  [`hopscotch-routine.md`](./hopscotch-routine.md).
- [`plans/s5-forward-hop-and-landing-quality.md`](./plans/s5-forward-hop-and-landing-quality.md) is the
  S5.1 plan and is now a **historical artifact**; S5.2 and S5.3 were never planned in writing.
- `scripts/infer_policy.py` cannot run on Windows (`termios`/`tty`), never overrides the MJCF's
  placeholder `kp≈0.5` gains, and — unverified — **almost certainly has the same phase-carrier gap
  `hop_eval.py` just had**, since it drives the twist slots as a velocity command from the keyboard.
- The eval battery's 100%-fallen rate contradicts training's 910–960-step episodes. Actuator gap is the
  likely cause, but it is unproven.

## Where the project is going

[`hopscotch-routine.md`](./hopscotch-routine.md) specs all 14 steps of a human hopscotch turn against
Microduck's capabilities. [`plans/abridged-court-demo.md`](./plans/abridged-court-demo.md) is the
recommended first demo: a 3-square court, drop-not-throw marker, two-foot hops. **Only E3 blocks it**,
and its non-training steps need no GPU.

## Read these, in this order

1. [`../CLAUDE.md`](../CLAUDE.md) — **upstream's playbook.** Env-building workflow, invariants, joint
   layout, and the reward-design lessons. Authoritative; read it before touching rewards.
2. [`microduck-hopscotch-project-brief.md`](../microduck-hopscotch-project-brief.md) — the intent. What
   we're building and why. Success #1 is *"Microduck intentionally hops forward and lands upright"*.
3. [`microduck-hopscotch-architecture.md`](../microduck-hopscotch-architecture.md) — the decisions, with
   rationale, rejected alternatives, spikes and open questions. **The load-bearing doc.**
4. [`hopscotch-routine.md`](./hopscotch-routine.md) — **all 14 steps of a human hopscotch turn**, mapped
   to Microduck capabilities. This is what the project is building toward; slice Phase 1 against it.
5. [`plans/abridged-court-demo.md`](./plans/abridged-court-demo.md) — the recommended first demo:
   3-square court, dropped marker, two-foot hops. Only E3 blocks it.
6. [`command-block.md`](./command-block.md) — the 13D command block, where hop intent goes, and the
   `nominal_height` discrepancy.
7. [`prior-art-hop.md`](./prior-art-hop.md) — a community Microduck hop policy, what it proves and
   what it leaves open.
8. [`tickets/microduck-hopscotch-phase-0.md`](./tickets/microduck-hopscotch-phase-0.md) — Phase 0 work:
   MD-1, MD-2, MD-3.

## Constraints that are expensive to rediscover

**The duck is blind — by choice, not by nature.** All Microduck policies share a fixed **61-dimensional**
observation: 48 proprioception + a 13D command block `[twist(3), head_pose(4), body_pose(6)]`. No vision,
no height scan, no terrain sensing. So hopscotch is **choreography** — a sequence of commands over flat
ground — not perception. Never delete a command slot; unused slots are zero-padded to keep input neurons
alive.

**A slot's SEMANTICS can change even though its width cannot.** The forward hop env now carries a phase
clock in the twist slots, exactly as ground_pick, sit_stand, roller_crouch and spin already do. The 61D
contract is untouched, but every off-policy consumer — the eval battery, `infer_policy.py`, any future
runtime driver — must write what the policy was trained to read, and **feeding the wrong semantics fails
silently as a bad-policy verdict.**

The nuance that matters under the sim-only scope: blindness is a *deferred decision*, not a physical
limit. mjlab 1.3.0 ships a `height_scan` terrain ray-scan by default and
`microduck_velocity_env_cfg.py:533-537` deletes it from both groups, because *the real robot has no such
sensor*. We stay at 61D because it costs nothing until the duck needs to aim at a cell.

**Hybrid, not all-remote.** Only *training* needs CUDA. Verified on Windows without a GPU: `uv sync`
succeeds, the full CPU test suite passes in ~20 s, and CPU MuJoCo loads the real model — so config
errors, reward signs and physics checks are caught **locally, in under a minute, for free**. Spike **S3**
is answered: stay off WSL2, iterate locally, submit only real training runs.

**This repo is a fork of upstream, pinned.** `pollen-robotics/microduck_rl` (Apache 2.0) is kept as a
git remote and pulled deliberately, not continuously. Its distilled sim2real and reward-design playbook
is the repo-root [`CLAUDE.md`](../CLAUDE.md) — **read it before touching rewards**; it encodes months of
hard-won lessons and is more trustworthy than reasoning from first principles here.

**The core new work is one reward term.** mjlab's stock `feet_air_time` rewards *alternating*
single-foot air time — ordinary walking. A hop needs **simultaneous** flight, both feet off at once.
That term does not exist upstream and is not a reweighting of one that does.

**Verify physics on CPU before spending GPU.** Upstream calls this the single biggest time-saver, and
it paid immediately — see [`s1-flight-probe.md`](./s1-flight-probe.md). Two metric traps live here:
contact-loss is not flight (a duck falling over loses both contacts and logs great "air time"), and
`current_air_time` is *per-foot* (a normal walk reports 125–300 ms). Simultaneous flight is
`n_contact == 0`, gated on tilt and trunk rise.

**The open project risk is physics — now the *quality* of the forward half.** Whether an ~800 g biped on
compliant, backlash-heavy XL330 servos can leave the ground is answered. Whether it can travel through
the air, on a chosen heading, to a beat, is what the S5.x runs are still chasing.

**The course is a free variable.** With no physical course to match, cell size is ours to choose. Measure
the hop distance the policy achieves, then size the cells to it — never the other way round.

## Working conventions

- Spike tickets are done when they produce a **trustworthy decision**, not a working feature.
- Every `Episode_Reward/<penalty>` term must log ≤ 0 throughout training. Upstream calls this check
  infallible for catching reward-sign inversions.
- Expect 2–5 reward-hacking iterations before convergence. That is normal, not failure.
- Never hand-export a checkpoint — the observation normalizer must be baked in by `scripts/export.py`.
  In-sim play hides the bug because it applies the normalizer anyway.
- Keep `pyproject.toml` honest: `uv sync` is ground truth because HF Jobs runs it. Anything that works
  only via manually-installed local packages dies remotely.
- **Judge a run from the video AND the battery, never from the reward curves alone.** Every course
  correction in this project so far came from watching it move.
