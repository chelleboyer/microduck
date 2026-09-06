# Microduck Hopscotch

Teach Pollen Robotics' Microduck to hop, then to hopscotch — trained in MuJoCo simulation on Hugging
Face Jobs. **Sim-only** as of session 3; hardware is deferred, not dropped (see Scope).

> **Start here (fresh session).** Read this file, then
> [`../microduck-hopscotch-architecture.md`](../microduck-hopscotch-architecture.md) (decisions) and
> [`hopscotch-routine.md`](./hopscotch-routine.md) (what we're ultimately building). Current state and
> the next action are in **Session 6 — START HERE** below. Everything after that section is standing
> context; everything in it is live.

## Scope (session 3, 2026-09-04)

**Sim-only.** Getting hopscotch working in simulation is what matters right now. Physical deployment is
**deferred, not dropped** — BAM, domain randomization and the backlash twins stay on, because they cost
nothing to keep (already wired; the one known working Microduck hop was trained *with* them) and dropping
them is a one-way door. What the scope change actually unlocks is that two hardware-justified switches are
now ours to flip if we want them: mjlab's native `height_scan`, and promoting `base_lin_vel` from
critic-only to the actor. Neither is flipped yet — see S6 in the architecture doc.

## Session 6 — START HERE (state as of 2026-09-06, opened session 6)

**Nothing has run since S5.4.** Working tree clean, 2 commits ahead of `origin/develop` (`4f1e836`
S5.4, `21cdb13` the renderer), 368 CPU tests green in ~13 s. No new training submitted.

Session 6 opened by re-measuring rather than by changing rewards, and that turned up **two findings
that reorder the next steps below**. Both are recorded in full in "The instrument problem" and "The
heading term is unobservable" sections that follow. The short version:

1. **The eval battery's per-second metrics are measuring the first ~1.1 s of an 8 s episode**, because
   it stops the clock at the fall — and its fall rate is the one number the project already agreed not
   to trust. `hops per cycle` and `heading drift deg/s` are both normalised by that truncated window,
   so both remaining failing requirements are currently being judged on a contaminated instrument.
2. **`heading_hold` prices a quantity the actor cannot observe.** Absolute heading is not in the 61D
   obs and the network has no memory, so the policy cannot steer back to a heading it cannot see.
   Raising `HEADING_HOLD_WEIGHT` — the previous session's first recommendation — pushes harder on a
   loop that cannot close. **Do that only together with the observability fix.**

### Next steps, in order

| # | Action | Why it is first |
|---|---|---|
| 1 | **Port `hop_eval.py`'s measurements onto `render_policy.py`'s BAM path** | Both open requirements are measured by contaminated numbers. Free, local, CPU, and the BAM setup is already proven (`scripts/infer_policy.py:load_mujoco_with_bam`). Do not tune cadence against 7.51 hops/cycle until this lands |
| 2 | **Make heading observable, then reprice it** | Write the wrapped heading error into the free twist slot (`mdp.py:5000` hard-zeros `vel_command_b[:, 2]` under the phase carrier — a spare input inside the 61D contract). Then `HEADING_HOLD_WEIGHT` 1.5 → ~3.5 is a real dial rather than added variance |
| 3 | **Re-measure cadence, then fix the design gap** | The gap the last session named is real — the cadence factor removes the *reward* for extra hops but never *charges* for them. Judge it on **takeoff phase lock (0.34)**, which is duration-free, not on hops/cycle |

Items 1 and 2 are independent and can land in either order; item 3 depends on item 1.

---

## What S5.4 achieved (end of session 5)

**S5.4 ran and the duck HOPS.** User's verdict on the video: *"that's a hop! not perfect but a hop
nonetheless."* First time in the project that the behaviour has been called a hop rather than a bounce,
a buzz or a scoot.

| | Run | wandb | Hub |
|---|---|---|---|
| S5.4 | 4096 × 1500, fresh, commit `4f1e836` | `bb8t7j2e` | `chelleboyer/s54-heading-cadence-v2` |

### What S5.4 fixed, measured

**The head, and it was a STEERING artifact — the hypothesis is confirmed.** Head weights were
deliberately left untouched; constraining the heading fixed the head by itself:

| Joint (mean signed error) | S5.3 | S5.4 |
|---|---|---|
| neck_pitch | −20.4° | **−2.0°** |
| head_pitch | +28.1° | **+5.6°** |
| head_yaw | +21.1° | **+4.4°** |

The counter-fold collapsed and the yaw crank went with it; `head_pose_bias` is 3.6× less negative
(−0.81 → −0.22) at the *same* weight of 3.0. **Three runs of escalating head penalties failed at this;
the fourth succeeded by not touching the head at all.** Record this as the lesson: when a penalty
DIVERGES under a fixed weight, the behaviour is buying something elsewhere — find the buyer, don't
raise the price.

The rest, all at unchanged weights unless noted:

| | S5.3 | S5.4 |
|---|---|---|
| `hop_settle` | 0.051 | **0.278** (5.4×) — the pause finally exists |
| `hop_landing_quality` | 0.073 | **0.253** (3.5×) — weakest term for three runs, now real |
| `action_rate` | −1.43 | **−0.74** — half the jitter |
| airborne fraction | 27% | **12%** |
| best consecutive hops | 3 | **6** |
| landing tilt / upright | 8.5° / 98% | **7.4° / 98%** |
| episode length | 910 | 914 — the repricing did NOT break it |

`hop_displacement` reads 0.901 against S5.3's 3.965, which is **expected and not a regression**: the
weight went 10 → 25 while the cadence cut the duty cycle ~10×, so raw is 0.036 vs 0.397. Judge it
against the cadence, not the level.

### What S5.4 did NOT fix

1. **Heading is still drifting: +107.5 → +62.8 deg/s** (but see the caveat on that unit below), and
   `heading_hold` plateaued at **0.315 of a possible 1.5**. It is the only one of the user's four
   requirements still failing outright. **The reason is observability, not weight — see below.**
2. **Cadence barely moved.** Takeoff phase lock 0.23 → 0.34 against a 1.00 target. The design gap is
   real: the cadence factor removes the *reward* for extra hops but never *charges* for them, so
   bouncing is still free apart from `action_rate`. The fix is to make an off-beat takeoff cost
   something, or to pay the hold enough that hopping off-beat loses to standing.

### The instrument problem — worse than "forward travel is untrustworthy" (session 6)

Three measurements of the same policy disagree, and the disagreement is systematic:

| Harness | Actuators | Says |
|---|---|---|
| Training (mjlab/warp) | BAM + DR + noise + delay | episodes run 914/1000 steps (~18 s) |
| `render_policy.py` | BAM, no DR/noise/delay | S5.4 fell at 6.62 s of 8 s |
| `hop_eval.py` | **position servos** | **100% of episodes fall** |

The ordering is monotonic in actuator fidelity, which is the tell. What session 6 added is **how far
that contaminates the report**, and it is further than the previous session assumed.

`hop_eval.py` **breaks out of the episode loop when tilt exceeds 70°** (`hop_eval.py:399-401`), and
then divides the per-second metrics by the elapsed time (`policy_s`, `hop_eval.py:519-525`). So a
harness that falls early does not merely report a bad fall rate — it reports every rate over a short
and self-selected window. Backing the number out of S5.4's own report:

```
102 genuine hops / 7.51 per cycle = 13.58 cycles x 1.6 s = 21.7 s across 20 episodes
                                  = 1.09 s per episode, of a nominal 8 s
```

**The battery is characterising the first 1.1 seconds of behaviour** — 14% of its own episode, and
~6% of the 18.3 s episodes training actually runs. At 5.10 hops in 1.09 s that is one takeoff every
213 ms with a median 32 ms flight: the drop-in transient and the topple, not steady state. That is
why "7.51 hops/cycle" disagrees with training's `hop_settle` rising 5.4×. **Both were right about
different windows.** The conflict the last session flagged is now explained.

**The corrected trust table** (this supersedes the one in "The eval battery" section below):

- **Trust — duration-free** (a ratio, a per-hop statistic, or a circular statistic): takeoff phase
  lock, air-vs-ground travel share, landing tilt, upright landing rate.
- **Do NOT trust — divided by the truncated window**: `hops per cycle`, `heading drift deg/s`,
  whole-run head error (a per-step mean over the same window), and best-consecutive (a count the
  fall cuts short).
- **Do NOT trust — the actuator gap directly**: forward travel per hop, apex rise, fall rate.

That leaves **takeoff phase lock as the only trustworthy cadence number in the report**, and it says
0.34 — cadence is genuinely failing, we just cannot currently size by how much.

So item 1 in the next-steps table is not a nice-to-have: it is the instrument for both remaining
requirements. A residual sim-to-sim gap will remain regardless (CPU MuJoCo has no DR, observation
noise or command delay), so BAM-on-CPU is a better instrument, not a perfect one.

### The heading term prices something the policy cannot see (session 6)

`heading_hold_reward` (`mdp.py:4874`) rewards `exp(-wrap(yaw - yaw_spawn)² / std²)`. The actor's
observation, dumped from the built cfg, is:

```
base_ang_vel(3)  projected_gravity(3)  joint_pos(14)  joint_vel(14)  actions(14)   = 48
+ twist(3, phase carrier)  head_command(4)  body_command(6)                        = 61
```

**Absolute heading is in none of it.** `projected_gravity` is yaw-invariant by construction — rotating
about the world z axis does not change the gravity projection — and `base_ang_vel` is the gyro, which
gives yaw *rate* only. The actor is a plain MLP (`hidden_dims=(512, 256, 128)`, no recurrence), so it
cannot integrate that rate into a heading estimate either. Spawn yaw is randomised full-circle, so
even the reference direction is unknowable.

Three consequences, and they explain the plateau exactly:

- **The policy cannot steer back**, which is the entire argument the docstring makes for choosing an
  angle-based term over a yaw-rate penalty. The only thing it *can* learn is to remove a systematic
  yaw bias from its gait — which is precisely what a yaw-rate penalty asks for. **The instrument has
  degenerated into the tool it was chosen over.**
- **The critic cannot see it either** (its extra terms are `base_lin_vel`, foot contacts — none
  yaw-bearing), so this term is unpredictable from the value function's input. It therefore enters
  the advantage as variance rather than signal. **Raising the weight 1.5 → 3.5 scales that noise**,
  which is an argument that the obvious next move would make learning worse, not merely ineffective.
- **The measured plateau is about what chance predicts.** Raw `heading_hold` is 0.315/1.5 = 0.210.
  A yaw scattered uniformly around the circle scores 0.113 at std 0.4; a held heading scores 1.0. The
  policy captured **11% of the range above chance** — consistent with "shaved some bias off, then
  stopped", and not with "the weight is nearly enough".

This is the same lesson S5.4 already booked, applied one level down: *when a term will not move, find
out what is actually happening before raising the price.* There it was a buyer elsewhere; here it is
a policy that is blind to what it is being charged for.

**The fix is free and stays inside the 61D contract.** `GroundPickPhaseCommand.compute` hard-zeros
the third twist slot (`mdp.py:5000`: `vel_command_b[:, 2] = 0.0`) — the phase carrier only needs two
slots for `[cos, sin]`. That third slot is a live input neuron carrying a constant, i.e. a wasted
input the standing rules already warn about. Writing `wrap(yaw - yaw_spawn)` (or its sine, which is
smooth across the wrap) there makes `heading_hold` a closable loop at zero cost to the observation
contract and zero risk to policy hot-swapping. It is also not a sim-only cheat: the real robot can
integrate its gyro for the ~18 s of an episode, so this does not spend the deferred hardware path.

**Do not treat this as decided** — it is a session-6 diagnosis, not a trained result. The cheap A/B
is the observability fix at the *current* weight, so the next run reads cleanly on whether
observability was the blocker, exactly as S5.4 held the head weights fixed to read cleanly on
steering.

### Deliberately abandoned (2026-09-06)

- **Chasing better video of the original bunny hop** (run `s5-forward-hop`). Its 12 training clips are
  320×240 at camera distance 3.0 because that run passed no viewer flags. It is renderable at any
  resolution now via `render_policy.py --no-phase`, and one was produced
  (`logs/bunnyhop/bunnyhop-hires.mp4`, in which it falls at 0.34 s under BAM), but the thread is
  CLOSED — S5.4 supersedes it. The bunny hop remains a historical artifact, not a target.
- The first S5.4 submission (`mcdojmw1`, repo `chelleboyer/s54-heading-cadence`) was **cancelled at
  iteration 166** because it was launched without viewer flags and rendered unwatchable 320×240 video.
  Ignore that repo; `-v2` is the real run.

### The requirement, in the user's words

Watching the S5.3 run's video: *"he's kind of hopping, but more scooting himself around in a circle"*,
and the target is **"he needs to hop straight ahead, pause and then repeat, all with his head up."**

Four properties, and the honest status of each **after S5.4** (the previous version of this table was
S5.3-era; each row now cites an instrument that survives the session-6 audit above):

| Requirement | Status | Evidence, and which instrument |
|---|---|---|
| **Hop** (leaves the ground) | **partial — and the user calls it a hop** | 66% of travel still happens with a foot down (air-share, a ratio → trustworthy). Airborne fraction 27% → 12% in training |
| **Straight ahead** | **failing** | `heading_hold` raw 0.210 vs 0.113 chance and 1.0 held — **from wandb, not the battery**, so it is clean. Only 11% of the range above chance. Root cause is observability, not weight |
| **Pause, then repeat** | **failing** | Takeoff phase lock 0.23 → 0.34 of 1.00 (circular statistic → duration-free → trustworthy). The "hops per cycle" figure is **not** trustworthy — see the instrument problem |
| **Head up** | **fixed, as a side effect** | Per-joint error collapsed with no head change at all: neck_pitch −20.4° → −2.0°, head_pitch +28.1° → +5.6°, head_yaw +21.1° → +4.4° |

### The runs so far (all `Mjlab-HopForward-Flat-MicroDuck`, 4096 envs)

| Increment | Run | wandb | Hub repo | Result |
|---|---|---|---|---|
| S5 | `s5-forward-hop`, 1500 it | — | `chelleboyer/s5-forward-hop` | Forward bunny hop, confirmed on video. Airborne 52% of its life |
| S5.1 | 1500 it | `5b77zi33` | `chelleboyer/s51-forward-hop` | Displacement-at-landing replaces air time |
| S5.2 | ~700 it (crashed at 692) + rerun | `8j6xu1nk` | `chelleboyer/s52-head-rhythm` | Head to walker strength; pause-scaled hop. **Rhythm failed measurably** |
| S5.3 | 2500 it, COMPLETED | `tbs1k86h` | `chelleboyer/s53-phase-hop` | Phase cycle + crouch + apex. **Crouch works; rhythm and heading do not** |
| **S5.4** | **1500 it, COMPLETED** | **`bb8t7j2e`** | **`chelleboyer/s54-heading-cadence-v2`** | Heading hold + binding cadence + repricing. **The user calls it a hop. Head fixed as a side effect; heading and cadence still failing** |

`chelleboyer/s53-phase-hop` holds 11 checkpoints, 21 videos and `exported/policy.onnx` — the in-job
auto-export is confirmed working end to end. Local copies: `logs/s53/`, `logs/s54/` (the latter also
holds `S54-bam-720p.mp4` and `S54-slowmo.mp4` from the local renderer, plus `eval_final.json`).

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

### S5.4 — what was built (results above)

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

These predictions were made before the run and are recorded because they held, which is weak evidence
the model of this env is getting better: displacement dropped sharply then recovered; `hop_settle`
became non-zero for the first time; episode length did not collapse under the repricing. The one that
FAILED: `heading_hold` was predicted to climb toward ~1.5 and plateaued at 0.315.

### The standard run commands

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
#
# THE VIEWER FLAGS ARE NOT OPTIONAL. mjlab's defaults are 320x240 at camera
# distance 3.0, which renders a 25 cm robot as a handful of pixels in the middle
# of an empty floor — unwatchable, and video is this project's primary judgment
# instrument. 640x480 at distance 0.9 with elevation -12 is the framing every
# usable clip so far was shot with. Omitting them cost a restart on 2026-09-06.
PYTHONIOENCODING=utf-8 uv run train Mjlab-HopForward-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 1500 --hf-jobs --video True \
    --env.viewer.distance 0.9 --env.viewer.elevation -12 \
    --env.viewer.width 640 --env.viewer.height 480
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

**Still open — the actuator gap, and it reaches further than this section used to claim.** The harness
drives plain MJCF **position servos**; training drives **BAM** (voltage model, back-EMF). On the S5
policy it reported −2.2 mm/hop while training logged ~95% of the velocity cap and the video plainly
showed forward hopping.

**The trust table lives in "The instrument problem" above and supersedes what used to be here.** The
correction session 6 made: an earlier version of this list put *hops per cycle* and *heading drift* in
the "do trust" column on the grounds that they are geometry and contact rather than torque. That is
true of the numerators and false of the denominators — both are divided by an elapsed time the fall
truncates, and the fall is the untrusted quantity. **Anything per-second in this report inherits the
actuator gap.** Per-hop statistics, ratios and the circular phase-lock statistic do not.

Closing that gap — or accepting it permanently — is the top item in the next-steps table.

## Tooling (all CPU, all free)

- `scripts/hopscotch/hop_eval.py` — headless eval battery; per-hop displacement, flight duration, apex
  rise, landing tilt, upright rate, consecutive streaks, plus (2026-09-06) hops per cycle, takeoff
  phase lock, heading drift, air-vs-ground travel share and whole-run head error.
- **`scripts/hopscotch/render_policy.py` (NEW, 2026-09-06)** — renders any exported ONNX to mp4
  locally on CPU: any resolution, tracking camera, **BAM actuators by default** (`--no-bam` for
  hop_eval's position-servo dynamics), `--no-phase` for pre-S5.3 policies, `--fps` below 50 for slow
  motion. This is the answer to "I can't see the policy without a GPU" and to "the training clips are
  320×240 because of flags passed hours ago". Validated on `pollen-robotics/microduck-policies`'
  official walker, which stays upright a full 8 s under it. Raises the scene's offscreen framebuffer
  on the loaded model, which is what allows resolutions the training clips could never reach.
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
- ~~`scripts/infer_policy.py` cannot run on Windows.~~ **FALSE, corrected 2026-09-06.** It guards the
  import (`try: import termios, tty / except: ... = None`), so it runs here — only keyboard control is
  disabled. `--help` works, and its `load_bam_model` / `load_mujoco_with_bam` are what
  `render_policy.py` now builds on. Two things about it remain true: it never overrides the MJCF's
  placeholder `kp≈0.5` gains on the legacy path, and it drives the twist slots as a **velocity
  command**, so it has the same phase-carrier gap `hop_eval.py` had — do not use it to drive an S5.3+
  policy. Its `--record` writes a **pickle of observations**, not video.
- **Prior art recalibrated (user's visual judgement, 2026-09-06):**
  `joanfox/microduck-happy-hop` is *"not even as good as some of my prior attempts"*. It is still
  evidence that a hop is reachable under BAM + backlash, which is what closed S1, but it should no
  longer be treated as a quality bar. See [`prior-art-hop.md`](./prior-art-hop.md).
- **A useful external benchmark for pacing:** upstream PR #28 reached a working hop in **600
  iterations**, `joanfox` in **1255**. Both were vertical hops from a standing entry. Nobody in the
  public record has done forward-plus-cadence, so there is no reference for what we are doing now.

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
