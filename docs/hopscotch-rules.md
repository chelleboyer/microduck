# Microduck Hopscotch

Teach Pollen Robotics' Microduck to hop, then to hopscotch — trained in MuJoCo simulation on Hugging
Face Jobs. **Sim-only** as of session 3; hardware is deferred, not dropped (see Scope).

> **Start here (fresh session).** Read this file, then
> [`../microduck-hopscotch-architecture.md`](../microduck-hopscotch-architecture.md) (decisions) and
> [`hopscotch-routine.md`](./hopscotch-routine.md) (what we're ultimately building). Current state and
> the next action are in **Session 4 — start here** below. Everything above that section is standing
> context; everything in it is live.

## Scope (session 3, 2026-09-04)

**Sim-only.** Getting hopscotch working in simulation is what matters right now. Physical deployment is
**deferred, not dropped** — BAM, domain randomization and the backlash twins stay on, because they cost
nothing to keep (already wired; the one known working Microduck hop was trained *with* them) and dropping
them is a one-way door. What the scope change actually unlocks is that two hardware-justified switches are
now ours to flip if we want them: mjlab's native `height_scan`, and promoting `base_lin_vel` from
critic-only to the actor. Neither is flipped yet — see S6 in the architecture doc.

## Historical: state at end of session 2 (2026-09-04) — superseded by Session 4 above

**The remote pipeline works end-to-end, and the hop env exists and is GPU-validated.**

- Upstream merged and pinned at `1e79c29` ([`upstream-pin.md`](./upstream-pin.md)). 199 CPU tests pass
  on Windows in ~13 s, no GPU.
- **HF Jobs proven.** Namespace `chelleboyer` (personal; Pro active, so Jobs is available — the org
  `context-course` is NOT Enterprise and would fail). wandb entity `chelleboyer-road-ranger`.
  Stock velocity smoke test ran to completion, checkpoints landed in a private Hub model repo, wandb
  streamed live.
- **`Mjlab-Hop-Flat-MicroDuck`** registered (+ `-Backlash-` twin) with two new rewards:
  `simultaneous_flight` (binary, weight 5.0) and `bilateral_foot_clearance` (dense ramp, weight 2.0).
  Smoke-tested on GPU twice: builds, steps NaN-free, all 18 terms compute, every penalty logs ≤ 0,
  and both new terms read non-zero under a random policy (so neither is silently dead).
- **Prior art found:** someone has already trained a Microduck hop —
  [`prior-art-hop.md`](./prior-art-hop.md). **Read it before spending GPU on S1.**

**S1 was re-scoped in session 3 and is now CLOSED without spending** — the prior art plus the CPU probe
answer "can it leave the ground" in the affirmative for sim. **Next action: S5 — can it hop *forward* and
land upright?** That is the brief's Success #1, and the prior art explicitly does not answer it (vertical
hop only, from a standing entry). Forward intent is encoded as un-commanded forward progress while
airborne (E1); commanded hop distance (E3) comes in Phase 1. Budget 2-3 runs, then re-plan. Full rationale
and decision rule in [`../microduck-hopscotch-architecture.md`](../microduck-hopscotch-architecture.md).

The gap to close before that run: **landing quality is unmeasured.** Every gate in the hop env checks the
robot *during* flight (tilt, trunk height), so a forward hop that reliably face-plants scores well today.

Two small Windows fixes in `hf_jobs.py` are known and unmade: the log streamer dies on non-ASCII
output unless `PYTHONIOENCODING=utf-8` is set (the job itself is unaffected, but the local command
reports failure), and the `[wandb] forwarding API key from ~/.netrc` message names the wrong file when
the key came from `_netrc`.

> **The encoding bug is broader than "the log streamer".** It also kills `uv run train --help`
> outright, locally, with the same `UnicodeEncodeError: 'charmap' codec`. Prefix
> `PYTHONIOENCODING=utf-8` on any `train` invocation on Windows.

## Session 4 — START HERE (state as of 2026-09-05, end of session 3)

**S5 ran to completion and the duck hops — but the reward shape is wrong, and fixing it is the next
job.** Run `s5-forward-hop` (job `6a9bf2e6259f8e97255e28a5`), `Mjlab-HopForward-Flat-MicroDuck`,
4096 envs × 1500 iters, COMPLETED. Artifacts in `chelleboyer/s5-forward-hop`: `model_1499.pt`,
9 training videos, and a local export at `logs/dl/policy.onnx`.

### What the run produced

- **A working forward bunny hop.** Confirmed on video by the user. Training logged
  `forward_flight_progress` at **~95% of the 0.4 m/s cap** — the cap is SATURATED, so the metric can no
  longer distinguish good from great and `FORWARD_VEL_CAP` needs raising next run.
- **The eval battery measured** 104 genuine hops, 5.2/episode, 8 consecutive, 98% upright landings.
- **Keep this policy as the "bunny hop" artifact.** It is the first thing this project trained that
  visibly does the thing.

### The two changes that define the next run — NOW IMPLEMENTED (2026-09-05, session 4)

Both are built and locked in by tests (278 CPU tests, ~14 s). `Mjlab-HopForward-Flat-MicroDuck`
evolved **in place** — the hop-in-place baseline is still the untouched A/B reference, and the S5
recipe stays reproducible from git plus its wandb run, so a third variant would have been a flag
matrix for nothing. What shipped:

- **`hop_displacement`** (`mdp.py`) — forward distance from takeoff to touchdown, paid across the
  landing window, capped at 10 cm, measured along the **takeoff heading** so a turn-and-drift scores
  nothing. It carries a takeoff latch (this repo's first stateful hop term; it follows roulade's state
  idiom and is reset-aware inline like `head_pose_bias_penalty`'s EMA).
- **The handover is a curriculum, not a swap.** `simultaneous_flight` 5.0 → 3.0 → 2.0 while
  `hop_displacement` 0 → 5.0 → 10.0, sharing a boundary at iter 300. Displacement is unearnable until
  flight exists, so it must not lead before then — and flight must not be demoted before then either.
- **`forward_flight_progress`** — cap 0.4 → **0.8 m/s** (it was saturated at ~95%), weight 1.5 → 0.5.
  It pays per airborne step, which is the shape displacement replaces, so it is now a ramp, not a driver.
- **Head priced at touchdown**: a head-upright factor inside `hop_landing_quality` (std 0.35,
  deliberately wide), the term raised 1.0 → 2.0, and `head_pose_bias` returning on a late gentle ramp
  (0.5/1.0 at iters 800/1200 vs the walker's 1.0/2.0/3.0). `head_pose_tracking` untouched.

**Two things a CPU test cannot prove, and this machine cannot run:** the env has NOT been built or
stepped (no CUDA here), so the mandatory **64-env / 5-iteration smoke test is still owed** before any
real run:

```bash
# SMOKE TEST — cents, ~minutes. Never launch the real run without it.
# PYTHONIOENCODING is mandatory on Windows: without it the log streamer (and
# even `train --help`) dies on non-ASCII with a charmap codec error.
PYTHONIOENCODING=utf-8 uv run train Mjlab-HopForward-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5 --hf-jobs

# THEN the real run (~1500 iters, ~$5-10 on l4x1):
PYTHONIOENCODING=utf-8 uv run train Mjlab-HopForward-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 1500 --hf-jobs --video
```

The smoke test only proves it builds, steps NaN-free, all terms compute and ONNX exports — 5
iterations show no behaviour. In the **real** run, watch three things:

- **`Episode_Reward/hop_displacement` must go non-zero after iter 300** (when the curriculum hands it
  the lead). A sparse, latched, gated term is exactly the kind that logs a silent `0.0000` — the trap
  `forward_flight_progress` sat in through all of S5's smoke tests. If it is still exactly 0, the gate
  is unreachable and the term needs LOOSENING, not tuning.
- **`Episode_Reward/simultaneous_flight` should fall well below half its S5 value** (2.58). If the
  airborne fraction stays ~50%, the handover did not take.
- Every penalty term ≤ 0, throughout.

**One judgment call worth re-examining before spending:** `HOP_DISP_WEIGHT = 10.0` is sized by reward
MASS, not by the face value of the terms around it — a once-per-hop payment against per-step
competitors. The derivation (a hop used to earn ~30 from air time; it now earns ~32 from a 45 mm
displacement) is in the cfg constant's comment. If the smoke test shows the term dominating or
vanishing, that constant is the dial.

### The two changes, as originally diagnosed

**1. Stop rewarding "airborne" as the accomplishment (user's call, and the data agrees).**
`simultaneous_flight` pays **1.0 per step** while airborne, so air time IS the objective — and the
policy ended up **airborne ~52% of its life** (`simultaneous_flight` 2.58 ÷ weight 5.0). That is
bouncing, not hopping, and it explains why `hop_landing_quality` was the weakest term (0.138): the
policy earns from being in the air, not from landing anywhere. The flight duration of a single hop was
capped, but the *fraction of life spent flying* was not.

Replace it with **"take off HERE → land THERE"**: a per-hop DISPLACEMENT reward paid **once at
landing**, not per-step while airborne. This is also exactly what E3 (commanded hop distance,
`body_pose[0]`) was designed to be, so the two merge rather than compete. `simultaneous_flight` should
demote to a small enabling term or disappear.

**2. Head up on landing (user's call).** The head rides low because deviation 3 deliberately FREED it
(`head_pose_tracking` 2.0 → 0.5, `head_pose_bias` curriculum removed) so its 280 g could act as a
countermovement. Do **not** fix this by raising `head_pose_tracking` — `microduck_velocity_env_cfg.py:729-737`
records that tightening it made the policy stop moving entirely. The right tools are:
  - a **head-upright factor inside `hop_landing_quality`**, pricing posture only AT TOUCHDOWN and
    leaving mid-flight swing free; and
  - re-introducing **`head_pose_bias`** (L1 on a 1 s EMA), which was built for exactly this droop and
    charges DC bias while letting oscillation cancel.

### Tooling added this session (all CPU, all free)

- `scripts/hopscotch/hop_eval.py` — headless eval battery; per-hop displacement, flight duration, apex
  rise, landing tilt, upright rate, consecutive streaks, S5 verdict.
- `scripts/hopscotch/training_montage.py` — stitches a run's clips into one labelled progression video.
- `scripts/hopscotch/flight_probe.py --view` — watch the best open-loop hop in slow motion.
- `scripts/export.py` **works on CPU** — no GPU needed to export a checkpoint.

### ⚠ Do NOT trust the eval battery's FORWARD verdict

It drives **position servos**; training drives **BAM** (voltage model, back-EMF). On this policy it
reported median **−2.2 mm/hop** while training logged ~95% of the velocity cap and the video plainly
showed forward hopping. **The harness is the outlier.** Its hop count, consecutive streaks, landing-tilt
distribution and fall rate ARE reliable (geometry and contact, not torque). Closing that gap — or
accepting it permanently — is an open task.

A related trap it already caught the hard way: it defaulted to raw accelerometer while training uses
`USE_PROJECTED_GRAVITY = True`, and failed as a convincing BAD-POLICY verdict rather than an error. The
only tell was a physically impossible 0.0 mm apex rise. **Trust the impossible number, not the verdict.**

### Known-stale / loose ends

- ~~The detailed S5 implementation plan is gitignored.~~ **DONE** — it is tracked at
  [`plans/s5-forward-hop-and-landing-quality.md`](./plans/s5-forward-hop-and-landing-quality.md).
  A stale duplicate still sits in the ignored `.claude/plans/`; delete it.
- [`tickets/microduck-hopscotch-phase-0.md`](./tickets/microduck-hopscotch-phase-0.md) still frames MD-3
  around the **closed** S1 question. Phase 1 is still unsliced — now slice it against
  [`hopscotch-routine.md`](./hopscotch-routine.md).
- `scripts/infer_policy.py` cannot run on Windows (`termios`/`tty`), and never overrides the MJCF's
  placeholder `kp≈0.5` gains — so the deployment rehearsal likely inherits the same trap `hop_eval.py`
  had to fix. Unverified.

### Where the project is going

[`hopscotch-routine.md`](./hopscotch-routine.md) specs all 14 steps of a human hopscotch turn against
Microduck's capabilities. [`plans/abridged-court-demo.md`](./plans/abridged-court-demo.md) is the
recommended first demo: a 3-square court, drop-not-throw marker, two-foot hops. **Only E3 blocks it**,
and its non-training steps need no GPU.

- **Video works end-to-end** — the first mp4s ever recovered from a job
  (`chelleboyer/s5-video-smoke2`). It took three attempts, because `--video` needs a GL backend the
  stock image lacks: **GLFW fails** (no Xlib), and **EGL fails worse** (no vendor library, so
  `import mujoco` itself dies — breaking *every* job, not just video ones). `osmesa` plus an
  apt-installed `libosmesa6` is the working answer, gated behind `_wants_video()` so ordinary runs
  keep the stock import path untouched.
- **New rewards** in `mdp.py`: `forward_flight_progress` (E1), `hop_landing_quality`,
  `hop_landing_impact_penalty`. Test suite 199 → 247.
- **The S5 threshold is measured, not asserted**: open-loop forward travel is **8.0 mm/hop**, so the
  old 5 cm placeholder was ~6× beyond what physics delivers and would have failed a genuinely
  successful run. Pass mark ≥25 mm, fail ≤8 mm.
- **OPEN — is `forward_flight_progress` actually reachable?** It logged `0.0000` for all 5 smoke
  iterations. That is consistent with alive-but-below-4-decimal-precision (a random policy is airborne
  ~0.05% of steps, and E1 additionally needs positive forward velocity), and its unit tests prove it
  pays 1.0 in the right state — but it is **not confirmed non-zero on real data**. Check this first on
  the real run: if it is still exactly 0 once `simultaneous_flight` climbs, the gate is unreachable and
  the term needs loosening, not tuning.

## Read these, in this order

1. [`../CLAUDE.md`](../CLAUDE.md) — **upstream's playbook.** Env-building workflow, invariants, joint
   layout, and the reward-design lessons. Authoritative; read it before touching rewards.
2. [`microduck-hopscotch-project-brief.md`](../microduck-hopscotch-project-brief.md) — the intent. What
   we're building and why. Success #1 is *"Microduck intentionally hops forward and lands upright"*.
3. [`microduck-hopscotch-architecture.md`](../microduck-hopscotch-architecture.md) — the decisions, with
   rationale, rejected alternatives, spikes and open questions. **The load-bearing doc.**
4. [`tickets/microduck-hopscotch-phase-0.md`](./tickets/microduck-hopscotch-phase-0.md) —
   Phase 0 work: MD-1, MD-2, MD-3.
5. [`prior-art-hop.md`](./prior-art-hop.md) — a community Microduck hop policy, what it proves and
   what it leaves open. Changes the shape of S1.
5b. [`hopscotch-routine.md`](./hopscotch-routine.md) — **all 14 steps of a human hopscotch turn**, mapped
   to Microduck capabilities, with the audit of what exists vs what is new. This is what the project is
   building toward; slice Phase 1 against it.
5c. [`plans/abridged-court-demo.md`](./plans/abridged-court-demo.md) — the recommended first demo:
   3-square court, dropped marker, two-foot hops. Only E3 blocks it.
6. [`command-block.md`](./command-block.md) — the 13D command block, where hop intent goes, and the
   `nominal_height` discrepancy.

## Constraints that are expensive to rediscover

**The duck is blind — by choice, not by nature.** All Microduck policies share a fixed **61-dimensional**
observation: 48 proprioception + a 13D command block `[twist(3), head_pose(4), body_pose(6)]`. No vision,
no height scan, no terrain sensing. So hopscotch is **choreography** — a sequence of commands over flat
ground — not perception. Never delete a command slot; unused slots are zero-padded to keep input neurons
alive.

The nuance that matters under the sim-only scope: this is a *deferred decision*, not a physical limit.
mjlab 1.3.0 ships a `height_scan` terrain ray-scan by default and `microduck_velocity_env_cfg.py:533-537`
deletes it from both groups, because *the real robot has no such sensor*. We stay at 61D because it costs
nothing until the duck needs to aim at a cell — not because perception is impossible. Breaking the
contract is a real decision with real costs (policies stop being hot-swappable, hardware becomes a
rewrite); it just isn't a closed one.

**Hybrid, not all-remote.** Only *training* needs CUDA. Verified 2026-09-04 on Windows without a GPU:
`uv sync` succeeds, upstream's 149 CPU config tests pass in 55 s, and CPU MuJoCo loads the real model —
so config errors, reward signs and physics checks are caught **locally, in under a minute, for free**.
Spike **S3** is effectively answered: stay off WSL2, iterate locally, submit only real training runs.
This demotes MD-2 from load-bearing to an optimization — its CPU stage no longer needs to be remote at
all, and only its in-job 64-env smoke test still earns its place before `l4x1` ($0.80/hr).

**This repo is a fork of upstream, pinned.** `pollen-robotics/microduck_rl` (Apache 2.0) is kept as a
git remote and pulled deliberately, not continuously. It is young and moving fast (69 commits on `main`
in August 2026 alone). Its distilled sim2real and reward-design playbook is the repo-root
[`CLAUDE.md`](../CLAUDE.md) — **read it before touching rewards**; it encodes months of hard-won
lessons and is more trustworthy than reasoning from first principles here.

**The core new work is one reward term.** mjlab's stock `feet_air_time` rewards *alternating*
single-foot air time — ordinary walking. A hop needs **simultaneous** flight, both feet off at once.
That term does not exist upstream and is not a reweighting of one that does.

**Verify physics on CPU before spending GPU.** Upstream calls this the single biggest time-saver, and
it paid immediately — see [`s1-flight-probe.md`](./s1-flight-probe.md). Two metric traps live here:
contact-loss is not flight (a duck falling over loses both contacts and logs great "air time"), and
`current_air_time` is *per-foot* (a normal walk reports 125–300 ms). Simultaneous flight is
`n_contact == 0`, gated on tilt and trunk rise.

**The open project risk is physics — now the *forward* half of it.** Whether an ~800 g biped on
compliant, backlash-heavy XL330 servos can leave the ground is answered (prior art + probe). Whether it
can travel while airborne and land upright is not. A negative result on S5 is a *successful* spike — it
pivots hopscotch from jumping to stepping into cells, and reshapes roughly a third of the remaining
backlog. **Phase 1 is deliberately unsliced until S5 reports.**

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
