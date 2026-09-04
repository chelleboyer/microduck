# Architecture — Microduck Hopscotch

> Intent: [microduck-hopscotch-project-brief.md](./microduck-hopscotch-project-brief.md)
> Tickets: [docs/tickets/microduck-hopscotch-phase-0.md](./docs/tickets/microduck-hopscotch-phase-0.md) (Phase 0 only)
> Status: decided 2026-09-04. Pre-implementation. No code written yet.

## Problem & goals

Teach Pollen Robotics' Microduck to perform hopscotch — starting in MuJoCo simulation and ending on
the physical robot. The first milestone is deliberately modest: **Microduck intentionally hops forward
and lands upright.** Accurate landings, consecutive hops, and the full pattern build from there.
Training runs on Hugging Face Jobs because the development machine has no GPU.

Every decision below is judged against that first milestone, and against the constraint that whatever
we train must eventually run on real hardware.

## What already exists (and reframes the work)

The brief assumes more greenfield than there is. [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl)
(Apache 2.0) already provides:

- **14 task families** on mjlab (MuJoCo Warp) + rsl_rl PPO — velocity, velstand, standup, sitstand,
  roulade, ballkick, groundpick, spin, roller and swizzle variants. Each has a `-Backlash-` twin.
- **A working HF Jobs path.** `--hf-jobs` on the train command. Default flavor `l4x1`, 12h timeout,
  a background uploader pushing `.pt` checkpoints to a private Hub repo every 60s, wandb credentials
  forwarded as a secret. Their own docs state `uv sync` is ground truth *because* HF Jobs runs it.
- **A sim2real stack** — BAM M6 actuator model for the Dynamixel XL330 (voltage law, back-EMF,
  load-dependent friction), domain randomization over battery voltage, sag, command delay and friction,
  IMU misalignment, encoder bias, backlash twins, NaN guards.
- **`CLAUDE.md`** (repo root, post-merge) — a distilled reward-design and sim2real playbook encoding
  months of hard-won lessons. Every reference to "`CLAUDE.md`" below means this upstream file; our own
  fork rules live in `docs/hopscotch-rules.md`.
- **An export/publish path** — ONNX with the observation normalizer baked in, schema-2 manifest, Hub upload.

Two consequences:

1. **The HF Jobs milestone shrinks from a build to a smoke test.** The brief's "prove HF Jobs can run
   normal training first" is still the right first move, but it is hours of verification, not weeks of
   pipeline work.
2. **The risk moves to physics and perception**, not compute. See Spikes.

### The binding constraint: the duck is blind

All Microduck policies share a fixed **61-dimensional observation**: 48 proprioception + a 13D command
block `[twist(3), head_pose(4), body_pose(6)]`. There is **no exteroception** — no vision, no height
scan, no terrain sensing. Policies are hot-swapped at runtime behind this shared contract, so slots are
never deleted, and unused slots are zero-padded with tiny sampling ranges to keep input neurons alive.

**Microduck cannot see a hopscotch course.** Any design where the robot perceives and navigates cells
is outside the existing contract. Hopscotch must therefore be expressed as *commanded* motion — the
course lives in whatever drives the commands, not in the robot's senses.

## Approaches considered

**A. Commanded hop-gait on flat ground, inside the 61D contract** *(recommended)*
Hopping becomes a gait mode in a velocity-derived environment, driven through the existing command
block. No physical course in simulation; the course is painted visual markers and a sequence of
commands issued by the runtime.
*For:* stays inside the observation contract, so policies remain hot-swappable and publishable; inherits
the full DR / obs-noise / delay / NaN-guard stack automatically; matches how the real robot is actually
driven; cheapest path to Success #1.
*Against:* not "true" autonomous hopscotch — the duck executes a choreography rather than reading a
course. Landing accuracy is open-loop, bounded by command-tracking precision.

**B. Physical course geometry + extended observations**
Build real cell geometry (box primitives, the way `slope_terrain.py` builds ramps, with difficulty
scaling for curriculum), and add cell-relative position observations so the duck can aim.
*For:* the most literal and most capable hopscotch; genuine closed-loop foot placement.
*Against:* breaks the 61D contract, so policies stop being hot-swappable in the runtime. Worse, it
requires a real-world position source the robot does not have — there is no path to deploy it on
hardware without adding localization that doesn't exist. Sim-only result.

**C. Episodic trick library + daemon choreography**
Train each hop type as a separate short episodic policy (like roulade / groundpick), and let the daemon
sequence them into a routine.
*For:* each policy is simple and independently verifiable; matches the existing episodic-trick pattern.
*Against:* stitching independent episodic policies into a continuous rhythm is brittle at the seams;
pays for the same discovery work repeatedly; clashes with the velocity template.

## Recommended approach

**Fork `microduck_rl`; add a velocity-derived hop-gait task; keep the duck on flat ground and inside
the 61D contract; run everything on HF Jobs.**

Hopping is modelled as a *continuous gait mode* rather than a one-shot trick. A rhythm is
self-reinforcing and easier for PPO to discover than an isolated jump, consecutive hops come free
rather than needing rework, and a single hop is simply the one-cycle case. Because the command stays
constant during execution, the resulting policy is publishable through the existing constant-command
publish path.

The hopscotch *course* is choreography: a sequence of commands issued to a hot-swappable policy, over
flat ground with the course painted on it for the humans watching.

## Key decisions

**Stack & libraries** — Inherited wholesale, deliberately. Python + `uv`; mjlab (MuJoCo Warp) for
GPU-parallel simulation; rsl_rl PPO; wandb for run tracking; ONNX for export; Hugging Face Hub for
artifacts and HF Jobs for compute. *Alternatives rejected:* MuJoCo Playground or a from-scratch Gymnasium
env — both would discard the BAM actuator model and DR stack that make sim2real work here, which is the
single hardest part of this problem and already solved upstream.

**Repo strategy** — Fork `microduck_rl` into this repo, upstream kept as a git remote to pull from.
The `--hf-jobs`, export, publish and DR tooling all assume the repo layout, so an overlay package would
have to port the full DR + obs-noise + NaN-guard stack (their CLAUDE.md warns about exactly this).
Cost accepted: our work is entangled with theirs and merges need care. Upstream is young and moving
fast, so we pin a known-good commit and pull deliberately, not continuously.

**Task template** — Build on the velocity family (`make_microduck_velocity*_env_cfg`). CLAUDE.md
recommends this explicitly: it keeps domain randomization, observation noise, command delays and NaN
guards in sync automatically. Roulade was the tempting alternative (the only existing explosive,
ballistic template, with motion-blocker regularizers already tuned low) but it is episodic and has no
locomotion command structure.

**Observation & command contract** — Stay at 61D. No new observation slots. The hop command is encoded
by repurposing existing command-block capacity (candidate: a `body_pose` vertical component as commanded
hop height). Exact slot semantics must be read from source after the fork — see Open questions. Per
CLAUDE.md, whichever slot is chosen must be non-zero from step 0 even at reward weight 0, or its input
weights die permanently; and the all-zero command (deployment idle) must be trained explicitly via
exact-zero sampling rather than left to uniform sampling.

**The core new reward: simultaneous flight** — mjlab's velocity lineage ships `feet_air_time`, but that
rewards *alternating* single-foot air time — ordinary walking. A hop requires **both feet off the ground
at once**. This is a genuinely new reward term, not a reweighting, and it is the central piece of new
work in the project. CLAUDE.md constrains its design: no "reach X" jackpots (use potential-based shaping,
charging for progress deltas); keep motion-blocker regularizers (body angular velocity, angular momentum,
pose std) *low*, because they penalize exactly what dynamic motion requires; introduce smoothness
penalties (action rate, torque rate) only *after* the skill exists, via curriculum, or "do nothing" wins
during exploration. All penalty terms must log ≤ 0 in wandb throughout — that check is infallible and
catches sign inversions.

**Compute & dev loop** — All-remote on HF Jobs; no local GPU and no WSL2 setup. Mitigated by a
**preflight-in-one-job** pattern: a single job runs CPU-runnable config tests → 64-env/5-iteration smoke
test → full training, failing fast before the GPU section starts. CLAUDE.md reports the smoke test catches
~95% of config errors, and `cpu-basic` is $0.01/hr, so a reward-sign typo dies in seconds rather than
after hours of training. One cold start per iteration.

**Budget posture** — Lean. Default `l4x1` at $0.80/hr. Flight-phase spike ~1000 iterations (~$3–5). A
full hop gait is a curriculum-heavy gait, so CLAUDE.md's 4000–6000 iteration budget applies (~$15–30 per
run), and 2–5 reward-hacking iterations before convergence is normal and expected. Early phase should
land well under $150.

**Data model** — Not a conventional data model; the durable shapes are the 61D observation contract
described above, and the artifact chain: wandb run → `.pt` checkpoint (auto-uploaded to a private Hub
model repo during training) → ONNX with normalizer baked in → schema-2 manifest → Hub policy repo →
loaded by the on-robot daemon.

**Boundaries & contracts** — `HF_TOKEN` as a job secret (with namespace controlling repos, uv-cache
bucket and billing); wandb credentials forwarded from `~/.netrc` as a secret; the Hub as the artifact
store; and the schema-2 policy manifest as the hard contract with the robot's Rust runtime. Never export
a checkpoint by hand — the normalizer must be baked in by `scripts/export.py`, and in-sim play hides the
bug because it applies the normalizer anyway.

## Missing pieces

- **A simultaneous-flight reward term** — the central new work (see above).
- **A hop command encoding** — which slot carries hop intent, and its sampling distribution, including
  explicit buckets for rare-but-important command regions (CLAUDE.md: turn-in-place spins emerged as ~2%
  of experience under uniform sampling and never trained).
- **A hop-gait environment config module** plus its registration and `-Backlash-` twin.
- **Landing-quality criteria** — what counts as "upright" and "accurate", as measurable terminations and
  reward gates rather than prose. CLAUDE.md warns to check *tilt*, not just height.
- **A curriculum** from hop-in-place → forward hop → consecutive hops → sequence, with stage boundaries.
- **A headless evaluation battery** — per-spawn-type outcomes, end-state clusters, air-time profiles —
  to judge runs before touching rewards.
- **The course choreography itself** — the command sequence that constitutes "hopscotch", and whatever
  drives it at runtime.
- **Painted course markers** for visual legibility in sim and on the floor.

## Spikes & experiments

**S1 — Can Microduck physically leave the ground?** *(blocking, do first)*
No existing task has a flight phase; roulade is the most dynamic and a roll never needs ground clearance.
Microduck is ~800g on low-torque, compliant, backlash-heavy XL330 servos. Whether a true flight phase is
achievable at all is genuinely unknown, and it determines the shape of the entire project.

```
Question:      Can Microduck achieve a real flight phase (both feet off ground, measurable air time)?
Spike:         Velocity-derived env, reward air-time + upright landing only, minimal regularizers.
               ~1000 iters @ 4096 envs on l4x1. Roughly 2-4h, under $5.
Decision rule: >80ms consistent air time with upright landing  -> true hop track, hopscotch as designed.
               Ground contact never breaks / air time <30ms    -> pivot to "stepping" hopscotch
                                                                  (foot placement into cells, no flight).
```

**S2 — Does the stock pipeline run end-to-end on our HF account?**
Cheap, and the brief already calls for it. Run stock `Mjlab-Velocity-Flat-MicroDuck` via `--hf-jobs`,
confirm namespace/billing, checkpoint upload, wandb streaming, and the 12h timeout behaviour. Decision
rule: if checkpoints land in the Hub repo and wandb streams, the pipeline is trusted and we never
revisit it. Run before or alongside S1.

**S3 — Is the all-remote loop actually tolerable?**
Our dev-loop choice is the one with real uncertainty. After ~5 preflight-pattern iterations, judge it.
Decision rule: if config errors are reliably caught in the CPU preflight stage and turnaround is
acceptable, stay all-remote. If we are repeatedly burning GPU minutes on errors a local test would have
caught, revisit WSL2 + CPU-only tests.

**S4 — Does a hop survive the reality gap?** *(defer until a hop exists)*
Ballistic motion is far more sensitive to actuator fidelity, backlash and command delay than walking is.
Before trusting any hop on hardware, compare the base task against its `-Backlash-` twin, then rehearse
via `scripts/infer_policy.py` before deploying to the physical robot.

## Open questions

- **Exact command-block semantics.** The 13D layout is `[twist(3), head_pose(4), body_pose(6)]`, but the
  precise meaning of each `body_pose` component must be read from `microduck_constants.py` and
  `microduck_velocity_env_cfg.py` after forking. This settles the hop command encoding.
- **Does the fork pin or track upstream?** Upstream is actively developed. Default is pin-and-pull-
  deliberately; revisit if they ship something we want.
- **Is "hopscotch" one policy or several?** A single commanded gait covering all hop types is cleanest,
  but distinct hops (single-foot cell vs two-foot straddle) may need separate policies hot-swapped by the
  daemon. Settled by how well one policy generalizes across hop types in training.
- **What does the real course look like?** Physical dimensions of the floor course constrain achievable
  hop distances, and should be measured against S1's results rather than assumed.
- **Does upstreaming matter?** The fork could be kept contribution-shaped to submit hopscotch back to
  Pollen Robotics. Not decided; costs discipline, and is reversible either way.

## Sources

- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) — README and CLAUDE.md
- [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) — on-robot Rust runtime
- [mujocolab/mjlab](https://github.com/mujocolab/mjlab) — training framework
- [HF Jobs configuration](https://huggingface.co/docs/hub/jobs-configuration) — flavors, pricing, timeouts, volumes
- [Pollen Robotics — Microduck](https://pollen-robotics.com/microduck/)
