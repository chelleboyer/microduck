# Architecture — Microduck Hopscotch

> Intent: [microduck-hopscotch-project-brief.md](./microduck-hopscotch-project-brief.md)
> Fork rules & current state: [docs/hopscotch-rules.md](./docs/hopscotch-rules.md)
> Tickets: [docs/tickets/microduck-hopscotch-phase-0.md](./docs/tickets/microduck-hopscotch-phase-0.md) (Phase 0 only)
> Status: decided 2026-09-04. **Revised 2026-09-04 (session 3)** — scope narrowed to sim-only, S1 closed
> and superseded by S5. Revision history at the end.

## Problem & goals

Teach Pollen Robotics' Microduck to perform hopscotch in MuJoCo simulation. The first milestone is
deliberately modest: **Microduck intentionally hops forward and lands upright.** Accurate landings,
consecutive hops, and the full pattern build from there. Training runs on Hugging Face Jobs because the
development machine has no GPU.

**Scope, as of session 3: sim-only.** Getting this working in simulation is what matters right now.
Physical deployment is **deferred, not abandoned** — see the scope decision below for exactly what that
does and does not change. Every decision here is judged against the first milestone, and against
keeping the hardware path cheap to resume rather than cheap to walk away from.

## What already exists (and reframes the work)

The brief assumes more greenfield than there is. [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl)
(Apache 2.0), merged into this repo and pinned at `1e79c29`, already provides:

- **14 task families** on mjlab (MuJoCo Warp) + rsl_rl PPO — velocity, velstand, standup, sitstand,
  roulade, ballkick, groundpick, spin, roller and swizzle variants. Each has a `-Backlash-` twin.
- **A working HF Jobs path**, proven end-to-end on our account (S2 below).
- **A sim2real stack** — BAM M6 actuator model for the Dynamixel XL330, domain randomization over
  battery voltage, sag, command delay and friction, IMU misalignment, encoder bias, backlash twins,
  NaN guards.
- **`CLAUDE.md`** (repo root) — a distilled reward-design and sim2real playbook encoding months of
  hard-won lessons. Every reference to "`CLAUDE.md`" below means this upstream file.
- **An export/publish path** — ONNX with the observation normalizer baked in, schema-2 manifest, Hub upload.
- **A hop environment** — `Mjlab-Hop-Flat-MicroDuck` and its `-Backlash-` twin, built during MD-3.

Two consequences:

1. **The HF Jobs milestone was a smoke test, not a build.** Resolved; see S2.
2. **The risk was never compute.** It is physics, and now behaviour design.

### Three properties of the inherited stack that drive the decisions below

**1. The 61D observation contract.** All Microduck policies share a fixed **61-dimensional**
observation: 48 proprioception + a 13D command block `[twist(3), head_pose(4), body_pose(6)]`. Policies
are hot-swapped at runtime behind this shared contract, so slots are never deleted, and unused slots are
zero-padded with tiny sampling ranges to keep input neurons alive. Semantics fully documented in
[`docs/command-block.md`](./docs/command-block.md).

**2. Exteroception is available in the framework, and was switched off deliberately.** mjlab 1.3.0 ships
a `height_scan` terrain ray-scan observation by default. `microduck_velocity_env_cfg.py:533-537` deletes
it from *both* the actor and critic groups, with the reason stated in the comment: *"The microduck has no
such body-mounted terrain sensor for the policy."* Terrain geometry is likewise a solved pattern —
`slope_terrain.py` builds difficulty-scaled boxes wired to a curriculum, which is the same shape a
hopscotch course needs.

**3. The stack is already asymmetric actor-critic, split on the hardware line.** `base_lin_vel` is
provided to the critic only, as privileged information (`microduck_velocity_env_cfg.py:539-543`). The
word *privileged* means "the real robot cannot measure this." In a sim-only project that distinction
carries no weight: anything the critic sees, the actor could see too.

Properties 2 and 3 exist because of hardware. They are the levers the scope change unlocks, and the
reason approach B below is no longer dead.

## Approaches considered

**A. Commanded hop-gait on flat ground, inside the 61D contract** *(chosen — see Recommended approach)*
Hopping is a gait mode in a velocity-derived environment, driven through the existing command block. No
physical course in simulation; the course is painted markers plus a sequence of commands.
*For:* stays inside the observation contract, so policies remain hot-swappable and the hardware path stays
open for free; inherits the full DR / obs-noise / delay / NaN-guard stack automatically; cheapest path to
Success #1; the environment already exists and is GPU-validated.
*Against:* not "true" autonomous hopscotch — the duck executes a choreography rather than reading a
course. Landing accuracy is open-loop and, without a course to measure against, not directly verifiable.

**B. Physical course geometry + extended observations** *(rejected in session 1 on a reason that no
longer holds; now a live deferred option)*
Build real cell geometry and give the actor course-relative observations so the duck can aim.
*For:* the most literal reading of the brief, which asks in as many words to *"create a simulated
hopscotch course"* and to *"progress through the course"*; genuine closed-loop foot placement; landing
accuracy becomes a measurable, trainable objective instead of open-loop hope. Cheaper to build than
session 1 assumed — `height_scan` is native to mjlab and is two deleted lines away, and `slope_terrain.py`
is a working template for the course geometry.
*Against:* **the original objection is void.** It was rejected because it "requires a real-world position
source the robot does not have… no path to deploy on hardware," which is not a defect in a sim-only
project. What remains against it is real but ordinary: it breaks the 61D contract, so policies stop being
hot-swappable and the hardware path becomes a rewrite rather than a port; and it is a substantially larger
learning problem (perception + locomotion + hopping at once), meaning more iterations and more money.

**C. Episodic trick library + scripted choreography** *(not chosen; strengthened, and now the fallback)*
Train each hop type as a separate short episodic policy (like roulade / groundpick) and sequence them.
*For:* **this is the shape of the one known working Microduck hop** ([`docs/prior-art-hop.md`](./docs/prior-art-hop.md)) —
evidence, not argument. Each policy is simple and independently verifiable. Its main recorded weakness —
"stitching independent episodic policies into a continuous rhythm is brittle at the seams" — is largely a
*runtime* problem, and sim-only softens it a lot: in simulation the sequencer is a script with perfect
switch timing that can guarantee the stable standing entry the prior-art policy requires.
*Against:* pays the discovery cost once per hop type, at $15-30 a run; no rhythm, so consecutive hops need
explicit work rather than coming free; the seams remain physically visible even when the switch is perfect
(the prior-art hop needs ~1 s of settling before it will fire).

## Recommended approach

**Prove the forward hop inside the 61D contract first; keep every other door open.**

Concretely: extend the existing commanded-gait hop environment (approach A) until we know whether
Microduck can hop *forward* and land upright. That question is a prerequisite for all three approaches —
you cannot aim at a cell (B) or sequence hops into a routine (C) before a forward hop exists — so it is
the one spend that cannot be wasted by a later change of direction.

Perception (B) is **deferred, not rejected**: re-decided at the aiming stage, once there is a hop to aim.
The episodic library (C) is the **fallback** if the commanded gait proves hard to condition.

The cost of this ordering is honest and small: if we later go to B, the forward-hop policy's *weights*
don't transfer across an observation-dimension change. The environment, rewards, terrain, curriculum and
evaluation all do. One ~$20 run is the price of not betting the project on an unproven motor skill.

## Key decisions

**Scope — sim-only, hardware deferred not dropped.** *(session 3)* The question is what to stop paying
for. Answer: **nothing, for now.** BAM, the backlash twins and domain randomization are kept.

- **BAM** is not sim2real overhead, it is the model of this robot's actuators. Removing it does not make
  the simulation easier to trust; it makes it a different robot.
- **DR** is genuine sim2real tax and stripping it would speed convergence — but the one known working
  Microduck hop was obtained *with* backlash, BAM, current saturation and DR, so DR is demonstrably not
  what blocks a hop. Stripping it would buy a speedup we have no evidence we need, discard the only
  conditions under which a hop is known to work, and lose comparability with the prior art.
- **Backlash twins** are generated by `make_backlash_variant`, so maintaining them costs approximately
  nothing. Kept and registered; not spent on.

If training later stalls, a DR-off ablation is a cheap **diagnostic**, and is named here as a lever rather
than adopted as a decision. **S4 (reality gap) is deferred, not deleted.**

**Observation & command contract — stay at 61D for now, as a deferred decision.** The contract costs
almost nothing today: hop intent already fits `body_pose[2]`, and the spike does not need to aim. It
starts costing the moment we want closed-loop cell placement. Recorded explicitly so nobody mistakes it
for a settled constraint: **two things become available the day we drop it** — mjlab's native
`height_scan`, and promoting `base_lin_vel` from critic-only to the actor (a ballistic hop genuinely
benefits from knowing its own linear velocity). Neither is taken now. Whichever slot carries a command
must be non-zero from step 0 even at reward weight 0, or its input weights die permanently; and the
all-zero command — the idle state — must be trained explicitly via exact-zero sampling.

**Command encoding for forward intent — E1 now, E3 for Phase 1.** *(session 3)* Three encodings were
weighed:

- **E1 — un-commanded forward progress** *(chosen for the spike)*: reward forward base velocity **while
  airborne**, capped, gated by the tilt and trunk-height screens the flight term already carries. Dense,
  stateless, and structurally immune to the walking exploit because it pays nothing while a foot is down.
- **E2 — forward velocity through the existing `twist.lin_vel_x`** *(rejected)*: the most natural "gait"
  and it reuses the velocity env's tracking rewards, but velocity tracking has a strictly easier solution
  than hopping — walking — and it is the solution the base environment is tuned to find. The hop env
  crushed the twist ranges to ±0.05 and cut `air_time` 3.0 → 0.5 for exactly this reason: *"At 3.0 a good
  stride out-earns any hop the robot can currently produce, so walking is simply the better policy."*
  This repo has a second receipt for the same failure shape at
  `microduck_velocity_env_cfg.py:729-737` — *"Standing still scored higher, so it stood still."*
- **E3 — commanded hop distance in `body_pose[0]`** *(deferred to Phase 1)*: a per-hop displacement
  target, latched takeoff → landing. The most hopscotch-shaped encoding, because cells are discrete
  distances rather than velocities. Deferred because it asks the policy to learn the skill and its
  command-conditioning simultaneously, and needs latching logic before any hop exists. **E3 is E1 plus
  latching plus command gating**, so E1 is an increment toward it, not throwaway work. When E3 lands it
  must *not* go through `body_pose_tracking`: commanding an x-offset and rewarding pose tracking rewards
  leaning forward, the same trap already documented for the z slot.

**The course is a free variable — size cells to the duck, not the duck to the cells.** *(session 3)*
Session 1 listed "what does the real course look like?" as an open question, on the grounds that physical
dimensions constrain achievable hop distances. Sim-only reverses the causality: there is no physical
course, so cell size is ours to choose. Measure the hop distance the policy actually achieves, then size
the course to it. This removes an entire class of "the hop isn't big enough" failure, and makes the
forward-hop spike the thing that dimensions the course.

**Stack & libraries — inherited wholesale, deliberately.** Python + `uv`; mjlab (MuJoCo Warp) for
GPU-parallel simulation; rsl_rl PPO; wandb for run tracking; ONNX for export; Hugging Face Hub for
artifacts and HF Jobs for compute. *Alternatives rejected:* MuJoCo Playground or a from-scratch Gymnasium
env — both would discard the BAM actuator model and DR stack, which are the hardest part of this problem
and already solved upstream. Unchanged by the scope narrowing: none of this was chosen *for* hardware.

**Task template — build on the velocity family** (`make_microduck_velocity*_env_cfg`). CLAUDE.md
recommends this explicitly: it keeps domain randomization, observation noise, command delays and NaN
guards in sync automatically. The existing hop env already does this, and enumerates its seven deliberate
deviations from the walker in its module docstring. Roulade was the tempting alternative (the only
existing explosive, ballistic template) but it is episodic and has no locomotion command structure.

**The core new rewards.** `simultaneous_flight` (binary, `n_contact == 0`, gated on tilt and trunk rise)
and `bilateral_foot_clearance` (dense ramp, target 0.035 m) are **built and GPU-validated**. Each rejects
the other's exploit: clearance can be farmed by tucking the feet while the trunk sags, which the flight
term's contact condition rejects; flight can be reached by toppling, which clearance rejects because a
falling duck's feet do not rise. CLAUDE.md's constraints continue to bind on anything added: no "reach X"
jackpots, motion-blocker regularizers kept low because they penalize exactly what dynamic motion requires,
smoothness penalties introduced only *after* the skill exists, and every penalty term logging ≤ 0 in wandb
throughout — a check CLAUDE.md calls infallible for catching sign inversions.

**Compute & dev loop — hybrid, settled by S3.** Iterate locally on CPU (config tests, physics probes,
reward signs — seconds, free, no GPU); submit only real training to HF Jobs. Measured job scheduling
latency of 46 / 0 / 12 minutes across three submissions dominates wall-clock more than config errors do,
which is what demoted the MD-2 preflight harness from load-bearing to optional.

**Budget posture — 2-3 runs, then re-plan.** *(session 3)* Default `l4x1` at $0.80/hr; the forward-hop
spike is ~$5-10 a run. CLAUDE.md's expectation of 2-5 reward-hacking iterations before convergence is
normal and expected, so a single run would likely land mid-tuning rather than at an answer. Three is
enough to distinguish "untuned" from "wrong approach"; stopping there forces a checkpoint before the spend
compounds. Early phase stays well under $150.

**Data model.** Not a conventional data model; the durable shapes are the 61D observation contract above,
and the artifact chain: wandb run → `.pt` checkpoint (auto-uploaded to a private Hub model repo during
training) → ONNX with normalizer baked in → schema-2 manifest → Hub policy repo. Under sim-only the
manifest's role as the hard contract with the robot's Rust runtime is dormant, but the export path is still
the only correct way to produce a policy for playback and evaluation — never hand-convert a checkpoint, as
in-sim play hides the bug by applying the normalizer anyway.

**Boundaries & contracts.** `HF_TOKEN` as a job secret, with the namespace (`chelleboyer`, personal, Pro
active) governing repos, uv-cache bucket **and billing**; wandb credentials forwarded from `~/.netrc` as a
secret; the Hub as the artifact store.

**The flight reward's shape is wrong, and S5's run proved it.** *(session 3, 2026-09-05)*
`simultaneous_flight` pays **1.0 per step while airborne**, which makes AIR TIME the objective. The S5
run drove that to its logical end: the policy spent **~52% of its life airborne** (`simultaneous_flight`
2.58 ÷ weight 5.0), and `hop_landing_quality` stayed the weakest term in the stack (0.138). It bounces;
it does not hop *to* anywhere. The per-flight duration was capped, but the *fraction of life spent
flying* never was — CLAUDE.md's no-jackpot rule applied to one axis and not the other.

**Decision: the main term becomes "take off HERE → land THERE"** — a per-hop DISPLACEMENT reward paid
**once at landing**, not per-step in the air. Hopscotch is about landing in a specific square, so the
reward should pay for arrival, not for hang time. This converges with **E3** (commanded hop distance in
`body_pose[0]`), which was already the Phase 1 plan: the two become one term rather than competing.
`simultaneous_flight` demotes to a small enabling term, or goes away — flight is a *means*, and paying
for it directly is what produced the bounce.

**Landing posture includes the head.** *(session 3)* The head rides low because the hop env deliberately
freed it (deviation 3: `head_pose_tracking` 2.0 → 0.5, bias curriculum removed) so its 280 g could serve
as a countermovement. That trade is still right in flight and wrong at touchdown. Fix it with a
head-upright factor **inside the landing term** — pricing posture only at touchdown — plus the existing
`head_pose_bias` (L1 on a 1 s EMA), which charges DC droop while letting oscillation cancel. Do **not**
raise `head_pose_tracking`: `microduck_velocity_env_cfg.py:729-737` records that tightening it made the
policy stop moving entirely.

## Missing pieces

Built since session 1, no longer missing: the simultaneous-flight reward, the bilateral-clearance ramp,
the hop environment and its `-Backlash-` twin, and the command-block semantics.

Still missing, in dependency order:

- **A forward-progress-while-airborne reward** (E1) — the spike's one new term.
- **Landing-quality criteria** — *the required new work, and currently a hole.* Every gate in the hop env
  measures the robot **during** flight (tilt, trunk height). "Lands upright" is unmeasured, so a forward
  hop that reliably face-plants scores well today. CLAUDE.md is explicit that upright checks must measure
  tilt, not just height.
- **A curriculum** from hop-in-place → forward hop → consecutive hops, with stage boundaries phase-aligned
  to what the policy has actually learned.
- **A headless evaluation battery** — per-spawn-type outcomes, end-state clusters, air-time and
  displacement profiles — to judge runs from evidence before changing rewards.
- **The visual half of that battery — training video that survives the job.** CLAUDE.md is explicit that
  *"sim metrics can pass while the video fails the human eye"*, so numbers alone cannot judge a run. The
  dev machine reports `cuda available: False`, so nothing renders locally and `uv run play` cannot run
  here at all — remote video is the only way to see a policy move. mjlab already records it
  (`--video`, `--video-length`, `--video-interval`), but `scripts/hf/uploader.py` ships only
  `model_*.pt` and `params/*`, so today those clips die with the container. A one-line glob closes it.
- *(deferred, approach-dependent)* Course terrain geometry; course-relative observations; the choreography
  itself; painted markers.

## Spikes & experiments

**S5 — Can Microduck hop forward and land upright?** *(open, blocking, do next)*
The brief's Success #1, and the one question every approach depends on. Prior art does **not** answer it:
its own model card reports a vertical hop from a standing entry and states it was *"not trained to take off
directly from an active walking stride."*

```
Question:      Can Microduck hop forward and land upright?
Spike:         Extend the hop env with forward-progress-while-airborne (E1) plus a
               landing-quality term. ~1000-1500 iters @ 4096 envs on l4x1. ~$5-10/run,
               2-3 runs before re-planning.
Decision rule: >=25 mm net displacement per hop, landing upright, repeatable over >=3
               consecutive hops  -> forward-hop track; size cells to the measured
                                    distance; proceed to commanded distance (E3).
               <=8 mm, or upright landings <50%
                                 -> no better than open-loop; the hop is vertical-only,
                                    and hopscotch becomes hop-in-place-into-cells or
                                    the stepping pivot.
```

**The threshold is now measured, not asserted** (2026-09-04, session 3). The extended CPU probe sweeps a
rearward push blended into the extension and reports the open-loop forward baseline:
**8.0 mm of travel per hop, over a 38 ms flight, at a mean airborne speed of 0.211 m/s.** Two consequences:

- **The original 5 cm placeholder was ~6x beyond anything open-loop physics delivers**, and would have
  failed a genuinely successful run. It is withdrawn.
- **25 mm is scaled off the only measured policy-vs-open-loop ratio available**: the prior-art hop reached
  30-35 mm bilateral clearance where this probe's open-loop rise tops out at 7-11 mm — PPO bought ~3-4x.
  The same multiple on the forward baseline is what "learning added something real" looks like. The fail
  mark is the baseline itself: matching 8 mm means learning added nothing.

Also measured: adding the rearward push **improved flight from 32 ms to 42 ms**, so the push direction is
not merely a forward lever — it finds better hops outright. Measure displacement between takeoff and
landing over the flight interval only, and measure uprightness by **tilt**, not height.

**S6 — Is perception worth breaking the 61D contract for?** *(deferred; trigger: S5 passes)*
Only meaningful once a forward hop exists. Give the actor course-relative observations (or mjlab's native
`height_scan`) and compare cell-placement accuracy against the open-loop choreographed policy.
Decision rule: if closed-loop placement is not materially more accurate than open-loop commands, stay at
61D and keep the hardware path — the contract is only worth breaking if aiming actually buys accuracy.

**S4 — Does a hop survive the reality gap?** *(deferred by the sim-only scope; not deleted)*
Ballistic motion is far more sensitive to actuator fidelity, backlash and command delay than walking.
Whenever hardware returns to scope: compare the base task against its `-Backlash-` twin, then rehearse via
`scripts/infer_policy.py` before deploying. Keeping BAM, DR and the backlash twins on is what keeps this
spike cheap to run later rather than a rebuild.

**S1 — Can Microduck physically leave the ground?** *(CLOSED 2026-09-04, superseded by S5)*
Answered in the affirmative for simulation, by two independent pieces of evidence, without spending the
budgeted GPU run:

- A free CPU open-loop probe ([`docs/s1-flight-probe.md`](./docs/s1-flight-probe.md)) found ~34 ms of
  genuine simultaneous flight, ~11 mm rise, landing upright — torque-saturated, so a real ceiling for
  open-loop, but optimistic (no BAM back-EMF, backlash or DR). It also caught two metric traps that would
  each have cost GPU money: contact-loss is not flight (a toppling duck loses both contacts and logs
  excellent "air time" — 384 apparent successes were topples), and `current_air_time` is *per-foot*
  (an ordinary walk reports 125-300 ms and would read as a pass).
- Prior art ([`docs/prior-art-hop.md`](./docs/prior-art-hop.md)) — a community policy reporting a complete
  two-foot hop with crouch, takeoff, landing absorption and stable recovery, trained under backlash + BAM +
  current saturation + DR. Sim-only, self-reported, never hardware-tested.

**Decision recorded: do not spend the ~$5 re-asking "can it leave the ground."** It buys little. The
budget moves to S5. Note that the prior art's upstream base `d424a0c` is not our pin `1e79c29`, and
whether it is ahead of ours is unchecked — one `git log` against the `upstream` remote settles it.

**S2 — Does the stock pipeline run end-to-end on our HF account?** *(RESOLVED 2026-09-04: yes.)*
Namespace `chelleboyer`, Pro active. Stock velocity task submitted via `--hf-jobs`, ran to completion,
`.pt` checkpoints landed in a private Hub model repo, wandb streamed live. The pipeline is trusted. Only
the 12h timeout behaviour remains unobserved — see [`docs/upstream-pin.md`](./docs/upstream-pin.md).

**S3 — Is the all-remote loop tolerable?** *(RESOLVED 2026-09-04: the loop is HYBRID, not all-remote.)*
`uv sync` and the full CPU test suite (199 tests) pass on Windows with no GPU in ~13 s. Iterate locally;
submit only real training.

## Open questions

- **Is "hopscotch" one policy or several?** A single commanded gait covering all hop types is cleanest,
  but distinct hops (single-foot cell vs two-foot straddle) may need separate policies sequenced by a
  script. Settled by how well one policy generalizes across hop types in training — and note that the
  prior art is evidence for the several-policies shape.
- ~~**Exact command-block semantics.**~~ **RESOLVED** — [`docs/command-block.md`](./docs/command-block.md).
  `body_pose` is `[x, y, z, roll, pitch, yaw]`, a delta from nominal standing, carried at reward weight 0
  in the velocity env, so the whole 6D block is free. Hop intent goes in `body_pose[2]`, and
  `body_pose_tracking` **stays at weight 0** — tracking a commanded height rewards standing tall, which
  beats flight. A test asserts this.
- ~~**What does the real course look like?**~~ **DISSOLVED by the sim-only scope** — the course is a free
  variable, sized to the measured hop. See Key decisions.
- **What is the right forward-hop distance threshold?** The 5 cm in S5 is a placeholder. Settled by
  measuring foot length and body height against the achieved distance.
- **`nominal_height = 0.095` does not match the model** (measured: trunk z 120.0 mm authored, 116.7 mm
  settled). Harmless only while `body_pose_tracking` is at weight 0. Anyone raising that weight inherits a
  ~22 mm error. Documented in [`docs/command-block.md`](./docs/command-block.md); re-measure before
  trusting it.
- **Does the fork pin or track upstream?** Default is pin-and-pull-deliberately; revisit if upstream ships
  something we want.
- **Does upstreaming matter?** The fork could be kept contribution-shaped to submit hopscotch back to
  Pollen Robotics. Not decided; costs discipline, reversible either way. Note the sim-only scope makes our
  work less directly useful to them, since their interest is hardware.

## Revision history

- **2026-09-04, session 1** — original decisions. Approach A chosen; B rejected as undeployable; C
  rejected as brittle. S1 blocking.
- **2026-09-04, session 2** — S2 and S3 resolved. Command-block semantics resolved. Hop env and both
  flight rewards built. Prior art found, contesting A vs C on evidence.
- **2026-09-05, session 3 (later)** — **S5 ran to completion.** The duck hops forward (confirmed on
  video; training logged forward travel at ~95% of the velocity cap, so the cap is saturated and needs
  raising). But the run exposed the flight reward's shape as wrong — 52% of life airborne — so the main
  term becomes per-hop displacement paid at landing, converging with E3, plus head-upright at touchdown.
  Eval battery, training-montage builder and probe viewer added; all CPU, all free. Recorded a hard
  limitation: the eval battery drives position servos while training drives BAM, so its FORWARD numbers
  are not trustworthy (it read −2.2 mm/hop against a policy the video shows hopping forward).
- **2026-09-04, session 3** — **scope narrowed to sim-only, hardware deferred.** B's rejection reason
  voided and B reopened as a deferred option; C recorded as the fallback. S1 closed without spending and
  superseded by **S5 (forward hop)**; **S6 (perception)** added, deferred. Forward-intent encoding decided
  (E1 now, E3 later, E2 rejected with receipts). Course sizing dissolved into a free variable. Budget
  posture set at 2-3 runs.
