# Microduck Hopscotch

Teach Pollen Robotics' Microduck to hop, then to hopscotch — trained in MuJoCo simulation on Hugging
Face Jobs, ultimately deployed to the physical robot.

## Scope (session 3, 2026-09-04)

**Sim-only.** Getting hopscotch working in simulation is what matters right now. Physical deployment is
**deferred, not dropped** — BAM, domain randomization and the backlash twins stay on, because they cost
nothing to keep (already wired; the one known working Microduck hop was trained *with* them) and dropping
them is a one-way door. What the scope change actually unlocks is that two hardware-justified switches are
now ours to flip if we want them: mjlab's native `height_scan`, and promoting `base_lin_vel` from
critic-only to the actor. Neither is flipped yet — see S6 in the architecture doc.

## Current state (2026-09-04, end of session 2)

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

## Session 3 state (2026-09-05)

**S5 is implemented and training.** Run `s5-forward-hop`, job `6a9bf2e6259f8e97255e28a5`,
`Mjlab-HopForward-Flat-MicroDuck`, 4096 envs × 1500 iters, video on; checkpoints and mp4s land in
`chelleboyer/s5-forward-hop`.

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
