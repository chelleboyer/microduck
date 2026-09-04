# Microduck Hopscotch

Teach Pollen Robotics' Microduck to hop, then to hopscotch — trained in MuJoCo simulation on Hugging
Face Jobs, ultimately deployed to the physical robot.

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

**Next action: re-scope S1 before running it.** The original question ("can this robot leave the
ground") is substantially answered by the prior art, so a ~$5 run to re-ask it buys little. The open
question is whether a hop can be **commanded and repeatable** through the 13D block rather than a
one-shot episodic trick. Decide that, then run.

Two small Windows fixes in `hf_jobs.py` are known and unmade: the log streamer dies on non-ASCII
output unless `PYTHONIOENCODING=utf-8` is set (the job itself is unaffected, but the local command
reports failure), and the `[wandb] forwarding API key from ~/.netrc` message names the wrong file when
the key came from `_netrc`.

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

**The duck is blind.** All Microduck policies share a fixed **61-dimensional** observation: 48
proprioception + a 13D command block `[twist(3), head_pose(4), body_pose(6)]`. No vision, no height
scan, no terrain sensing. Microduck *cannot see a course*. Hopscotch is therefore **choreography** — a
sequence of commands over flat ground — not perception. Designs where the robot reads its environment
are unreachable on real hardware. Never delete a command slot; unused slots are zero-padded to keep
input neurons alive.

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

**The open project risk is physics.** No existing Microduck task has a flight phase. It is genuinely
unknown whether an ~800g biped on compliant, backlash-heavy XL330 servos can leave the ground. MD-3
answers this for ~$5. A negative result is a *successful* spike — it pivots hopscotch from jumping to
stepping into cells, and reshapes roughly a third of the remaining backlog. **Phase 1 is deliberately
unsliced until MD-3 reports.**

## Working conventions

- Spike tickets are done when they produce a **trustworthy decision**, not a working feature.
- Every `Episode_Reward/<penalty>` term must log ≤ 0 throughout training. Upstream calls this check
  infallible for catching reward-sign inversions.
- Expect 2–5 reward-hacking iterations before convergence. That is normal, not failure.
- Never hand-export a checkpoint — the observation normalizer must be baked in by `scripts/export.py`.
  In-sim play hides the bug because it applies the normalizer anyway.
- Keep `pyproject.toml` honest: `uv sync` is ground truth because HF Jobs runs it. Anything that works
  only via manually-installed local packages dies remotely.
