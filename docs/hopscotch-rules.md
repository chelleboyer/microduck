# Microduck Hopscotch

Teach Pollen Robotics' Microduck to hop, then to hopscotch — trained in MuJoCo simulation on Hugging
Face Jobs, ultimately deployed to the physical robot.

## Current state (2026-09-04)

**Upstream merged and pinned; MD-1 in progress.** Upstream `microduck_rl` is merged at
`1e79c29` (see [`upstream-pin.md`](./upstream-pin.md)). No hopscotch code written yet.

**Next action: finish MD-1** — `uv sync`, HF auth and namespace, then run stock
`Mjlab-Velocity-Flat-MicroDuck` via `--hf-jobs` end-to-end.

## Read these, in this order

1. [`../CLAUDE.md`](../CLAUDE.md) — **upstream's playbook.** Env-building workflow, invariants, joint
   layout, and the reward-design lessons. Authoritative; read it before touching rewards.
2. [`microduck-hopscotch-project-brief.md`](../microduck-hopscotch-project-brief.md) — the intent. What
   we're building and why. Success #1 is *"Microduck intentionally hops forward and lands upright"*.
3. [`microduck-hopscotch-architecture.md`](../microduck-hopscotch-architecture.md) — the decisions, with
   rationale, rejected alternatives, spikes and open questions. **The load-bearing doc.**
4. [`tickets/microduck-hopscotch-phase-0.md`](./tickets/microduck-hopscotch-phase-0.md) —
   Phase 0 work: MD-1, MD-2, MD-3.

## Constraints that are expensive to rediscover

**The duck is blind.** All Microduck policies share a fixed **61-dimensional** observation: 48
proprioception + a 13D command block `[twist(3), head_pose(4), body_pose(6)]`. No vision, no height
scan, no terrain sensing. Microduck *cannot see a course*. Hopscotch is therefore **choreography** — a
sequence of commands over flat ground — not perception. Designs where the robot reads its environment
are unreachable on real hardware. Never delete a command slot; unused slots are zero-padded to keep
input neurons alive.

**No local GPU; all-remote by choice.** Training requires CUDA (mjlab runs on MuJoCo Warp) and the dev
machine is Windows without a GPU. We deliberately chose an all-remote loop over WSL2 setup. MD-2's
preflight harness is what makes that affordable — CPU config tests and a 64-env smoke test run before
GPU training in the *same* job, so a bad reward sign dies in seconds on `cpu-basic` ($0.01/hr) instead
of hours into `l4x1` ($0.80/hr).

**This repo is a fork of upstream, pinned.** `pollen-robotics/microduck_rl` (Apache 2.0) is kept as a
git remote and pulled deliberately, not continuously. It is young and moving fast (69 commits on `main`
in August 2026 alone). Its distilled sim2real and reward-design playbook is the repo-root
[`CLAUDE.md`](../CLAUDE.md) — **read it before touching rewards**; it encodes months of hard-won
lessons and is more trustworthy than reasoning from first principles here.

**The core new work is one reward term.** mjlab's stock `feet_air_time` rewards *alternating*
single-foot air time — ordinary walking. A hop needs **simultaneous** flight, both feet off at once.
That term does not exist upstream and is not a reweighting of one that does.

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
