# Prior art — someone already trained a Microduck hop

> **⚠ RECALIBRATED 2026-09-06.** After watching this policy's own videos next to our S5.x runs, the
> user's judgement was that it is *"not even as good as some of my prior attempts."* Two consequences:
> it is still valid evidence that **a hop is reachable** under BAM + backlash + DR, which is what
> closed S1 and is why we did not spend a GPU run re-asking that — but it is **not a quality bar**, and
> nothing in our reward design should be tuned to match it. Its `bilateral_foot_clearance` target
> (0.035 m) was already borrowed and remains fine as a clearance scale, not as an aspiration.
> Media for comparison: `logs/community/happy-hop-sim.mp4` and `happy-hop-REAL-ROBOT.mp4`.
> Note it is also **ONNX-only** — no `.pt`, so it cannot be used as a training warm start.

> Found 2026-09-04 while surveying community policies. This materially changes the **S1** spike in
> [`../microduck-hopscotch-architecture.md`](../microduck-hopscotch-architecture.md). Read before
> spending GPU on S1.

## The artifact

[`joanfox/microduck-happy-hop`](https://huggingface.co/joanfox/microduck-happy-hop) — a Hugging Face
model repo containing `policy.onnx`, `manifest.json`, and two videos.

From its manifest:

| | |
|---|---|
| task_id | `Mjlab-HappyHop-WalkTransition-Clearance-Flat-Backlash-MicroDuck` |
| kind | **episodic**, `duration_s` 3.0, `entry_pose` standing |
| obs / action | 61 / 14 — the standard contract |
| objective | `bilateral_foot_clearance_target_m` **0.035**, success threshold **0.030** |
| environment | `backlash: true`, BAM, `current_limit_a` 1.75, `action_filter: none` |
| training | PPO, 1255 iterations, upstream `microduck_rl@d424a0c` |
| entry state | `walking_backlash_model_5000.onnx` after 1.0 s of exact-zero velocity command |
| status | `sim-only-hardware-candidate` — *"never tested on hardware"* |
| reported result | *"complete crouch, two-foot hop, landing absorption, and stable recovery"* |
| known limits | *"modest jump height; requires a stable standing entry; not trained to take off directly from an active walking stride"* |

## What it changes

**S1's original question is substantially answered: yes, in sim, under realistic actuation.** A two-foot
hop with ~30–35 mm bilateral clearance was reached under backlash + BAM + current saturation + DR —
the conditions our own CPU probe was explicitly optimistic about
([`s1-flight-probe.md`](./s1-flight-probe.md) found 34 ms / 11 mm open-loop, no back-EMF). The
stepping-hopscotch pivot recedes accordingly.

**Do not spend the ~$5 S1 run purely to re-ask "can it hop."** That buys little now. The unresolved
question for hopscotch is narrower and more useful:

> Can a hop be made **commanded and repeatable** — driven through the 13D command block — rather than
> a one-shot episodic trick triggered by an external policy switch?

Their policy zeroes every command slot (`twist`, `head`, `body` all "unused; zeros") and is triggered
by an external one-shot switch after a stable standing entry. That is architecture-doc **approach C**
(episodic trick library), which we rejected in favour of **A** (commanded gait). Their result is real
evidence that C is what someone got working first; our reasoning for A still stands *for hopscotch
specifically* (a rhythm makes consecutive hops free), but the choice is now contested by evidence
rather than settled by argument.

## What it does not answer

- **Hardware.** Explicitly never tested on a real robot. S4 (does a hop survive the reality gap) is
  untouched by this.
- **Forward motion.** Vertical hop only, and explicitly not trained to take off from a walking stride.
- **Reproducibility.** Self-reported, read from its own model card. Not reproduced here.
- **Their upstream base `d424a0c` is not our pin `1e79c29`.** Unchecked whether it is ahead of ours,
  i.e. whether they built on env features we do not have. One `git log` against the `upstream` remote
  settles it.

## What we took from it

`bilateral_foot_clearance` (`tasks/mdp.py`, commit `e96d5d9`) — the clearance metric, at their 0.035 m
target, as the dense ramp under our binary `simultaneous_flight` term. See that function's docstring
for why both are kept: each rejects the other's exploit.
