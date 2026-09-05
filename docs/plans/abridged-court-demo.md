# Plan — the abridged 3-square hopscotch demo

> The first thing that looks like hopscotch to a human. Deliberately small: it exercises the whole
> routine STRUCTURE using only primitives that land soonest, so it ships while the hard skills
> (single-foot hop, straddle) are still being spiked.
>
> Routine spec: [hopscotch-routine.md](../hopscotch-routine.md) · Architecture:
> [architecture](../../microduck-hopscotch-architecture.md)

## Goal

Microduck completes a recognizable hopscotch turn on a 3-square court, in simulation, on video.

**Not** a full 8-square game, not accurate landings, not a thrown marker. The bar is *"a person watching
says 'that's hopscotch'"* — which is a real milestone, and the brief's success ladder puts a recognizable
sequence well before a perfect one.

## The abridged court

```
        ( HOME )        <- turn-around
      +----+----+
      | 3  |            <- single, one/two foot
      +----+
      | 2  |            <- MARKER square this turn
      +----+
      | 1  |            <- single
      +----+
       start
```

Marker goes in square 2, which is what makes it a *turn* rather than a walk-through: the duck must skip
2 on the way out, and pick the marker up from square 1 on the way back.

## The command sequence

Eight beats. Every one is a command issued open-loop by the sequencer — the duck cannot see the court.

| beat | action | how |
|---|---|---|
| 1 | idle at the start line | all-zero command, walk/stand policy |
| 2 | **drop the marker into square 2** | `GroundPick` + release (see Substitution below) |
| 3 | hop into square 1 | one commanded forward hop |
| 4 | hop OVER square 2 into square 3 | one longer hop — needs commanded distance (E3) |
| 5 | turn 180° in HOME | `ang_vel_z`, already a trained bucket |
| 6 | hop back into square 1 | one commanded forward hop |
| 7 | **pick the marker up** | `GroundPick`, unmodified |
| 8 | hop out over the start line | one commanded forward hop |

## What it needs that we have

- **Forward hop + upright landing** — S5, training now.
- **Turn in place** — `rel_turn_in_place_envs` is an explicit command bucket in the velocity family.
- **Crouch to the ground with the mouth tip** — `Mjlab-GroundPick-*`, which is beat 7 exactly.
- **Policy hot-swapping behind the 61D contract** — the runtime already does this; `infer_policy.py`
  rehearses it.

## What it needs that we don't have

**1. Commanded hop distance (E3)** — beat 4 is "hop *further*, to clear a square". Without distance as an
input, every hop is the same length and there is no skip. Already designed: `body_pose[0]`, latched
takeoff→landing, explicitly *not* routed through `body_pose_tracking`. **This is the blocking item.**

**2. A marker the duck can release** — see below.

**3. The sequencer** — a script that issues the eight beats with fixed timing, swaps policies between
them, and records the video. In sim this is genuinely easy: perfect timing, guaranteed entry poses,
privileged state for detecting fouls. No perception needed.

**4. A painted court** — three coloured boxes in the scene, purely visual. `slope_terrain.py` is the
pattern for adding geometry, but these can be non-colliding decals so they cannot trip the duck.

## Two deliberate substitutions

**Drop, don't throw.** A thrown marker landing in a specific square is open-loop aiming with no feedback
— the duck cannot see where it went. A marker carried and *dropped* at a commanded moment reads
identically on camera and is a far easier skill. Throwing is a later upgrade, not a prerequisite.

**Two feet, not one.** The real game uses one foot on single squares. The single-foot hop is the hardest
unbuilt primitive (a different balance problem, gating 3 of 14 routine steps). The abridged demo uses the
two-foot hop everywhere. A viewer reads the *sequence* as hopscotch; the foot count is a refinement.

Both substitutions are recorded here so nobody later mistakes them for oversights.

## Order of work

1. **Judge S5** (imminent) — the whole demo rests on the forward hop being real.
2. **E3: commanded hop distance.** The one blocking capability. Roughly S5's shape: extend the reward,
   register a variant, cfg tests, smoke test, train. Budget 2-3 runs.
3. **Marker drop.** Reuse the `BallKick` ball model (70 mm / 15 g, `ball.xml` already in the tree) as the
   marker; extend `GroundPick` with a release. Verify in CPU MuJoCo before training anything.
4. **Court geometry** — visual only, cheap.
5. **The sequencer** — `scripts/hopscotch/court_demo.py`: load the ONNX policies, issue the eight beats,
   record. CPU MuJoCo, so it runs locally and free, exactly like `hop_eval.py`.
6. **Record and watch.** Iterate on timing, which is free.

Steps 3-5 need no GPU at all. Only step 2 costs money.

## Risks

- **E3 may need more than 2-3 runs.** It asks the policy to learn the skill *and* its command
  conditioning. If it stalls, a fallback exists: fix a single hop distance and space the court to it —
  less flexible, but it still demos.
- **Landing accuracy is unverifiable.** Open-loop commands mean the duck lands where the policy's
  variance puts it. Sizing squares generously to the *measured* hop distribution (not its mean) is the
  mitigation, and is free because the court is a free variable.
- **Beat timing is brittle** at the policy seams — the prior-art hop needs ~1 s of stable standing entry.
  Build slack into the sequencer between beats rather than optimizing for a snappy routine.

## Definition of done

A single continuous video in which Microduck drops a marker, hops out over the marker square, turns,
hops back, picks the marker up, and exits — with the court visible. Fouls tolerated and reported, not
prevented.
