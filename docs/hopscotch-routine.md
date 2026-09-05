# The hopscotch routine — every step, and what Microduck needs for it

> The choreography this project is ultimately building toward. Written 2026-09-05, before Phase 1 is
> sliced, so the routine drives the ticket list rather than being retrofitted to it.
>
> Companion docs: [architecture](../microduck-hopscotch-architecture.md) (decisions),
> [command-block](./command-block.md) (the 13D contract), [hopscotch-rules](./hopscotch-rules.md) (state).

## Ground rule: the duck is blind, so this is choreography

Microduck has **no exteroception** — 48D proprioception + a 13D command block, no vision, no height scan,
no localization. It cannot see the court, the marker, or where it landed. So every step below is a
**command sequence issued to a hot-swappable policy**, timed open-loop, over a course painted on flat
ground for the humans watching.

This is the single most important constraint on the routine: **anything requiring the duck to know where
it is, is out of reach** unless we break the 61D contract (spike S6, deferred). Accuracy comes from
command precision and repeatability, not from aiming.

## The court

Classic single-player layout, 8 squares plus a home arc:

```
            ( HOME )          <- turn-around arc
          +----+----+
          | 7  | 8  |         <- side-by-side  -> TWO FEET, one per square
          +----+----+
            | 6  |            <- single        -> ONE FOOT
          +----+----+
          | 4  | 5  |         <- side-by-side  -> TWO FEET
          +----+----+
            | 3  |            <- single        -> ONE FOOT
          +----+----+
            | 2  |            <- single        -> ONE FOOT
          +----+----+
            | 1  |            <- single        -> ONE FOOT
          +----+----+
             ^^^
           start line
```

**Cell size is a free variable** (sim-only scope): there is no physical court, so squares get sized to
the hop the duck actually achieves, not the other way round. The measured open-loop forward reach is
8 mm/hop and the S5 pass mark is 25 mm — so a "square" may end up only a few centimetres. Scale the
whole court to the measured hop, and it still reads as hopscotch on camera.

## The full run, step by step

One complete turn for marker-square **N**. A full game repeats this for N = 1…8.

| # | Human step | Microduck equivalent | Policy / capability | Exists? |
|---|---|---|---|---|
| 1 | Stand at the start line | Idle stand, stable entry pose | walk policy, all-zero command | ✅ |
| 2 | **Toss the marker into square N** | Pick up carried marker, release/throw it forward | `GroundPick` + a **toss** | ⚠️ partial |
| 3 | Hop into square 1 (skip it if N=1) | One commanded forward hop | `HopForward` | 🔄 training |
| 4 | Continue 2, 3 — one foot each | Consecutive single-foot hops | **single-foot hop** | ❌ |
| 5 | Land 4 & 5 — one foot per square | Two-foot straddle landing | **two-foot straddle** | ❌ |
| 6 | Square 6 — one foot | Single-foot hop | single-foot hop | ❌ |
| 7 | Land 7 & 8 — straddle | Two-foot straddle | two-foot straddle | ❌ |
| 8 | **Turn around in HOME** | 180° turn in place | turn-in-place (`ang_vel_z`) | ✅ |
| 9 | Hop back 8/7, 6, 5/4, 3, 2 | Reverse sequence, same primitives | as above | partial |
| 10 | **Pause at square N+1, balance on one foot** | Hold a stable one-foot stance | **one-foot balance hold** | ❌ |
| 11 | **Bend down and pick up the marker** | Crouch, grasp, return to stance | `GroundPick` | ✅ |
| 12 | Hop over square N, land past it | One longer hop (skip a cell) | commanded hop **distance** (E3) | ❌ |
| 13 | Hop out over the start line | Final forward hop | `HopForward` | 🔄 |
| 14 | Next turn: N+1 | Sequencer increments, repeat | choreography daemon | ❌ |

**Fouls** (a human loses their turn; for us these are the failure modes to detect and report):
marker misses the square · foot touches a line · the non-hopping foot touches down · loss of balance ·
a square gets skipped.

## What this implies — the capability list

Ordered by how much new work each represents.

**Already exist**
- Forward hop and upright landing — S5, training now.
- Turn in place — a trained command bucket in the velocity family (`rel_turn_in_place_envs`), because
  uniform sampling buried it at ~2% of experience.
- Crouch and touch the ground with the mouth tip — `Mjlab-GroundPick-*`, which is step 11 almost exactly.

**New, and the real Phase 1 work**
1. **Commanded hop distance (E3).** Steps 3, 12 and the whole idea of "land *in* a square" need
   distance to be an input, not an emergent property. Already designed: `body_pose[0]`, latched
   takeoff→landing, and explicitly *not* routed through `body_pose_tracking`.
2. **Single-foot hop.** Steps 4, 6, 9. **This is not a tuning step** — hopping and landing on one leg is
   a different balance problem from the two-foot hop, and nothing in the current reward stack touches it.
   Expect it to be the hardest single item on this list.
3. **Two-foot straddle.** Steps 5, 7. Land with feet *apart*, one per square — the opposite of the
   bilateral-clearance shape the current reward encourages (which pays for both feet rising *together*).
   Probably a lateral foot-separation term at touchdown.
4. **One-foot balance hold.** Step 10. A static skill, closer to `SitStand`'s hold than to a hop, but on
   one leg. Prerequisite for the pickup, since the marker foot must stay up.
5. **The marker toss.** Step 2. `GroundPick` proves the duck can reach the ground with its mouth; a
   *toss* additionally needs a release with forward momentum. The object physics exist —
   `Mjlab-BallKick-Flat-MicroDuck` already simulates a 70 mm / 15 g ball, and `ball.xml` / `scene_ball.xml`
   are in the tree. A carried-and-dropped marker is a strictly easier version of a thrown one, and is
   worth doing first.
6. **The choreography daemon.** Step 14. The sequencer that issues the command series, hot-swaps
   policies, and counts squares. In sim this is a script with perfect timing — which is precisely why
   the sim-only scope makes the episodic approach (architecture doc, approach C) more attractive than it
   looked in session 1.

## Sequencing recommendation

Do **not** attempt the full routine as one policy. The realistic order:

1. **Finish S5** — forward hop, commanded distance (E3). Gets steps 3, 12, 13.
2. **A "hopscotch primitives" env family** — single-foot hop, two-foot straddle, one-foot balance. Each
   is its own spike with its own decision rule, because each may fail independently.
3. **Marker handling** — drop first, toss second, reusing the ball model.
4. **The daemon** — sequence what exists into a partial routine, and show it, before every primitive is
   perfect. A three-square run that works is better evidence than an eight-square plan.

An **abridged court** is a legitimate first demo: squares 1-2-3 out, turn, back, with a dropped marker.
It exercises the full *structure* of the routine (toss → hop out → turn → hop back → pause → pick up →
hop over → exit) with only the primitives that exist soonest.

## Open questions this raises

- **Does a single-foot hop survive this robot's balance?** Genuinely unknown, and it gates steps 4/6/9.
  It deserves a spike with the same shape as S1/S5 — measure before committing a curriculum.
- **One policy or several?** Already the architecture doc's open question, and this routine sharpens it:
  the primitives are different enough (hop / straddle / balance / pick) that hot-swapping several
  policies looks more likely than one policy generalizing across all of them.
- **Does the marker need to be a physical object at all?** A dropped marker that the duck then hops over
  reads correctly on camera; a *thrown* marker landing in a specific square is open-loop aiming without
  feedback, which is a much harder ask than it looks.
- **How is a foul detected?** With no perception, the duck cannot know it faulted. Foul detection is the
  daemon's job, from privileged sim state — fine in sim, impossible on hardware.
