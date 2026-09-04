# S1 — Can Microduck leave the ground? (CPU probe, preliminary)

> Spike **S1** in [`../microduck-hopscotch-architecture.md`](../microduck-hopscotch-architecture.md).
> Run 2026-09-04. Probe: [`../scripts/hopscotch/flight_probe.py`](../scripts/hopscotch/flight_probe.py).
> **This does not close S1.** It is a free CPU pre-check that sharpens what MD-3 has to measure.

## Result

**~34 ms of genuine simultaneous flight, ~11 mm apex rise, landing upright (max tilt 17°.)**

Best of 1728 open-loop countermovements (crouch → fast two-leg extension) on `scene_walk.xml`:

| flight | apex rise | max tilt | crouch | extend |
|---|---|---|---|---|
| 34 ms | 11.3 mm | 17.2° | −0.45 over 0.40 s | +0.60 over 0.03 s |

Against the architecture's decision rule (`>80 ms` → true hop track; `<30 ms` → pivot to stepping),
**34 ms lands in the ambiguous band, at its low end.**

## Why the number is trustworthy as a *bound*

Extension durations from 10 ms to 40 ms all produce the same 34 ms of flight. The result is
**plateaued on the actuator torque ceiling** (`forcerange ±0.96 N·m`, the real XL330 limit, enforced by
the MJCF), not on the search grid. Driving the legs faster buys nothing because torque saturates.

## Why it is optimistic

- **No BAM.** Training drives these joints with the BAM voltage model (`kp_fw=200`, `vin` 6.5–8.2 V).
  The probe uses a plain position servo. BAM's back-EMF cuts available torque as joint speed rises —
  exactly the regime a hop needs. This will reduce the 34 ms.
- **No backlash, no domain randomization, no observation noise, no command delay.**

So 34 ms is a **ceiling on the open-loop countermovement**, and the realistic figure is lower.

## Why a trained policy may still beat it

- **The head is ~280 g of a 737 g robot (38%).** The probe holds the neck and head fixed at `STAND`.
  A policy can swing that mass as a countermovement and add real impulse — a lever the probe never
  pulls. Upstream already documents the head's mass dominating dynamics ("a 280 g head must oscillate
  while stepping").
- The probe is open-loop, symmetric, and starts from rest. PPO can use asymmetric timing, ankle
  push-off shaping, and momentum from an approach.

## Two traps this probe caught (both would have cost GPU money)

1. **Contact-loss is not flight.** The first run reported "162 ms of air time" that was the duck
   *falling over* — feet leave the floor when you topple. Screening on tilt and trunk rise turned 384
   apparent successes into 0. MD-3 must gate its flight metric on tilt + rise, not contact alone.
2. **Per-foot air time is not simultaneous flight.** `current_air_time` / `last_air_time` are per-foot,
   and the velocity env's `air_time` reward already targets `[0.125, 0.300] s` with a walking policy
   scoring `1.01`. Reading those fields would report 125–300 ms **for an ordinary walk** and look like
   a pass. The metric must be duration of `n_contact == 0`.

`single_support_reward` (`mdp.py:4700`) already computes `n_contact` and enumerates the
`0 blades down (flight/hop)` case — the primitive for the simultaneous-flight reward exists.

## A third trap, in the model itself

The MJCF ships `kp=0.55, kd=0` position servos, which **cannot hold `STAND`** — the robot topples in
0.6 s untouched. `XmlPositionActuatorCfg` is commented out in `microduck_constants.py`; the XML gains
are placeholders because training uses BAM. Any CPU-side experiment must set its own gains and
verify the settle first.

Relatedly, the intuitive leg pattern (`hip +1, knee −2, ankle +1`) is **wrong for this model**: foot
pitch goes as `(−hip + knee − ankle)`, so that pattern rotates the foot ~228°/rad and topples the
robot at ±0.1 rad quasi-statically. The correct flat-foot direction, from the leg Jacobian, is
`(−0.515, −1.0, −0.485)` — the probe now derives it rather than assuming it.

## What this changes

- **S1 is not answered, but it is now cheap to answer well.** MD-3's GPU spike should be judged on
  `n_contact == 0` duration gated by tilt and rise, with the 34 ms open-loop figure as the
  no-learning baseline to beat.
- **Head/neck motion is a first-class part of the hop**, not a regularization nuisance. Do not add
  motion-blocker penalties on the neck early.
- If PPO cannot beat ~34 ms under BAM, the stepping-hopscotch pivot is the answer and the ~$5 spike
  will have said so quickly.

## Reproduce

```bash
uv run python scripts/hopscotch/flight_probe.py            # coarse grid + verdict
uv run python scripts/hopscotch/flight_probe.py --csv /tmp/best.csv
```
