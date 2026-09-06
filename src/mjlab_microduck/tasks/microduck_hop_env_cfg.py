"""Microduck HOP environment — the S1 spike: can this robot leave the ground?

**This is a spike env. Its deliverable is a DECISION, not a gait.** "Done" is
not "the duck hops well"; it is "we know whether it can, with data good enough
to bet the next phase on". See docs/tickets/microduck-hopscotch-phase-0.md
(MD-3) and the decision rule at the bottom of this docstring.

Built on make_microduck_velocity_env_cfg so domain randomization, obs noise,
command delays, encoder bias, IMU misalignment and the NaN guards stay in sync
by construction (CLAUDE.md: building standalone means porting that whole stack
by hand).

WHAT IS DIFFERENT FROM THE WALKER, AND WHY
------------------------------------------
Everything below is a deliberate deviation. The velocity recipe is tuned to
make WALKING optimal, and several of its terms make walking beat hopping.

1. **simultaneous_flight is the main term** (weight 5.0). mjlab's air_time
   rewards ALTERNATING single-foot air time — a healthy walker already scores
   1.01 on it with 125-300 ms per-foot air times while never leaving the floor.
   It cannot detect a hop and is not a substitute (docs/s1-flight-probe.md).

1b. **bilateral_foot_clearance is its dense ramp** (weight 2.0). The flight
   term is binary and pays nothing until the robot is ALREADY airborne, which
   is no gradient at all through the long stretch where it is learning to load
   and extend. Clearance pays partial credit for a partial lift. The pairing is
   deliberate: clearance can be farmed by tucking the feet while the trunk
   sags, which the flight term's contact condition rejects; flight can be
   reached by toppling, which clearance rejects because a falling duck's feet
   do not rise. Each covers the other's exploit, and both carry the trunk gates.

2. **air_time drops 3.0 → 0.5.** At 3.0 a good stride out-earns any hop the
   robot can currently produce, so walking is simply the better policy. Not
   zeroed: lifting a foot at all is a prerequisite skill, and a small stride
   reward keeps that bootstrap alive.

3. **The head is FREED, not regularized.** head_pose_tracking 2.0 → 0.5 and
   the head_pose_bias curriculum is removed entirely. The S1 probe holds the
   neck rigid and still reaches 34 ms; the head is 280 g of a 737 g robot
   (38%), and swinging it is the most plausible lever a policy has to beat the
   open-loop bound. A term that pulls the head to a commanded pose blocks
   exactly that countermovement. MD-3 is explicit: no motion-blocker penalties
   on the neck.

4. **Motion-blockers cut ~5x** (body_ang_vel -0.05 → -0.01, angular_momentum
   -0.02 → -0.005). CLAUDE.md: these penalize what dynamic motion physically
   requires; keep them LOW for dynamic tasks. Anti-violence pressure belongs on
   impacts, not on rotation rate — a 25 cm robot tumbles at 3.5-5.5 rad/s
   naturally.

5. **action_rate starts at ZERO**, ramped in only from iter 400 (see the
   curriculum below). CLAUDE.md: any attempt-tax active while a hard skill is
   being explored makes "do nothing" win. Smoothness comes AFTER discovery.

6. **pose 1.0 → 0.3.** The pose reward pulls leg joints toward HOME, which is
   the opposite of the crouch-then-extend countermovement a hop needs.

7. **Hop in place.** Twist ranges shrink to near-zero and half the envs are
   commanded to stand. The spike asks one question; adding locomotion splits
   the policy's capacity across two. Ranges stay NON-ZERO so the twist input
   neurons survive for the forward-hop curriculum that follows.

THE FORWARD VARIANT (forward=True) — SPIKE S5
---------------------------------------------
Everything above describes the hop-in-place baseline, which is unchanged and
stays registered as Mjlab-Hop-Flat-MicroDuck so backlash and forward A/Bs have
a fixed reference. ``forward=True`` adds three more deviations and registers as
Mjlab-HopForward-Flat-MicroDuck.

S5 asks the brief's actual Success #1 — "hops FORWARD and lands upright" —
which the prior art does NOT answer (vertical hop only, from a standing entry,
explicitly not from a walking stride).

8. **forward_flight_progress** (weight 1.5), encoding E1: capped forward speed
   paid ONLY while airborne. Un-commanded by decision — E3 (commanded hop
   distance in body_pose[0]) is Phase 1 work, and E3 is this term plus latching
   plus command gating, so nothing here is throwaway. Gating on flight is what
   makes it unfarmable: walking is a strictly easier way to earn forward
   velocity than hopping, so an ungated version would just retrain the walker.

9. **hop_landing_quality** (1.0) and **hop_landing_impact_penalty** (+0.5,
   self-negating). Every other gate in this file screens the robot DURING
   flight, so before these a forward hop that reliably face-planted scored
   exactly as well as one that stuck. The impact term's weight is POSITIVE
   because the function returns <= 0 — see its docstring, and note that
   roulade's *_penalty functions use the opposite convention.

10. **track_linear_velocity 2.0 -> 0.3.** THE fix that makes E1 reachable, and
   the least obvious line in this file. The walker's tracking term
   (weight 2.0, std sqrt(0.1) = 0.316) is inherited unchanged, and with this
   env's near-zero commanded velocity it pays ~2.0/step for standing still and
   2.0*exp(-(0.4/0.316)^2) = 0.41/step at 0.4 m/s. So hopping forward COSTS
   ~1.6 reward/step against a forward term that can pay at most 1.5 — and
   test_flight_is_the_dominant_reward caps every positive term below
   simultaneous_flight's 5.0 anyway. E1 loses before it starts unless tracking
   steps back. Kept non-zero (0.3) so it still discourages aimless drift.
   This is CLAUDE.md's "compare reward mass, not weights" rule, exactly.

   NOT fixed by widening the twist ranges: that is encoding E2, rejected in the
   architecture doc because velocity tracking has a strictly easier solution
   than hopping.

S5.1 — THE RESHAPE AFTER S5's RUN (forward=True, 2026-09-05)
------------------------------------------------------------
S5 ran to completion and the duck hops forward — confirmed on video, with
forward_flight_progress at ~95% of its cap. It also proved the reward SHAPE
wrong, and deviations 11-13 are that fix. The env evolves in place rather than
forking a third variant: the hop-in-place baseline is the A/B reference (and is
still untouched), the S5 recipe is reproducible from git plus its wandb run,
and a task id that means "the current best forward hop" is more useful than a
growing flag matrix.

11. **hop_displacement is now the main term** (weight 10.0 after handover),
   and simultaneous_flight demotes 5.0 -> 2.0. simultaneous_flight pays 1.0 PER
   STEP airborne, so air time WAS the objective: the run ended up airborne ~52%
   of its life (2.58 / 5.0) while hop_landing_quality stayed the weakest term
   in the stack (0.138). That is bouncing, not hopping. Displacement pays for
   "took off HERE, landed THERE", once per hop, capped — hanging in the air
   earns nothing by itself. This converges with E3 (commanded hop distance)
   instead of competing with it: E3 is this measurement plus a command.

   The handover is a CURRICULUM, not a swap, because the two terms are phase-
   aligned to different skills. Flight must be DISCOVERED before displacement
   means anything (a term that pays only at a landing from a genuine flight is
   silent until flight exists), so flight leads until iter 300 and displacement
   takes over after. CLAUDE.md: never introduce a term before the skill it
   prices exists.

   Weights are set by reward MASS, not by their face values. Under S5 a single
   ~0.12 s hop earned roughly 5.0 x 6 steps = 30 from flight alone; a 45 mm hop
   here earns 10.0 x 0.45 x ~7 window steps = ~32. The same money, paid for
   arriving instead of for hanging.

12. **The head is priced AT TOUCHDOWN, and only there.** hop_landing_quality
   gains a head-upright factor (std 0.35, deliberately wide) and head_pose_bias
   returns on a late, gentle ramp — 0.5/1.0 against the walker's 1.0/2.0/3.0.
   The head rides low because deviation 3 freed it on purpose, and that trade
   is still right in flight; it is wrong at landing. Raising
   head_pose_tracking is NOT the fix — see note 3 and the receipt at
   microduck_velocity_env_cfg.py:729-737.

13. **FORWARD_VEL_CAP 0.4 -> 0.8 m/s, and the term demotes 1.5 -> 0.5.** The run
   logged ~95% of the 0.4 cap: SATURATED, so the metric could no longer tell a
   good hop from a great one. It also pays per airborne step, which is the same
   shape the displacement term exists to replace — so it keeps its job as the
   dense in-flight ramp (the role bilateral_foot_clearance plays under flight)
   and loses its claim to being a driver.

COMMAND ENCODING
----------------
Hop intent lives in **body_pose[2]** (the z component), per docs/command-block.md.
The 6D body_pose block is carried at reward weight 0 in the velocity env, so
the whole block is free. body_pose_tracking STAYS at weight 0 here: it is a
pose-HOLDING reward, and rewarding a commanded height makes "extend the legs
and stand tall" strictly better than ballistic flight — which is easier and
scores higher. Flight must be paid by the flight term alone.

No explicit hop bucket is needed, unlike TURN_IN_PLACE_FRACTION. Turn-in-place
was rare because it is a CONJUNCTION in a 3D command space (lin≈0 AND |ang|
high) that independent uniform sampling produces ~2% of the time. A single
scalar sampled uniformly over (0, HOP_CMD_MAX) is well covered by construction.
The idle state still needs explicit training, so zero_command_prob handles the
exact-zero command that uniform sampling never produces.

DECISION RULE (architecture doc, S1)
------------------------------------
    >80 ms consistent simultaneous flight, upright landing -> true hop track.
    contact never breaks / <30 ms                          -> pivot to
                                                              "stepping" hopscotch.

Measure duration of ``n_contact == 0`` GATED ON TILT AND RISE. Do not read
per-foot current_air_time: it reports 125-300 ms for an ordinary walk and will
look like a pass. Baseline to beat: ~34 ms open-loop, torque-saturated and
WITHOUT BAM back-EMF, so the realistic no-learning figure is lower.
"""

import math
from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

NUM_STEPS_PER_ENV = 24

# Max commanded hop intent, metres of body_pose z delta. Not a target height
# the policy must reach — body_pose_tracking is at weight 0 — just the scale
# over which "how much hop do you want" is expressed.
HOP_CMD_MAX = 0.06
# Fraction of resamples that yield the exact all-zero command. Uniform sampling
# essentially never produces it, and all-zero is the deployment idle state.
HOP_ZERO_CMD_PROB = 0.15

# Flight gates. min_height MEASURED, not assumed: the walk model authors STAND
# at trunk z = 120.0 mm and settles to 116.7 mm under load, so 100 mm allows a
# compressed takeoff while excluding anything resting on the trunk. Re-measure
# after any model revision.
FLIGHT_MIN_HEIGHT = 0.10
FLIGHT_MAX_TILT_DEG = 30.0
FLIGHT_MIN_S = 0.02
FLIGHT_MAX_S = 0.30

# Bilateral foot clearance — the dense ramp toward flight. Target matches the
# one known working Microduck hop (joanfox/microduck-happy-hop: target 0.035 m,
# success 0.030 m). Measured on the walk model: foot sites rest at 2.9 mm at
# STAND, so clearance is scored ABOVE that and 0.035 means real lift.
CLEARANCE_TARGET = 0.035
FOOT_REST_HEIGHT = 0.0029

# Smoothness is introduced only after the skill exists (see docstring note 5).
ACTION_RATE_KICKIN_ITER = 400

# ── S5 / forward=True constants (docstring notes 8-10) ────────────────────────
# Forward speed at which the E1 term saturates, m/s. RAISED 0.4 -> 0.8 after the
# S5 run logged ~95% of the old cap: a saturated metric cannot distinguish good
# from great, and the term is flat exactly where the policy now lives. Too low
# caps a good hop; too high leaves the term flat across the achievable range and
# kills its gradient. Re-derive whenever a run pins it again.
FORWARD_VEL_CAP = 0.8
# MEASURED walk-model settle height. NOT the velocity env's nominal_height
# (0.095), which is ~22 mm low and survives only because body_pose_tracking
# runs at weight 0 — see docs/command-block.md.
LANDING_TARGET_Z = 0.1167
LANDING_WINDOW_S = 0.15
# Impact allowance: free-fall from the S1 probe's ~11 mm apex is ~0.46 m/s, so
# a hop the robot can currently perform costs nothing.
IMPACT_FREE_SPEED = 0.5
IMPACT_WINDOW_S = 0.04
# Down from the walker's 2.0. See docstring note 10 — this is the line that
# makes E1 reachable at all.
HOP_TRACK_LIN_VEL_WEIGHT = 0.3

# ── S5.1 constants (docstring notes 11-13) ────────────────────────────────────
# Per-hop forward travel at which the displacement reward saturates, m.
# DERIVED, and to be re-derived: the S5 policy flew at ~0.38 m/s (95% of the old
# 0.4 cap) for flight phases of roughly 0.1-0.15 s, i.e. ~40-60 mm per hop. A
# 0.10 m cap leaves ~2x headroom above that so the term keeps a gradient as the
# policy improves, while S5's 25 mm pass mark still scores a visible 0.25.
HOP_DISP_CAP = 0.10
# Final weight of the main term. Sized by reward MASS against what a hop used to
# earn from air time — see docstring note 11 — not by comparison with the face
# value of the per-step terms around it.
HOP_DISP_WEIGHT = 10.0
# Iteration at which displacement takes over from flight. Phase alignment, not a
# schedule: flight leads while the skill is being discovered, displacement leads
# once it exists. Keyed to ACTION_RATE_KICKIN_ITER (400), which S5 established as
# "the skill is consolidated by roughly here"; set a little earlier because a
# resumed run arrives with the skill already in hand.
DISP_HANDOVER_ITER = 300
# Flight after the handover. Not zero: leaving the ground stays a prerequisite,
# and displacement is silent until it happens. Must stay ABOVE the clearance
# ramp below, or a deep tuck outranks the goal it is supposed to ramp toward.
FLIGHT_WEIGHT_AFTER_HANDOVER = 2.0
CLEARANCE_WEIGHT_FORWARD = 1.5
# Head-upright factor at touchdown. WIDE on purpose (~55° of droop at 1 e-fold):
# a multiplicative factor tighter than the current policy's error collapses the
# product and the gradient with it, and hop_landing_quality was already the
# weakest term in the S5 stack. Tighten once the head actually comes up.
LANDING_HEAD_UPRIGHT_STD = 0.35

# ── S5.2: head up ALWAYS, and a hop RHYTHM (user call, 2026-09-05) ────────────
# Watching the S5.1 run's video: "his head is just hanging down, we need to get
# that head up all the time" and "head up, hop, stay in place for a sec, then
# hop again". Both are behaviour requirements, and both are reward changes.
#
# HEAD. Pricing the head only at touchdown (S5.1) was too little: the duck
# spends most of its life between landings, and that is where it looks wrong.
# The instrument for "up ON AVERAGE" is head_pose_bias — L1 on a 1 s EMA, which
# charges the DC droop while letting the countermovement oscillation cancel.
# It is brought to the WALKER's full strength (1/2/3) because the walker's head
# looks right, and it arrives early rather than at iter 800.
#
# The documented trap is NOT this: microduck_velocity_env_cfg.py:729-737 records
# that tightening head_pose_tracking's STD made the policy stop moving. That was
# an instantaneous tolerance a 280 g head cannot escape while stepping. The
# weight is a different dial, and the DC term is the escapable half — "at the
# optimum this costs a walking policy nothing".
HEAD_BIAS_STAGES = ((300, 1.0), (600, 2.0), (900, 3.0))
# Partial restore of the instantaneous term, 0.5 -> 1.0. Still half the walker's
# 2.0, so the head keeps room to swing as a countermovement mid-flight.
HOP_HEAD_TRACK_WEIGHT = 1.0

# RHYTHM. Seconds of standing still before a takeoff for a hop to be paid in
# full. Scaled smoothly, not gated (see hop_displacement) — a cliff would pay 0
# for every hop a bouncing policy can currently produce and the term would go
# silent. 0.5 s is "a sec" at the scale of a 25 cm robot: long enough to read as
# a deliberate pause on video, short enough to keep a routine watchable.
HOP_MIN_GROUND_S = 0.5
# How long after a landing the hold is worth paying for. Matched to
# HOP_MIN_GROUND_S so the payment stops exactly when the next hop becomes
# fully-paid: standing longer earns nothing, hopping does.
HOP_SETTLE_WINDOW_S = 0.5
# Horizontal speed at which "still" scores zero. A hop that skids on landing is
# not a hold.
HOP_SETTLE_MAX_SPEED = 0.15
HOP_SETTLE_WEIGHT = 1.5

# ── S5.3: make it LOOK like a hop — phase cycle, crouch, apex ─────────────────
# S5.2 fixed the head and failed the rhythm: hop_settle earned ~0, the policy was
# airborne 53% of its life, and displacement collapsed 7x because the 0.5 s pause
# it demanded from step 0 was a requirement the current behaviour could not meet.
# The user's verdict on the video was blunter and more useful: "he isn't really
# doing a hopscotch hop at all".
#
# The survey of what everyone else does explains why. TWO independent working
# hops exist on this exact robot, and BOTH explicitly shape a crouch and a
# vertical rise:
#   - joanfox/microduck-happy-hop (episodic, hardware-tested 2026-09-01):
#     "stabilize, crouch through the hips and knees, extend for takeoff, absorb
#     the landing with knee flexion, return toward HOME".
#   - upstream PR #28 (open, phase-driven periodic): a reward LADDER of
#     crouch (phase 0.10-0.30, trunk z -> 0.106) / launch (0.28-0.38, vz) /
#     airtime, reporting 256/256 envs achieving liftoff and a trunk rise of
#     0.120 -> 0.187 m (67 mm) over 0.21 s of air, in 600 iterations.
# We shaped NEITHER, and hoped the countermovement would emerge from a travel
# reward. It did not. Nothing in the S5.x stack ever paid for going UP.
#
# So S5.3 takes the structure the field converged on and keeps the parts of ours
# that are ahead of it (displacement paid at landing; landing quality; the head).
HOP_PERIOD = 1.6          # s per hop cycle: crouch, launch, fly, land, HOLD
HOP_CROUCH_PHASE = (0.10, 0.30)   # matches PR #28's window
HOP_CROUCH_Z = 0.106      # PR #28's measured crouch target, m
HOP_CROUCH_WEIGHT = 3.0
# Under PR #28's measured 67 mm so the target is demanding, not impossible.
HOP_APEX_TARGET = 0.05
HOP_APEX_WEIGHT = 6.0
# The phase clock now provides the pause, so the displacement term stops
# policing it. S5.2 proved a 0.5 s requirement imposed from step 0 just silences
# the term; a short backstop is enough to keep a bounce from banking travel.
HOP_MIN_GROUND_S_S53 = 0.15
# Flight is now purely instrumental — displacement and apex both REQUIRE it, so
# paying per airborne step on top is double-paying, and it is the measured
# bounce engine (53% of life in the air at weight 2.0).
FLIGHT_WEIGHT_S53 = 0.25


def make_microduck_hop_env_cfg(
    play: bool = False, rough: bool = False, forward: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # ── Commands: hop in place ────────────────────────────────────────────────
    # Ranges stay non-zero so the twist input neurons stay alive for the
    # forward-hop curriculum after S1 reports.
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.05, 0.05)
    twist.ranges.lin_vel_y = (-0.05, 0.05)
    twist.ranges.ang_vel_z = (-0.10, 0.10)
    twist.rel_standing_envs = 0.5
    twist.rel_turn_in_place_envs = 0.0

    # Hop intent in body_pose[2]; other components keep the velocity env's tiny
    # keep-alive ranges. Asymmetric on z — negative hop intent is meaningless.
    cfg.commands["body_pose"].ranges = (
        (-0.005, 0.005),      # x (m)
        (-0.005, 0.005),      # y (m)
        (0.0, HOP_CMD_MAX),   # z (m) — HOP INTENT
        (-0.05, 0.05),        # roll (rad)
        (-0.05, 0.05),        # pitch (rad)
        (-0.05, 0.05),        # yaw (rad)
    )
    cfg.commands["body_pose"].zero_command_prob = HOP_ZERO_CMD_PROB

    # ── The main term ─────────────────────────────────────────────────────────
    cfg.rewards["simultaneous_flight"] = RewardTermCfg(
        func=microduck_mdp.simultaneous_flight,
        weight=5.0,
        params={
            "sensor_name": "feet_ground_contact",
            "min_flight_s": FLIGHT_MIN_S,
            "max_flight_s": FLIGHT_MAX_S,
            "min_height": FLIGHT_MIN_HEIGHT,
            "max_tilt_deg": FLIGHT_MAX_TILT_DEG,
            # Ungated by command for the spike: the question is whether flight
            # is reachable at all, and scaling the only positive signal by an
            # intent the policy cannot yet act on dilutes that experience.
            # Phase 1 sets command_name="body_pose" to make hopping commandable.
            "command_name": None,
        },
    )

    # Dense bootstrap for the flight term, which is binary and pays nothing
    # until the robot is already airborne. Subordinate weight: clearance is the
    # ramp, flight is the goal. See bilateral_foot_clearance in mdp.py for the
    # convergent evidence (an independently trained Microduck hop optimised
    # exactly this metric, target 0.035 m).
    cfg.rewards["bilateral_foot_clearance"] = RewardTermCfg(
        func=microduck_mdp.bilateral_foot_clearance,
        weight=2.0,
        params={
            "target_height": CLEARANCE_TARGET,
            "rest_height": FOOT_REST_HEIGHT,
            "min_trunk_height": FLIGHT_MIN_HEIGHT,
            "max_tilt_deg": FLIGHT_MAX_TILT_DEG,
        },
    )

    # body_pose_tracking stays at 0 — see COMMAND ENCODING above. Asserted in
    # tests/test_hop_cfg.py because raising it is the single most tempting and
    # most wrong change to this file.
    cfg.rewards["body_pose_tracking"].weight = 0.0

    # ── Retuned inherited terms (rationale in the module docstring) ───────────
    cfg.rewards["air_time"].weight = 0.5
    cfg.rewards["pose"].weight = 0.3
    cfg.rewards["head_pose_tracking"].weight = 0.5
    cfg.rewards["body_ang_vel"].weight = -0.01
    cfg.rewards["angular_momentum"].weight = -0.005
    cfg.rewards["action_rate_l2"].weight = 0.0  # ramped by curriculum below

    # ── S5: forward travel + landing quality (docstring notes 8-10) ───────────
    if forward:
        # THE MAIN TERM (note 11). Starts at 0 and is handed the lead by the
        # curriculum below — displacement is unearnable until flight exists, so
        # weighting it from step 0 would just be a term that logs zero while
        # the policy learns something else.
        cfg.rewards["hop_displacement"] = RewardTermCfg(
            func=microduck_mdp.hop_displacement,
            weight=0.0,
            params={
                "sensor_name": "feet_ground_contact",
                "disp_cap": HOP_DISP_CAP,
                "min_flight_s": FLIGHT_MIN_S,
                # Same window as hop_landing_quality: one event, one window.
                "landing_window_s": LANDING_WINDOW_S,
                "max_tilt_deg": FLIGHT_MAX_TILT_DEG,
                # S5.2: a hop is only worth full marks if it followed a pause.
                "min_ground_s": HOP_MIN_GROUND_S,
            },
        )

        # S5.2: pay for the hold itself, so the pause has a gradient and not
        # just a precondition. Sits below the hop payment by design — a hop is
        # always worth more than standing after one.
        cfg.rewards["hop_settle"] = RewardTermCfg(
            func=microduck_mdp.hop_settle,
            weight=HOP_SETTLE_WEIGHT,
            params={
                "sensor_name": "feet_ground_contact",
                "settle_window_s": HOP_SETTLE_WINDOW_S,
                "max_speed": HOP_SETTLE_MAX_SPEED,
                "min_flight_s": FLIGHT_MIN_S,
            },
        )

        cfg.rewards["forward_flight_progress"] = RewardTermCfg(
            func=microduck_mdp.forward_flight_progress,
            # Demoted 1.5 -> 0.5 (note 13): it pays per airborne step, which is
            # the shape displacement replaces. It stays as the dense in-flight
            # ramp, the role clearance plays under flight.
            weight=0.5,
            params={
                "sensor_name": "feet_ground_contact",
                "vel_cap": FORWARD_VEL_CAP,
                "min_flight_s": FLIGHT_MIN_S,
                "max_flight_s": FLIGHT_MAX_S,
                "min_height": FLIGHT_MIN_HEIGHT,
                "max_tilt_deg": FLIGHT_MAX_TILT_DEG,
            },
        )

        # 1.0 -> 2.0: it was the weakest term in the S5 stack (0.138) precisely
        # because nothing else paid for the landing either. Now that
        # displacement pays at the same instant, posture at touchdown needs
        # enough mass to shape HOW the duck arrives, not just THAT it does.
        cfg.rewards["hop_landing_quality"] = RewardTermCfg(
            func=microduck_mdp.hop_landing_quality,
            weight=2.0,
            params={
                "sensor_name": "feet_ground_contact",
                "target_height": LANDING_TARGET_Z,
                # Note 12 — the head is priced HERE, at touchdown, and nowhere
                # else in flight.
                "head_upright_std": LANDING_HEAD_UPRIGHT_STD,
                "min_flight_s": FLIGHT_MIN_S,
                "landing_window_s": LANDING_WINDOW_S,
            },
        )

        # POSITIVE weight on a SELF-NEGATING function — the microduck
        # convention, enforced for any *_penalty reward key by
        # test_every_penalty_term_has_a_sign_that_can_only_log_negative.
        cfg.rewards["hop_landing_impact_penalty"] = RewardTermCfg(
            func=microduck_mdp.hop_landing_impact_penalty,
            weight=0.5,
            params={
                "sensor_name": "feet_ground_contact",
                "free_speed": IMPACT_FREE_SPEED,
                "min_flight_s": FLIGHT_MIN_S,
                "impact_window_s": IMPACT_WINDOW_S,
            },
        )

        # The line that makes E1 reachable — docstring note 10.
        cfg.rewards["track_linear_velocity"].weight = HOP_TRACK_LIN_VEL_WEIGHT

        # Clearance stays the dense ramp but steps back with the term it ramps
        # toward, so the documented ordering survives the handover:
        # clearance < flight < displacement at every stage.
        cfg.rewards["bilateral_foot_clearance"].weight = CLEARANCE_WEIGHT_FORWARD

        # S5.2: partial restore of the instantaneous head term (weight, NOT
        # std — the std is what stopped the walker moving).
        cfg.rewards["head_pose_tracking"].weight = HOP_HEAD_TRACK_WEIGHT

        # ── S5.3: the phase cycle, the crouch, and the rise ──────────────────
        # The twist slot becomes a cyclic phase carrier, exactly as ground_pick
        # and sitstand already do — same 3 command slots, different semantics,
        # so the 61D contract is untouched.
        # Copy only the fields the phase cfg actually declares: the velocity env's
        # twist cfg is a microduck SUBCLASS carrying extras (rel_turn_in_place_envs)
        # that UniformVelocityCommandCfg — and so GroundPickPhaseCommandCfg —
        # does not have, and an unfiltered vars() spread dies on them.
        twist_cfg = cfg.commands["twist"]
        _allowed = {f.name for f in fields(microduck_mdp.GroundPickPhaseCommandCfg)}
        cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
            **{
                **{k: v for k, v in vars(twist_cfg).items() if k in _allowed},
                "class_type": microduck_mdp.GroundPickPhaseCommand,
                "period": HOP_PERIOD,
            }
        )

        # MANDATORY with a phase carrier: these terms read the twist slot as a
        # VELOCITY command. Left in, the policy is paid for matching cos(2*pi*t)
        # as a target speed, which is nonsense. ground_pick deletes exactly
        # these for exactly this reason.
        for name in ("track_linear_velocity", "track_angular_velocity", "air_time"):
            cfg.rewards.pop(name, None)

        cfg.rewards["hop_crouch"] = RewardTermCfg(
            func=microduck_mdp.hop_crouch_by_phase,
            weight=HOP_CROUCH_WEIGHT,
            params={
                "command_name": "twist",
                "crouch_z": HOP_CROUCH_Z,
                "stand_z": LANDING_TARGET_Z,
                "phase_lo": HOP_CROUCH_PHASE[0],
                "phase_hi": HOP_CROUCH_PHASE[1],
            },
        )

        # THE term that should make it read as a hop rather than a buzz.
        cfg.rewards["hop_apex_rise"] = RewardTermCfg(
            func=microduck_mdp.hop_apex_rise,
            weight=HOP_APEX_WEIGHT,
            params={
                "sensor_name": "feet_ground_contact",
                "rise_target": HOP_APEX_TARGET,
                "min_flight_s": FLIGHT_MIN_S,
                "landing_window_s": LANDING_WINDOW_S,
                "max_tilt_deg": FLIGHT_MAX_TILT_DEG,
            },
        )

        # The phase clock now owns the rhythm; the pause factor steps back to a
        # backstop so it cannot silence the travel term the way S5.2's did.
        cfg.rewards["hop_displacement"].params["min_ground_s"] = HOP_MIN_GROUND_S_S53

    # ── Curricula ─────────────────────────────────────────────────────────────
    # The velocity env ramps head_pose_bias from iter 600. Drop it: it is a
    # posture-precision tax on the head, and the head is a load-bearing part of
    # the hop (docstring note 3).
    cfg.curriculum.pop("head_pose_bias_weight", None)

    # Smoothness AFTER discovery. reward_weight is a step function, not an
    # interpolation, so the ramp is discretized into stages.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": ACTION_RATE_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": -0.02},
                {"step": 800 * NUM_STEPS_PER_ENV, "weight": -0.05},
            ],
        },
    )

    if forward:
        # THE HANDOVER (note 11). Two halves of one transition, deliberately
        # sharing a boundary: flight leads while the skill is discovered, then
        # displacement leads. Splitting the boundary would leave a window where
        # neither term is the objective.
        cfg.curriculum["hop_displacement_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "hop_displacement",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": DISP_HANDOVER_ITER * NUM_STEPS_PER_ENV,
                     "weight": HOP_DISP_WEIGHT * 0.5},
                    {"step": 2 * DISP_HANDOVER_ITER * NUM_STEPS_PER_ENV,
                     "weight": HOP_DISP_WEIGHT},
                ],
            },
        )
        cfg.curriculum["flight_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "simultaneous_flight",
                "weight_stages": [
                    {"step": 0, "weight": 5.0},
                    {"step": DISP_HANDOVER_ITER * NUM_STEPS_PER_ENV, "weight": 1.0},
                    # S5.3: flight is instrumental — displacement and apex both
                    # REQUIRE it, so paying per airborne step is double-paying,
                    # and it is the measured bounce engine.
                    {"step": 2 * DISP_HANDOVER_ITER * NUM_STEPS_PER_ENV,
                     "weight": FLIGHT_WEIGHT_S53},
                ],
            },
        )

        # S5.2 — HEAD UP ALL THE TIME. head_pose_bias returns at the WALKER's
        # full strength and early (see HEAD_BIAS_STAGES): it is the DC term, so
        # it charges a permanently drooping head while leaving the in-flight
        # countermovement swing free to cancel out. Still held at 0 through the
        # first stage so it is not a posture tax during skill discovery.
        cfg.rewards["head_pose_bias"].weight = 0.0
        cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "head_pose_bias",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    *[
                        {"step": it * NUM_STEPS_PER_ENV, "weight": w}
                        for it, w in HEAD_BIAS_STAGES
                    ],
                ],
            },
        )

    return cfg


MicroduckHopRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Symmetry OFF, as everywhere else in this repo by default. A hop is
        # left/right symmetric, so the mirror loss is defensible here later —
        # but not while the question is whether the behavior exists at all.
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="hop",
    run_name="hop",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=50_000,
)

# S5. Identical hyperparameters to the baseline on purpose: the forward run is
# an A/B against hop-in-place, and changing the optimizer at the same time as
# the reward stack would confound it. Only the experiment/run names differ —
# sharing them would overwrite the baseline's logs/<experiment_name>/ directory
# and collide in wandb, destroying the comparison this variant exists for.
MicroduckHopForwardRlCfg = replace(
    MicroduckHopRlCfg,
    experiment_name="hop_forward",
    run_name="hop_forward",
)
