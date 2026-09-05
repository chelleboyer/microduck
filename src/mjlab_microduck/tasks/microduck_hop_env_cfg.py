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
from dataclasses import replace

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
# Forward speed at which the E1 term saturates, m/s. Matched to the velocity
# env's walking command ceiling; re-derive against the flight probe's measured
# open-loop forward reach. Too low caps a good hop, too high leaves the term
# flat across the achievable range and kills its gradient.
FORWARD_VEL_CAP = 0.4
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
        cfg.rewards["forward_flight_progress"] = RewardTermCfg(
            func=microduck_mdp.forward_flight_progress,
            weight=1.5,
            params={
                "sensor_name": "feet_ground_contact",
                "vel_cap": FORWARD_VEL_CAP,
                "min_flight_s": FLIGHT_MIN_S,
                "max_flight_s": FLIGHT_MAX_S,
                "min_height": FLIGHT_MIN_HEIGHT,
                "max_tilt_deg": FLIGHT_MAX_TILT_DEG,
            },
        )

        cfg.rewards["hop_landing_quality"] = RewardTermCfg(
            func=microduck_mdp.hop_landing_quality,
            weight=1.0,
            params={
                "sensor_name": "feet_ground_contact",
                "target_height": LANDING_TARGET_Z,
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
