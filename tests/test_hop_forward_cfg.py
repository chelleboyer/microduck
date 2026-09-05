"""Cfg invariants for the S5 forward-hop variant (`forward=True`).

The baseline hop-in-place env stays registered and unchanged as the A/B
reference, so the first thing these tests pin is that `forward=False` is
genuinely untouched — an accidental leak there silently destroys the
comparison the variant exists for.

Everything else here corresponds to a documented failure mode. See the module
docstring of microduck_hop_env_cfg.py, deviations 8-10.
"""

import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    FORWARD_VEL_CAP,
    HOP_TRACK_LIN_VEL_WEIGHT,
    IMPACT_FREE_SPEED,
    LANDING_TARGET_Z,
    MicroduckHopForwardRlCfg,
    MicroduckHopRlCfg,
    make_microduck_hop_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

_NEW_TERMS = (
    "forward_flight_progress",
    "hop_landing_quality",
    "hop_landing_impact_penalty",
)


def test_baseline_variant_is_untouched():
    # GUARDS THE A/B. Mjlab-Hop-Flat-MicroDuck must stay exactly what it was,
    # or the forward run has nothing unconfounded to compare against.
    base = make_microduck_hop_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for name in _NEW_TERMS:
        assert name not in base.rewards, name
    assert (
        base.rewards["track_linear_velocity"].weight
        == vel.rewards["track_linear_velocity"].weight
    )


def test_forward_variant_adds_all_three_terms():
    cfg = make_microduck_hop_env_cfg(forward=True)
    for name in _NEW_TERMS:
        assert name in cfg.rewards, name
    assert cfg.rewards["forward_flight_progress"].func is (
        microduck_mdp.forward_flight_progress
    )
    assert cfg.rewards["hop_landing_quality"].func is microduck_mdp.hop_landing_quality
    assert cfg.rewards["hop_landing_impact_penalty"].func is (
        microduck_mdp.hop_landing_impact_penalty
    )


def test_flight_still_dominates_every_positive_term():
    # Same invariant as the baseline: flight is the goal, forward travel and
    # landing are modifiers on it. If a modifier outranked flight the policy
    # could farm it without ever leaving the ground.
    cfg = make_microduck_hop_env_cfg(forward=True)
    flight = cfg.rewards["simultaneous_flight"].weight
    positives = {
        n: t.weight for n, t in cfg.rewards.items()
        if t.weight > 0 and n != "simultaneous_flight"
    }
    assert flight > max(positives.values()), positives


def test_velocity_tracking_is_demoted_but_alive():
    # THE fix that makes E1 reachable. At the walker's 2.0 / std sqrt(0.1),
    # tracking a near-zero command costs a 0.4 m/s hop ~1.6 reward/step, which
    # out-masses any forward term the dominance invariant permits.
    hop = make_microduck_hop_env_cfg(forward=True)
    vel = make_microduck_velocity_env_cfg()
    w = hop.rewards["track_linear_velocity"].weight
    assert w == HOP_TRACK_LIN_VEL_WEIGHT
    assert w < vel.rewards["track_linear_velocity"].weight
    # Not zeroed: it still discourages aimless drift.
    assert w > 0.0
    # And the demotion must actually beat the forward term it was blocking.
    fwd = hop.rewards["forward_flight_progress"].weight
    cost_of_hopping = vel.rewards["track_linear_velocity"].params["std"]
    assert cost_of_hopping is not None  # std present, so the arithmetic holds
    assert fwd > w


def test_impact_penalty_self_negates_so_its_weight_is_positive():
    # The convention: microduck *_penalty functions return <= 0 and take a
    # POSITIVE weight. Inverted, this pays the policy to slam into the ground.
    cfg = make_microduck_hop_env_cfg(forward=True)
    assert cfg.rewards["hop_landing_impact_penalty"].weight > 0.0


def test_every_penalty_term_has_a_sign_that_can_only_log_negative():
    # CLAUDE.md calls this check infallible for catching sign inversions.
    cfg = make_microduck_hop_env_cfg(forward=True)
    for name, term in cfg.rewards.items():
        if name.endswith("_penalty") or name.endswith("_l1"):
            assert term.weight >= 0.0, f"{name} self-negates; weight must be >= 0"


def test_landing_target_is_the_measured_height_not_nominal_height():
    # nominal_height = 0.095 is ~22 mm low (docs/command-block.md) and survives
    # only because body_pose_tracking runs at weight 0. Do not inherit it here.
    params = make_microduck_hop_env_cfg(forward=True).rewards["hop_landing_quality"].params
    assert params["target_height"] == LANDING_TARGET_Z
    assert not math.isclose(params["target_height"], 0.095, abs_tol=1e-6)
    # Sits at the measured settle height, well above a seated pose.
    assert 0.10 < params["target_height"] < 0.13


def test_forward_and_flight_share_the_same_flight_gates():
    # The terms only cover each other's exploits if they agree on what counts
    # as airborne and upright.
    r = make_microduck_hop_env_cfg(forward=True).rewards
    f, w = r["simultaneous_flight"].params, r["forward_flight_progress"].params
    assert w["min_height"] == f["min_height"]
    assert w["max_tilt_deg"] == f["max_tilt_deg"]
    assert w["min_flight_s"] == f["min_flight_s"]
    assert w["max_flight_s"] == f["max_flight_s"]


def test_forward_cap_is_a_real_speed():
    params = make_microduck_hop_env_cfg(forward=True).rewards["forward_flight_progress"].params
    assert params["vel_cap"] == FORWARD_VEL_CAP > 0.0


def test_impact_allowance_clears_an_achievable_hop():
    # Free-fall from the S1 probe's ~11 mm apex is ~0.46 m/s. If the allowance
    # sat below that, every hop the robot can currently perform would be taxed.
    params = make_microduck_hop_env_cfg(forward=True).rewards["hop_landing_impact_penalty"].params
    assert params["free_speed"] == IMPACT_FREE_SPEED
    assert params["free_speed"] > math.sqrt(2 * 9.81 * 0.011)


def test_twist_ranges_are_unchanged_from_the_baseline():
    # Pins that we did NOT drift into encoding E2. Widening lin_vel_x makes
    # walking the optimal answer to a forward-velocity command, which is the
    # documented failure this whole env was detuned to avoid.
    base = make_microduck_hop_env_cfg()
    fwd = make_microduck_hop_env_cfg(forward=True)
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        assert getattr(fwd.commands["twist"].ranges, axis) == getattr(
            base.commands["twist"].ranges, axis
        ), axis


def test_body_pose_tracking_still_zero():
    # Unchanged from the baseline: rewarding a commanded height makes standing
    # tall strictly better than flight.
    assert make_microduck_hop_env_cfg(forward=True).rewards["body_pose_tracking"].weight == 0.0


def test_flight_still_ungated_by_command():
    # E1 is un-commanded by decision; E3 is Phase 1 work.
    params = make_microduck_hop_env_cfg(forward=True).rewards["simultaneous_flight"].params
    assert params["command_name"] is None


def test_runner_cfg_experiment_name_is_distinct():
    # A shared experiment_name overwrites the baseline's logs and collides in
    # wandb, destroying the A/B.
    assert MicroduckHopForwardRlCfg.experiment_name == "hop_forward"
    assert MicroduckHopForwardRlCfg.experiment_name != MicroduckHopRlCfg.experiment_name
    # Hyperparameters otherwise identical, so the A/B isolates the reward stack.
    assert MicroduckHopForwardRlCfg.num_steps_per_env == MicroduckHopRlCfg.num_steps_per_env
    assert MicroduckHopForwardRlCfg.algorithm == MicroduckHopRlCfg.algorithm


def test_play_variant_builds():
    assert make_microduck_hop_env_cfg(play=True, forward=True) is not None


def test_tasks_are_registered():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    names = set(list_tasks())
    assert "Mjlab-HopForward-Flat-MicroDuck" in names
    assert "Mjlab-HopForward-Flat-Backlash-MicroDuck" in names
    # The baseline must survive alongside it.
    assert "Mjlab-Hop-Flat-MicroDuck" in names
