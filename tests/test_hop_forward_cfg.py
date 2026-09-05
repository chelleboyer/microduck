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
    CLEARANCE_WEIGHT_FORWARD,
    DISP_HANDOVER_ITER,
    FLIGHT_WEIGHT_AFTER_HANDOVER,
    FORWARD_VEL_CAP,
    HOP_DISP_CAP,
    HOP_DISP_WEIGHT,
    HOP_TRACK_LIN_VEL_WEIGHT,
    IMPACT_FREE_SPEED,
    LANDING_HEAD_UPRIGHT_STD,
    LANDING_TARGET_Z,
    LANDING_WINDOW_S,
    MicroduckHopForwardRlCfg,
    MicroduckHopRlCfg,
    make_microduck_hop_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

_NEW_TERMS = (
    "hop_displacement",
    "forward_flight_progress",
    "hop_landing_quality",
    "hop_landing_impact_penalty",
)


def _final_weight(cfg, reward_name: str) -> float:
    """Weight after every curriculum stage has elapsed.

    Terms handed their weight by a curriculum read 0 (or their pre-handover
    value) in the cfg, so comparing raw cfg weights would pin the wrong
    invariant entirely — it would "prove" the main term is worthless.
    """
    for term in cfg.curriculum.values():
        params = getattr(term, "params", None) or {}
        if params.get("reward_name") == reward_name:
            return params["weight_stages"][-1]["weight"]
    return cfg.rewards[reward_name].weight


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
    # S5.1 must not leak backwards: the baseline keeps flight as its objective
    # at full weight, and its head stays free.
    assert base.rewards["simultaneous_flight"].weight == 5.0
    assert "flight_weight" not in base.curriculum
    assert "head_pose_bias_weight" not in base.curriculum


def test_forward_variant_adds_all_four_terms():
    cfg = make_microduck_hop_env_cfg(forward=True)
    for name in _NEW_TERMS:
        assert name in cfg.rewards, name
    assert cfg.rewards["hop_displacement"].func is microduck_mdp.hop_displacement
    assert cfg.rewards["forward_flight_progress"].func is (
        microduck_mdp.forward_flight_progress
    )
    assert cfg.rewards["hop_landing_quality"].func is microduck_mdp.hop_landing_quality
    assert cfg.rewards["hop_landing_impact_penalty"].func is (
        microduck_mdp.hop_landing_impact_penalty
    )


def test_displacement_is_the_dominant_term_after_the_handover():
    # THE S5.1 INVERSION. Air time was the objective and the policy spent ~52%
    # of its life airborne; "took off HERE, landed THERE" is the objective now.
    # If flight (or any per-step term) outranked displacement again, the same
    # bounce is the optimal policy.
    cfg = make_microduck_hop_env_cfg(forward=True)
    disp = _final_weight(cfg, "hop_displacement")
    assert disp == HOP_DISP_WEIGHT > 0.0
    others = {
        n: _final_weight(cfg, n) for n, t in cfg.rewards.items()
        if n != "hop_displacement" and _final_weight(cfg, n) > 0
    }
    assert disp > max(others.values()), others


def test_flight_demotes_but_stays_above_the_clearance_ramp():
    # Flight is a MEANS now, not the goal — but it must not fall below the ramp
    # that exists to bootstrap it, or a deep two-foot tuck outranks leaving the
    # ground at all.
    cfg = make_microduck_hop_env_cfg(forward=True)
    base = make_microduck_hop_env_cfg()
    flight = _final_weight(cfg, "simultaneous_flight")
    clearance = _final_weight(cfg, "bilateral_foot_clearance")
    assert flight == FLIGHT_WEIGHT_AFTER_HANDOVER
    assert flight < base.rewards["simultaneous_flight"].weight
    assert 0.0 < clearance == CLEARANCE_WEIGHT_FORWARD < flight


def test_the_handover_is_phase_aligned_not_a_swap():
    # Displacement is unearnable until flight exists, so it must not lead
    # before then; flight must not be demoted before then either. CLAUDE.md:
    # never introduce a term before the skill it prices exists.
    cfg = make_microduck_hop_env_cfg(forward=True)
    disp = cfg.curriculum["hop_displacement_weight"].params["weight_stages"]
    flight = cfg.curriculum["flight_weight"].params["weight_stages"]
    assert disp[0]["step"] == 0 and disp[0]["weight"] == 0.0
    assert cfg.rewards["hop_displacement"].weight == 0.0  # matches stage 0
    assert flight[0]["step"] == 0 and flight[0]["weight"] == 5.0
    assert cfg.rewards["simultaneous_flight"].weight == 5.0
    # Same boundary, or there is a window where neither term is the objective.
    assert disp[1]["step"] == flight[1]["step"] == DISP_HANDOVER_ITER * 24
    # Monotone in opposite directions: one hands over to the other.
    assert [s["weight"] for s in disp] == sorted(s["weight"] for s in disp)
    assert [s["weight"] for s in flight] == sorted(
        (s["weight"] for s in flight), reverse=True
    )


def test_displacement_pays_at_the_landing_not_per_airborne_step():
    # The whole point of S5.1. A displacement term keyed to anything other than
    # a genuine landing would reintroduce the per-step air payment under a new
    # name.
    p = make_microduck_hop_env_cfg(forward=True).rewards["hop_displacement"].params
    assert p["min_flight_s"] > 0.0            # a real flight must have happened
    assert p["landing_window_s"] == LANDING_WINDOW_S
    # One event, one window: landing quality prices the same instant.
    q = make_microduck_hop_env_cfg(forward=True).rewards["hop_landing_quality"].params
    assert q["landing_window_s"] == p["landing_window_s"]
    assert q["min_flight_s"] == p["min_flight_s"]


def test_displacement_cap_has_headroom_over_the_s5_pass_mark():
    # Capped, or a dive outscores every controlled hop. But the cap must sit
    # well above what the policy already does, or it saturates and stops
    # distinguishing good from great — exactly what happened to FORWARD_VEL_CAP.
    p = make_microduck_hop_env_cfg(forward=True).rewards["hop_displacement"].params
    assert p["disp_cap"] == HOP_DISP_CAP
    assert p["disp_cap"] > 0.025 * 2   # S5's 25 mm pass mark, with headroom
    assert p["disp_cap"] > 0.008       # ...and far above the 8 mm open-loop floor


def test_forward_cap_was_raised_off_saturation():
    # The S5 run logged ~95% of the old 0.4 m/s cap. A saturated metric cannot
    # tell a good hop from a great one.
    cfg = make_microduck_hop_env_cfg(forward=True)
    assert FORWARD_VEL_CAP > 0.4
    assert cfg.rewards["forward_flight_progress"].params["vel_cap"] == FORWARD_VEL_CAP
    # And it demotes to a ramp: it pays per airborne step, which is the shape
    # displacement replaces.
    assert cfg.rewards["forward_flight_progress"].weight < 1.5


def test_head_is_priced_at_touchdown_only():
    # Note 12. The head stays FREE in flight (its 280 g is the countermovement)
    # and is priced only in the landing window. Raising head_pose_tracking is
    # the documented wrong fix — it made the policy stop moving entirely.
    fwd = make_microduck_hop_env_cfg(forward=True)
    base = make_microduck_hop_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    params = fwd.rewards["hop_landing_quality"].params
    assert params["head_upright_std"] == LANDING_HEAD_UPRIGHT_STD
    # Wide on purpose: a factor tighter than the current droop collapses the
    # product and kills the gradient.
    assert params["head_upright_std"] >= 0.3
    assert (
        fwd.rewards["head_pose_tracking"].weight
        == base.rewards["head_pose_tracking"].weight
        < vel.rewards["head_pose_tracking"].weight
    )


def test_head_pose_bias_returns_for_forward_only_and_gently():
    fwd = make_microduck_hop_env_cfg(forward=True)
    vel = make_microduck_velocity_env_cfg()
    assert "head_pose_bias_weight" not in make_microduck_hop_env_cfg().curriculum
    stages = fwd.curriculum["head_pose_bias_weight"].params["weight_stages"]
    vel_stages = vel.curriculum["head_pose_bias_weight"].params["weight_stages"]
    # Off during discovery, and gentler than the walker's ramp at the end: the
    # head is load-bearing here, and the touchdown factor does the primary work.
    assert stages[0]["weight"] == 0.0
    assert stages[-1]["weight"] < vel_stages[-1]["weight"]
    # Arrives after the handover, never during skill discovery.
    assert stages[1]["step"] > DISP_HANDOVER_ITER * 24


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
