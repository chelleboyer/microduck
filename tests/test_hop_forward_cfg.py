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
    FLIGHT_WEIGHT_S53,
    FORWARD_VEL_CAP,
    HOP_DISP_CAP,
    HOP_DISP_WEIGHT,
    HOP_HEAD_TRACK_WEIGHT,
    HOP_APEX_TARGET,
    HOP_MIN_GROUND_S,
    HOP_MIN_GROUND_S_S53,
    HOP_PERIOD,
    HOP_SETTLE_WEIGHT,
    HOP_SETTLE_WINDOW_S,
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


def test_flight_becomes_instrumental_not_an_objective():
    # S5.3. Paying per airborne step is DOUBLE-paying once displacement and apex
    # both require flight to collect — and it is the measured bounce engine: at
    # weight 2.0 the S5.2 policy lived 53% of its life in the air. It is not
    # zeroed, because a residual signal for leaving the ground is still the
    # cheapest bootstrap, but it must no longer outrank anything real.
    cfg = make_microduck_hop_env_cfg(forward=True)
    base = make_microduck_hop_env_cfg()
    flight = _final_weight(cfg, "simultaneous_flight")
    assert flight == FLIGHT_WEIGHT_S53
    assert 0.0 < flight < base.rewards["simultaneous_flight"].weight
    for goal in ("hop_displacement", "hop_apex_rise"):
        assert flight < _final_weight(cfg, goal), goal
    # The clearance ramp may now sit ABOVE it — that is intended, not drift:
    # clearance shapes the lift, flight merely observes it.
    assert _final_weight(cfg, "bilateral_foot_clearance") == CLEARANCE_WEIGHT_FORWARD


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


def test_the_hop_rhythm_is_encoded_not_hoped_for():
    # S5.2 (user call): "head up, hop, stay in place for a sec, then hop again".
    # A cadence has to be IN the reward — CLAUDE.md: encode what counts as the
    # maneuver in state-based structure, not in small nudges.
    cfg = make_microduck_hop_env_cfg(forward=True)
    disp = cfg.rewards["hop_displacement"].params
    # S5.3: the RHYTHM now comes from a phase clock, which is structure rather
    # than something the policy has to stumble into. S5.2 proved the latter
    # fails — demanding a 0.5 s pause from step 0 just silenced the term
    # (hop_settle earned ~0 and displacement collapsed 7x).
    twist = cfg.commands["twist"]
    assert isinstance(twist, microduck_mdp.GroundPickPhaseCommandCfg)
    assert twist.period == HOP_PERIOD > 0.0
    # The pause factor stays only as a backstop, well below the old demand, so
    # it cannot silence the travel term again.
    assert 0.0 < disp["min_ground_s"] == HOP_MIN_GROUND_S_S53 < HOP_MIN_GROUND_S
    # The hold is still paid for directly.
    assert cfg.rewards["hop_settle"].func is microduck_mdp.hop_settle
    assert cfg.rewards["hop_settle"].weight == HOP_SETTLE_WEIGHT > 0.0


def test_the_hop_has_a_crouch_and_a_rise_which_is_what_makes_it_look_like_one():
    # S5.3, and the whole point of it. Nothing in S5/S5.1/S5.2 ever paid for the
    # trunk going UP or for loading the legs first, so a 2 mm vibration scored
    # like a jump. BOTH independent working hops on this robot shape both.
    cfg = make_microduck_hop_env_cfg(forward=True)
    crouch = cfg.rewards["hop_crouch"]
    apex = cfg.rewards["hop_apex_rise"]
    assert crouch.func is microduck_mdp.hop_crouch_by_phase
    assert apex.func is microduck_mdp.hop_apex_rise
    # The crouch target must be BELOW the settled standing height, or this pays
    # for standing.
    assert crouch.params["crouch_z"] < crouch.params["stand_z"]
    # And it fires in the loading window, before the launch.
    assert 0.0 <= crouch.params["phase_lo"] < crouch.params["phase_hi"] < 0.5
    # The rise target sits under the 67 mm an independent phase-driven hop
    # actually measured on this robot: demanding, not impossible.
    assert 0.0 < apex.params["rise_target"] == HOP_APEX_TARGET < 0.067


def test_holding_still_never_outearns_hopping():
    # If the hold paid more than the hop, the optimal policy is to hop once and
    # then stand in the window forever. It must sit below the hop payment.
    cfg = make_microduck_hop_env_cfg(forward=True)
    assert cfg.rewards["hop_settle"].weight < _final_weight(cfg, "hop_displacement")


def test_forward_cap_was_raised_off_saturation():
    # The S5 run logged ~95% of the old 0.4 m/s cap. A saturated metric cannot
    # tell a good hop from a great one.
    cfg = make_microduck_hop_env_cfg(forward=True)
    assert FORWARD_VEL_CAP > 0.4
    assert cfg.rewards["forward_flight_progress"].params["vel_cap"] == FORWARD_VEL_CAP
    # And it demotes to a ramp: it pays per airborne step, which is the shape
    # displacement replaces.
    assert cfg.rewards["forward_flight_progress"].weight < 1.5


def test_head_is_priced_at_touchdown_and_on_average():
    # S5.2 (user call, watching the S5.1 video): "his head is just hanging down,
    # we need to get that head up all the time". Touchdown-only pricing was too
    # little — the duck spends most of its life between landings, and that is
    # where it looked wrong. So BOTH instruments are live: the touchdown factor
    # and the DC-bias term.
    fwd = make_microduck_hop_env_cfg(forward=True)
    params = fwd.rewards["hop_landing_quality"].params
    assert params["head_upright_std"] == LANDING_HEAD_UPRIGHT_STD
    # Wide on purpose: a factor tighter than the current droop collapses the
    # product and kills the gradient.
    assert params["head_upright_std"] >= 0.3
    assert fwd.rewards["head_pose_bias"] is not None
    assert "head_pose_bias_weight" in fwd.curriculum


def test_the_head_fix_is_weight_not_std():
    # THE receipt, and the one thing that must never drift:
    # microduck_velocity_env_cfg.py:729-737 — tightening head_pose_tracking's
    # STD (fine_std=0.1) made the policy stop moving entirely by iter 300,
    # because a 280 g head MUST oscillate while the robot is moving and an
    # instantaneous tolerance is therefore unescapable. The WEIGHT is a
    # different dial, and the DC bias is the escapable half.
    fwd = make_microduck_hop_env_cfg(forward=True)
    vel = make_microduck_velocity_env_cfg()
    assert fwd.rewards["head_pose_tracking"].params["std"] == (
        vel.rewards["head_pose_tracking"].params["std"]
    )
    # Weight raised off the baseline's 0.5, but still under the walker's, so the
    # head keeps room to swing as a countermovement mid-flight.
    base = make_microduck_hop_env_cfg()
    assert (
        base.rewards["head_pose_tracking"].weight
        < fwd.rewards["head_pose_tracking"].weight
        == HOP_HEAD_TRACK_WEIGHT
        <= vel.rewards["head_pose_tracking"].weight
    )


def test_head_pose_bias_returns_at_walker_strength_for_forward_only():
    fwd = make_microduck_hop_env_cfg(forward=True)
    vel = make_microduck_velocity_env_cfg()
    # The baseline hop-in-place env keeps its head free — S5.2 must not leak.
    assert "head_pose_bias_weight" not in make_microduck_hop_env_cfg().curriculum
    stages = fwd.curriculum["head_pose_bias_weight"].params["weight_stages"]
    vel_stages = vel.curriculum["head_pose_bias_weight"].params["weight_stages"]
    # Off during discovery — a posture tax while a hard skill is being explored
    # makes "do nothing" win — then the WALKER's full strength, because the
    # walker's head is what "looks right" looks like.
    assert stages[0]["weight"] == 0.0
    assert stages[-1]["weight"] >= vel_stages[-1]["weight"]
    # ...and it arrives EARLIER than the walker's, because the droop is the
    # complaint rather than a refinement.
    assert stages[-1]["step"] < vel_stages[-1]["step"]
    assert [s["weight"] for s in stages] == sorted(s["weight"] for s in stages)


def test_velocity_terms_are_deleted_once_twist_carries_the_phase():
    # MANDATORY, and the way this env can break most quietly. Under S5.3 the
    # twist slot carries [cos(2*pi*phase), sin(2*pi*phase), 0]. Any term that
    # reads that slot as a VELOCITY command is then paying the policy to track a
    # cosine as a target speed. ground_pick deletes exactly these three for
    # exactly this reason.
    hop = make_microduck_hop_env_cfg(forward=True)
    for name in ("track_linear_velocity", "track_angular_velocity", "air_time"):
        assert name not in hop.rewards, name
    # The baseline hop-in-place env still has a real velocity command, so it
    # must KEEP them — this is the A/B reference.
    base = make_microduck_hop_env_cfg()
    for name in ("track_linear_velocity", "air_time"):
        assert name in base.rewards, name
    assert base.rewards["track_linear_velocity"].weight > 0.0


def test_the_baseline_keeps_the_demoted_tracking_weight():
    # S5's "least obvious line": the walker's 2.0 tracking term makes hopping
    # forward COST ~1.6 reward/step against a forward term capped at 1.5, so E1
    # loses before it starts unless tracking steps back. The forward variant now
    # deletes the term outright, but the constant still documents the finding
    # and the baseline still demonstrates the demotion.
    vel = make_microduck_velocity_env_cfg()
    assert 0.0 < HOP_TRACK_LIN_VEL_WEIGHT < vel.rewards["track_linear_velocity"].weight


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
