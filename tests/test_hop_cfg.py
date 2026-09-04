"""Cfg invariants for the hop spike env (MD-3).

These lock in the deviations from the velocity recipe that make hopping
learnable at all. Every assertion below corresponds to a documented failure
mode — if one of these drifts back to its velocity-env value, the run silently
becomes a worse walker instead of a hop spike.
"""

import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    HOP_CMD_MAX,
    MicroduckHopRlCfg,
    make_microduck_hop_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_flight_is_the_dominant_reward():
    cfg = make_microduck_hop_env_cfg()
    assert "simultaneous_flight" in cfg.rewards
    flight = cfg.rewards["simultaneous_flight"]
    assert flight.func is microduck_mdp.simultaneous_flight
    # Must out-weigh every other positive term, or walking stays optimal.
    positives = {
        n: t.weight for n, t in cfg.rewards.items()
        if t.weight > 0 and n != "simultaneous_flight"
    }
    assert flight.weight > max(positives.values()), positives


def test_body_pose_tracking_stays_at_zero():
    # THE tempting wrong change: command z = +0.05 and reward tracking it, and
    # the optimal policy extends its legs and stands tall — strictly easier
    # than flight, and it scores better. Flight must be paid by the flight term.
    cfg = make_microduck_hop_env_cfg()
    assert cfg.rewards["body_pose_tracking"].weight == 0.0


def test_alternating_air_time_is_demoted_below_the_walker():
    # Stock air_time rewards single-foot swing — ordinary walking. At the
    # velocity env's 3.0 a good stride out-earns any reachable hop.
    hop = make_microduck_hop_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    assert hop.rewards["air_time"].weight < vel.rewards["air_time"].weight
    # Not zeroed: lifting a foot is a prerequisite skill worth bootstrapping.
    assert hop.rewards["air_time"].weight > 0.0


def test_head_is_freed_not_regularized():
    # The head is 280 g of a 737 g robot and swinging it is the most plausible
    # lever for beating the 34 ms open-loop bound. MD-3: no motion-blocker
    # penalties on the neck.
    hop = make_microduck_hop_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    assert hop.rewards["head_pose_tracking"].weight < vel.rewards["head_pose_tracking"].weight
    assert hop.rewards["head_pose_bias"].weight == 0.0
    # ...and nothing may ramp it back up.
    assert "head_pose_bias_weight" not in hop.curriculum


def test_motion_blockers_are_low_for_a_dynamic_task():
    hop = make_microduck_hop_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for name in ("body_ang_vel", "angular_momentum"):
        # Still penalties (negative), but far weaker than the walker's.
        assert hop.rewards[name].weight < 0.0, name
        assert abs(hop.rewards[name].weight) < abs(vel.rewards[name].weight), name


def test_smoothness_starts_at_zero_and_arrives_later():
    # An attempt-tax active during exploration makes "do nothing" win.
    cfg = make_microduck_hop_env_cfg()
    assert cfg.rewards["action_rate_l2"].weight == 0.0
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0 and stages[0]["weight"] == 0.0
    assert stages[-1]["step"] > 0 and stages[-1]["weight"] < 0.0
    # Monotonically more negative, and never positive (a positive weight here
    # would pay the policy to thrash).
    weights = [s["weight"] for s in stages]
    assert weights == sorted(weights, reverse=True)
    assert all(w <= 0.0 for w in weights)


def test_hop_command_lives_in_body_pose_z_and_is_alive_from_step_zero():
    cfg = make_microduck_hop_env_cfg()
    ranges = cfg.commands["body_pose"].ranges
    assert len(ranges) == 6
    lo, hi = ranges[2]  # z — the hop slot
    assert hi == HOP_CMD_MAX > 0.0
    assert lo == 0.0  # negative hop intent is meaningless
    # Every other slot keeps a non-zero keep-alive range: a command input that
    # is never non-zero has dead weights forever.
    for i, (a, b) in enumerate(ranges):
        assert b > a, i


def test_zero_command_is_explicitly_sampled():
    # Uniform sampling essentially never produces the all-zero command, which
    # is exactly the deployment idle state.
    cfg = make_microduck_hop_env_cfg()
    assert cfg.commands["body_pose"].zero_command_prob > 0.0


def test_twist_shrinks_but_stays_alive():
    # Hop in place — but the twist neurons must survive for the forward-hop
    # curriculum that follows S1.
    hop = make_microduck_hop_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        h = getattr(hop.commands["twist"].ranges, axis)
        v = getattr(vel.commands["twist"].ranges, axis)
        assert h[1] < v[1], axis
        assert h[1] > 0.0, axis


def test_flight_gates_match_the_measured_model():
    # Walk model authors STAND at 120.0 mm, settles at 116.7 mm. The height
    # gate must sit below standing (so a compressed takeoff still counts) and
    # well above a seated pose.
    params = make_microduck_hop_env_cfg().rewards["simultaneous_flight"].params
    assert 0.07 < params["min_height"] < 0.1167
    assert params["max_tilt_deg"] <= 30.0
    # Must still be paying across the 34 ms open-loop baseline it has to beat.
    assert params["min_flight_s"] < 0.034 < params["max_flight_s"]
    assert params["sensor_name"] == "feet_ground_contact"


def test_every_penalty_term_has_a_sign_that_can_only_log_negative():
    # CLAUDE.md calls this check infallible for catching sign inversions: a
    # negative weight on a self-negating penalty double-negates into a reward
    # for the violation. mjlab-base cost functions return >= 0 -> negative
    # weight; microduck *_penalty / *_l1 self-negate -> positive weight.
    cfg = make_microduck_hop_env_cfg()
    for name, term in cfg.rewards.items():
        if name.endswith("_penalty") or name.endswith("_l1"):
            assert term.weight >= 0.0, f"{name} self-negates; weight must be >= 0"


def test_flight_reward_is_ungated_by_command_for_the_spike():
    # Deliberate: the spike asks whether flight is reachable AT ALL. Phase 1
    # sets command_name to make hopping commandable.
    params = make_microduck_hop_env_cfg().rewards["simultaneous_flight"].params
    assert params["command_name"] is None


def test_runner_cfg_is_distinct_from_the_walker():
    # A shared experiment_name would overwrite the walker's logs.
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
    assert MicroduckHopRlCfg.experiment_name == "hop"
    assert MicroduckHopRlCfg.experiment_name != MicroduckRlCfg.experiment_name
    # Symmetry off while the question is whether the behavior exists at all.
    assert MicroduckHopRlCfg.algorithm.symmetry_cfg is None


def test_play_variant_builds():
    assert make_microduck_hop_env_cfg(play=True) is not None


def test_tasks_are_registered():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    names = set(list_tasks())
    assert "Mjlab-Hop-Flat-MicroDuck" in names
    assert "Mjlab-Hop-Flat-Backlash-MicroDuck" in names
