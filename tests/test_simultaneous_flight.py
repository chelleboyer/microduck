"""`simultaneous_flight` must pay for a HOP and nothing else.

The S1 CPU probe (docs/s1-flight-probe.md) found two metrics that look like a
hop and are not:

  1. Contact-loss is not flight — a duck falling over loses both foot contacts
     and logs excellent air time. 384 of the probe's apparent successes were
     topples.
  2. Per-foot air time is not simultaneous flight — an ordinary walking policy
     reports 125-300 ms per foot and scores 1.01 on the stock feet_air_time
     reward while never leaving the ground.

Each test below pins one of those, or one of the gates that blocks it.
"""

import math

import torch

from mjlab_microduck.tasks.mdp import simultaneous_flight

STAND_Z = 0.1167  # measured: walk model settles here holding STAND
UPRIGHT = (1.0, 0.0, 0.0, 0.0)


class _SensorData:
    def __init__(self, air_time):
        self.current_air_time = air_time


class _Sensor:
    def __init__(self, air_time):
        self.data = _SensorData(air_time)


class _AssetData:
    def __init__(self, n, z, quat):
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_pos_w[:, 2] = torch.as_tensor(z, dtype=torch.float32)
        self.root_link_quat_w = torch.as_tensor(quat, dtype=torch.float32).expand(n, 4).clone()


class _Asset:
    def __init__(self, data):
        self.data = data


class _Terrain:
    def __init__(self, n):
        self.env_origins = torch.zeros(n, 3)


class _Scene:
    def __init__(self, sensor, asset, n):
        self._items = {"contact": sensor, "robot": asset}
        self.terrain = _Terrain(n)

    def __getitem__(self, key):
        return self._items[key]


class _Env:
    def __init__(self, air_time, z=STAND_Z, quat=UPRIGHT, step=100):
        air_time = torch.as_tensor(air_time, dtype=torch.float32)
        n = air_time.shape[0]
        self.scene = _Scene(_Sensor(air_time), _Asset(_AssetData(n, z, quat)), n)
        self.episode_length_buf = torch.full((n,), step, dtype=torch.long)


def _quat_pitch(deg):
    """Trunk pitched `deg` about y — the axis a topple actually rotates about."""
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def _reward(env, **kw):
    return simultaneous_flight(env, sensor_name="contact", **kw)


def test_pays_while_both_feet_airborne():
    # Both feet up for 60 ms, upright, at standing height: a real hop.
    assert _reward(_Env([[0.06, 0.06]])).tolist() == [1.0]


def test_ignores_alternating_single_foot_air_time():
    # THE CENTRAL CASE: an ordinary walk. One foot swinging 200 ms while the
    # other is planted must pay ZERO, or the term reports walking as hopping.
    assert _reward(_Env([[0.20, 0.0], [0.0, 0.20]])).tolist() == [0.0, 0.0]


def test_min_across_feet_not_max():
    # Guards the specific bug of reading the wrong reduction: max() would call
    # this flight (0.20), min() correctly sees the planted foot.
    assert _reward(_Env([[0.20, 0.0]])).tolist() == [0.0]


def test_toppling_is_not_flight():
    # Feet genuinely off the ground, but the duck is falling over. This is the
    # failure that produced 162 ms of fake "air time" in the first probe run.
    env = _Env([[0.12, 0.12]], quat=_quat_pitch(50.0))
    assert _reward(env).tolist() == [0.0]


def test_sitting_with_feet_lifted_is_not_flight():
    # Trunk resting on the ground, both feet tucked up: no contact, upright,
    # going nowhere. Blocked by the height gate.
    assert _reward(_Env([[0.10, 0.10]], z=0.05)).tolist() == [0.0]


def test_contact_flicker_below_threshold_pays_nothing():
    # A one-step solver blip during a stride must not read as a hop.
    assert _reward(_Env([[0.005, 0.005]])).tolist() == [0.0]


def test_credit_stops_past_max_flight():
    # No jackpot: a ballistic launch longer than any real hop stops earning
    # rather than outscoring every honest one.
    assert _reward(_Env([[0.5, 0.5]])).tolist() == [0.0]


def test_reward_is_flat_not_proportional_to_elapsed():
    # Paying elapsed time each step would be quadratic in duration — the
    # jackpot shape. Every in-range flight step is worth exactly the same.
    short = _reward(_Env([[0.03, 0.03]]))
    long = _reward(_Env([[0.25, 0.25]]))
    assert short.tolist() == long.tolist() == [1.0]


def test_freshly_reset_env_banks_nothing():
    # A robot spawned clear of the floor passes every physical gate on step 0.
    assert _reward(_Env([[0.10, 0.10]], step=1)).tolist() == [0.0]


def test_never_negative():
    # Positive-weight term: CLAUDE.md's sign check reads Episode_Reward/<term>,
    # and a self-negating value here would double-negate into a penalty.
    for env in (_Env([[0.06, 0.06]]), _Env([[0.0, 0.0]]), _Env([[0.5, 0.5]])):
        assert (_reward(env) >= 0.0).all()


def test_nan_air_time_does_not_pay():
    # NaN must not propagate into the reward sum — rsl_rl's global check_nan
    # kills the run, and nan_to_num(0.0) is the repo's convention.
    out = _reward(_Env([[float("nan"), 0.06]]))
    assert torch.isfinite(out).all() and out.tolist() == [0.0]


def test_batched_envs_are_independent():
    # hop / walking / toppling / sitting in one batch.
    env = _Env([[0.06, 0.06], [0.20, 0.0], [0.06, 0.06], [0.06, 0.06]])
    env.scene["robot"].data.root_link_quat_w[2] = torch.tensor(_quat_pitch(50.0))
    env.scene["robot"].data.root_link_pos_w[3, 2] = 0.05
    assert _reward(env).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_beats_the_open_loop_baseline_only_above_34ms():
    # The S1 decision rule is measured in flight DURATION, so the term must
    # still be paying across the 34 ms open-loop baseline it has to beat.
    assert _reward(_Env([[0.034, 0.034]])).tolist() == [1.0]
