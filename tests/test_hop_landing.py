"""Landing quality and landing impact — the unmeasured half of Success #1.

Every other hopscotch gate screens the robot DURING flight (tilt, trunk
height), so before these terms a forward hop that reliably face-planted scored
exactly as well as one that stuck the landing.

Two failure modes drive the design and are pinned below:

  1. A landing reward that fires without a preceding flight is just a "stand
     upright" bonus, double-paying alongside the inherited `upright` term
     (already weight 2.0). Hence the min_flight_s condition.
  2. `hop_landing_impact_penalty` is SELF-NEGATING and takes a POSITIVE weight.
     Inverting that pays the policy to slam into the ground, and the cfg test
     that enforces the convention only checks the weight's sign — so the
     function's sign has to be pinned here.
"""

import math

import torch

from mjlab_microduck.tasks.mdp import (
    hop_landing_impact_penalty,
    hop_landing_quality,
)

STAND_Z = 0.1167   # measured: walk model settles here holding STAND
UPRIGHT = (1.0, 0.0, 0.0, 0.0)
FLEW = 0.06        # a genuine 60 ms simultaneous flight just ended
FREE = 0.5         # impact allowance (m/s)


class _SensorData:
    def __init__(self, last_air, contact_t):
        self.last_air_time = last_air
        self.current_contact_time = contact_t


class _Sensor:
    def __init__(self, last_air, contact_t):
        self.data = _SensorData(last_air, contact_t)


class _AssetData:
    def __init__(self, n, z, quat, vz):
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_pos_w[:, 2] = torch.as_tensor(z, dtype=torch.float32)
        self.root_link_quat_w = (
            torch.as_tensor(quat, dtype=torch.float32).expand(n, 4).clone()
        )
        self.root_link_lin_vel_w = torch.zeros(n, 3)
        self.root_link_lin_vel_w[:, 2] = torch.as_tensor(vz, dtype=torch.float32)


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
    """last_air / contact_t are per-foot (left, right), like the real sensor."""

    def __init__(self, last_air, contact_t, z=STAND_Z, quat=UPRIGHT, vz=0.0, step=100):
        last_air = torch.as_tensor(last_air, dtype=torch.float32)
        contact_t = torch.as_tensor(contact_t, dtype=torch.float32)
        n = last_air.shape[0]
        self.scene = _Scene(
            _Sensor(last_air, contact_t), _Asset(_AssetData(n, z, quat, vz)), n
        )
        self.episode_length_buf = torch.full((n,), step, dtype=torch.long)


def _quat_pitch(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def _landed(**kw):
    """A clean touchdown 20 ms ago, after a real 60 ms flight."""
    kw.setdefault("last_air", [[FLEW, FLEW]])
    kw.setdefault("contact_t", [[0.02, 0.02]])
    return _Env(**kw)


def _quality(env, **kw):
    return hop_landing_quality(env, sensor_name="contact", **kw)


def _impact(env, **kw):
    kw.setdefault("free_speed", FREE)
    return hop_landing_impact_penalty(env, sensor_name="contact", **kw)


# ---------------------------------------------------------------- quality ----

def test_pays_on_upright_landing_after_real_flight():
    assert _quality(_landed()).item() > 0.9


def test_landing_without_a_real_flight_pays_nothing():
    # THE central case: both feet down after a mere contact flicker. Without
    # this condition the term becomes a standing bonus that double-pays with
    # the inherited `upright` reward.
    env = _Env(last_air=[[0.001, 0.001]], contact_t=[[0.02, 0.02]])
    assert _quality(env).tolist() == [0.0]


def test_min_across_feet_not_max():
    # An ordinary stride: one foot swung 200 ms, the other never left. max()
    # would call this a landing from flight; min() correctly rejects it.
    env = _Env(last_air=[[0.20, 0.0]], contact_t=[[0.02, 0.02]])
    assert _quality(env).tolist() == [0.0]


def test_still_airborne_pays_nothing():
    # contact_time 0 on a foot means it is not down yet.
    env = _Env(last_air=[[FLEW, FLEW]], contact_t=[[0.02, 0.0]])
    assert _quality(env).tolist() == [0.0]


def test_long_after_landing_pays_nothing():
    # Outside the window this would become a per-step standing annuity.
    env = _Env(last_air=[[FLEW, FLEW]], contact_t=[[0.9, 0.9]])
    assert _quality(env).tolist() == [0.0]


def test_tilted_landing_scores_far_below_upright():
    good = _quality(_landed()).item()
    bad = _quality(_landed(quat=_quat_pitch(45.0))).item()
    assert bad < 0.25 * good


def test_crouched_landing_scores_below_a_clean_one():
    good = _quality(_landed()).item()
    low = _quality(_landed(z=0.07)).item()
    assert low < 0.25 * good


def test_freshly_reset_env_banks_nothing():
    assert _quality(_landed(step=1)).tolist() == [0.0]


def test_quality_never_negative():
    for env in (_landed(), _landed(quat=_quat_pitch(80.0)), _landed(z=0.02)):
        assert (_quality(env) >= 0.0).all()


def test_quality_nan_does_not_pay():
    env = _Env(last_air=[[float("nan"), FLEW]], contact_t=[[0.02, 0.02]])
    out = _quality(env)
    assert torch.isfinite(out).all() and out.tolist() == [0.0]


# ----------------------------------------------------------------- impact ----

def test_impact_is_never_positive():
    # THE sign invariant. This term self-negates and takes a POSITIVE weight;
    # a positive return would pay the policy to slam into the ground.
    for env in (
        _landed(vz=-3.0),
        _landed(vz=0.0),
        _landed(vz=+3.0),
        _Env(last_air=[[0.0, 0.0]], contact_t=[[0.0, 0.0]], vz=-3.0),
    ):
        assert (_impact(env) <= 0.0).all()


def test_soft_landing_is_free():
    # A hop to the S1 probe's ~11 mm apex lands at ~0.46 m/s — under the
    # allowance, so a hop the robot can actually perform costs nothing.
    assert _impact(_landed(vz=-0.46, contact_t=[[0.02, 0.02]])).tolist() == [0.0]


def test_hard_landing_is_penalized_linearly_in_excess():
    out = _impact(_landed(vz=-(FREE + 0.8), contact_t=[[0.02, 0.02]]))
    assert math.isclose(out.item(), -0.8, rel_tol=1e-5)


def test_rising_costs_nothing():
    # Upward velocity is takeoff, not impact.
    assert _impact(_landed(vz=+2.0, contact_t=[[0.02, 0.02]])).tolist() == [0.0]


def test_no_penalty_without_a_real_flight():
    env = _Env(last_air=[[0.001, 0.001]], contact_t=[[0.02, 0.02]], vz=-5.0)
    assert _impact(env).tolist() == [0.0]


def test_no_penalty_outside_the_impact_window():
    # The window is ~2 control steps so it samples touchdown, not the whole
    # settle. 150 ms after landing there is no impact left to charge for.
    env = _Env(last_air=[[FLEW, FLEW]], contact_t=[[0.15, 0.15]], vz=-5.0)
    assert _impact(env).tolist() == [0.0]


def test_impact_nan_does_not_pay():
    out = _impact(_landed(vz=float("nan"), contact_t=[[0.02, 0.02]]))
    assert torch.isfinite(out).all() and out.tolist() == [0.0]


def test_batched_envs_are_independent():
    # clean landing / hard landing / no-flight touchdown
    env = _Env(
        last_air=[[FLEW, FLEW], [FLEW, FLEW], [0.0, 0.0]],
        contact_t=[[0.02, 0.02], [0.02, 0.02], [0.02, 0.02]],
        vz=[-0.1, -(FREE + 1.0), -5.0],
    )
    out = _impact(env)
    assert out[0].item() == 0.0
    assert math.isclose(out[1].item(), -1.0, rel_tol=1e-5)
    assert out[2].item() == 0.0
