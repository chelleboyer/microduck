"""`forward_flight_progress` must pay for TRAVEL DURING FLIGHT and nothing else.

Encoding E1 from the architecture doc: forward intent is un-commanded for the
S5 spike, so travel is paid directly rather than tracked against a command.

The exploit this term exists to reject is walking. The velocity recipe this env
inherits already pays for forward velocity, and walking is a strictly easier way
to earn it than hopping — so a forward-velocity reward that is not gated on
flight simply retrains the walker. Every test below pins one way that gate could
leak.
"""

import math

import torch

from mjlab_microduck.tasks.mdp import forward_flight_progress

STAND_Z = 0.1167  # measured: walk model settles here holding STAND
UPRIGHT = (1.0, 0.0, 0.0, 0.0)
CAP = 0.4


class _SensorData:
    def __init__(self, air_time):
        self.current_air_time = air_time


class _Sensor:
    def __init__(self, air_time):
        self.data = _SensorData(air_time)


class _AssetData:
    def __init__(self, n, z, quat, v_fwd):
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_pos_w[:, 2] = torch.as_tensor(z, dtype=torch.float32)
        self.root_link_quat_w = (
            torch.as_tensor(quat, dtype=torch.float32).expand(n, 4).clone()
        )
        self.root_link_lin_vel_b = torch.zeros(n, 3)
        self.root_link_lin_vel_b[:, 0] = torch.as_tensor(v_fwd, dtype=torch.float32)


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
    def __init__(self, air_time, v_fwd=CAP, z=STAND_Z, quat=UPRIGHT, step=100):
        air_time = torch.as_tensor(air_time, dtype=torch.float32)
        n = air_time.shape[0]
        self.scene = _Scene(
            _Sensor(air_time), _Asset(_AssetData(n, z, quat, v_fwd)), n
        )
        self.episode_length_buf = torch.full((n,), step, dtype=torch.long)


def _quat_pitch(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def _reward(env, **kw):
    kw.setdefault("vel_cap", CAP)
    return forward_flight_progress(env, sensor_name="contact", **kw)


def test_pays_for_forward_velocity_while_airborne():
    # Both feet up 60 ms, upright, at height, moving forward at the cap.
    assert _reward(_Env([[0.06, 0.06]])).tolist() == [1.0]


def test_walking_forward_pays_nothing():
    # THE central case. Full forward speed, one foot planted — an ordinary
    # stride. If this pays, the term just retrains the walker.
    assert _reward(_Env([[0.20, 0.0]])).tolist() == [0.0]
    assert _reward(_Env([[0.0, 0.20]])).tolist() == [0.0]


def test_standing_still_in_flight_pays_nothing():
    # A purely vertical hop travels nowhere and earns nothing HERE (it is paid
    # by simultaneous_flight instead). Keeps the two questions separable.
    assert _reward(_Env([[0.06, 0.06]], v_fwd=0.0)).tolist() == [0.0]


def test_backward_flight_pays_zero_not_negative():
    # Positive-weight term: a negative return would break the sign invariant.
    out = _reward(_Env([[0.06, 0.06]], v_fwd=-CAP))
    assert out.tolist() == [0.0]


def test_partial_velocity_gives_partial_credit():
    assert math.isclose(_reward(_Env([[0.06, 0.06]], v_fwd=CAP / 2)).item(), 0.5,
                        rel_tol=1e-5)


def test_velocity_is_capped():
    # No jackpot: a forward dive must not out-earn a controlled hop.
    assert _reward(_Env([[0.06, 0.06]], v_fwd=5.0)).tolist() == [1.0]


def test_toppling_forward_pays_nothing():
    # Falling forward is fast, airborne, and travelling — and is not a hop.
    env = _Env([[0.12, 0.12]], v_fwd=CAP, quat=_quat_pitch(50.0))
    assert _reward(env).tolist() == [0.0]


def test_sitting_and_sliding_pays_nothing():
    # Trunk on the ground, feet tucked, moving forward: blocked by the height
    # gate, not by the contact condition.
    assert _reward(_Env([[0.10, 0.10]], z=0.05)).tolist() == [0.0]


def test_contact_flicker_pays_nothing():
    # A one-step solver blip mid-stride must not read as flight.
    assert _reward(_Env([[0.005, 0.005]])).tolist() == [0.0]


def test_credit_stops_past_max_flight():
    assert _reward(_Env([[0.5, 0.5]])).tolist() == [0.0]


def test_freshly_reset_env_banks_nothing():
    assert _reward(_Env([[0.10, 0.10]], step=1)).tolist() == [0.0]


def test_never_negative():
    for env in (
        _Env([[0.06, 0.06]], v_fwd=-2.0),
        _Env([[0.0, 0.0]], v_fwd=-2.0),
        _Env([[0.5, 0.5]], v_fwd=CAP),
    ):
        assert (_reward(env) >= 0.0).all()


def test_nan_does_not_pay():
    assert torch.isfinite(_reward(_Env([[float("nan"), 0.06]]))).all()
    out = _reward(_Env([[0.06, 0.06]], v_fwd=float("nan")))
    assert torch.isfinite(out).all() and out.tolist() == [0.0]


def test_batched_envs_are_independent():
    # hop-forward / walking / toppling / vertical-only hop
    env = _Env([[0.06, 0.06], [0.20, 0.0], [0.06, 0.06], [0.06, 0.06]])
    env.scene["robot"].data.root_link_quat_w[2] = torch.tensor(_quat_pitch(50.0))
    env.scene["robot"].data.root_link_lin_vel_b[3, 0] = 0.0
    assert _reward(env).tolist() == [1.0, 0.0, 0.0, 0.0]
