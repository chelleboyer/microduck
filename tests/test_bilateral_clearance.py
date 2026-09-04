"""`bilateral_foot_clearance` must pay for BOTH feet rising, and nothing else.

The dense companion to simultaneous_flight: binary flight gives no gradient
until the robot is already airborne, so this term scores a partial lift. It has
its own exploit — tucking both feet while sitting on the trunk scores full
marks on clearance alone — which the trunk gates block.

Metric borrowed from the one known working Microduck hop
(joanfox/microduck-happy-hop), which optimised bilateral clearance at a
0.035 m target under backlash and BAM.
"""

import math

import torch

from mjlab_microduck.tasks.mdp import bilateral_foot_clearance

STAND_Z = 0.1167   # measured: trunk settles here holding STAND
REST = 0.0029      # measured: foot sites rest here at STAND
TARGET = 0.035
UPRIGHT = (1.0, 0.0, 0.0, 0.0)


class _AssetData:
    def __init__(self, foot_z, trunk_z, quat):
        n = len(foot_z)
        self.site_pos_w = torch.zeros(n, 2, 3)
        self.site_pos_w[:, :, 2] = torch.as_tensor(foot_z, dtype=torch.float32)
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_pos_w[:, 2] = torch.as_tensor(trunk_z, dtype=torch.float32)
        self.root_link_quat_w = (
            torch.as_tensor(quat, dtype=torch.float32).expand(n, 4).clone()
        )


class _Asset:
    def __init__(self, data):
        self.data = data


class _Terrain:
    def __init__(self, n):
        self.env_origins = torch.zeros(n, 3)


class _Scene:
    def __init__(self, asset, n):
        self._a = asset
        self.terrain = _Terrain(n)

    def __getitem__(self, _key):
        return self._a


class _Env:
    def __init__(self, foot_z, trunk_z=STAND_Z, quat=UPRIGHT, step=100):
        n = len(foot_z)
        self.scene = _Scene(_Asset(_AssetData(foot_z, trunk_z, quat)), n)
        self.episode_length_buf = torch.full((n,), step, dtype=torch.long)


class _FeetCfg:
    name = "robot"
    site_ids = [0, 1]


def _reward(env, **kw):
    kw.setdefault("feet_cfg", _FeetCfg())
    kw.setdefault("rest_height", REST)
    kw.setdefault("target_height", TARGET)
    return bilateral_foot_clearance(env, **kw)


def test_both_feet_at_target_scores_full():
    assert _reward(_Env([[TARGET, TARGET]])).tolist() == [1.0]


def test_feet_at_rest_score_zero():
    assert _reward(_Env([[REST, REST]])).tolist() == [0.0]


def test_single_foot_lift_pays_nothing():
    # THE central case: an ordinary stride. One foot at 20 mm, one planted.
    # min() over feet means the planted foot sets the score.
    out = _reward(_Env([[0.020, REST]]))
    assert out.tolist() == [0.0]


def test_lower_foot_sets_the_score():
    # Asymmetric lift is scored by the laggard, not the leader.
    out = _reward(_Env([[TARGET, 0.019]]))
    expected = (0.019 - REST) / (TARGET - REST)
    assert math.isclose(out.item(), expected, rel_tol=1e-5)


def test_partial_lift_gives_partial_credit():
    # The whole point: a gradient before flight ever happens.
    half = REST + (TARGET - REST) / 2
    assert math.isclose(_reward(_Env([[half, half]])).item(), 0.5, rel_tol=1e-5)


def test_exceeding_target_earns_nothing_extra():
    # No jackpot: an uncapped height reward buys arbitrary violence.
    assert _reward(_Env([[0.5, 0.5]])).tolist() == [1.0]


def test_sitting_with_feet_tucked_is_rejected():
    # Clearance's OWN exploit: feet at full target while the trunk rests on the
    # ground. Scores 1.0 on clearance alone; the trunk-height gate kills it.
    assert _reward(_Env([[TARGET, TARGET]], trunk_z=0.05)).tolist() == [0.0]


def test_toppling_is_rejected():
    h = math.radians(50.0) / 2.0
    quat = (math.cos(h), 0.0, math.sin(h), 0.0)
    assert _reward(_Env([[TARGET, TARGET]], quat=quat)).tolist() == [0.0]


def test_freshly_reset_env_banks_nothing():
    assert _reward(_Env([[TARGET, TARGET]], step=1)).tolist() == [0.0]


def test_never_negative_even_below_rest():
    # A foot pressed into the terrain (or a noisy origin) must not pay a
    # negative reward — this is a positive-weight term.
    out = _reward(_Env([[-0.01, -0.01], [REST, REST], [TARGET, TARGET]]))
    assert (out >= 0.0).all()


def test_nan_foot_height_does_not_pay():
    out = _reward(_Env([[float("nan"), TARGET]]))
    assert torch.isfinite(out).all() and out.tolist() == [0.0]


def test_batched_envs_are_independent():
    # hop / stride / sitting-with-feet-tucked
    env = _Env([[TARGET, TARGET], [0.020, REST], [TARGET, TARGET]])
    env.scene["robot"].data.root_link_pos_w[2, 2] = 0.05
    assert _reward(env).tolist() == [1.0, 0.0, 0.0]
