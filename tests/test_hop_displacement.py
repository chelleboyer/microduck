"""Per-hop displacement — the S5.1 term that replaces air time as the objective.

The S5 run's policy spent ~52% of its life airborne because `simultaneous_flight`
pays per step in the air, so air time WAS the objective. `hop_displacement` pays
for "took off HERE, landed THERE" instead, once per hop.

That makes this a LATCHED term in a file of otherwise stateless ones, and the
latch is where the bugs live. Pinned below, each corresponding to a way the
measurement can silently read zero (or read a teleport as a hop):

  1. The landing edge must be settled BEFORE the takeoff point is re-armed, or
     touchdown overwrites the takeoff pose with the landing pose and every hop
     measures exactly 0.
  2. An env that resets mid-flight must not bank the jump to its spawn.
  3. `min` across feet, not `max` — an ordinary stride swings one foot for
     200 ms while the other stays planted, and `max` would call that a hop.
"""

import math

import torch

from mjlab_microduck.tasks.mdp import hop_displacement, hop_settle

CAP = 0.10
UPRIGHT = (1.0, 0.0, 0.0, 0.0)
FLIGHT = 0.06      # a 60 ms simultaneous flight
GROUNDED_T = 0.30  # both feet down for a while


class _SensorData:
    def __init__(self, n):
        self.current_air_time = torch.zeros(n, 2)
        self.last_air_time = torch.zeros(n, 2)
        self.current_contact_time = torch.full((n, 2), GROUNDED_T)


class _Sensor:
    def __init__(self, n):
        self.data = _SensorData(n)


class _AssetData:
    def __init__(self, n):
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_pos_w[:, 2] = 0.1167
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_quat_w[:, 0] = 1.0
        self.root_link_lin_vel_w = torch.zeros(n, 3)


class _Asset:
    def __init__(self, n):
        self.data = _AssetData(n)


class _Terrain:
    def __init__(self, n):
        self.env_origins = torch.zeros(n, 3)


class _Scene:
    def __init__(self, n):
        self._items = {"contact": _Sensor(n), "robot": _Asset(n)}
        self.terrain = _Terrain(n)

    def __getitem__(self, key):
        return self._items[key]


class _Rig:
    """A steppable fake env. Only what the latch and its gates actually read."""

    def __init__(self, n=1):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02  # 50 Hz control, as everywhere in this repo
        self.common_step_counter = 0
        self.episode_length_buf = torch.full((n,), 100, dtype=torch.long)
        self.scene = _Scene(n)

    # -- state the tests drive -------------------------------------------------
    @property
    def _sensor(self):
        return self.scene["contact"].data

    @property
    def _asset(self):
        return self.scene["robot"].data

    def place(self, x=None, y=None, quat=None):
        if x is not None:
            self._asset.root_link_pos_w[:, 0] = torch.as_tensor(x, dtype=torch.float32)
        if y is not None:
            self._asset.root_link_pos_w[:, 1] = torch.as_tensor(y, dtype=torch.float32)
        if quat is not None:
            self._asset.root_link_quat_w[:] = torch.as_tensor(
                quat, dtype=torch.float32
            ).expand(self.num_envs, 4)

    def ground(self, contact_t=GROUNDED_T, last_air=None):
        self._sensor.current_air_time[:] = 0.0
        self._sensor.current_contact_time[:] = torch.as_tensor(
            contact_t, dtype=torch.float32
        ).reshape(-1, 1) if isinstance(contact_t, (list, tuple)) else contact_t
        if last_air is not None:
            self._sensor.last_air_time[:] = torch.as_tensor(
                last_air, dtype=torch.float32
            ).reshape(self.num_envs, -1)

    def fly(self, air_t):
        self._sensor.current_air_time[:] = torch.as_tensor(
            air_t, dtype=torch.float32
        ).reshape(self.num_envs, -1)
        self._sensor.current_contact_time[:] = 0.0

    def reward(self, **kw):
        kw.setdefault("disp_cap", CAP)
        return hop_displacement(self, sensor_name="contact", **kw)

    def step(self, **kw):
        """Advance one control step and score it."""
        self.common_step_counter += 1
        return self.reward(**kw)


def _yaw(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def _pitch(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def _hop(rig, dx=0.0, dy=0.0, flight_s=FLIGHT, land_quat=None, **kw):
    """One full hop from wherever the rig stands. Returns the landing reward."""
    x0 = rig._asset.root_link_pos_w[:, 0].clone()
    y0 = rig._asset.root_link_pos_w[:, 1].clone()
    rig.step(**kw)  # a grounded step arms the takeoff latch

    n_air = max(int(round(flight_s / 0.02)), 1)
    for i in range(1, n_air + 1):
        frac = i / n_air
        rig.fly(flight_s * frac)
        rig.place(x=x0 + dx * frac, y=y0 + dy * frac)
        rig.step(**kw)

    rig.ground(contact_t=0.02, last_air=[[flight_s, flight_s]] * rig.num_envs)
    if land_quat is not None:
        rig.place(quat=land_quat)
    return rig.step(**kw)


def _fresh(n=1):
    return _Rig(n)


# ------------------------------------------------------------------ paying ----

def test_a_forward_hop_pays_in_proportion_to_the_distance_travelled():
    r = _fresh()
    out = _hop(r, dx=0.05)
    assert math.isclose(out.item(), 0.5, rel_tol=1e-4)  # 50 mm of a 100 mm cap


def test_hopping_in_place_earns_essentially_nothing():
    # THE S5.1 CASE. Under the old stack this hop paid full flight reward for
    # its hang time; here going nowhere is worth nothing.
    assert _hop(_fresh(), dx=0.0).item() < 1e-6


def test_a_longer_flight_that_goes_nowhere_still_earns_nothing():
    # Air time is a means now, not the objective: tripling the hang time
    # changes nothing if the duck lands where it took off.
    assert _hop(_fresh(), dx=0.0, flight_s=0.20).item() < 1e-6


def test_backward_travel_pays_zero_and_never_negative():
    # A positive-weight term that can return negative breaks the sign invariant
    # CLAUDE.md calls infallible.
    out = _hop(_fresh(), dx=-0.08)
    assert out.item() == 0.0


def test_distance_is_capped_so_a_dive_beats_no_controlled_hop():
    at_cap = _hop(_fresh(), dx=CAP).item()
    way_over = _hop(_fresh(), dx=3 * CAP).item()
    assert math.isclose(at_cap, 1.0, rel_tol=1e-4)
    assert math.isclose(way_over, at_cap, rel_tol=1e-6)


# ------------------------------------------------------------------- latch ----

def test_each_hop_is_measured_from_its_own_takeoff_not_the_episode_start():
    # If the takeoff latch were armed once, hop 2 would bank hop 1's distance
    # again and consecutive hops would inflate without end.
    r = _fresh()
    first = _hop(r, dx=0.05)
    for _ in range(10):       # settle, well past the landing window
        r.ground()
        r.step()
    second = _hop(r, dx=0.03)
    assert math.isclose(first.item(), 0.5, rel_tol=1e-4)
    assert math.isclose(second.item(), 0.3, rel_tol=1e-4)


def test_touchdown_does_not_overwrite_the_takeoff_point():
    # The ordering bug that would make every hop measure exactly zero: re-arming
    # the latch before settling the landing edge.
    assert _hop(_fresh(), dx=0.06).item() > 0.0


def test_an_env_reset_mid_flight_banks_nothing():
    # Otherwise the teleport from the old position to the spawn reads as a
    # spectacular hop.
    r = _fresh()
    r.step()
    r.fly(0.04)
    r.place(x=0.5)
    r.step()
    r.episode_length_buf[:] = 0          # reset happened; spawn is elsewhere
    r.place(x=0.0)
    r.ground(contact_t=0.02, last_air=[[0.04, 0.04]])
    assert r.step().tolist() == [0.0]
    # ...and the fresh episode's own first hop is still measured correctly.
    r.episode_length_buf[:] = 100
    assert math.isclose(_hop(r, dx=0.04).item(), 0.4, rel_tol=1e-4)


def test_reading_the_latch_twice_in_one_step_does_not_advance_it():
    # E3 will read the same latch in the same step. Double-advancing would
    # re-arm the takeoff point mid-measurement.
    r = _fresh()
    out = _hop(r, dx=0.05)
    again = r.reward()  # same step counter
    assert math.isclose(out.item(), again.item(), rel_tol=1e-9)


# ------------------------------------------------------------------- gates ----

def test_a_contact_flicker_is_not_a_hop():
    # Below min_flight_s the "flight" is a contact-solver blip during a stride.
    assert _hop(_fresh(), dx=0.05, flight_s=0.005).item() == 0.0


def test_min_across_feet_not_max():
    # An ordinary stride: one foot swings 200 ms, the other never leaves.
    r = _fresh()
    r.step()
    r._sensor.current_air_time[:] = torch.tensor([[0.20, 0.0]])
    r._sensor.current_contact_time[:] = torch.tensor([[0.0, 0.10]])
    r.place(x=0.05)
    r.step()
    r.ground(contact_t=0.02, last_air=[[0.20, 0.0]])
    assert r.step().tolist() == [0.0]


def test_a_face_planting_hop_pays_nothing_however_far_it_went():
    # Tilt, not height: at touchdown the trunk is legitimately compressed, but
    # a fallen robot is exactly what tilt catches.
    assert _hop(_fresh(), dx=0.09, land_quat=_pitch(60.0)).item() == 0.0


def test_travel_is_measured_along_the_takeoff_heading():
    # Body-frame forward, like forward_flight_progress: a duck that turns 90°
    # and drifts sideways has not hopped forward.
    facing_y = _fresh()
    facing_y.place(quat=_yaw(90.0))
    assert math.isclose(_hop(facing_y, dy=0.05).item(), 0.5, rel_tol=1e-3)

    drifting = _fresh()
    drifting.place(quat=_yaw(90.0))
    assert drifting._asset.root_link_quat_w is not None
    assert _hop(drifting, dx=0.05).item() < 1e-3


def test_pays_across_the_landing_window_then_stops():
    r = _fresh()
    assert _hop(r, dx=0.05).item() > 0.0
    r.ground(contact_t=0.10)          # ~2 steps in: still inside the 0.15 s window
    assert r.step().item() > 0.0
    for _ in range(10):               # long settled — this would be an annuity
        r.ground(contact_t=0.40)
        r.step()
    assert r.step().tolist() == [0.0]


def test_a_stale_hop_cannot_be_re_collected_by_tapping_one_foot():
    # THE farm the latch-driven window exists to block. `_hop_just_landed`
    # reopens whenever the MINIMUM per-foot contact time is small, which one
    # foot lifting and returning is enough to produce — so a sensor-driven
    # window would re-pay this hop's distance indefinitely without the duck
    # travelling another millimetre.
    r = _fresh()
    assert _hop(r, dx=0.09).item() > 0.0
    for _ in range(12):               # let the real window age out
        r.ground()
        r.step()
    assert r.step().tolist() == [0.0]

    for _ in range(6):                # right foot swings 60 ms and comes back
        r._sensor.current_air_time[:] = torch.tensor([[0.0, 0.06]])
        r._sensor.current_contact_time[:] = torch.tensor([[0.50, 0.0]])
        r.step()
    r._sensor.current_air_time[:] = 0.0
    r._sensor.current_contact_time[:] = torch.tensor([[0.52, 0.02]])
    r._sensor.last_air_time[:] = torch.tensor([[0.09, 0.06]])
    assert r.step().tolist() == [0.0]


# ------------------------------------------------------------------ rhythm ----
# S5.2 (user call): "head up, hop, stay in place for a sec, then hop again".
# The pause is part of what a hop IS, so it is priced at the moment the hop is
# paid — using the grounded clock frozen at takeoff.

PAUSE = 0.5  # HOP_MIN_GROUND_S


def _stand(rig, seconds):
    for _ in range(int(round(seconds / rig.step_dt))):
        rig.ground()
        rig.step(min_ground_s=PAUSE)


def test_a_hop_after_a_full_pause_pays_in_full():
    r = _fresh()
    _stand(r, PAUSE)
    assert math.isclose(_hop(r, dx=0.05, min_ground_s=PAUSE).item(), 0.5, rel_tol=1e-3)


def test_a_bounce_with_no_pause_pays_almost_nothing():
    # THE behaviour being removed: continuous bouncing. Same 50 mm of travel,
    # but taken straight off a landing, is worth a fraction of the same hop
    # taken from a stand.
    r = _fresh()
    _stand(r, PAUSE)
    _hop(r, dx=0.05, min_ground_s=PAUSE)          # a proper hop
    bounce = _hop(r, dx=0.05, min_ground_s=PAUSE)  # relaunch immediately
    assert bounce.item() < 0.1 * 0.5


def test_the_pause_scales_smoothly_rather_than_gating():
    # A cliff would pay exactly 0 for every hop a bouncing policy can currently
    # produce, and the term would go silent — the failure mode
    # forward_flight_progress sat in through all of S5's smoke tests.
    scores = []
    for held in (0.0, 0.25 * PAUSE, 0.5 * PAUSE, PAUSE):
        r = _fresh()
        _stand(r, held)
        scores.append(_hop(r, dx=0.05, min_ground_s=PAUSE).item())
    assert scores == sorted(scores), scores
    assert scores[0] < scores[-1]
    assert 0.0 < scores[2] < scores[-1]   # partial credit really is partial


def test_pausing_longer_than_required_earns_no_more():
    # Otherwise "stand still forever" beats hopping.
    short, long = _fresh(), _fresh()
    _stand(short, PAUSE)
    _stand(long, 4 * PAUSE)
    a = _hop(short, dx=0.05, min_ground_s=PAUSE).item()
    b = _hop(long, dx=0.05, min_ground_s=PAUSE).item()
    assert math.isclose(a, b, rel_tol=1e-6)


def test_air_time_does_not_count_as_pause():
    # The grounded clock must freeze in flight, or a long hang would qualify
    # the NEXT hop and the bounce comes straight back.
    r = _fresh()
    _stand(r, PAUSE)
    _hop(r, dx=0.05, flight_s=0.28, min_ground_s=PAUSE)
    assert _hop(r, dx=0.05, flight_s=0.28, min_ground_s=PAUSE).item() < 0.1 * 0.5


def test_a_freshly_reset_env_banks_nothing():
    r = _fresh()
    r.episode_length_buf[:] = 1
    assert _hop(r, dx=0.05).tolist() == [0.0]


def test_nan_positions_do_not_pay():
    r = _fresh()
    r.step()
    r.fly(0.04)
    r.place(x=float("nan"))
    r.step()
    r.ground(contact_t=0.02, last_air=[[0.04, 0.04]])
    r.place(x=0.05)
    out = r.step()
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------ settle ----
# The rhythm's other half: paying for the hold itself, so the pause has a
# gradient toward it and not only a precondition attached to it.

def _settle(rig, **kw):
    return hop_settle(rig, sensor_name="contact", **kw)


def test_the_hold_after_a_hop_pays():
    r = _fresh()
    _stand(r, PAUSE)
    _hop(r, dx=0.05, min_ground_s=PAUSE)
    r.ground(contact_t=0.02)
    assert _settle(r).item() > 0.9


def test_standing_without_ever_hopping_pays_nothing():
    # The window only opens on a genuine landing, so "never hop, just stand"
    # earns zero — this is what stops the hold being farmable.
    r = _fresh()
    _stand(r, 3.0)
    assert _settle(r).tolist() == [0.0]


def test_the_hold_stops_paying_once_the_window_closes():
    r = _fresh()
    _stand(r, PAUSE)
    _hop(r, dx=0.05, min_ground_s=PAUSE)
    assert _settle(r).item() > 0.0
    for _ in range(int(round((PAUSE + 0.1) / r.step_dt))):
        r.ground()
        r.step()
    assert _settle(r).tolist() == [0.0]


def test_skidding_on_landing_is_not_a_hold():
    r = _fresh()
    _stand(r, PAUSE)
    _hop(r, dx=0.05, min_ground_s=PAUSE)
    r.ground(contact_t=0.02)
    r._asset.root_link_lin_vel_w[:, 0] = 0.4   # sliding forward
    assert _settle(r).item() == 0.0


def test_a_toppled_landing_is_not_a_hold():
    r = _fresh()
    _stand(r, PAUSE)
    _hop(r, dx=0.05, min_ground_s=PAUSE)
    r.ground(contact_t=0.02)
    r.place(quat=_pitch(50.0))
    assert _settle(r).item() == 0.0


def test_settle_is_never_negative():
    r = _fresh()
    _stand(r, PAUSE)
    _hop(r, dx=0.05, min_ground_s=PAUSE)
    for v in (0.0, 0.3, -2.0, float("nan")):
        r._asset.root_link_lin_vel_w[:, 0] = v
        out = _settle(r)
        assert torch.isfinite(out).all() and (out >= 0.0).all()


def test_batched_envs_are_independent():
    r = _fresh(3)
    x0 = torch.tensor([0.0, 0.0, 0.0])
    r.step()
    for i in (1, 2, 3):
        r.fly([[FLIGHT * i / 3] * 2] * 3)
        r.place(x=x0 + torch.tensor([0.09, 0.0, -0.05]) * (i / 3))
        r.step()
    r.ground(contact_t=0.02, last_air=[[FLIGHT, FLIGHT]] * 3)
    out = r.step()
    assert math.isclose(out[0].item(), 0.9, rel_tol=1e-4)   # 90 mm forward
    assert out[1].item() < 1e-6                              # in place
    assert out[2].item() == 0.0                              # backward
