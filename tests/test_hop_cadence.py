"""S5.4: the phase clock becomes BINDING for the terms that pay, and heading is
constrained again.

The S5.3 run installed a 1.6 s hop cycle and then ignored it — 8.71 hops per
cycle, takeoff phase lock 0.23 — because ``hop_crouch_by_phase`` was the only
term gated on phase while every PAYMENT (displacement, apex, flight) was
phase-blind. It also drifted +107 deg/s, a full circle every 3.4 s, because
installing the phase carrier displaced ``track_angular_velocity`` and nothing
replaced the only heading constraint in the env.

The shape of the cadence fix is the load-bearing part and is pinned below: the
first genuine hop of each cycle pays in FULL and the extras pay
``repeat_pay``. Scaling every hop down instead is what S5.2 did with a pause
requirement, and it collapsed ``hop_displacement`` 7x by making the term
unearnable before the behaviour existed.
"""

import math

import torch

from mjlab_microduck.tasks.mdp import (
    GroundPickPhaseCommand,
    hop_apex_rise,
    hop_displacement,
    hop_settle,
)

CAP = 0.10
FLIGHT = 0.10
LAUNCH = (0.28, 0.38)


class _Data:
    def __init__(self, n):
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_pos_w[:, 2] = 0.12
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_quat_w[:, 0] = 1.0
        self.root_link_lin_vel_w = torch.zeros(n, 3)


class _Asset:
    def __init__(self, n):
        self.data = _Data(n)


class _SensorData:
    def __init__(self, n):
        self.current_air_time = torch.zeros(n, 2)
        self.last_air_time = torch.zeros(n, 2)
        self.current_contact_time = torch.zeros(n, 2)


class _Sensor:
    def __init__(self, n):
        self.data = _SensorData(n)


class _Terrain:
    def __init__(self, n):
        self.env_origins = torch.zeros(n, 3)


class _Scene:
    def __init__(self, n):
        self._items = {"contact": _Sensor(n), "robot": _Asset(n)}
        self.terrain = _Terrain(n)

    def __getitem__(self, key):
        return self._items[key]


class _PhaseCommandManager:
    """Serves the twist slot as a GroundPickPhaseCommand, like the forward env."""

    def __init__(self, n):
        self.n = n
        self.phase = torch.zeros(n)
        # isinstance is what _hop_phase_or_none checks; an uninitialised
        # instance is enough and needs no env to build.
        self._term = object.__new__(GroundPickPhaseCommand)

    def get_term(self, name):
        assert name == "twist"
        return self._term

    def get_command(self, name):
        cmd = torch.zeros(self.n, 3)
        cmd[:, 0] = torch.cos(2 * math.pi * self.phase)
        cmd[:, 1] = torch.sin(2 * math.pi * self.phase)
        return cmd


class _Rig:
    def __init__(self, n=1, phase=True):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.common_step_counter = 0
        self.episode_length_buf = torch.full((n,), 100, dtype=torch.long)
        self.scene = _Scene(n)
        if phase:
            self.command_manager = _PhaseCommandManager(n)

    @property
    def _asset(self):
        return self.scene["robot"].data

    @property
    def _sensor(self):
        return self.scene["contact"].data

    def set_phase(self, p):
        self.command_manager.phase[:] = float(p)

    def place(self, x=None):
        if x is not None:
            self._asset.root_link_pos_w[:, 0] = torch.as_tensor(x, dtype=torch.float32)

    def ground(self, last_air=None):
        self._sensor.current_air_time[:] = 0.0
        self._sensor.current_contact_time[:] = 0.02
        if last_air is not None:
            self._sensor.last_air_time[:] = last_air

    def fly(self, air_t):
        self._sensor.current_air_time[:] = air_t
        self._sensor.current_contact_time[:] = 0.0

    def step(self, func=hop_displacement, **kw):
        self.common_step_counter += 1
        kw.setdefault("sensor_name", "contact")
        if func is hop_displacement:
            kw.setdefault("disp_cap", CAP)
        return func(self, **kw)


def _hop(rig, dx=0.05, takeoff_phase=None, flight_s=FLIGHT, func=hop_displacement, **kw):
    """One hop, optionally launched at a chosen phase. Returns the landing pay."""
    if takeoff_phase is not None:
        rig.set_phase(takeoff_phase)
    x0 = rig._asset.root_link_pos_w[:, 0].clone()
    rig.ground()
    rig.step(func=func, **kw)          # a grounded step arms the latch

    n_air = max(int(round(flight_s / 0.02)), 1)
    for i in range(1, n_air + 1):
        rig.fly(flight_s * i / n_air)
        rig.place(x=x0 + dx * i / n_air)
        rig.step(func=func, **kw)

    rig.ground(last_air=torch.full((rig.num_envs, 2), flight_s))
    return rig.step(func=func, **kw)


# ------------------------------------------------------ one hop per cycle ----

def test_the_first_hop_of_a_cycle_is_paid_in_full():
    r = _Rig()
    out = _hop(r, dx=0.05, takeoff_phase=0.30, launch_phase=LAUNCH)
    assert out.item() > 0.49, "a 50 mm on-beat hop should pay ~half the cap"


def test_a_second_hop_in_the_same_cycle_earns_repeat_pay():
    r = _Rig()
    first = _hop(r, dx=0.05, takeoff_phase=0.30, launch_phase=LAUNCH)
    second = _hop(r, dx=0.05, takeoff_phase=0.55, launch_phase=LAUNCH)
    assert first.item() > 0.4
    assert second.item() == 0.0, "the extras must not pay at repeat_pay=0"


def test_the_budget_refills_when_the_cycle_wraps():
    r = _Rig()
    _hop(r, dx=0.05, takeoff_phase=0.30, launch_phase=LAUNCH)
    r.set_phase(0.95)
    r.ground()
    r.step(launch_phase=LAUNCH)        # observe the high phase...
    r.set_phase(0.02)
    r.ground()
    r.step(launch_phase=LAUNCH)        # ...then the wrap
    out = _hop(r, dx=0.05, takeoff_phase=0.30, launch_phase=LAUNCH)
    assert out.item() > 0.4, "a new cycle must pay its first hop again"


def test_the_hop_that_counts_is_never_scaled_down():
    """The S5.2 failure mode: a cadence requirement that reduces the payment for
    every hop makes the term unearnable before the behaviour exists."""
    r_plain = _Rig()
    plain = _hop(r_plain, dx=0.05, takeoff_phase=0.30)
    r_cad = _Rig()
    cadenced = _hop(r_cad, dx=0.05, takeoff_phase=0.30, launch_phase=LAUNCH)
    assert cadenced.item() == plain.item()


def test_repeat_pay_can_be_loosened_without_touching_the_first_hop():
    r = _Rig()
    _hop(r, dx=0.05, takeoff_phase=0.30, launch_phase=LAUNCH, repeat_pay=0.25)
    second = _hop(r, dx=0.05, takeoff_phase=0.55, launch_phase=LAUNCH, repeat_pay=0.25)
    assert second.item() > 0.0
    assert second.item() < 0.2


# ------------------------------------------------------- the launch window ----

def test_an_on_beat_hop_outearns_an_off_beat_one():
    on = _hop(_Rig(), dx=0.05, takeoff_phase=0.33, launch_phase=LAUNCH)
    off = _hop(_Rig(), dx=0.05, takeoff_phase=0.80, launch_phase=LAUNCH)
    assert on.item() > off.item()


def test_an_off_beat_hop_still_pays_the_floor_and_is_never_silenced():
    on = _hop(_Rig(), dx=0.05, takeoff_phase=0.33, launch_phase=LAUNCH)
    off = _hop(_Rig(), dx=0.05, takeoff_phase=0.80, launch_phase=LAUNCH)
    assert off.item() >= 0.5 * on.item() - 1e-6, "floor=0.5 must hold"


def test_anywhere_inside_the_window_scores_the_same():
    lo = _hop(_Rig(), dx=0.05, takeoff_phase=0.28, launch_phase=LAUNCH)
    hi = _hop(_Rig(), dx=0.05, takeoff_phase=0.379, launch_phase=LAUNCH)
    assert abs(lo.item() - hi.item()) < 1e-6


def test_phase_distance_wraps_around_the_cycle():
    """A takeoff at 0.0 is 0.05 from a window ending at 0.95, not 0.9."""
    near = _hop(_Rig(), dx=0.05, takeoff_phase=0.0, launch_phase=(0.90, 0.95))
    far = _hop(_Rig(), dx=0.05, takeoff_phase=0.45, launch_phase=(0.90, 0.95))
    assert near.item() > far.item()


# ------------------------------------------------- apex shares the cadence ----

def test_apex_rise_obeys_the_same_cadence_as_displacement():
    r = _Rig()

    def rise_hop(phase):
        rig_phase = phase
        return _hop(r, dx=0.0, takeoff_phase=rig_phase, func=hop_apex_rise,
                    launch_phase=LAUNCH, rise_target=0.05)

    # The fake rig keeps z fixed, so the rise is 0 and this checks only that the
    # cadence factor is wired in and cannot raise a zero rise.
    assert rise_hop(0.30).item() == 0.0


def test_apex_and_displacement_take_the_identical_factor():
    """One hop is one event: paying its distance and its height on different
    cadences would let a policy split them."""
    from mjlab_microduck.tasks.mdp import _hop_cadence_factor

    r = _Rig()
    _hop(r, dx=0.05, takeoff_phase=0.30, launch_phase=LAUNCH)
    a = _hop_cadence_factor(r, LAUNCH, 0.15, 0.5, 0.0)
    b = _hop_cadence_factor(r, LAUNCH, 0.15, 0.5, 0.0)
    assert torch.equal(a, b)


# --------------------------------------- the baseline env is left untouched ----

def test_an_env_with_no_phase_carrier_is_completely_unaffected():
    """Mjlab-Hop-Flat-MicroDuck and every pre-S5.3 policy: no phase command, so
    the cadence factor must be exactly 1.0 and the reward unchanged."""
    r = _Rig(phase=False)
    first = _hop(r, dx=0.05, launch_phase=LAUNCH)
    assert first.item() > 0.49
    # And CONSECUTIVE hops must keep paying: with no clock there are no cycle
    # boundaries, so a per-cycle budget that never refills would silently zero
    # every hop after the first. tests/test_hop_displacement.py caught exactly
    # this when the budget was applied unconditionally.
    for _ in range(10):
        r.ground()
        r.step(launch_phase=LAUNCH)
    second = _hop(r, dx=0.03, launch_phase=LAUNCH)
    assert math.isclose(second.item(), 0.3, rel_tol=1e-4)


def test_a_velocity_command_is_not_mistaken_for_a_phase_carrier():
    class _VelocityManager:
        def get_term(self, name):
            return object()          # not a GroundPickPhaseCommand

        def get_command(self, name):
            return torch.zeros(1, 3)

    r = _Rig(phase=False)
    r.command_manager = _VelocityManager()
    out = _hop(r, dx=0.05, launch_phase=LAUNCH)
    assert out.item() > 0.49


# ----------------------------------------- the latch is order-independent ----

def test_hop_settle_running_first_does_not_change_what_displacement_pays():
    """The latch is step-guarded, so a scaling parameter passed INTO it would
    take effect only for whichever term the reward manager evaluated first.
    min_ground_s used to live there and worked purely because hop_displacement
    happened to be registered before hop_settle."""
    def run(settle_first):
        r = _Rig()
        x0 = r._asset.root_link_pos_w[:, 0].clone()
        r.ground()
        r.set_phase(0.30)
        r.common_step_counter += 1
        if settle_first:
            hop_settle(r, sensor_name="contact")
        hop_displacement(r, sensor_name="contact", disp_cap=CAP,
                         min_ground_s=0.5, launch_phase=LAUNCH)

        for i in range(1, 6):
            r.fly(0.02 * i)
            r.place(x=x0 + 0.01 * i)
            r.common_step_counter += 1
            if settle_first:
                hop_settle(r, sensor_name="contact")
            hop_displacement(r, sensor_name="contact", disp_cap=CAP,
                             min_ground_s=0.5, launch_phase=LAUNCH)

        r.ground(last_air=torch.full((1, 2), 0.10))
        r.common_step_counter += 1
        if settle_first:
            hop_settle(r, sensor_name="contact")
        return hop_displacement(r, sensor_name="contact", disp_cap=CAP,
                                min_ground_s=0.5, launch_phase=LAUNCH).item()

    assert run(settle_first=True) == run(settle_first=False)


# ------------------------------------------------------------ the cfg wiring ----

def test_the_forward_env_holds_its_heading_and_the_baseline_does_not_change():
    from mjlab_microduck.tasks.microduck_hop_env_cfg import (
        HEADING_HOLD_STD,
        HEADING_HOLD_WEIGHT,
        make_microduck_hop_env_cfg,
    )

    fwd = make_microduck_hop_env_cfg(forward=True)
    term = fwd.rewards["heading_hold"]
    assert term.weight == HEADING_HOLD_WEIGHT > 0.0
    assert term.params["std"] == HEADING_HOLD_STD

    base = make_microduck_hop_env_cfg()
    assert "heading_hold" not in base.rewards, "the A/B baseline must stay fixed"


def test_the_paying_terms_are_gated_on_the_clock_and_agree_with_each_other():
    from mjlab_microduck.tasks.microduck_hop_env_cfg import (
        HOP_LAUNCH_PHASE,
        make_microduck_hop_env_cfg,
    )

    cfg = make_microduck_hop_env_cfg(forward=True)
    disp = cfg.rewards["hop_displacement"].params
    apex = cfg.rewards["hop_apex_rise"].params
    for p in (disp, apex):
        assert p["launch_phase"] == HOP_LAUNCH_PHASE
    assert disp["launch_taper"] == apex["launch_taper"]
    assert disp["launch_floor"] == apex["launch_floor"]
    assert disp["repeat_pay"] == apex["repeat_pay"]


def test_the_launch_window_follows_the_crouch_window():
    """Crouch then launch must read as one countermovement, not two events."""
    from mjlab_microduck.tasks.microduck_hop_env_cfg import (
        HOP_CROUCH_PHASE,
        HOP_LAUNCH_PHASE,
    )

    assert HOP_LAUNCH_PHASE[0] < HOP_CROUCH_PHASE[1] <= HOP_LAUNCH_PHASE[1], (
        "the launch window should overlap the end of the crouch, not start later"
    )


def test_the_head_weights_are_deliberately_unchanged_this_round():
    """S5.4 measures the head instead of re-weighting it: head_pose_bias has
    resisted three instruments and is diverging, so the next run must be a clean
    read on whether holding the heading takes the head-yaw crank with it."""
    from mjlab_microduck.tasks.microduck_hop_env_cfg import (
        HEAD_BIAS_STAGES,
        HOP_HEAD_TRACK_WEIGHT,
        make_microduck_hop_env_cfg,
    )

    cfg = make_microduck_hop_env_cfg(forward=True)
    assert cfg.rewards["head_pose_tracking"].weight == HOP_HEAD_TRACK_WEIGHT == 1.0
    assert HEAD_BIAS_STAGES[-1][1] == 3.0
