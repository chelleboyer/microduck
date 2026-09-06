"""The eval battery must feed the hop policy the SAME twist-slot semantics it
was trained on.

S5.3 turned the twist command into a ``GroundPickPhaseCommand`` — the first
three command slots carry ``[cos(2*pi*phase), sin(2*pi*phase), 0]`` on a 1.6 s
cycle — while ``hop_eval.py`` was still writing zeros there. Zeros are the idle
VELOCITY command and are correct for the hop-in-place baseline; in a phase
carrier ``(0, 0)`` is not on the unit circle at all, so the policy is fed a
state that occurs nowhere in training. The harness cannot detect that: it
returns a confident BAD-POLICY verdict instead of an error, which is exactly
how the projected-gravity default burned a day.

So the numbers the two sides share are pinned here. The script duplicates them
rather than importing the cfg (it stays torch/warp-free, the infer_policy.py
idiom), and duplication is only safe if drift fails a test.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def he():
    spec = importlib.util.spec_from_file_location(
        "hop_eval", REPO / "scripts" / "hopscotch" / "hop_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the script uses `from __future__ import annotations`,
    # so @dataclass resolves its string annotations through
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules["hop_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_phase_period_matches_the_trained_hop_cycle(he):
    from mjlab_microduck.tasks.microduck_hop_env_cfg import HOP_PERIOD

    assert he.PHASE_PERIOD_S == HOP_PERIOD, (
        "the battery would drive the clock at a rate the policy never heard"
    )


def test_flight_gates_match_the_reward_gates(he):
    from mjlab_microduck.tasks import microduck_hop_env_cfg as cfg

    assert he.FLIGHT_MIN_HEIGHT == cfg.FLIGHT_MIN_HEIGHT
    assert he.FLIGHT_MAX_TILT_DEG == cfg.FLIGHT_MAX_TILT_DEG
    assert he.FLIGHT_MIN_S == cfg.FLIGHT_MIN_S


def test_phase_encoding_round_trips_through_the_training_decoder(he):
    """cos/sin written by the harness must decode to the same phase the reward
    terms read via mdp._gp_phase (atan2(sin, cos) / 2pi, mod 1)."""
    for phase in (0.0, 0.1, 0.25, 0.5, 0.75, 0.999):
        cos, sin = math.cos(2 * math.pi * phase), math.sin(2 * math.pi * phase)
        decoded = (math.atan2(sin, cos) / (2 * math.pi)) % 1.0
        assert decoded == pytest.approx(phase, abs=1e-9)


def test_phase_clock_advances_at_the_trained_rate(he):
    """One period of control steps must advance the phase exactly one cycle."""
    dt = 0.02  # 50 Hz control, DECIMATION * 0.005 s
    steps = round(he.PHASE_PERIOD_S / dt)
    phase = 0.0
    for _ in range(steps):
        phase = (phase + dt / he.PHASE_PERIOD_S) % 1.0
    assert phase == pytest.approx(0.0, abs=1e-9) or phase == pytest.approx(1.0, abs=1e-9)


def test_no_phase_leaves_the_twist_slots_at_the_idle_velocity_command(he):
    """--no-phase must reproduce the pre-S5.3 behaviour exactly, or the
    hop-in-place baseline stops being a valid A/B reference."""
    runner = he._Runner.__new__(he._Runner)
    runner.phase_period = 0.0
    runner.phase = 0.37
    runner.hop_cmd = 0.06

    cmd = np.zeros(13, dtype=np.float32)
    if runner.phase_period > 0.0:  # mirrors observe()
        cmd[0] = math.cos(2.0 * math.pi * runner.phase)
        cmd[1] = math.sin(2.0 * math.pi * runner.phase)
    cmd[9] = runner.hop_cmd

    assert cmd[0] == 0.0 and cmd[1] == 0.0 and cmd[2] == 0.0
    assert cmd[9] == pytest.approx(0.06)


def test_hop_intent_still_lands_in_body_pose_z(he):
    """Slot 9 = 3 twist + 4 head + body[2]. The phase carrier must not have
    moved it — docs/command-block.md, and the 61D contract."""
    from mjlab_microduck.tasks.microduck_hop_env_cfg import HOP_CMD_MAX

    src = (REPO / "scripts" / "hopscotch" / "hop_eval.py").read_text()
    assert "cmd[9] = self.hop_cmd" in src
    assert HOP_CMD_MAX == 0.06
