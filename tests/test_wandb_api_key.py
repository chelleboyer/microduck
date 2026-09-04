"""`_wandb_api_key` must find the key that `wandb login` actually wrote.

On Windows `wandb login` writes ~/_netrc (the Windows netrc convention) and
never ~/.netrc. The submitter only checked the dotted name, so a correctly
logged-in Windows machine hit "no API key found" and hf_jobs.submit() aborted
with exit 1 before uploading anything (2026-09-04).

These lock the lookup order — env var, then ~/.netrc, then ~/_netrc — so an
upstream merge that restores the POSIX-only path fails here instead of at
submit time.
"""

import pytest

from mjlab_microduck.hf_jobs import _wandb_api_key
from mjlab_microduck import hf_jobs

KEY = "0123456789abcdef0123456789abcdef01234567"


def _write_netrc(path, key=KEY, machine="api.wandb.ai"):
    path.write_text(f"machine {machine}\n  login user\n  password {key}\n")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty HOME with no ambient wandb credentials."""
    monkeypatch.setattr(hf_jobs.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return tmp_path


def test_reads_dot_netrc(home):
    _write_netrc(home / ".netrc")
    assert _wandb_api_key() == KEY


def test_reads_underscore_netrc(home):
    # THE REGRESSION: this is the only file `wandb login` writes on Windows.
    _write_netrc(home / "_netrc")
    assert _wandb_api_key() == KEY


def test_env_var_wins_over_netrc(home, monkeypatch):
    # submit() reports the source to the user; env must stay first.
    _write_netrc(home / ".netrc", key="from-netrc")
    monkeypatch.setenv("WANDB_API_KEY", "from-env")
    assert _wandb_api_key() == "from-env"


def test_dot_netrc_wins_over_underscore(home):
    _write_netrc(home / ".netrc", key="from-dot")
    _write_netrc(home / "_netrc", key="from-underscore")
    assert _wandb_api_key() == "from-dot"


def test_malformed_dot_netrc_falls_through_to_underscore(home):
    # A broken ~/.netrc (any machine, not just wandb) must not mask a good
    # ~/_netrc — otherwise the fallback is dead on exactly the machines it
    # exists for.
    (home / ".netrc").write_text("this-is-not-a-valid-netrc-token\n")
    _write_netrc(home / "_netrc")
    assert _wandb_api_key() == KEY


def test_none_when_no_credentials_anywhere(home):
    assert _wandb_api_key() is None


def test_none_when_netrc_has_no_wandb_machine(home):
    # A netrc holding only e.g. GitHub credentials is not a wandb login.
    _write_netrc(home / ".netrc", machine="github.com")
    assert _wandb_api_key() is None
