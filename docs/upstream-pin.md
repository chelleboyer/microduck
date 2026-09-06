# Upstream pin

> Part of **MD-1**. See [`tickets/microduck-hopscotch-phase-0.md`](./tickets/microduck-hopscotch-phase-0.md).

## The pin

| | |
|---|---|
| Upstream | [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl) (Apache 2.0) |
| Remote | `upstream` |
| Pinned commit | `29e887ecfbf5d37144759e5a9f8a176dfb83d547` |
| Date | 2026-09-02 |
| Subject | *Merge pull request #30 from peterschade/sim-body-server* |
| Merged into | `develop` on 2026-09-04 (`--allow-unrelated-histories`); re-pulled 2026-09-05 from `upstream/develop` (44 commits) |

> **NOTE the branch.** Upstream's default branch is `develop`, NOT `main` — `upstream/HEAD` points
> there and `main` has not moved since the original fork. A `git log 1e79c29..upstream/main` reports
> ZERO new commits and looks like "nothing to pull", while `develop` was 44 ahead. Always compare
> against `upstream/develop`.

## Why this commit

It was `upstream/main` HEAD at fork time, and it sits at a natural stopping point: the preceding ~10
commits are release tidy-up — README rewrite, Apache 2.0 license added, `MicroDuck` → `Microduck`
naming, removal of retired envs and of two large unused binaries. Pinning mid-release-prep would have
captured a half-renamed tree.

There are **no tags** upstream, so a SHA is the only available pin.

## Why pin at all

69 commits landed on `upstream/main` in August 2026. The architecture doc's decision is
pin-and-pull-deliberately, not track-continuously: upstream is young and moving fast, and our work is
entangled with theirs by the fork strategy.

## Clone, not GitHub fork

We merged upstream's history into this existing repo rather than using GitHub's fork button, so our
planning-doc commits stay at the root of history and the repo stands on its own rather than being
presented as a fork. Practically identical: `upstream` is a normal remote and updates are a normal
merge with a real merge base.

`origin` is **public** at [`chelleboyer/microduck`](https://github.com/chelleboyer/microduck).
Apache 2.0 is satisfied by the retained `LICENSE` and the preserved upstream history.

Reversible — if upstreaming hopscotch to Pollen Robotics becomes worthwhile (an open question in the
architecture doc), fork then and push the branch. Apache 2.0 attribution is preserved by the merged
history and the retained `LICENSE`.

## How to take an upstream update

```bash
git fetch upstream
git log --oneline <pinned-sha>..upstream/main    # read what changed first
git merge <new-sha>                              # deliberate, then update this file
```

Expect conflicts in `CLAUDE.md` only at the appended fork section at its end — that placement is
deliberate, to keep upstream's edits merging cleanly above it.

## Layout note

Upstream's tree lives at **our repo root** (`pyproject.toml`, `uv.lock`, `src/`, `scripts/`, `tests/`),
matching their README's plain `git clone && cd` quickstart. This is required, not cosmetic:
`uv run` / `uv sync` resolve against a root `pyproject.toml`, and the HF Jobs submitter snapshots
`git ls-files` **from the repo root** into `src-<stamp>.tar.gz`, so a nested layout would produce a
tarball the container cannot use.

## Conflict resolution taken at merge time

`CLAUDE.md` was the only collision (add/add).

- Upstream's playbook kept **verbatim at repo root**, where their README links it and where their
  own agent conventions expect it.
- Our hopscotch project rules moved to [`hopscotch-rules.md`](./hopscotch-rules.md).
- A short pointer section appended to the **end** of root `CLAUDE.md` linking the two.

## Verification status (rest of MD-1)

- [x] Upstream retained as a git remote, pinned, pin recorded with rationale
- [x] `uv sync` succeeds and is honest — verified on Windows 2026-09-04, exit 0
- [x] Upstream's CPU test suite passes locally: **149 passed in 55 s, no GPU**
- [x] HF auth working; namespace chosen deliberately — **`chelleboyer`** (personal). Pro is active, so
      Jobs is available; the one org `context-course` is NOT Enterprise, and Jobs for orgs requires
      Enterprise, so running under it would fail *and* bill the org. wandb entity is
      `chelleboyer-road-ranger`.
- [~] Stock `Mjlab-Velocity-Flat-MicroDuck` submitted via `--hf-jobs` — **submitted and completed**
      (job `6a9b0c79`, 64 envs / 5 iters). That is a smoke test, not a usable checkpoint; the full
      4096-env run has not been done.
- [x] `.pt` checkpoints confirmed landing in the private Hub model repo during training —
      `chelleboyer/mjlab-velocity-flat-microduck-20260904-132240`, `model_0.pt` + `model_4.pt`
      (4.7 MB each) plus `params/{agent,env}.yaml`, repo private.
- [x] wandb confirmed streaming live — run `7rl454wv` under `chelleboyer-road-ranger/mjlab_microduck`.
- [ ] 12h timeout behaviour understood and written down — **not exercised.** No run has approached it.
- [ ] Walking policy plays back correctly — the known-good reference for later A/Bs. Blocked on the
      full-length run above.

**Scheduling latency is variable and worth planning around:** three jobs queued 46 min, ~0 min, and
~12 min before starting. Compute for a 64-env/5-iter smoke test is ~5 min, so wall-clock is dominated
by scheduling, not training. This strengthens the hybrid loop (spike **S3**): prove everything
provable on CPU locally, submit only what genuinely needs a GPU.
