# Upstream pin

> Part of **MD-1**. See [`tickets/microduck-hopscotch-phase-0.md`](./tickets/microduck-hopscotch-phase-0.md).

## The pin

| | |
|---|---|
| Upstream | [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl) (Apache 2.0) |
| Remote | `upstream` |
| Pinned commit | `1e79c29c97d8b38aee9eefde77a545860ba7658e` |
| Date | 2026-08-25 |
| Subject | *license section: hardware design files are CC BY-SA-NC (same wording as reachy_mini)* |
| Merged into | `develop` on 2026-09-04, via `git merge --allow-unrelated-histories` |

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

We merged upstream's history into this existing repo rather than using GitHub's fork button, so
`origin` can be private and our planning-doc commits stay at the root of history. Practically
identical: `upstream` is a normal remote and updates are a normal merge with a real merge base.

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
- [ ] HF auth working; namespace chosen deliberately (governs repos, uv-cache bucket, **and billing**)
- [ ] Stock `Mjlab-Velocity-Flat-MicroDuck` submitted via `--hf-jobs`, run to a usable checkpoint
- [ ] `.pt` checkpoints confirmed landing in the private Hub model repo during training
- [ ] wandb confirmed streaming live
- [ ] 12h timeout behaviour understood and written down
- [ ] Walking policy plays back correctly — the known-good reference for later A/Bs
