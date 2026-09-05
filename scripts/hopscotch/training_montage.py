"""Stitch a run's training clips into ONE continuous progression video.

This is how a training montage is actually made: mjlab does not record
continuously — `--video-interval N` grabs a `--video-length`-step clip every N
env steps, so a run leaves behind a numbered pile of short mp4s. The montage is
those clips, ordered by step and concatenated, each labelled with its iteration
so the viewer can see the policy improve.

Pulls the clips straight from the run's Hub checkpoint repo (where
scripts/hf/uploader.py puts them), so no manual downloading.

Usage:
    uv run python scripts/hopscotch/training_montage.py chelleboyer/s5-forward-hop
    uv run python scripts/hopscotch/training_montage.py <repo> -o progress.mp4 --steps-per-iter 24

Requires ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def _step_of(name: str) -> int:
    m = re.search(r"step-(\d+)", name)
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="HF model repo holding the run, e.g. user/s5-forward-hop")
    ap.add_argument("-o", "--out", type=Path, default=Path("training_montage.mp4"))
    ap.add_argument("--steps-per-iter", type=int, default=24,
                    help="NUM_STEPS_PER_ENV, to label clips by iteration (default 24)")
    ap.add_argument("--label", action="store_true", default=True,
                    help="burn an iteration label into each clip")
    ap.add_argument("--no-label", dest="label", action="store_false")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")

    files = [f for f in HfApi().list_repo_files(args.repo, repo_type="model")
             if f.endswith(".mp4")]
    if not files:
        raise SystemExit(f"no .mp4 files in {args.repo} — was the run launched with --video True?")
    files.sort(key=lambda f: _step_of(Path(f).name))
    print(f"{len(files)} clips in {args.repo}")

    tmp = Path(tempfile.mkdtemp())
    clips: list[Path] = []
    for f in files:
        step = _step_of(Path(f).name)
        local = Path(hf_hub_download(args.repo, f, repo_type="model", local_dir=tmp))
        if args.label:
            it = step // max(args.steps_per_iter, 1)
            labelled = tmp / f"lab_{step:09d}.mp4"
            # drawtext needs a font; fall back to an unlabelled copy if the
            # filter is unavailable rather than losing the clip entirely.
            rc = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(local),
                 "-vf", f"drawtext=text='iteration {it}':x=12:y=12:fontsize=24:"
                        f"fontcolor=white:box=1:boxcolor=black@0.5",
                 "-c:a", "copy", str(labelled)],
                capture_output=True,
            ).returncode
            clips.append(labelled if rc == 0 and labelled.exists() else local)
            if rc != 0:
                print(f"  (label failed for step {step}, using raw clip)")
        else:
            clips.append(local)
        print(f"  step {step:>7}  ->  iteration {step // max(args.steps_per_iter,1)}")

    listing = tmp / "clips.txt"
    listing.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Re-encode rather than stream-copy: the clips are independently encoded and
    # concat demuxer copy fails on any parameter mismatch between them.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.out)],
        check=True,
    )
    print(f"\nmontage -> {args.out}  ({len(clips)} clips, "
          f"iterations {_step_of(Path(files[0]).name)//args.steps_per_iter}"
          f"-{_step_of(Path(files[-1]).name)//args.steps_per_iter})")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
