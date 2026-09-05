"""Checkpoint + video uploader run inside an HF Job.

Watches `logs/rsl_rl/**/` for `model_*.pt`, the dumped `params/*`, and any
training videos, and uploads new/updated files to the target HF Model repo.
Designed to be `nohup uv run`-launched from the job bootstrap, with auth coming
from the HF_TOKEN secret injected by `hf jobs run`.

Videos matter as much as checkpoints: CLAUDE.md is explicit that "sim metrics
can pass while the video fails the human eye", and the dev machine has no CUDA,
so nothing renders locally — footage recovered from the job is the ONLY way to
watch a policy move. Enable it with `--video True` on the train command.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd

# A file untouched for this long is treated as finished being written. mjlab
# writes a video progressively across its frames, so without this guard a clip
# caught mid-write uploads truncated, then uploads again when its mtime moves —
# a wasted commit and a corrupt artifact in between.
SETTLE_S = 5.0


def main() -> int:
    repo_id = os.environ.get("CKPT_REPO")
    if not repo_id:
        print("[uploader] CKPT_REPO not set, exiting", flush=True)
        return 1

    poll_interval = float(os.environ.get("CKPT_POLL_INTERVAL", "60"))
    root = Path(os.environ.get("CKPT_ROOT", "logs/rsl_rl"))

    one_shot = os.environ.get("CKPT_ONE_SHOT") == "1"

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    mode = "one-shot" if one_shot else f"every {poll_interval}s"
    print(f"[uploader] watching {root} -> {repo_id} ({mode})", flush=True)

    sent: dict[Path, float] = {}
    while True:
        try:
            files = list(root.glob("**/model_*.pt"))
            # also pick up the dumped configs once
            files += [p for p in root.glob("**/params/*.yaml")]
            files += [p for p in root.glob("**/params/*.json")]
            # training videos (mjlab writes <log_dir>/videos/train/*.mp4 when
            # the run is launched with --video True)
            files += [p for p in root.glob("**/videos/**/*.mp4")]

            now = time.time()
            to_upload: list[CommitOperationAdd] = []
            for f in files:
                try:
                    stat = f.stat()
                except FileNotFoundError:
                    continue
                mtime = stat.st_mtime
                if sent.get(f) == mtime:
                    continue
                # Still being written? Leave it for the next poll. Skipped in
                # one-shot mode (the final upload as the job exits), where
                # there IS no next poll and a partial file beats no file.
                if not one_shot and (now - mtime) < SETTLE_S:
                    continue
                # use path-in-repo relative to logs/rsl_rl so the repo mirrors run dirs
                rel = f.relative_to(root)
                to_upload.append(
                    CommitOperationAdd(path_in_repo=str(rel), path_or_fileobj=str(f))
                )
                sent[f] = mtime

            if to_upload:
                msg = f"upload {len(to_upload)} file(s)"
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="model",
                    operations=to_upload,
                    commit_message=msg,
                )
                print(f"[uploader] pushed {len(to_upload)} file(s)", flush=True)
        except Exception as e:
            print(f"[uploader] error: {e}", flush=True)

        if one_shot:
            return 0
        time.sleep(poll_interval)


if __name__ == "__main__":
    sys.exit(main())
