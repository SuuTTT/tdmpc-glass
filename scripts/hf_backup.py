#!/usr/bin/env python3
"""Push milestone checkpoints to HuggingFace so worker teardown never loses them.

Usage:
  # one-shot upload of a file or dir to a logical path in the repo
  python hf_backup.py push <local_path> <repo_subpath> [--msg "..."]

  # watch a checkpoints dir and upload any new/updated *.pkl every N seconds
  python hf_backup.py watch <ckpt_dir> <repo_subpath_prefix> [--interval 600]

Repo: $HF_MILESTONE_REPO (default Dannibal/tdmpc-glass-milestones), private model repo.
Token: $HF_TOKEN.  Idempotent; safe to re-run.
"""
import os, sys, time, argparse, hashlib
from pathlib import Path
from huggingface_hub import HfApi

REPO = os.environ.get("HF_MILESTONE_REPO", "Dannibal/tdmpc-glass-milestones")
TOKEN = os.environ.get("HF_TOKEN")

def api():
    if not TOKEN:
        sys.exit("HF_TOKEN not set")
    a = HfApi(token=TOKEN)
    a.create_repo(repo_id=REPO, repo_type="model", private=True, exist_ok=True)
    return a

def push(local, subpath, msg=None):
    a = api(); p = Path(local)
    if p.is_dir():
        a.upload_folder(folder_path=str(p), path_in_repo=subpath, repo_id=REPO,
                        repo_type="model", commit_message=msg or f"backup {subpath}")
    else:
        a.upload_file(path_or_fileobj=str(p), path_in_repo=subpath, repo_id=REPO,
                      repo_type="model", commit_message=msg or f"backup {subpath}")
    print(f"[hf_backup] pushed {local} -> {REPO}/{subpath}", flush=True)

def _sig(p):
    st = p.stat(); return f"{st.st_size}-{int(st.st_mtime)}"

def watch(ckpt_dir, prefix, interval):
    a = api(); seen = {}
    print(f"[hf_backup] watching {ckpt_dir} -> {REPO}/{prefix} every {interval}s", flush=True)
    while True:
        for p in sorted(Path(ckpt_dir).rglob("*.pkl")):
            sig = _sig(p)
            if seen.get(str(p)) == sig:
                continue
            rel = p.relative_to(ckpt_dir)
            try:
                a.upload_file(path_or_fileobj=str(p), path_in_repo=f"{prefix}/{rel}",
                              repo_id=REPO, repo_type="model",
                              commit_message=f"milestone {prefix}/{rel}")
                seen[str(p)] = sig
                print(f"[hf_backup] {p} -> {prefix}/{rel}", flush=True)
            except Exception as e:
                print(f"[hf_backup] FAIL {p}: {e}", flush=True)
        time.sleep(interval)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("push"); pp.add_argument("local"); pp.add_argument("subpath"); pp.add_argument("--msg")
    wp = sub.add_parser("watch"); wp.add_argument("ckpt_dir"); wp.add_argument("prefix"); wp.add_argument("--interval", type=int, default=600)
    args = ap.parse_args()
    if args.cmd == "push": push(args.local, args.subpath, args.msg)
    else: watch(args.ckpt_dir, args.prefix, args.interval)
