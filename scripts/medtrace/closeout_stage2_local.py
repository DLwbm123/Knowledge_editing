#!/usr/bin/env python3
"""One bounded delivery continuation on the authenticated local host; no cron.

The remote GPU pipeline is independently detached. This process waits only for
its terminal artifact, then fetches an explicit public report allowlist and pushes.
Local sleep/network loss can delay publication, never stop the remote experiment.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request

REPORTS = (
    "STAGE1_FULL_BEHAVIOR_ADDENDUM.md", "STAGE1_FULL_BEHAVIOR.csv", "STAGE1_FULL_BEHAVIOR_AUDIT.json",
    "STAGE1_FULL_BEHAVIOR_SUPPORT_BY_EDIT.csv", "STAGE2_AMENDMENT_V2.md", "METHOD_CONFIG_LOCKS.json",
    "NEW_EPISODE_MANIFEST_PUBLIC.json", "SOURCE_EXPOSURE_SUMMARY.json", "SOURCE_PREPARATION_REPORT.md",
    "BALANCEDIT_MATCHED_DEV_REPORT.md", "BALANCEDIT_MATCHED_DEV_RESULTS.csv", "STAGE2_BEHAVIOR_BY_EDIT.csv",
    "STAGE2_BEHAVIOR_MACRO.csv", "METHOD_COSTS.csv", "NEW_EDIT_FORCED_ON_RESULTS.csv",
    "NEW_EDIT_SYSTEM_RESULTS.csv", "NEW_EDIT_CONFIRMATION_REPORT.md", "NEW_PAIRED_EDIT_EFFECTS.csv",
    "QUEUE_COMPLETION.json", "JUDGE_CLOSURE.json", "GPU_AND_TIMING.json", "RUN_COMPLETION.json", "GPT_PRO_REVIEW.md",
)
RELDIR = "reports/medtrace_stage2_20260908"
FORBIDDEN = {"question", "reference", "gold_answer", "raw_answer", "raw_base_answer", "raw_token_ids",
             "tokens", "logp", "patient_id", "image_path", "source_qid", "opaque_query_id", "private_key"}


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, timeout=120, **kwargs).strip()


def check_public(path):
    if path.stat().st_size > 10_000_000:
        raise ValueError("oversize public report")
    text = path.read_text()
    if any(term in text for term in ("/remote-home/", "/Users/", "gho_", "github_pat_", "BEGIN PRIVATE KEY")):
        raise ValueError("private path or credential-shaped value in report")
    if path.suffix == ".json":
        def inspect(value):
            if isinstance(value, dict):
                if FORBIDDEN & value.keys():
                    raise ValueError("private record fields in aggregate report")
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
        inspect(json.loads(text))


def push(repo, paths, remote, branch, message):
    if command(["git", "-C", str(repo), "branch", "--show-current"]) != branch:
        raise RuntimeError("delivery branch changed")
    if command(["git", "-C", str(repo), "diff", "--cached", "--name-only"]):
        raise RuntimeError("existing staged work: do not include unrelated changes")
    subprocess.run(["git", "-C", str(repo), "add", "--", *paths], check=True)
    if command(["git", "-C", str(repo), "diff", "--cached", "--name-only"]):
        subprocess.run(["git", "-C", str(repo), "commit", "-m", message], check=True)
    subprocess.run(["git", "-C", str(repo), "push", remote, f"HEAD:refs/heads/{branch}"], check=True, timeout=120)
    head = command(["git", "-C", str(repo), "rev-parse", "HEAD"])
    actual = command(["git", "ls-remote", remote, f"refs/heads/{branch}"]).split()[0]
    if head != actual:
        raise RuntimeError("remote branch verification failed")
    return head


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--remote-public-dir", required=True)
    p.add_argument("--research", required=True, type=Path)
    p.add_argument("--public", required=True, type=Path)
    p.add_argument("--state", required=True, type=Path)
    args = p.parse_args()
    # Inputs are operator-frozen paths, not arbitrary shell snippets.
    if any(c in args.run_root+args.remote_public_dir for c in "'\n\r\x00"):
        raise ValueError("invalid remote path")
    deadline = time.time()+25*3600
    status = None
    while time.time() < deadline:
        try:
            raw = command(["ssh", "-o", "ConnectTimeout=10", "my-gpu", f"test -f '{args.run_root}/RUN_COMPLETION.json' && head -c 100000 '{args.run_root}/RUN_COMPLETION.json'"])
            status = json.loads(raw)
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
            time.sleep(60)
    if status is None:
        raise TimeoutError("bounded delivery window ended; remote experiment is independent")
    with tempfile.TemporaryDirectory(prefix="stage2-public-closeout-") as temporary:
        temporary = Path(temporary)
        received = []
        for name in REPORTS:
            result = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=10", f"my-gpu:{args.remote_public_dir}/{name}", str(temporary / name)], timeout=120, capture_output=True)
            if result.returncode:
                continue  # Missing optional/unfinished reports remain visibly absent.
            check_public(temporary / name)
            received.append(name)
        if "RUN_COMPLETION.json" not in received:
            raise RuntimeError("terminal report not available")
        for repo in (args.research, args.public):
            destination = repo / RELDIR
            destination.mkdir(parents=True, exist_ok=True)
            for name in received:
                shutil.copyfile(temporary / name, destination / name)
        paths = [RELDIR+"/"+name for name in received]
        research_sha = push(args.research, paths, "https://github.com/DLwbm123/Knowledge_editing.git", "medtrace-stage2-20260908", "Publish bounded Stage2 result closure")
        public_sha = push(args.public, paths, "origin", "main", "Publish Stage2 aggregate results and completion ledger")
        url = f"https://raw.githubusercontent.com/DLwbm123/Knowledge_editing-public/{public_sha}/{RELDIR}/RUN_COMPLETION.json"
        with urllib.request.urlopen(url, timeout=30) as response:
            observed = json.load(response)
        if observed != status:
            raise RuntimeError("anonymous terminal report differs")
        args.state.write_text(json.dumps(dict(status="PUBLIC_DELIVERY_VERIFIED", experiment=status,
            research_sha=research_sha, public_sha=public_sha, anonymous_report_url=url), indent=2)+"\n")
        print("PUBLIC_DELIVERY_VERIFIED", public_sha, flush=True)


if __name__ == "__main__":
    main()
