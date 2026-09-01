#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from launch_local_review_fast import ACTION, CONFIDENCE, ISSUE, RELATION
from resolve_authorized_images import atomic_new, build_reviewer_b_queue, canonical, flag_counts, load_jsonl, load_roots, outcome, sha256_file, verify_all, verify_chain

EXPECTED_STATUS = "M3BENCH_C0R_FAST_REVIEW_UI_READY__REVIEW_NOT_STARTED"


def order_hash(order):
    return hashlib.sha256(canonical(order).encode()).hexdigest()


def layout(run_root=None):
    run = Path(run_root).resolve() if run_root else Path(__file__).resolve().parents[4]
    review = run / "review_reconstruction"
    resolver = review / "local_console/package/resolver"
    return {
        "run": run,
        "review": review,
        "resolver": resolver,
        "package_dir": review / "local_console/package",
        "package_zip": review / "public_metadata_only/M3BENCH_FORMAL_TARGET_REVIEW_RECONSTRUCTION_200.zip",
        "session": review / "sessions/Reviewer_A/REVIEW_SESSION_MANIFEST.json",
        "roots": review / "sessions/Reviewer_A/DATASET_ROOTS.json",
        "output": review / "sessions/Reviewer_A/REVIEWER_A_OUTPUT.jsonl",
        "events": review / "sessions/Reviewer_A/SESSION_EVENTS.jsonl",
        "amendment": review / "tooling_amendments/FAST_REVIEW_UI_V1/TOOLING_AMENDMENT.json",
        "state": review / "local_console/private_state",
    }


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verdict_rows(paths):
    if not paths["output"].exists():
        return []
    if not verify_chain(paths["output"]):
        raise ValueError("output hash chain failure")
    rows = load_jsonl(paths["output"])
    if len({row["review_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate active verdicts")
    return rows


def verify_environment(paths, images=True):
    session = load(paths["session"]); amendment = load(paths["amendment"])
    input_path = paths["package_dir"] / "FORMAL_TARGET_REVIEW_INPUT_200.jsonl"
    items = load_jsonl(input_path); roots = load_roots(paths["roots"])
    checks = {
        "package_hash": sha256_file(paths["package_zip"]),
        "input_hash": sha256_file(input_path),
        "order_hash": order_hash(session["review_order"]),
        "session_manifest_hash": sha256_file(paths["session"]),
    }
    for key, value in checks.items():
        if value != amendment[key]:
            raise ValueError(f"{key} mismatch")
    if len(items) != 200 or len(session["review_order"]) != 200 or len({row["review_id"] for row in items}) != 200:
        raise ValueError("review input or order count mismatch")
    image_report = verify_all(items, roots) if images else {"status": "NOT_RUN", "image_hashes_exact": 0}
    if images and image_report["status"] != "PASS":
        raise ValueError("image verification failure")
    rows = verdict_rows(paths)
    return {
        "status": "PASS",
        **checks,
        "items": 200,
        "images_exact": image_report["image_hashes_exact"],
        "verdict_count": len(rows),
        "unique_verdicts": len({row["review_id"] for row in rows}),
        "output_chain": "PASS" if not rows or verify_chain(paths["output"]) else "FAIL",
        "server_pid_owned": pid_matches(paths, read_pid(paths)) if read_pid(paths) else True,
    }


def state_paths(paths):
    paths["state"].mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(paths["state"], 0o700)
    return paths["state"] / "review.pid", paths["state"] / "review.log"


def save_verify_state(paths, report):
    target = paths["state"] / "last_verify.json"; target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {**report, "verified_at_epoch": time.time()}; tmp = target.with_name(f".{target.name}.{os.getpid()}")
    with open(tmp, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
    os.chmod(tmp, 0o600); os.replace(tmp, target)


def fresh_verify_state(paths):
    target = paths["state"] / "last_verify.json"
    try:
        report = load(target)
        session = load(paths["session"]); input_path = paths["package_dir"] / "FORMAL_TARGET_REVIEW_INPUT_200.jsonl"
        current = {"package_hash": sha256_file(paths["package_zip"]), "input_hash": sha256_file(input_path), "order_hash": order_hash(session["review_order"]), "session_manifest_hash": sha256_file(paths["session"])}
        return report if time.time() - report["verified_at_epoch"] < 3600 and report["images_exact"] == 200 and all(report[key] == value for key, value in current.items()) else None
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None


def read_pid(paths):
    pidfile, _ = state_paths(paths)
    try:
        return int(pidfile.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def pid_matches(paths, pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        cwd = (Path("/proc") / str(pid) / "cwd").resolve()
        session_id = load(paths["session"])["session_id"]
        return cwd == paths["resolver"] and "review-console" in cmd and "launch_local_review_fast.py" in cmd and session_id in cmd
    except (OSError, FileNotFoundError):
        return False


def counts(rows):
    result = {"high": 0, "medium": 0, "low": 0, "valid": 0, "confirmed_invalid": 0, "unresolved": 0}
    for row in rows:
        result[row["confidence"]] += 1
        label = outcome(row)
        result[{"VALID": "valid", "CONFIRMED_INVALID": "confirmed_invalid", "UNRESOLVED": "unresolved"}[label]] += 1
    return result


def status(paths):
    session = load(paths["session"]); rows = verdict_rows(paths); pid = read_pid(paths)
    done = {row["review_id"] for row in rows}
    return {
        "server_running": pid_matches(paths, pid),
        "pid": pid if pid_matches(paths, pid) else None,
        "completed": len(rows),
        "remaining": 200 - len(rows),
        "flagged": sum(rid not in done for rid in flag_counts(paths["events"])),
        **counts(rows),
        "hash_chain": "PASS" if not rows or verify_chain(paths["output"]) else "FAIL",
        "last_committed_review_id": rows[-1]["review_id"] if rows else None,
        "session_id": session["session_id"],
    }


def token_url(paths, port):
    return f"http://127.0.0.1:{port}/{load(paths['session'])['session_token']}/"


def start(paths, port):
    current = status(paths)
    if current["server_running"]:
        return {"remote_port": port, "url": token_url(paths, port), "pid": current["pid"], "already_running": True}
    report = fresh_verify_state(paths)
    if report is None:
        report = verify_environment(paths, images=True); save_verify_state(paths, report)
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError("port unavailable") from exc
    pidfile, logfile = state_paths(paths); session_id = load(paths["session"])["session_id"]
    package_alias = paths["state"] / "package.zip"
    if package_alias.exists() or package_alias.is_symlink():
        if not package_alias.is_symlink() or package_alias.resolve() != paths["package_zip"].resolve():
            raise RuntimeError("package alias target mismatch")
    else:
        package_alias.symlink_to(paths["package_zip"])
    if pidfile.exists():
        pidfile.unlink()
    args = [
        "review-console", "launch_local_review_fast.py", "--package-dir", "..", "--package-zip", "../../private_state/package.zip",
        "--session-manifest", "../../../sessions/Reviewer_A/REVIEW_SESSION_MANIFEST.json", "--roots", "../../../sessions/Reviewer_A/DATASET_ROOTS.json", "--session-id", session_id, "--preverified-images", "--port", str(port),
    ]
    with open(logfile, "ab", buffering=0) as log:
        os.chmod(logfile, 0o600)
        process = subprocess.Popen(args, executable=sys.executable or os.readlink("/proc/self/exe"), cwd=paths["resolver"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_new(pidfile, f"{process.pid}\n".encode(), 0o600)
    url = token_url(paths, port)
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("review server exited during startup")
        try:
            with urllib.request.urlopen(url + "health", timeout=.2) as response:
                if response.status == 200:
                    return {"remote_port": port, "url": url, "pid": process.pid, "already_running": False}
        except Exception:
            time.sleep(.1)
    stop(paths)
    raise RuntimeError("review server health timeout")


def stop(paths):
    pid = read_pid(paths)
    if not pid_matches(paths, pid):
        raise RuntimeError("PID ownership check failed")
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        try:
            os.kill(pid, 0)
            time.sleep(.1)
        except OSError:
            break
    else:
        raise RuntimeError("review server did not stop after SIGTERM")
    pidfile, _ = state_paths(paths)
    pidfile.unlink(missing_ok=True)
    return {"stopped": True, "pid": pid}


def validate_frozen_row(row):
    required = {"review_id", "valid", "confidence", "relation", "issue_type", "recommended_action", "reason", "created_at_utc", "prev_entry_sha256", "entry_sha256"}
    if not required.issubset(row) or not isinstance(row["valid"], bool) or row["confidence"] not in CONFIDENCE or row["relation"] not in RELATION or row["issue_type"] not in ISSUE or row["recommended_action"] not in ACTION:
        raise ValueError("invalid frozen verdict schema")
    minimum = 12 if not row["valid"] or outcome(row) == "UNRESOLVED" else 1
    if not minimum <= len(row["reason"].strip()) <= 2000:
        raise ValueError("invalid frozen verdict reason")


def validate_freeze_rows(rows):
    if len(rows) != 200 or len({row.get("review_id") for row in rows}) != 200:
        raise RuntimeError("freeze requires 200 unique verdicts")
    for row in rows:
        validate_frozen_row(row)


def freeze(paths):
    current = status(paths)
    if current["completed"] != 200:
        raise RuntimeError("freeze requires 200 unique verdicts")
    if current["server_running"]:
        stop(paths)
    report = verify_environment(paths, images=True); rows = verdict_rows(paths)
    validate_freeze_rows(rows)
    output_hash = sha256_file(paths["output"]); session_dir = paths["session"].parent
    manifest = {"status": "PASS", "records": 200, "unique_review_ids": 200, "missing": 0, "duplicates": 0, "output_sha256": output_hash, "unresolved": counts(rows)["unresolved"], "package_hash": report["package_hash"], "input_hash": report["input_hash"], "order_hash": report["order_hash"], "image_export_count": 0}
    atomic_new(session_dir / "REVIEWER_A_FREEZE_MANIFEST.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    atomic_new(session_dir / "REVIEWER_A_FREEZE_REPORT.md", (f"# Reviewer A freeze\n\nStatus: PASS\n\n- Records: 200\n- Unresolved: {manifest['unresolved']}\n- Image exports: 0\n").encode())
    atomic_new(session_dir / "REVIEWER_A_OUTPUT.sha256", f"{output_hash}  REVIEWER_A_OUTPUT.jsonl\n".encode())
    return manifest


def build_b(paths, selection_input=None):
    session_dir = paths["session"].parent
    if not (session_dir / "REVIEWER_A_FREEZE_MANIFEST.json").exists():
        raise RuntimeError("Reviewer A must be frozen first")
    source = Path(selection_input) if selection_input else paths["review"] / "private/REVIEWER_B_SELECTION_INPUT.json"
    if not source.exists():
        raise RuntimeError("private Reviewer B selection input is not available")
    selection = load(source); rows = verdict_rows(paths); seed = secrets.randbits(64)
    queue = build_reviewer_b_queue(selection["focus_ids"], rows, selection["control_ids"], seed)
    target = paths["review"] / "sessions/Reviewer_B_PENDING"; target.mkdir(parents=True, exist_ok=False)
    atomic_new(target / "REVIEW_QUEUE.jsonl", "".join(json.dumps({"review_id": rid}, sort_keys=True) + "\n" for rid in queue).encode())
    atomic_new(target / "SESSION_MANIFEST.json", (json.dumps({"status": "READY_NOT_STARTED", "count": len(queue), "selection_reasons_exposed": False, "image_export_count": 0}, indent=2, sort_keys=True) + "\n").encode())
    return {"status": "READY_NOT_STARTED", "count": len(queue), "selection_reasons_exposed": False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    start_parser = commands.add_parser("start"); start_parser.add_argument("--port", type=int, default=8765)
    commands.add_parser("status"); url_parser = commands.add_parser("url"); url_parser.add_argument("--port", type=int, default=8765)
    commands.add_parser("stop"); commands.add_parser("freeze-reviewer-a")
    b_parser = commands.add_parser("build-reviewer-b"); b_parser.add_argument("--selection-input")
    args = parser.parse_args(); paths = layout(args.run_root)
    if args.command == "verify":
        result = verify_environment(paths, images=True); save_verify_state(paths, result)
    elif args.command == "start": result = start(paths, args.port)
    elif args.command == "status": result = status(paths)
    elif args.command == "url":
        if not status(paths)["server_running"]: raise RuntimeError("server is not running")
        print(token_url(paths, args.port)); return
    elif args.command == "stop": result = stop(paths)
    elif args.command == "freeze-reviewer-a": result = freeze(paths)
    else: result = build_b(paths, args.selection_input)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
