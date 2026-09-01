#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_roots(path):
    roots = json.load(open(path, encoding="utf-8"))
    if not roots or not all(isinstance(k, str) and isinstance(v, str) for k, v in roots.items()):
        raise ValueError("dataset roots must be a non-empty string map")
    return {key: Path(value).resolve(strict=True) for key, value in roots.items()}


def resolve_item(item, roots, verify_hash=True):
    dataset = item["source_dataset_identifier"]
    if dataset not in roots:
        raise ValueError(f"missing dataset root: {dataset}")
    relative = Path(item["dataset_relative_image_id_or_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("image ID must be dataset-relative")
    root = roots[dataset]
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("image path escaped the authorized root")
    if verify_hash and sha256_file(path) != item["image_sha256"]:
        raise ValueError(f"image hash mismatch: {item['review_id']}")
    return path


def verify_all(items, roots):
    failures = []
    for item in items:
        try:
            resolve_item(item, roots, verify_hash=True)
        except Exception as exc:
            failures.append({"review_id": item.get("review_id"), "error": type(exc).__name__})
    return {
        "dataset_roots_supplied": bool(roots),
        "review_items": len(items),
        "image_ids_resolvable": len(items) - len(failures),
        "image_hashes_exact": len(items) - len(failures),
        "root_escapes": sum(x["error"] == "ValueError" for x in failures),
        "image_copies": 0,
        "external_network_dependencies": 0,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def atomic_new(path, data, mode=0o444):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path.name)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_verdict(path, verdict):
    path = Path(path)
    previous = "0" * 64
    if path.exists():
        rows = load_jsonl(path)
        if rows:
            previous = rows[-1]["entry_sha256"]
    entry = dict(verdict)
    entry["prev_entry_sha256"] = previous
    entry["entry_sha256"] = hashlib.sha256(canonical(entry).encode()).hexdigest()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (canonical(entry) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return entry


def verify_chain(path):
    previous = "0" * 64
    for entry in load_jsonl(path):
        digest = entry["entry_sha256"]
        payload = dict(entry)
        del payload["entry_sha256"]
        if payload.get("prev_entry_sha256") != previous or hashlib.sha256(canonical(payload).encode()).hexdigest() != digest:
            return False
        previous = digest
    return True


def build_reviewer_b_queue(focus_ids, a_verdicts, controls, seed):
    selected = set(focus_ids)
    selected.update(row["review_id"] for row in a_verdicts if not row.get("valid") or row.get("confidence") != "high")
    selected.update(controls[:20])
    queue = sorted(selected)
    random.Random(seed).shuffle(queue)
    return queue


def snapshot_complete(components):
    required = {"clean_base", "checkpoint", "method_state", "optimizer_scheduler", "rng", "data_cursor", "formal_index", "code_config_environment"}
    return required.issubset({key for key, value in components.items() if value})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--roots", required=True)
    args = parser.parse_args()
    report = verify_all(load_jsonl(args.input), load_roots(args.roots))
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
