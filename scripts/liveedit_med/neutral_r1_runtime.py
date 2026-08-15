#!/usr/bin/env python3
"""Environment-driven neutral-path dispatcher for long R1 GPU processes."""
from __future__ import annotations

import os
import sys

N = os.environ.get("R1_NEUTRAL_ROOT", "/dev/shm/.r1-346")
sys.path.insert(0, N + "/repo")
task = os.environ["R1_TASK"]
gpu = os.environ.get("R1_GPU", "2")
worker = os.environ.get("R1_WORKER", "0")

if task == "cache_regular":
    from scripts.liveedit_med.cache_router_r1 import main
    split = os.environ["R1_SPLIT"]
    sys.argv = ["w.py", "worker", "--source-records", N + "/s.json", "--checkpoint", N + "/c",
        "--split", split, "--physical-gpu", gpu, "--worker-index", worker, "--worker-count", "2",
        "--out", f"{N}/o/.runtime_cache/{split}_{worker}"]
elif task == "cache_hard":
    from scripts.liveedit_med.cache_router_r1_hard_negatives import main
    sys.argv = ["w.py", "worker", "--source-records", N + "/s.json", "--nearest", N + "/o/data/nearest_neighbor_audit.json",
        "--regular-manifest", N + "/o/cache/representation_cache_manifest.json", "--physical-gpu", gpu,
        "--worker-index", worker, "--worker-count", "2", "--out", f"{N}/o/.runtime_cache/hard_{worker}"]
elif task == "cache_parity":
    from scripts.liveedit_med.verify_router_r1_cache import main
    sys.argv = ["w.py", "--source-records", N + "/s.json", "--representation-manifest",
        N + "/o/cache/representation_cache_manifest.json", "--strict-checkpoint", N + "/c",
        "--physical-gpu", gpu, "--out", N + "/o/cache/cache_parity_report.json"]
elif task == "train":
    from scripts.liveedit_med.train_router_r1 import main
    sys.argv = ["w.py", "--representation-manifest", N + "/o/cache/representation_cache_manifest.json",
        "--expert-manifest", N + "/o/cache/expert_cache_manifest.json", "--hard-cache",
        N + "/o/cache/hard_negative_cache_manifest.json", "--nearest", N + "/o/data/nearest_neighbor_audit.json",
        "--membership", N + "/o/data/repository_membership_training.jsonl", "--strict-checkpoint", N + "/c",
        "--run-dir", N + "/o/training", "--physical-gpu", "2", "--suffix-physical-gpu", "3"]
elif task == "evaluate":
    from scripts.liveedit_med.evaluate_router_r1_checkpoint import main
    split = os.environ["R1_SPLIT"]; step = int(os.environ["R1_STEP"])
    sys.argv = ["w.py", "worker", "--source-records", N + "/s.json", "--split", split,
        "--representation-manifest", N + "/o/cache/representation_cache_manifest.json", "--hard-cache",
        N + "/o/cache/hard_negative_cache_manifest.json", "--nearest", N + "/o/data/nearest_neighbor_audit.json",
        "--checkpoint", f"{N}/o/training/checkpoint_{step:04d}", "--physical-gpu", gpu,
        "--worker-index", worker, "--worker-count", os.environ.get("R1_WORKER_COUNT", "1"),
        "--out", os.environ["R1_OUT"]]
elif task == "reproducibility":
    from scripts.liveedit_med.verify_router_r1_candidate import main
    step = int(os.environ["R1_STEP"])
    sys.argv = ["w.py", "worker", "--source-records", N + "/s.json", "--representation-manifest",
        N + "/o/cache/representation_cache_manifest.json", "--nearest", N + "/o/data/nearest_neighbor_audit.json",
        "--checkpoint", f"{N}/o/training/checkpoint_{step:04d}", "--physical-gpu", gpu,
        "--process-index", worker, "--out", os.environ["R1_OUT"]]
elif task == "record953_regression":
    from scripts.liveedit_med.run_router_r1_record953_regression import main
    step = int(os.environ["R1_STEP"])
    sys.argv = ["w.py", "--checkpoint", f"{N}/o/training/checkpoint_{step:04d}",
        "--source-records", N + "/s.json", "--frozen-external-split", N + "/x",
        "--strict-stage-q", N + "/q", "--physical-gpu", gpu, "--out", os.environ["R1_OUT"]]
else:
    raise RuntimeError(f"UNKNOWN_R1_NEUTRAL_TASK:{task}")

main()
