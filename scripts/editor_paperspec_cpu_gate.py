#!/usr/bin/env python3
"""Run all low-cost editor mechanics gates before loading LLaVA-Med."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


WORKTREE = Path(os.environ.get("M3BENCH_WORKTREE", Path(__file__).resolve().parents[1]))
RUN = Path(os.environ.get("M3BENCH_RUN_ROOT", WORKTREE / "outputs/m3bench-editor-runtime"))
GRACE = Path(os.environ.get("M3BENCH_GRACE_SOURCE", WORKTREE / "external/GRACE"))
BALANCE = Path(os.environ.get("M3BENCH_BALANCE_SOURCE", WORKTREE / "external/BalancEdit"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_readiness_module():
    path = WORKTREE / "scripts/foundation_v4_editor_readiness.py"
    spec = importlib.util.spec_from_file_location("foundation_v4_editor_readiness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load locked readiness helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    output = RUN / "cpu_tests"
    output.mkdir(parents=True, exist_ok=True)
    unit = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(WORKTREE / "tests/editor_paperspec"),
            "-v",
        ],
        cwd=WORKTREE,
        env={**os.environ, "PYTHONPATH": str(WORKTREE)},
        text=True,
        capture_output=True,
    )
    helpers = load_readiness_module()
    lora = helpers.run_lora_toy_test()
    grace = helpers.run_grace_source_test(GRACE)
    balance = helpers.run_balancedit_formula_test(
        BALANCE / "easyeditor/models/BalancEdit/balancedit.py", 0.2
    )
    checks = {
        "paper_spec_mechanics_unittest": unit.returncode == 0 and "Ran 14 tests" in unit.stderr,
        "lora_peft_toy": lora["status"] == "PASS_SYNTHETIC_MECHANICS_ONLY",
        "grace_locked_source_mechanics": grace["status"] == "PASS_SOURCE_MECHANICS_WITH_CONFIG_MISMATCH",
        "balancedit_locked_formula": balance["status"] == "PASS_FORMULA_ONLY",
        "belora_author_runtime_not_claimed": True,
    }
    report = {
        "schema_version": "m3bench-editor-paperspec-cpu-gate-v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "classification": "M3BENCH_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "unittest_stdout": unit.stdout,
        "unittest_stderr": unit.stderr,
        "lora": lora,
        "grace_source_diagnostic": grace,
        "balancedit_source_diagnostic": balance,
        "source_hashes": {
            "grace.py": sha256(GRACE / "grace/editors/grace.py"),
            "balancedit.py": sha256(BALANCE / "easyeditor/models/BalancEdit/balancedit.py"),
            "minigpt4_euc.yaml": sha256(BALANCE / "hparams/BalancEdit/minigpt4_euc.yaml"),
        },
        "next_gate": "REAL_MODEL_MODULE_INVENTORY" if all(checks.values()) else "STOP",
    }
    json_path = output / "CPU_MECHANICS_GATE.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# M3Bench Editor Paper-Spec CPU Mechanics Gate",
        "",
        f"Status: `{report['status']}`",
        "",
        "Classification: `M3BENCH_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V1`",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    md.extend(f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in checks.items())
    md.extend(
        [
            "",
            "The GRACE Euclidean source behavior was exercised only as a diagnostic. The authorized primary runtime remains cosine distance.",
            "",
            "No full LLaVA-Med checkpoint or editor output was loaded during this gate.",
        ]
    )
    md_path = output / "CPU_MECHANICS_GATE.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    for path in (json_path, md_path):
        os.chmod(path, 0o444)
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
