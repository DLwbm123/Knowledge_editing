#!/usr/bin/env python3
"""Summarize DSCA edit-step phase profile JSONL logs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dsca_edit_step_profile.jsonl")
    parser.add_argument("--profile-jsonl", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize(records: List[Dict[str, Any]]) -> str:
    done = [row for row in records if row.get("event") == "done" and row.get("elapsed_sec") is not None]
    by_step: Dict[Any, float] = defaultdict(float)
    by_phase: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for row in done:
        step = row.get("step")
        phase = str(row.get("phase"))
        elapsed = float(row.get("elapsed_sec", 0.0))
        by_step[step] += elapsed
        by_phase[phase] += elapsed
        counts[phase] += 1

    slowest = sorted(done, key=lambda row: float(row.get("elapsed_sec", 0.0)), reverse=True)[:20]
    phase_totals = sorted(by_phase.items(), key=lambda item: item[1], reverse=True)
    bottleneck = phase_totals[0][0] if phase_totals else "<none>"

    lines: List[str] = []
    lines.append("DSCA Profile Summary")
    lines.append("====================")
    lines.append("")
    lines.append(f"records: {len(records)}")
    lines.append(f"completed phases: {len(done)}")
    lines.append(f"bottleneck phase: {bottleneck}")
    lines.append("")
    lines.append("Top 20 slowest phase events:")
    for row in slowest:
        lines.append(
            f"- step={row.get('step')} phase={row.get('phase')} elapsed={float(row.get('elapsed_sec', 0.0)):.4f}s"
        )
    lines.append("")
    lines.append("Per-step total timed phase seconds:")
    for step, elapsed in sorted(by_step.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9):
        lines.append(f"- step={step}: {elapsed:.4f}s")
    lines.append("")
    lines.append("Phase totals:")
    for phase, elapsed in phase_totals:
        mean = elapsed / max(counts[phase], 1)
        lines.append(f"- {phase}: total={elapsed:.4f}s count={counts[phase]} mean={mean:.4f}s")
    lines.append("")
    lines.append("Suggested diagnosis:")
    if bottleneck in {"residualized_pca", "initialize_basis_if_ready", "refine_subspaces"}:
        lines.append("- PCA/basis initialization dominates; inspect repeated basis recomputation and PCA buffer shapes.")
    elif bottleneck == "backward":
        lines.append("- Backward dominates; inspect graph size, enabled DSAM residual path, and loss ablations.")
    elif bottleneck.endswith("forward_dsca_enabled") or "forward" in bottleneck:
        lines.append("- Forward dominates; inspect DSCA hook routing/residual work and backbone sequence lengths.")
    elif bottleneck == "<none>":
        lines.append("- No completed phase events found; inspect the last `event=start` record for the stuck phase.")
    else:
        lines.append("- Inspect the slowest phase and its neighboring start/done records in the JSONL log.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    path = Path(args.profile_jsonl)
    text = summarize(load_records(path))
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
