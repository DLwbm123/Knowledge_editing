#!/usr/bin/env python3
"""Minimal, method-blind diagnostic scoring for an existing M3Bench event manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


LABEL = "PROVISIONAL_AS_RUN_ORIGINAL_SEQUENCE_DIAGNOSTIC"
SUCCESS = "M3BENCH_MINIMAL_DIAGNOSTIC_METRICS_READY"
TASKS = ("T0", "T1L", "T1G", "T2L", "T2G", "T3L", "T3G", "T4L", "T4G", "T5")
PREFIXES = ("1", "50", "100", "200")
EVENT_KEY_FIELDS = (
    "mode", "checkpoint", "task_id", "formal_edit_position", "probe_id", "anchor_record_id"
)
YES_NO = {"yes", "no", "true", "false", "是", "否", "有", "无", "正确", "错误"}
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
SYNONYMS = {
    "x ray": "xray", "x-ray": "xray", "magnetic resonance imaging": "mri",
    "computed tomography": "ct", "ultrasonography": "ultrasound", "sonography": "ultrasound",
}
JUDGE_MODEL = "gpt-5.6-sol"
JUDGE_REASONING = "low"
JUDGE_TEMPERATURE = "provider_default_fixed"
NORMALIZATION_RULE = "m3bench-gpt56sol-v1"
JUDGE_PROMPT = """You are the fixed semantic correctness judge for a method-blind medical QA diagnostic.
Return only the JSON required by the supplied schema. For each item, set correct=true only when the
raw model answer semantically entails the gold/reference in the context of the question. Accept
faithful concise paraphrases and standard synonyms. Reject contradictions; wrong anatomy,
laterality, polarity, number, or modality; omission of critical content; non-responsive or empty
answers; and visibly incomplete/truncated answers. Judge each raw answer as-is. Do not compare
items and do not infer any method, group, formal position, anomaly, or expected performance.
"""


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_answer(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    for source, target in SYNONYMS.items():
        text = text.replace(source, target)
    text = re.sub(r"[^\w\u3400-\u9fff.+-]+", " ", text, flags=re.UNICODE)
    words = [NUMBER_WORDS.get(word, word) for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


def answer_tokens(value: str) -> list[str]:
    normalized = normalize_answer(value)
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+(?:\.[0-9]+)?", normalized)


def token_f1(left: str, right: str) -> float:
    a, b = Counter(answer_tokens(left)), Counter(answer_tokens(right))
    overlap = sum((a & b).values())
    if not a or not b or not overlap:
        return 0.0
    precision, recall = overlap / sum(a.values()), overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall)


def deterministic_gold(gold: str) -> bool:
    normalized = normalize_answer(gold)
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", normalized))
    if normalized in YES_NO:
        return True
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:\s*[a-z%]+)?", normalized):
        return True
    return (
        len(normalized) <= 60
        and len(answer_tokens(normalized)) <= 3
        and cjk_count <= 12
        and not re.search(r"[,，;；/、\n]", gold)
    )


def deterministic_correct(answer: str, gold: str) -> bool:
    return normalize_answer(answer) == normalize_answer(gold) or token_f1(answer, gold) >= 0.80


def event_key(row: dict) -> tuple:
    return tuple(row.get(field) for field in EVENT_KEY_FIELDS)


def validate_events(events: list[dict], expected_count: int) -> dict:
    if len(events) != expected_count:
        raise RuntimeError(f"event count mismatch: {len(events)} != {expected_count}")
    groups = sorted({str(row["anonymous_group_id"]) for row in events})
    if len(groups) != 4:
        raise RuntimeError(f"expected four anonymous groups, got {groups}")
    keys_by_group = {group: {event_key(row) for row in events if str(row["anonymous_group_id"]) == group}
                     for group in groups}
    if any(len(keys) != len([r for r in events if str(r["anonymous_group_id"]) == group])
           for group, keys in keys_by_group.items()):
        raise RuntimeError("duplicate event key within anonymous group")
    first = keys_by_group[groups[0]]
    if any(keys != first for keys in keys_by_group.values()):
        raise RuntimeError("anonymous group event keys are not symmetric")
    required = {"question", "gold_or_reference", "raw_model_answer", "task_id", "dataset"}
    for row in events:
        if not required <= row.keys() or not all(isinstance(row[field], str) for field in required):
            raise RuntimeError("required raw scoring field missing or unreadable")
    return {
        "event_count": len(events),
        "anonymous_groups": groups,
        "events_per_group": {group: len(keys_by_group[group]) for group in groups},
        "event_key_symmetry": True,
        "necessary_raw_fields_readable": True,
    }


def preflight(events_path: Path, raw_root: Path, expected_count: int) -> dict:
    events = load_jsonl(events_path)
    result = validate_events(events, expected_count)
    if not raw_root.is_dir():
        raise RuntimeError("raw root does not exist")
    readable = {}
    for group in result["anonymous_groups"]:
        row = next(item for item in events if str(item["anonymous_group_id"]) == group)
        relative = Path(row["source_raw_path"])
        target = (raw_root / relative).resolve()
        if raw_root.resolve() not in target.parents or not target.is_file():
            raise RuntimeError(f"necessary raw file unavailable for {group}")
        with target.open("rb") as handle:
            handle.read(1)
        readable[group] = True
    result.update({"raw_root_exists": True, "necessary_raw_files_readable": readable})
    return result


def semantic_payloads(events: list[dict]) -> tuple[dict[int, str], list[dict]]:
    event_to_id, seen, payloads = {}, {}, []
    for index, row in enumerate(events):
        if deterministic_gold(row["gold_or_reference"]):
            continue
        key = (
            row["question"], row["gold_or_reference"], row["raw_model_answer"],
            row["task_id"], row["dataset"],
        )
        opaque_id = seen.get(key)
        if opaque_id is None:
            opaque_id = f"judge_{len(seen) + 1:06d}"
            seen[key] = opaque_id
            payloads.append({
                "opaque_event_id": opaque_id,
                "question": row["question"],
                "gold_or_reference": row["gold_or_reference"],
                "raw_model_answer": row["raw_model_answer"],
                "task_metadata": {"task_id": row["task_id"], "dataset": row["dataset"]},
            })
        event_to_id[index] = opaque_id
    return event_to_id, payloads


def load_cache(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    if any(set(row) != {"opaque_event_id", "correct"} or not isinstance(row["correct"], bool) for row in rows):
        raise RuntimeError("invalid judge cache")
    return {row["opaque_event_id"]: row["correct"] for row in rows}


def judge_schema(path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "properties": {"opaque_event_id": {"type": "string"}, "correct": {"type": "boolean"}},
            "required": ["opaque_event_id", "correct"], "additionalProperties": False,
        }}},
        "required": ["items"], "additionalProperties": False,
    }
    path.write_text(json.dumps(schema), encoding="utf-8")


def run_judge(payloads: list[dict], cache_path: Path, codex_bin: Path, batch_size: int) -> dict[str, bool]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    schema_path = cache_path.parent / "judge_schema.json"
    judge_schema(schema_path)
    pending = [item for item in payloads if item["opaque_event_id"] not in cache]
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        expected = {item["opaque_event_id"] for item in batch}
        prompt = JUDGE_PROMPT + "\nFixed configuration:\n" + json.dumps({
            "model": JUDGE_MODEL,
            "temperature": JUDGE_TEMPERATURE,
            "normalization_rule": NORMALIZATION_RULE,
        }, ensure_ascii=False) + "\nItems:\n" + json.dumps(batch, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(prefix="audit-judge-", suffix=".json", delete=False) as tmp:
            output_path = Path(tmp.name)
        command = [
            str(codex_bin), "-a", "never", "exec", "--ignore-user-config", "--ignore-rules",
            "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-m", JUDGE_MODEL,
            "-c", f'model_reasoning_effort="{JUDGE_REASONING}"', "--output-schema", str(schema_path),
            "--output-last-message", str(output_path), "-",
        ]
        try:
            completed = subprocess.run(command, input=prompt, text=True, capture_output=True, timeout=600)
            if completed.returncode:
                raise RuntimeError(f"judge failed: {completed.stderr[-1000:]}")
            result = json.loads(output_path.read_text(encoding="utf-8"))
            rows = result.get("items", [])
            actual = {row.get("opaque_event_id") for row in rows}
            if actual != expected or any(not isinstance(row.get("correct"), bool) for row in rows):
                raise RuntimeError("judge output coverage/type mismatch")
            with cache_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    cache[row["opaque_event_id"]] = row["correct"]
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            output_path.unlink(missing_ok=True)
    expected_all = {item["opaque_event_id"] for item in payloads}
    if not expected_all <= cache.keys():
        raise RuntimeError("semantic judge coverage incomplete")
    return cache


def canonical_method(value) -> str:
    if isinstance(value, dict):
        value = value.get("method") or value.get("method_name") or value.get("name")
    normalized = re.sub(r"[^a-z]", "", str(value).casefold())
    names = {"lora": "LoRA", "grace": "GRACE", "balancedit": "BalanceEdit", "belora": "BELoRA"}
    if normalized not in names:
        raise RuntimeError(f"unknown method mapping value: {value!r}")
    return names[normalized]


def read_method_map(path: Path, groups: list[str]) -> dict[str, str]:
    obj = load_json(path)
    mapping = obj.get("group_mapping") or obj.get("anonymous_group_mapping") or obj.get("method_map")
    if not isinstance(mapping, dict):
        raise RuntimeError("method map not found")
    result = {str(group): canonical_method(value) for group, value in mapping.items() if str(group) in groups}
    if set(result) != set(groups) or set(result.values()) != {"LoRA", "GRACE", "BalanceEdit", "BELoRA"}:
        raise RuntimeError("method map coverage mismatch")
    return result


def checkpoint(row: dict) -> str:
    value = str(row["checkpoint"]).casefold()
    return "200" if value in {"final", "prefix-final", "prefix_200"} else value


def metric_cell(numerator: int, denominator: int) -> str:
    return "NA" if denominator == 0 else f"{numerator}/{denominator} ({numerator / denominator:.6f})"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(config: dict) -> dict:
    events = load_jsonl(Path(config["events_path"]))
    preflight_result = validate_events(events, int(config["expected_event_count"]))
    event_to_judge, payloads = semantic_payloads(events)
    verdicts = run_judge(
        payloads, Path(config["judge_cache_path"]), Path(config["codex_bin"]), int(config.get("batch_size", 50))
    )
    correct = []
    for index, row in enumerate(events):
        if index in event_to_judge:
            correct.append(verdicts[event_to_judge[index]])
        else:
            correct.append(deterministic_correct(row["raw_model_answer"], row["gold_or_reference"]))

    # The method map is intentionally opened only after every anonymous event verdict is frozen.
    groups = preflight_result["anonymous_groups"]
    method_map = read_method_map(Path(config["method_map_path"]), groups)
    counts = defaultdict(lambda: [0, 0])
    raw_by_key = defaultdict(dict)
    diagnostics = defaultdict(Counter)
    for row, is_correct in zip(events, correct):
        group, mode, task = str(row["anonymous_group_id"]), row["mode"], row["task_id"]
        cp = checkpoint(row)
        counts[(group, mode, cp, task)][1] += 1
        counts[(group, mode, cp, task)][0] += int(is_correct)
        raw_by_key[event_key(row)][group] = row["raw_model_answer"]
        answer = row["raw_model_answer"].strip()
        diagnostics[group]["events"] += 1
        diagnostics[group]["empty"] += int(not answer)
        diagnostics[group]["invalid_generation"] += int(bool(re.search(
            r"(?:traceback|cuda out of memory|generation error|exception:)", answer, re.I
        )))
        diagnostics[group]["truncation_suspect"] += int(
            len(answer) >= 64 and bool(re.search(r"(?:\.\.\.|[,，:：;；/\-])$", answer))
        )

    rows = []
    for group in groups:
        method = method_map[group]
        for view, mode, cp in (("single-edit", "single", "single"), ("sequential-final", "sequential", "200")):
            for task in TASKS:
                numerator, denominator = counts[(group, mode, cp, task)]
                rows.append((LABEL, view, cp, method, task, numerator, denominator,
                             "NA" if not denominator else f"{numerator / denominator:.6f}"))
        for cp in PREFIXES:
            for task in TASKS:
                numerator, denominator = counts[(group, "sequential", cp, task)]
                rows.append((LABEL, "sequential-trajectory", cp, method, task, numerator, denominator,
                             "NA" if not denominator else f"{numerator / denominator:.6f}"))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "PROVISIONAL_M3BENCH_METRICS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("result_label", "view", "prefix", "method", "metric", "numerator", "denominator", "value"))
        writer.writerows(rows)
    csv_hash = sha256(csv_path)

    trajectory_path = output_dir / "PROVISIONAL_M3BENCH_PREFIX_TRAJECTORY.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("result_label", "prefix", "method", "metric", "numerator", "denominator", "value"))
        writer.writerows((label, prefix, method, metric, numerator, denominator, value)
                         for label, view, prefix, method, metric, numerator, denominator, value in rows
                         if view == "sequential-trajectory")

    lookup = {(view, method, metric): (num, den) for _, view, _, method, metric, num, den, _ in rows
              if view in {"single-edit", "sequential-final"}}
    methods = ("LoRA", "GRACE", "BalanceEdit", "BELoRA")
    md = [
        "# Provisional M3Bench metrics", "", f"Status: `{SUCCESS}`", "",
        f"Result label: `{LABEL}`", "",
        "These implementation-diagnostic results use the original as-run target sequence. They are not the final audited/amended M3Bench result.", "",
    ]
    for title, view in (("Single-edit", "single-edit"), ("Sequential-final", "sequential-final")):
        md.extend((f"## {title}", "", "| Method | " + " | ".join(TASKS[:-1]) + " |", "|---|" + "---:|" * len(TASKS[:-1])))
        for method in methods:
            cells = [metric_cell(*lookup[(view, method, task)]) for task in TASKS[:-1]]
            md.append("| " + method + " | " + " | ".join(cells) + " |")
        md.extend(("", "T5: `NA` (no legal T5 events in the manifest).", ""))
    md.extend((
        "## Prefix trajectory", "",
        "See `PROVISIONAL_M3BENCH_PREFIX_TRAJECTORY.csv` for prefix-1, 50, 100, and final numerator/denominator/value rows.", "",
        "## Integrity", "",
        f"- parent commit: `{config['parent_commit']}`",
        f"- existing parent aggregate SHA-256: `{config['parent_aggregate_sha256']}`",
        f"- scoring code commit: `{config['scoring_code_commit']}`",
        f"- final CSV SHA-256: `{csv_hash}`", "",
        "BELoRA is a paper-spec independent reimplementation.", "",
    ))
    (output_dir / "PROVISIONAL_M3BENCH_METRICS.md").write_text("\n".join(md), encoding="utf-8")

    vectors = {method: tuple(lookup[(view, method, task)] for view in ("single-edit", "sequential-final")
                             for task in TASKS[:-1]) for method in methods}
    identical_pairs = [f"{left}={right}" for i, left in enumerate(methods) for right in methods[i + 1:]
                       if vectors[left] == vectors[right]]
    common = [answers for answers in raw_by_key.values() if len(answers) == 4]
    all_raw_identical = sum(len(set(answers.values())) == 1 for answers in common)
    zero_denominators = sorted({f"{view}:{task}" for view in ("single-edit", "sequential-final")
                                for task in TASKS if all(lookup[(view, method, task)][1] == 0 for method in methods)})
    near_zero = [method for method in methods if (lambda cell: cell[1] and cell[0] / cell[1] <= 0.05)(lookup[("single-edit", method, "T0")])]
    locality_collapse = [method for method in methods if (lambda cell: cell[1] and cell[0] / cell[1] <= 0.10)(lookup[("single-edit", method, "T1L")])]
    degradations = []
    for method in methods:
        for task in TASKS[:-1]:
            single, sequential = lookup[("single-edit", method, task)], lookup[("sequential-final", method, task)]
            if single[1] and sequential[1] and single[0] / single[1] - sequential[0] / sequential[1] >= 0.15:
                degradations.append(f"{method}:{task} ({single[0]/single[1]:.3f}->{sequential[0]/sequential[1]:.3f})")
    diagnostic_md = [
        "# Implementation diagnostic", "", f"Status: `{SUCCESS}`", "", f"Result label: `{LABEL}`", "",
        "## Findings", "",
        f"- T0 near zero (<=0.05): {', '.join(near_zero) if near_zero else 'none'}.",
        f"- Locality collapse on available T1L (<=0.10): {', '.join(locality_collapse) if locality_collapse else 'none'}.",
        f"- Single-to-sequential-final degradation >=0.15: {', '.join(degradations) if degradations else 'none'}.",
        f"- Exactly identical method metric vectors: {', '.join(identical_pairs) if identical_pairs else 'none'}.",
        f"- Zero-denominator metric families: {', '.join(zero_denominators) if zero_denominators else 'none'}.",
        f"- Same raw answer in all four anonymous groups: {all_raw_identical}/{len(common)} symmetric event keys.",
    ]
    for group in groups:
        method, item = method_map[group], diagnostics[group]
        diagnostic_md.append(
            f"- {method}: empty={item['empty']}/{item['events']}; truncation-suspect={item['truncation_suspect']}; "
            f"invalid-generation-marker={item['invalid_generation']}."
        )
    join_flag = len(common) > 0 and all_raw_identical / len(common) >= 0.95
    diagnostic_md.extend((
        f"- Method-output join error: {'POSSIBLE (>=95% four-way raw equality)' if join_flag else 'not indicated by key parity/raw-equality checks'}.", "",
        "The truncation count is a conservative suffix heuristic because the event manifest does not expose a finish reason.", "",
        "## Integrity", "",
        f"- parent commit: `{config['parent_commit']}`",
        f"- existing parent aggregate SHA-256: `{config['parent_aggregate_sha256']}`",
        f"- scoring code commit: `{config['scoring_code_commit']}`",
        f"- final CSV SHA-256: `{csv_hash}`", "",
        "Parent raw was read-only and unchanged; GPU editing was not rerun.", "",
        "These results are implementation diagnostics on the original as-run target sequence, not final audited/amended M3Bench results.", "",
    ))
    (output_dir / "IMPLEMENTATION_DIAGNOSTIC.md").write_text("\n".join(diagnostic_md), encoding="utf-8")
    return {
        "status": SUCCESS,
        "events": len(events),
        "deterministic_events": len(events) - len(event_to_judge),
        "semantic_events": len(event_to_judge),
        "unique_semantic_payloads": len(payloads),
        "csv_sha256": csv_hash,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--expected-event-count", type=int, default=38660)
    args = parser.parse_args()
    if args.preflight_only:
        if not args.events or not args.raw_root:
            parser.error("--preflight-only requires --events and --raw-root")
        print(json.dumps(preflight(args.events, args.raw_root, args.expected_event_count), sort_keys=True))
        return
    if not args.config:
        parser.error("--config is required")
    print(json.dumps(score(load_json(args.config)), sort_keys=True))


if __name__ == "__main__":
    main()
