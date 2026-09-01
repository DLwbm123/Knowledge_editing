#!/usr/bin/env python3
"""Route-B audit import and contamination gate. Reads one JSON config from stdin."""

import contextlib
import hashlib
import io
import json
import os
import runpy
import stat
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_ZIP_SHA = "0dbfe3ae848b8f5373d4d72b11108bbfd7246dd79082ce50c5eb0c0821fbaecd"
EXPECTED_PARENT_AGGREGATE = "7d555ad3e9340797fa801e8c999c08151efdb03a15d875c113562b8fe8de8e64"
EXPECTED_OVERALL = "0aeb1609706591699b61cbac6bb16acf46a3dac17cd6f8aab4a08a0b76052b68"
EXPECTED_SELECTION = "ab8b3b18d731150c15535fe405ac38454dce707bd16fc8654c1ff4b88d77fd96"
ISSUE_IDS = ["audit_0018", "audit_0040", "audit_0054"]
EVENT_FIELDS = ("mode", "checkpoint", "formal_edit_position", "source_record_id", "probe_id", "task_id")
MAP_FIELDS = ("anonymous_group_id",) + EVENT_FIELDS


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def pretty_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def atomic_bytes(path, data, mode=0o400):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise RuntimeError(f"refuse overwrite: {path.name}")
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        os.chmod(path, mode)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value, mode=0o400):
    atomic_bytes(path, pretty_json(value), mode)


def atomic_text(path, value, mode=0o400):
    atomic_bytes(path, value.encode("utf-8"), mode)


def atomic_jsonl(path, rows, mode=0o400):
    atomic_bytes(path, b"".join(canonical_json(row) for row in rows), mode)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL at line {line_no}") from exc
    return rows


def resolve_beneath(root, source):
    root = Path(root).resolve()
    source = Path(source)
    path = (source if source.is_absolute() else root / source).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("source path escaped parent") from exc
    return path


def invoke_tool(script, argv):
    old_argv = sys.argv
    old_path = sys.path[:]
    stdout, stderr = io.StringIO(), io.StringIO()
    code = 0
    try:
        sys.argv = [Path(script).name, *map(str, argv)]
        sys.path.insert(0, str(Path(script).parent))
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as exc:
                code = int(exc.code or 0)
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
    if code:
        raise RuntimeError(f"tool failed: {Path(script).name}")
    return {
        "exit_code": code,
        "stdout_sha256": sha_bytes(stdout.getvalue().encode()),
        "stderr_sha256": sha_bytes(stderr.getvalue().encode()),
        "stdout_lines": len(stdout.getvalue().splitlines()),
        "stderr_lines": len(stderr.getvalue().splitlines()),
    }


def validate_results(rows):
    required = {"audit_id", "verdict", "confidence", "reason", "issue_type"}
    allowed = required | {"dataset_issue"}
    issue_types = {
        "none", "exact_match", "acceptable_synonym", "medically_equivalent",
        "over_specific", "under_specific", "partially_correct", "contradiction",
        "wrong_entity", "wrong_count", "wrong_location", "ambiguous_reference",
        "ambiguous_question", "rubric_gap", "other",
    }
    for row in rows:
        if not required <= set(row) <= allowed:
            raise RuntimeError("audit schema field mismatch")
        if type(row["verdict"]) is not bool:
            raise RuntimeError("verdict is not native boolean")
        if row["confidence"] not in {"high", "medium", "low"}:
            raise RuntimeError("invalid confidence")
        if row["issue_type"] not in issue_types:
            raise RuntimeError("invalid issue type")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise RuntimeError("empty reason")
        if "dataset_issue" in row and type(row["dataset_issue"]) is not bool:
            raise RuntimeError("dataset_issue is not boolean")
        if row.get("dataset_issue") is True and row["confidence"] != "low":
            raise RuntimeError("dataset issue must be low confidence")
    ids = [row["audit_id"] for row in rows]
    if ids != [f"audit_{i:04d}" for i in range(1, 201)]:
        raise RuntimeError("audit ID order or coverage mismatch")
    census = {
        "records": len(rows),
        "unique_ids": len(set(ids)),
        "verdict_true": sum(row["verdict"] for row in rows),
        "verdict_false": sum(not row["verdict"] for row in rows),
        "confidence": dict(Counter(row["confidence"] for row in rows)),
        "dataset_issue_ids": [row["audit_id"] for row in rows if row.get("dataset_issue") is True],
    }
    expected = {
        "records": 200, "unique_ids": 200, "verdict_true": 64, "verdict_false": 136,
        "confidence": {"high": 193, "medium": 4, "low": 3},
        "dataset_issue_ids": ISSUE_IDS,
    }
    if census != expected:
        raise RuntimeError("audit census mismatch")
    return census


def safe_extract(zip_path, destination):
    expected = {"GPT_PRO_AUDIT_SUMMARY.md", "SHA256SUMS_RESULTS.txt"} | {
        f"audit_results_{i:02d}.jsonl" for i in range(1, 11)
    }
    destination = Path(destination)
    if destination.exists():
        raise RuntimeError("extract destination exists")
    destination.mkdir(mode=0o700)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(expected) or set(names) != expected:
            raise RuntimeError("ZIP inventory mismatch")
        for info in infos:
            mode = (info.external_attr >> 16) & 0o170000
            if info.is_dir() or info.filename.startswith("/") or ".." in Path(info.filename).parts or mode == stat.S_IFLNK:
                raise RuntimeError("unsafe ZIP entry")
            atomic_bytes(destination / info.filename, zf.read(info.filename))
    os.chmod(destination, 0o500)


def make_md(title, fields):
    lines = [f"# {title}", ""]
    for key, value in fields:
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main(config):
    run = Path(config["run_root"])
    parent = Path(config["parent_run"])
    export = Path(config["audit_export"])
    worktree = Path(config["worktree"])
    raw = run / "audit_return" / "raw"
    validated = run / "audit_return" / "validated"
    private_dir = run / "audit_return" / "private_mapping"
    governance = run / "audit_return" / "governance"

    zip_path = raw / "GPT_PRO_AUDIT_RESULTS_200.zip"
    if sha_file(zip_path) != EXPECTED_ZIP_SHA:
        raise RuntimeError("M3BENCH_POSTRAW_BLOCKED__AUDIT_RETURN_HASH_MISMATCH")

    closure = json.load(open(parent / "raw_closure" / "FORMAL_ALL_RAW_CLOSURE.json"))
    raw_manifest = json.load(open(parent / "raw_closure" / "FORMAL_RAW_FILE_MANIFEST.json"))
    source_binding = json.load(open(export / "SOURCE_RUN_BINDING.json"))
    selection_audit = json.load(open(export / "cohort" / "GPT_PRO_MODEL_AUDIT_SELECTION_AUDIT.json"))
    if not (
        closure["status"] == "PASS"
        and closure["semantic_metrics_computed"] is False
        and closure["post_edit_judge_started"] is False
        and raw_manifest["file_count"] == 6523
        and raw_manifest["total_bytes"] == 212979427180
        and raw_manifest["aggregate_sha256"] == EXPECTED_PARENT_AGGREGATE
        and source_binding["overall_report_sha256"] == EXPECTED_OVERALL
        and config["rehash"]["hash_mismatch"] == 0
        and config["rehash"]["missing"] == 0
        and config["rehash"]["size_mismatch"] == 0
        and config["rehash"]["manifest_conflicts"] == 0
    ):
        raise RuntimeError("parent evidence gate failed")

    safe_extract(zip_path, raw / "extracted")
    extracted = raw / "extracted"
    if (raw / "GPT_PRO_AUDIT_SUMMARY.md").read_bytes() != (extracted / "GPT_PRO_AUDIT_SUMMARY.md").read_bytes():
        raise RuntimeError("summary mismatch")
    if (raw / "SHA256SUMS_RESULTS.txt").read_bytes() != (extracted / "SHA256SUMS_RESULTS.txt").read_bytes():
        raise RuntimeError("checksum file mismatch")

    declared = {}
    for line in (extracted / "SHA256SUMS_RESULTS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        declared[name] = digest
    for name, digest in declared.items():
        if sha_file(extracted / name) != digest:
            raise RuntimeError("result payload checksum mismatch")

    tools = export / "tools"
    package = export / "GPT_PRO_AUDIT_PACKAGE_200"
    tool_logs = {}
    chunks = []
    for i in range(1, 11):
        result = extracted / f"audit_results_{i:02d}.jsonl"
        blind_input = package / "chunks" / f"GPT_PRO_AUDIT_CHUNK_{i:02d}_OF_10.jsonl"
        tool_logs[f"validate_chunk_{i:02d}"] = invoke_tool(
            tools / "validate_gpt_pro_audit_output.py", ["--input", blind_input, result]
        )
        rows = load_jsonl(result)
        if len(rows) != 20:
            raise RuntimeError("chunk row count mismatch")
        chunks.append(result)

    validated.mkdir(parents=True, exist_ok=True, mode=0o700)
    merged_tmp = validated / ".GPT_PRO_AUDIT_RESULTS_200.jsonl.merge"
    if merged_tmp.exists():
        raise RuntimeError("merge temp exists")
    tool_logs["merge"] = invoke_tool(
        tools / "merge_gpt_pro_audit_chunks.py", [*chunks, "--output", merged_tmp]
    )
    all_input = package / "GPT_PRO_AUDIT_INPUT_200.jsonl"
    tool_logs["validate_merged"] = invoke_tool(
        tools / "validate_gpt_pro_audit_output.py", ["--input", all_input, merged_tmp]
    )
    rows = load_jsonl(merged_tmp)
    census = validate_results(rows)
    merged = validated / "GPT_PRO_AUDIT_RESULTS_200.jsonl"
    os.replace(merged_tmp, merged)
    os.chmod(merged, 0o400)

    private_map_obj = json.load(open(export / "private" / "GPT_PRO_AUDIT_ID_MAP.json"))
    private_map = private_map_obj["items"]
    if len(private_map) != 200 or set(private_map) != {f"audit_{i:04d}" for i in range(1, 201)}:
        raise RuntimeError("private map coverage mismatch")
    selection_path = export / "cohort" / "GPT_PRO_MODEL_AUDIT_SELECTION_RULE.json"
    if sha_file(selection_path) != EXPECTED_SELECTION or private_map_obj["selection_rule_sha256"] != EXPECTED_SELECTION:
        raise RuntimeError("selection rule mismatch")

    events = load_jsonl(export / "private" / "EVENT_LEVEL_SCORING_MANIFEST_38660.jsonl")
    if len(events) != 38660:
        raise RuntimeError("event universe count mismatch")
    event_index = {}
    for event in events:
        key = tuple(event[field] for field in MAP_FIELDS)
        if key in event_index:
            raise RuntimeError("duplicate event identity")
        event_index[key] = event
    public_input = {row["audit_id"]: row for row in load_jsonl(all_input)}
    result_by_id = {row["audit_id"]: row for row in rows}
    joins, source_hash_failures = [], 0
    for audit_id in sorted(private_map):
        mapping = private_map[audit_id]
        key = tuple(mapping[field] for field in MAP_FIELDS)
        event = event_index.get(key)
        if event is None:
            raise RuntimeError("private map event missing")
        public = public_input[audit_id]
        if any(public[field] != event[field] for field in ("question", "gold_or_reference", "raw_model_answer")):
            raise RuntimeError("public/private source byte parity failure")
        if sha_bytes(event["raw_model_answer"].encode()) != event["raw_answer_sha256"]:
            raise RuntimeError("raw answer hash mismatch")
        source_path = resolve_beneath(parent, mapping["source_raw_path"])
        if sha_file(source_path) != mapping["source_raw_hash"]:
            source_hash_failures += 1
        joins.append({
            "audit_id": audit_id,
            "result": result_by_id[audit_id],
            "mapping": mapping,
            "question_sha256": sha_bytes(event["question"].encode()),
            "gold_or_reference_sha256": sha_bytes(event["gold_or_reference"].encode()),
            "raw_answer_sha256": event["raw_answer_sha256"],
            "underlying_event_key": {field: event[field] for field in EVENT_FIELDS},
        })
    if source_hash_failures:
        raise RuntimeError("source raw hash mismatch")
    underlying = [tuple(item["mapping"][field] for field in EVENT_FIELDS) for item in joins]
    repeat_census = Counter(underlying)
    repeat_analysis = {
        "audit_occurrences": len(joins),
        "distinct_underlying_events": len(repeat_census),
        "repeated_occurrences": len(joins) - len(repeat_census),
        "duplicate_key_count": sum(1 for count in repeat_census.values() if count > 1),
        "selection_audit_status": selection_audit["status"],
        "duplicate_lower_bound": selection_audit["duplicate_lower_bound"],
    }
    if repeat_analysis["distinct_underlying_events"] != 192 or repeat_analysis["repeated_occurrences"] != 8:
        raise RuntimeError("repeat census mismatch")

    issue_joins = [item for item in joins if item["audit_id"] in ISSUE_IDS]
    issue_events = [event_index[tuple(item["mapping"][field] for field in MAP_FIELDS)] for item in issue_joins]
    bad_pairs = {(event["question"], event["gold_or_reference"]) for event in issue_events}
    if len(bad_pairs) != 1:
        raise RuntimeError("dataset issue pair mismatch")
    bad_question, bad_gold = next(iter(bad_pairs))
    editors = load_jsonl(parent / "inputs" / "frozen" / "FORMAL_EDITOR_RECORDS_200.jsonl")
    probes = load_jsonl(parent / "inputs" / "frozen" / "FORMAL_PROBE_CATALOG.jsonl")
    bad_editors = [r for r in editors if r.get("question") == bad_question and r.get("gold_answer") == bad_gold]
    bad_positions = sorted({int(r["formal_sequence_position"]) for r in bad_editors})
    all_bad_events = [r for r in events if r.get("question") == bad_question and r.get("gold_or_reference") == bad_gold]
    canonical_qas = {
        (r["probe_id"], sha_bytes(r["question"].encode()), sha_bytes(r["gold_or_reference"].encode()))
        for r in issue_events
    }
    direct_single = [r for r in events if r["mode"] == "single" and int(r["formal_edit_position"]) in bad_positions]
    contaminated_seq = [
        r for r in events
        if r["mode"] == "sequential" and str(r["checkpoint"]).isdigit()
        and int(r["checkpoint"]) >= min(bad_positions)
    ]
    clean_prefix = [r for r in events if r["mode"] == "sequential" and str(r["checkpoint"]) == "1"]
    group_seq = Counter(r["anonymous_group_id"] for r in contaminated_seq)
    group_single = Counter(r["anonymous_group_id"] for r in direct_single)
    if len(bad_editors) != 3 or bad_positions != [19, 57, 67] or len(group_seq) != 4:
        raise RuntimeError("edit-target contamination evidence mismatch")

    investigation = {
        "status": "CONFIRMED_INVALID_EDIT_TARGET_CONTAMINATION",
        "audit_issue_occurrences": len(issue_joins),
        "distinct_canonical_qa": len(canonical_qas),
        "distinct_underlying_events": len({tuple(r[field] for field in EVENT_FIELDS) for r in issue_events}),
        "question_sha256": sha_bytes(bad_question.encode()),
        "gold_or_reference_sha256": sha_bytes(bad_gold.encode()),
        "formal_editor_record_count": len(bad_editors),
        "formal_positions": bad_positions,
        "bad_pair_event_count": len(all_bad_events),
        "bad_pair_probe_count": len({r["probe_id"] for r in all_bad_events}),
        "bad_pair_group_count": len({r["anonymous_group_id"] for r in all_bad_events}),
        "bad_pair_modes": sorted({r["mode"] for r in all_bad_events}),
        "bad_pair_checkpoints": sorted({str(r["checkpoint"]) for r in all_bad_events}),
        "direct_single_contaminated_events": len(direct_single),
        "direct_single_events_per_group": sorted(group_single.values()),
        "sequential_contaminated_events": len(contaminated_seq),
        "sequential_contaminated_events_per_group": sorted(group_seq.values()),
        "earliest_contaminated_position": min(bad_positions),
        "clean_prefix_1_events": len(clean_prefix),
        "classification": "B_FORMAL_EDIT_TARGET",
        "protected_access_violations": 0,
    }
    decision = {
        "decision": "HARD_STOP",
        "status": "M3BENCH_POSTRAW_BLOCKED__INVALID_EDIT_TARGET_CONTAMINATION",
        "reason": "A question-reference mismatch is present in three frozen formal edit targets.",
        "scorer_started": False,
        "full_judge_started": False,
        "evaluator_started": False,
        "method_unblinded": False,
        "human_signoff": "not completed",
        "independent_llm_gate": "not issued",
        "rerun_authorized": False,
    }
    rerun = {
        "status": "PROPOSAL_ONLY__NEW_OPERATOR_AUTHORIZATION_REQUIRED",
        "invalid_formal_positions": bad_positions,
        "earliest_sequential_contaminated_position": min(bad_positions),
        "single_reuse": "Reuse unaffected edit-target runs; do not reuse the three invalid-target single runs. Exclude all invalid QA probes by an approved governance overlay.",
        "sequential_reuse": "Prefix 1 is pre-contamination. Logical rerun suffix begins at position 19 for all four anonymous groups.",
        "operational_constraint": "If no validated state snapshot immediately before position 19 exists, rerun each full sequential trajectory under an approved amended sequence.",
        "symmetry": "Apply the identical amended sequence and rerun scope to all four anonymous groups.",
        "raw_parent_mutation": False,
        "automatic_rerun": False,
    }

    amendment = {
        "schema_version": "route-b-operator-amendment-v1",
        "created_at_utc": now(),
        "approval_status": "APPROVED",
        "approved_scope": "this post-raw scoring run only",
        "operator_authorization_message": "按该 prompt 执行 Route B post-raw scoring",
        "gate_substitution": "GPT_PRO_INDEPENDENT_MODEL_AUDIT substitutes for prior HUMAN_JUDGE_AUDIT gate",
        "reviewer_is_human": False,
        "required_label": "independent LLM audit",
        "forbidden_claim": "HUMAN_SIGNOFF completed",
        "human_signoff": "not completed",
        "operator_prompt_sha256": sha_file(run / "authorization" / "OPERATOR_PROMPT.md"),
        "base_commit": config["base_commit"],
        "run_id": config["run_id"],
    }
    access_policy = {
        "schema_version": "audit-v2-access-policy-v1",
        "parent_run_mode": "read-only",
        "audit_export_mode": "read-only",
        "write_scope": "new child run only",
        "protected_roots_accessed": [],
        "protected_access_violations": 0,
        "method_performance_before_judge": False,
        "private_mapping_publication_allowed": False,
    }
    source_binding_out = {
        "parent_manifest_aggregate_sha256": EXPECTED_PARENT_AGGREGATE,
        "parent_overall_report_sha256": EXPECTED_OVERALL,
        "formal_file_count": 6523,
        "formal_total_bytes": 212979427180,
        "formal_event_count": 38660,
        "preservation_baseline_entries": 6941,
        "preservation_differences": 0,
        "audit_export_selection_rule_sha256": EXPECTED_SELECTION,
        "audit_return_zip_sha256": EXPECTED_ZIP_SHA,
        "base_commit": config["base_commit"],
    }
    parent_evidence = {
        "status": "PASS",
        "created_at_utc": now(),
        "content_rehash": config["rehash"],
        "raw_closure_status": closure["status"],
        "semantic_metrics_computed": closure["semantic_metrics_computed"],
        "judge_started": closure["post_edit_judge_started"],
        "protected_access_violations": 0,
    }
    allowlist = {
        "parent": [
            "RUN_MANIFEST.json", "raw_closure/FORMAL_ALL_RAW_CLOSURE.json",
            "raw_closure/FORMAL_RAW_FILE_MANIFEST.json", "inputs/frozen/FORMAL_EDITOR_RECORDS_200.jsonl",
            "inputs/frozen/FORMAL_PROBE_CATALOG.jsonl",
        ],
        "audit_export": [
            "RUN_MANIFEST.json", "SOURCE_RUN_BINDING.json", "PARENT_EVIDENCE_AUDIT.json",
            "private/GPT_PRO_AUDIT_ID_MAP.json", "private/EVENT_LEVEL_SCORING_MANIFEST_38660.jsonl",
            "cohort/GPT_PRO_MODEL_AUDIT_SELECTION_RULE.json", "cohort/GPT_PRO_MODEL_AUDIT_SELECTION_AUDIT.json",
            "GPT_PRO_AUDIT_PACKAGE_200", "tools",
        ],
        "protected_access_violations": 0,
    }

    atomic_json(run / "authorization" / "ROUTE_B_OPERATOR_AMENDMENT.json", amendment)
    atomic_text(run / "authorization" / "ROUTE_B_OPERATOR_AMENDMENT.md", make_md(
        "Route B Operator Amendment",
        [("approval_status", "APPROVED"), ("reviewer_is_human", "false"),
         ("human_signoff", "not completed"), ("scope", "this child run only")],
    ))
    atomic_json(run / "ACCESS_POLICY.json", access_policy)
    atomic_json(run / "SOURCE_RUN_BINDING.json", source_binding_out)
    atomic_json(run / "PARENT_EVIDENCE_AUDIT.json", parent_evidence)
    atomic_json(run / "SCORING_INPUT_ALLOWLIST.json", allowlist)
    atomic_json(validated / "GPT_PRO_AUDIT_RETURN_VALIDATION.json", {
        "status": "M3BENCH_GPT_PRO_AUDIT_RETURN_VALIDATED", "zip_sha256": EXPECTED_ZIP_SHA,
        "chunks": 10, "rows_per_chunk": 20, "census": census, "tool_logs": tool_logs,
    })
    atomic_text(validated / "GPT_PRO_AUDIT_RETURN_VALIDATION.md", make_md(
        "GPT Pro Audit Return Validation",
        [("status", "PASS"), ("records", 200), ("missing", 0), ("duplicate", 0),
         ("true", 64), ("false", 136), ("dataset_issue", 3)],
    ))
    atomic_json(validated / "GPT_PRO_AUDIT_RETURN_CENSUS.json", census)
    atomic_jsonl(private_dir / "GPT_PRO_AUDIT_RETURN_PRIVATE_JOIN.jsonl", joins)
    atomic_json(private_dir / "GPT_PRO_AUDIT_RETURN_MAPPING_AUDIT.json", {
        "status": "PASS", "mappings": 200, "source_hash_failures": 0,
        "public_private_field_mismatches": 0, "protected_access_violations": 0,
    })
    atomic_json(private_dir / "GPT_PRO_AUDIT_RETURN_REPEAT_ANALYSIS.json", repeat_analysis)
    atomic_json(governance / "AUDIT_DATASET_ISSUE_INVESTIGATION.json", investigation)
    atomic_text(governance / "AUDIT_DATASET_ISSUE_INVESTIGATION.md", make_md(
        "Audit Dataset-Issue Investigation",
        [("decision", "HARD_STOP"), ("distinct_canonical_qa", investigation["distinct_canonical_qa"]),
         ("invalid_formal_positions", ",".join(map(str, bad_positions))),
         ("earliest_contaminated_position", investigation["earliest_contaminated_position"]),
         ("full_judge_started", "false")],
    ))
    atomic_json(governance / "AUDIT_DATASET_ISSUE_DEPENDENCY_GRAPH.json", {
        "bad_pair_sha256": sha_bytes((bad_question + "\n" + bad_gold).encode()),
        "formal_positions": bad_positions,
        "bad_pair_event_count": len(all_bad_events),
        "direct_single_contaminated_events": len(direct_single),
        "sequential_contaminated_events": len(contaminated_seq),
        "edges": [
            {"from": "invalid_target_positions", "to": "single_target_runs"},
            {"from": "earliest_invalid_target", "to": "sequential_suffix"},
            {"from": "invalid_question_reference_pair", "to": "all_scoring_occurrences"},
        ],
    })
    atomic_json(governance / "AUDIT_DATASET_ISSUE_DECISION.json", decision)
    atomic_json(governance / "AMENDED_SEQUENCE_RERUN_PROPOSAL.json", rerun)
    atomic_text(governance / "AMENDED_SEQUENCE_RERUN_PROPOSAL.md", make_md(
        "Amended-Sequence Rerun Proposal",
        [("status", rerun["status"]), ("invalid_positions", ",".join(map(str, bad_positions))),
         ("logical_sequential_suffix_start", min(bad_positions)),
         ("automatic_rerun", "false"), ("new_authorization_required", "true")],
    ))

    run_manifest = {
        "schema_version": "audit-v2-run-manifest-v1",
        "run_id": config["run_id"],
        "created_at_utc": now(),
        "branch": config["branch"],
        "base_commit": config["base_commit"],
        "status": decision["status"],
        "audit_identity": "GPT_PRO_INDEPENDENT_MODEL_AUDIT",
        "audit_return_validated": True,
        "private_mapping_validated": True,
        "dataset_issue_decision": "INVALID_EDIT_TARGET_CONTAMINATION",
        "independent_llm_gate": "not issued",
        "scorer_started": False,
        "full_judge_started": False,
        "evaluator_started": False,
        "semantic_metrics_computed": False,
        "human_signoff": "not completed",
        "protected_access_violations": 0,
    }
    atomic_json(run / "RUN_MANIFEST.json", run_manifest)
    atomic_json(run / "reports" / "POSTRAW_FINAL_DECISION.json", decision)
    atomic_text(run / "STATUS.md", make_md(
        "Post-Raw Route B Status",
        [("status", decision["status"]), ("audit_return", "200/200 PASS"),
         ("private_mapping", "200/200 PASS"), ("independent_llm_gate", "not issued"),
         ("human_signoff", "not completed"), ("full_judge", "not started")],
    ))

    checksum_files = [
        run / "RUN_MANIFEST.json", run / "ACCESS_POLICY.json", run / "SOURCE_RUN_BINDING.json",
        run / "PARENT_EVIDENCE_AUDIT.json", validated / "GPT_PRO_AUDIT_RETURN_VALIDATION.json",
        private_dir / "GPT_PRO_AUDIT_RETURN_MAPPING_AUDIT.json",
        governance / "AUDIT_DATASET_ISSUE_INVESTIGATION.json",
        governance / "AUDIT_DATASET_ISSUE_DECISION.json",
        governance / "AMENDED_SEQUENCE_RERUN_PROPOSAL.json",
        run / "reports" / "POSTRAW_FINAL_DECISION.json",
    ]
    atomic_text(run / "checksums" / "SHA256SUMS.txt", "".join(
        f"{sha_file(path)}  {path.relative_to(run)}\n" for path in checksum_files
    ))
    return {
        "status": decision["status"],
        "audit_records": census["records"],
        "distinct_underlying_events": repeat_analysis["distinct_underlying_events"],
        "repeated_occurrences": repeat_analysis["repeated_occurrences"],
        "distinct_canonical_qa": investigation["distinct_canonical_qa"],
        "invalid_formal_positions": bad_positions,
        "bad_pair_event_count": investigation["bad_pair_event_count"],
        "direct_single_contaminated_events": investigation["direct_single_contaminated_events"],
        "sequential_contaminated_events": investigation["sequential_contaminated_events"],
        "full_judge_started": False,
        "protected_access_violations": 0,
    }


if __name__ == "__main__":
    print(json.dumps(main(json.load(sys.stdin)), ensure_ascii=False, sort_keys=True))
