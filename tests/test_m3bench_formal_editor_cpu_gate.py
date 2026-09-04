import argparse
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from m3bench_repro.inference.llava_med import LlavaMedAdapter, redirect_official_vision_tower
from scripts.editor_paperspec_formal import (
    assert_authorized_device,
    assert_official_llavamed_source,
    load_single_events,
    normalized_generation_contract,
    normalized_inventory_contract,
    normalized_target_contract,
)
from scripts.m3bench_formal_editor_cpu_gate import TASKS, build, canonical_sha256, sha256


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FormalEditorCpuGateTests(unittest.TestCase):
    def test_no_model_end_to_end_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff, legacy, effect, output = root / "handoff", root / "legacy", root / "effect", root / "output"
            image = root / "image.png"
            image.write_bytes(b"synthetic-image")
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()

            inventory = []
            for index in range(11088):
                query_id = f"q{index + 1}"
                inventory.append({
                    "query_id": query_id, "dataset": "synthetic", "image_id": "image.png",
                    "image_path": str(image), "image_sha256": image_hash,
                    "question": f"question {index + 1}", "gold_answer": f"answer {index + 1}",
                    "role": "probe", "lineage": [],
                })
            legacy_rows = []
            for index in range(200):
                query = inventory[index] if index < 3 else inventory[0]
                legacy_rows.append({
                    "record_id": f"legacy-{index + 1}", "dataset": query["dataset"],
                    "question": query["question"], "gold_answer": query["gold_answer"],
                    "official_rephrase": f"rephrase {index + 1}", "image_path": str(image),
                    "relative_image_path": "image.png", "image_sha256": image_hash,
                    "formal_sequence_position": index + 1, "question_type": "synthetic",
                })
            for index in range(3):
                inventory[index]["lineage"] = [{"source_task": "T0", "relation_id": f"legacy-{index + 1}"}]

            write_jsonl(root / "inventory.jsonl", inventory)
            write_jsonl(legacy / "inputs/frozen/FORMAL_EDITOR_RECORDS_200.jsonl", legacy_rows)
            write_json(legacy / "locks/FORMAL_MODEL_AND_GENERATION_LOCK.json", {"status": "frozen"})
            method_configs = {}
            for method in ("lora", "grace", "balancedit", "belora"):
                config = {"method": method}
                method_configs[method] = {**config, "config_sha256": canonical_sha256(config)}
            write_json(effect / "locks/EFFECT_REPAIRED_METHOD_CONFIG_BUNDLE.json", {"method_configs": method_configs})
            write_json(effect / "runtime/LLAVA_MED_MODULE_INVENTORY.json", {"name": "inventory"})
            write_json(effect / "locks/EFFECT_REPAIRED_TARGET_MODULE_LOCK.json", {"target_lists_sha256": "targets"})
            write_json(effect / "locks/EFFECT_REPAIRED_MODEL_RUNTIME_LOCK.json", {
                "model_id": "llava_med_v1_5_mistral_7b", "language_block_count": 32,
                "model_class": "Model",
            })
            write_json(effect / "locks/EFFECT_REPAIRED_GENERATION_LOCK.json", {
                "batch_size": 1, "do_sample": False, "max_new_tokens": 10,
                "num_beams": 1, "temperature": None, "use_cache": True,
            })
            write_json(effect / "locks/EFFECT_REPAIRED_SOURCE_MANIFEST.json", {"commit": "effect"})

            t0 = []
            for position in range(1, 4):
                row = dict(inventory[position - 1])
                row.update({
                    "event_id": f"T0:q{position}", "edit_query_id": f"q{position}",
                    "task": "T0", "sequence_position": position, "amended_position": position,
                    "original_position": position, "method_outputs_used_for_selection": False,
                })
                t0.append(row)
            write_jsonl(handoff / "T0_FINAL_SEQUENCE.jsonl", t0)
            counts = {
                "T0": {"eligible_edit_count": 3, "eligible_probe_count": 3, "prefixes": [1, 3]},
            }
            next_query = 4
            for task in TASKS[1:]:
                edit_id = "q1" if task in {"T1L", "T1G", "T2G"} else f"q{next_query}"
                probe_id = f"q{next_query + 1}"
                next_query += 2
                write_jsonl(handoff / f"{task}_FORMAL_RECORDS.jsonl", [{
                    "event_id": f"{task}:1", "task": task, "edit_query_id": edit_id,
                    "probe_query_ids": [probe_id], "eligible": True,
                    "method_outputs_used_for_selection": False,
                }])
                counts[task] = {"eligible_edit_count": 1, "eligible_probe_count": 1}
            write_json(handoff / "FORMAL_TASK_COUNTS.json", counts)
            write_json(handoff / "FORMAL_CATALOG_MANIFEST.json", {
                "scope": "public-release-aligned T0-T4", "paper_exact_claim_permitted": False,
                "editing_methods_started": False, "method_outputs_used": False,
            })
            runtime_lock = {
                "selected_runtime": "runtime_b_official_native", "llava_med_code_commit": "source",
                "model_snapshot_sha": "model", "vision_snapshot_sha": "vision",
                "generation": {"batch_size": 1, "do_sample": False, "max_new_tokens": 10,
                               "num_beams": 1, "temperature": 0, "use_cache": True},
            }
            write_json(handoff / "CANONICAL_LLVAMED_RUNTIME_LOCK.json", runtime_lock)
            write_json(root / "official_snapshot.json", {
                "llava_med_code_commit": "source", "model_snapshot_sha": "model",
                "vision_snapshot_sha": "vision", "local_checkpoint_matches_official_snapshot": True,
                "architecture": ["Model"],
            })
            names = [path.name for path in handoff.iterdir() if path.is_file()]
            (handoff / "SHA256SUMS.txt").write_text(
                "".join(f"{sha256(handoff / name)}  {name}\n" for name in sorted(names)), encoding="utf-8"
            )

            result = build(argparse.Namespace(
                handoff_root=handoff, query_inventory=root / "inventory.jsonl",
                legacy_run_root=legacy, effect_repair_root=effect,
                official_snapshot_lock=root / "official_snapshot.json",
                output_root=output, skip_image_hashes=False,
            ))
            self.assertEqual(result["status"], "M3BENCH_FORMAL_EDITOR_INTEGRATION_CPU_GATE_PASS__GPU_APPROVAL_REQUIRED")
            self.assertEqual(result["single_event_count_per_method"], 11)
            self.assertEqual(result["single_raw_output_count_per_method"], 11)
            self.assertEqual(load_single_events(output, "T0")[0]["edit_record"]["official_rephrase"], "rephrase 1")
            self.assertEqual(result["router_positive_sources"], {
                "identity_fallback_no_frozen_rephrase": 5,
                "legacy_official_rephrase": 6,
            })
            self.assertTrue((output / "GPU_APPROVAL_REQUIRED").is_file())

    def test_runtime_contract_normalizes_disabled_temperature(self):
        source = {"batch_size": 1, "do_sample": False, "max_new_tokens": 10, "num_beams": 1, "temperature": None, "use_cache": True}
        self.assertEqual(normalized_generation_contract(source)["temperature"], 0)

    def test_runtime_topology_contract_ignores_only_host_provenance(self):
        frozen = {
            "classification": "old", "model_path": "/old", "device": "cuda:0",
            "model_class": "Model", "language_block_count": 1, "language_blocks": [0],
            "final_block_path": "model.layers.0", "final_mlp_path": "model.layers.0.mlp",
            "projector_candidates": ["mm_projector"], "projector_path": "mm_projector",
            "vision_encoder_candidates": ["vision_tower"], "vision_encoder_path": "vision_tower",
            "model_dtype": "torch.float16", "total_model_parameters": 10,
            "candidate_internal_linears": [{
                "path": "model.layers.0.mlp.up_proj", "block": 0, "projection": "up_proj",
                "in_features": 1, "out_features": 2, "bias": False,
                "parameter_count": 2, "dtype": "torch.float16", "device": "cuda:0",
            }],
        }
        current = {**frozen, "classification": "new", "model_path": "/new", "device": "cuda:7"}
        current["candidate_internal_linears"] = [{**frozen["candidate_internal_linears"][0], "device": "cuda:7"}]
        self.assertEqual(normalized_inventory_contract(frozen), normalized_inventory_contract(current))
        current["candidate_internal_linears"][0]["out_features"] = 3
        self.assertNotEqual(normalized_inventory_contract(frozen), normalized_inventory_contract(current))

    def test_target_contract_rejects_target_drift(self):
        frozen = {
            "lora": {"targets": ["a"]}, "grace": {"targets": ["b"]},
            "balancedit": {"targets": ["b"]}, "belora": {"targets": ["b", "c"]},
            "projector_excluded": "p", "vision_encoder_excluded": "v", "target_lists_sha256": "x",
        }
        current = json.loads(json.dumps(frozen))
        self.assertEqual(normalized_target_contract(frozen), normalized_target_contract(current))
        current["lora"]["targets"] = ["z"]
        self.assertNotEqual(normalized_target_contract(frozen), normalized_target_contract(current))

    def test_device_guard_accepts_only_allowed_uuid_bound_device(self):
        env = {
            "CUDA_VISIBLE_DEVICES": "3",
            "M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES": "3",
            "M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES": "2,3",
            "M3BENCH_FORMAL_EXPECTED_GPU_UUID": "GPU-test",
        }
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch("scripts.editor_paperspec_formal.torch.cuda.is_available", return_value=True), \
             mock.patch("scripts.editor_paperspec_formal.torch.cuda.device_count", return_value=1), \
             mock.patch("scripts.editor_paperspec_formal.subprocess.check_output", return_value="GPU-test\n"):
            assert_authorized_device()

    def test_invalid_loader_mode_is_rejected_without_loading(self):
        with self.assertRaises(ValueError):
            LlavaMedAdapter("model", "vision", load_mode="unknown")

    def test_official_vision_tower_redirect_is_scoped(self):
        calls = []

        class Tower:
            def __init__(self, path, *, delay_load=False):
                calls.append((path, delay_load))

        original = Tower.__init__
        with redirect_official_vision_tower(Tower, Path("/frozen/vision")):
            Tower("remote/id", delay_load=True)
        self.assertIs(Tower.__init__, original)
        self.assertEqual(calls, [("/frozen/vision", True)])

    def test_official_source_guard_runs_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"M3BENCH_EXPECTED_LLAVA_SOURCE": directory}
            replies = [
                "30697ca50b5c29a8e955c99330b259776aef27b9\n",
                "\n",
                "https://github.com/microsoft/LLaVA-Med.git\n",
            ]
            with mock.patch.dict(os.environ, env, clear=False), \
                 mock.patch("scripts.editor_paperspec_formal.subprocess.check_output", side_effect=replies), \
                 mock.patch("scripts.editor_paperspec_formal.torch.cuda.is_available") as cuda:
                self.assertEqual(assert_official_llavamed_source(), Path(directory).resolve())
                cuda.assert_not_called()

    def test_official_source_guard_rejects_dirty_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"M3BENCH_EXPECTED_LLAVA_SOURCE": directory}
            replies = [
                "30697ca50b5c29a8e955c99330b259776aef27b9\n",
                " M modified.py\n",
            ]
            with mock.patch.dict(os.environ, env, clear=False), \
                 mock.patch("scripts.editor_paperspec_formal.subprocess.check_output", side_effect=replies):
                with self.assertRaisesRegex(RuntimeError, "dirty"):
                    assert_official_llavamed_source()


if __name__ == "__main__":
    unittest.main()
