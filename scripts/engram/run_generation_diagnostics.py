#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from easyeditor.editors.multimodal_editor import MultimodalEditor  # noqa: E402
from easyeditor.models.engram import EngramMultimodalHparams  # noqa: E402
from easyeditor.models.engram.bank import EngramBank  # noqa: E402


def _load_records(path: Path) -> List[Dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Expected a non-empty JSON list in {path}.")
    return records


def _resolve_image(root: Path, rel_path: str) -> str:
    path = Path(rel_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists() and root.name == "images":
        rel = Path(rel_path)
        if rel.parts and rel.parts[0] == "images":
            path = root / Path(*rel.parts[1:])
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def _diagnostic_items(record: Dict, image_root: Path, case_index: int) -> List[Dict]:
    items = [
        {
            "name": "target",
            "case_index": case_index,
            "record_id": record.get("id"),
            "prompt": record["src"],
            "image": _resolve_image(image_root, record["image"]),
            "expected_new_answer": record.get("alt"),
            "old_answer": record.get("pred"),
        }
    ]
    if record.get("m_loc_q") and record.get("m_loc"):
        items.append(
            {
                "name": "multimodal_locality",
                "case_index": case_index,
                "record_id": record.get("id"),
                "prompt": record["m_loc_q"],
                "image": _resolve_image(image_root, record["m_loc"]),
                "expected_answer": record.get("m_loc_a"),
            }
        )
    return items


def _special_token_ids(tokenizer) -> set[int]:
    ids = {
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        getattr(tokenizer, "unk_token_id", None),
    }
    return {int(value) for value in ids if value is not None}


def _generate_llava_med(
    wrapper,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
) -> Dict:
    image = Image.open(image_path).convert("RGB")
    image_tensor = wrapper.process_images([image], wrapper.image_processor, wrapper.llava_model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(wrapper.lm_device, dtype=wrapper.dtype)

    prompt_text = wrapper._conversation_prompt(prompt, None)
    input_ids = wrapper.tokenizer_image_token(
        prompt_text,
        wrapper.tokenizer,
        wrapper.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(wrapper.lm_device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=wrapper.lm_device)

    generate_kwargs = {
        "images": image_tensor,
        "attention_mask": attention_mask,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": wrapper.tokenizer.eos_token_id,
    }
    if min_new_tokens is not None:
        generate_kwargs["min_new_tokens"] = min_new_tokens
    with torch.inference_mode():
        output_ids = wrapper.llava_model.generate(input_ids, **generate_kwargs)
    new_tokens = output_ids[0, input_ids.shape[1] :]
    generated_ids = [int(token_id) for token_id in new_tokens.detach().cpu().tolist()]
    decoded_raw = wrapper.tokenizer.decode(new_tokens, skip_special_tokens=False)
    decoded_skip_special = wrapper.tokenizer.decode(new_tokens, skip_special_tokens=True)
    decoded_stripped = decoded_skip_special.strip()
    special_ids = _special_token_ids(wrapper.tokenizer)
    only_special = bool(generated_ids) and all(token_id in special_ids for token_id in generated_ids)
    eos_id = wrapper.tokenizer.eos_token_id
    if generated_ids and eos_id is not None and generated_ids[0] == int(eos_id):
        stop_reason = "immediate_eos"
    elif eos_id is not None and int(eos_id) in generated_ids:
        stop_reason = "eos"
    elif len(generated_ids) >= max_new_tokens:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "unknown"
    generation_empty = decoded_stripped == ""
    reason_guess = None
    if generation_empty:
        if only_special or stop_reason in {"immediate_eos", "eos"}:
            reason_guess = "immediate EOS or only special tokens"
        else:
            reason_guess = "template mismatch or synthetic prompt unsuitable"
    return {
        "prompt_text": prompt_text,
        "input_token_ids": [int(token_id) for token_id in input_ids[0].detach().cpu().tolist()],
        "output_token_ids": [int(token_id) for token_id in output_ids[0].detach().cpu().tolist()],
        "generated_token_ids": generated_ids,
        "decoded_raw": decoded_raw,
        "decoded_skip_special": decoded_skip_special,
        "decoded_stripped": decoded_stripped,
        "stop_reason": stop_reason,
        "eos_token_id": int(eos_id) if eos_id is not None else None,
        "generated_only_eos_or_special": only_special,
        "generation_empty": generation_empty,
        "reason_guess": reason_guess,
    }


def _latest_edit_id(bank: EngramBank) -> str:
    edits = bank.list_edits()
    if not edits:
        raise RuntimeError(f"No edits in ENGRAM bank {bank.root}.")
    return edits[-1]["edit_id"]


def _edit_ids_for_records(
    bank: EngramBank,
    records: List[Dict],
    *,
    allow_positional_matching: bool = False,
) -> tuple[List[str], Dict]:
    edits = bank.list_edits()
    if not edits:
        raise RuntimeError(f"No edits in ENGRAM bank {bank.root}.")
    if len(edits) >= len(records):
        return bank.match_edit_ids_to_records(records, allow_positional_matching=allow_positional_matching)
    return [edits[-1]["edit_id"] for _ in range(len(records))], {
        "mode": "last_edit_repeated_fallback",
        "reason": f"bank has {len(edits)} edits for {len(records)} records",
    }


def _snapshot_modules(model: torch.nn.Module, module_names: List[str]) -> Dict[str, Dict[str, torch.Tensor | None]]:
    modules = dict(model.named_modules())
    snapshots: Dict[str, Dict[str, torch.Tensor | None]] = {}
    for name in module_names:
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"Bank module not found or not Linear: {name}")
        snapshots[name] = {
            "weight": module.weight.detach().clone().cpu(),
            "bias": module.bias.detach().clone().cpu() if module.bias is not None else None,
        }
    return snapshots


def _restore_modules(model: torch.nn.Module, snapshots: Dict[str, Dict[str, torch.Tensor | None]]) -> None:
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, tensors in snapshots.items():
            module = modules[name]
            module.weight.copy_(tensors["weight"].to(module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None and tensors["bias"] is not None:
                module.bias.copy_(tensors["bias"].to(module.bias.device, dtype=module.bias.dtype))


def _max_snapshot_diff(model: torch.nn.Module, snapshots: Dict[str, Dict[str, torch.Tensor | None]]) -> float:
    modules = dict(model.named_modules())
    diffs = []
    for name, tensors in snapshots.items():
        module = modules[name]
        diffs.append((module.weight.detach().cpu() - tensors["weight"]).abs().max().item())
        if module.bias is not None and tensors["bias"] is not None:
            diffs.append((module.bias.detach().cpu() - tensors["bias"]).abs().max().item())
    return float(max(diffs) if diffs else 0.0)


def _parse_bank_specs(values: List[str]) -> List[Dict]:
    specs = []
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"Bank spec must be alpha=path, got {value!r}.")
        alpha, path = value.split("=", 1)
        specs.append({"alpha": float(alpha), "bank_dir": path})
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Save deterministic before/after ENGRAM generations.")
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--bank", action="append", required=True, help="Repeated alpha=bank_dir entries.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--rollback-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--allow-positional-matching",
        action="store_true",
        help="Allow legacy bank positional edit/record matching when record_id metadata is missing.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = _load_records(Path(args.data_file))
    image_root = Path(args.image_root)
    items_by_case = [_diagnostic_items(record, image_root, idx) for idx, record in enumerate(records)]
    bank_specs = _parse_bank_specs(args.bank)

    hparams = EngramMultimodalHparams.from_hparams(args.hparams)
    hparams.device = int(args.device) if str(args.device).isdigit() else args.device
    editor = MultimodalEditor.from_hparams(hparams)
    wrapper = editor.model

    banks = []
    union_modules: List[str] = []
    for spec in bank_specs:
        bank = EngramBank(spec["bank_dir"])
        edit_ids, edit_record_matching = _edit_ids_for_records(
            bank,
            records,
            allow_positional_matching=args.allow_positional_matching,
        )
        update_names = []
        for edit_id in edit_ids:
            update_names.extend(list(bank.load_edit(edit_id)["updates"].keys()))
        for name in update_names:
            if name not in union_modules:
                union_modules.append(name)
        banks.append(
            {
                **spec,
                "bank": bank,
                "edit_ids": edit_ids,
                "module_names": sorted(set(update_names)),
                "edit_record_matching": edit_record_matching,
            }
        )

    snapshots = _snapshot_modules(wrapper, union_modules)

    def generate_all(case_index: int) -> Dict[str, Dict]:
        return {
            item["name"]: _generate_llava_med(
                wrapper,
                item["prompt"],
                item["image"],
                args.max_new_tokens,
                args.min_new_tokens,
            )
            for item in items_by_case[case_index]
        }

    baseline = [generate_all(case_index) for case_index in range(len(records))]
    results = []
    for bank_item in banks:
        case_results = []
        for case_index, edit_id in enumerate(bank_item["edit_ids"]):
            _restore_modules(wrapper, snapshots)
            bank_item["bank"].apply_edit(wrapper, edit_id)
            after_edit = generate_all(case_index)
            bank_item["bank"].rollback_edit(wrapper, edit_id)
            after_rollback = generate_all(case_index)
            max_diff = _max_snapshot_diff(wrapper, snapshots)
            case_results.append(
                {
                    "case_index": case_index,
                    "record_id": records[case_index].get("id"),
                    "edit_id": edit_id,
                    "edit_record_matching": bank_item["edit_record_matching"],
                    "rollback_max_abs_diff": max_diff,
                    "rollback_within_tolerance": max_diff <= args.rollback_tolerance,
                    "generations": {
                        name: {
                            "before": baseline[case_index][name],
                            "after_edit": after_edit[name],
                            "after_rollback": after_rollback[name],
                        }
                        for name in baseline[case_index]
                    }
                },
            )
        results.append(
            {
                "alpha": bank_item["alpha"],
                "bank_dir": str(bank_item["bank"].root),
                "edit_ids": bank_item["edit_ids"],
                "edit_record_matching": bank_item["edit_record_matching"],
                "module_names": bank_item["module_names"],
                "case_results": case_results,
                "rollback_within_tolerance": all(row["rollback_within_tolerance"] for row in case_results),
            }
        )

    _restore_modules(wrapper, snapshots)
    output = {
        "status": "pass" if all(row["rollback_within_tolerance"] for row in results) else "fail",
        "data_file": str(args.data_file),
        "image_root": str(args.image_root),
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": False,
        "temperature": None,
        "logprob_status": "unavailable_current_evaluator",
        "items_by_case": items_by_case,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "status": output["status"], "num_alpha": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
