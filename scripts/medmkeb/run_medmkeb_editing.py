#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medmkeb_editing.adapter import build_edit_batch, build_vlkeb_records, unresolved_image_count  # noqa: E402
from medmkeb_editing.asset_resolver import create_smoke_dataset  # noqa: E402
from medmkeb_editing.metrics import write_metrics_jsonl, write_summary_csv  # noqa: E402
from medmkeb_editing.paths import ensure_layout, get_paths, set_cache_env, utc_timestamp, write_json, write_simple_yaml  # noqa: E402


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value}")


def default_data_file(paths) -> Path:
    preferred = paths.raw / "eval_data_threehop_final.json"
    if preferred.exists():
        return preferred
    for candidate in sorted(paths.raw.glob("*.json")):
        if not candidate.name.startswith("._"):
            return candidate
    repo_candidate = paths.medmkeb_repo / "data" / "eval_data_threehop_final.json"
    return repo_candidate


def make_output_dir(paths, dry_run: bool) -> Path:
    tag = "dry_run" if dry_run else "run"
    base = paths.outputs / f"{utc_timestamp()}_{tag}"
    for counter in range(1000):
        output_dir = base if counter == 0 else paths.outputs / f"{base.name}_{counter}"
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            return output_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique output directory under {paths.outputs}")


def env_info(paths) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "root": str(paths.root),
        "HF_HOME": os.environ.get("HF_HOME"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        "TORCH_HOME": os.environ.get("TORCH_HOME"),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_status"] = f"unavailable: {type(exc).__name__}: {exc}"
        info["cuda_available"] = False
    return info


def write_run_artifacts(output_dir: Path, args: argparse.Namespace, paths, batch: Dict[str, Any]) -> None:
    write_json(output_dir / "env.json", env_info(paths))
    write_simple_yaml(
        output_dir / "run_config.yaml",
        {
            "args": vars(args),
            "source_file": batch.get("source_file"),
            "selected_records": batch.get("selected_records"),
            "unresolved_records_with_images": unresolved_image_count(batch),
        },
    )
    write_json(
        output_dir / "dry_run_batch_summary.json",
        {
            "source_file": batch.get("source_file"),
            "selected_records": batch.get("selected_records"),
            "total_records_in_file": batch.get("total_records_in_file"),
            "prompts_example": batch["prompts"][:2],
            "targets_example": batch["targets"][:2],
            "image_example": batch["image"][:2],
            "rephrase_prompt_example": batch["rephrase_prompts"][:2],
            "locality_inputs_example": {
                "text": {
                    "prompt": batch["locality_inputs"]["text"]["prompt"][:2],
                    "ground_truth": batch["locality_inputs"]["text"]["ground_truth"][:2],
                },
                "multimodal": {
                    "prompt": batch["locality_inputs"]["multimodal"]["prompt"][:2],
                    "ground_truth": batch["locality_inputs"]["multimodal"]["ground_truth"][:2],
                    "image": batch["locality_inputs"]["multimodal"]["image"][:2],
                },
            },
            "missing_images_example": batch["missing_images"][:5],
        },
    )


def build_batch_from_args(args: argparse.Namespace, paths):
    if args.smoke:
        smoke_path = paths.processed / "smoke" / "smoke_data.json"
        if not smoke_path.exists():
            create_smoke_dataset(paths)
        data_file = smoke_path
    else:
        data_file = Path(args.data_file) if args.data_file else default_data_file(paths)
    if not data_file.exists():
        raise SystemExit(f"Data file does not exist: {data_file}. Copy MedMKEB JSON files into {paths.raw}.")
    return build_edit_batch(
        paths,
        data_file=data_file,
        max_edits=args.max_edits,
        seed=args.seed,
        modality=args.modality,
        department=args.department,
        task=args.task,
        shuffle=args.shuffle,
        stream_type=args.stream_type,
        prompt_template=args.prompt_template,
        file_type=args.file_type,
        missing_image_policy=args.missing_image_policy,
    )


def make_mock_metrics(batch: Dict[str, Any], sequential_edit: bool) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for idx, metadata in enumerate(batch.get("metadata", [])):
        metrics.append(
            {
                "case_id": idx,
                "time": 0.0,
                "mock_edit": True,
                "sequential_edit": sequential_edit,
                "requested_rewrite": {
                    "prompt": batch["prompts"][idx],
                    "target": batch["targets"][idx],
                    "image": batch["image"][idx],
                },
                "post": {
                    "rewrite_acc": None,
                    "rephrase_acc": None,
                    "image_rephrase_acc": None,
                    "locality_acc": None,
                    "multimodal_locality_acc": None,
                    "portability_acc": None,
                    "mock_status": "not_evaluated_model_not_loaded",
                },
                "pre": {},
                "metadata": metadata,
            }
        )
    return metrics


def default_hparams_for_method(method: str) -> Path:
    method_key = method.upper().replace("_", "-")
    if method_key == "IKE":
        return PROJECT_ROOT / "hparams" / "IKE" / "blip2.yaml"
    if method_key == "MEND":
        return PROJECT_ROOT / "hparams" / "MEND" / "blip2.yaml"
    if method_key == "ASAM-MEND":
        return PROJECT_ROOT / "hparams" / "ASAM_MEND" / "blip2.yaml"
    if method_key in {"SERAC", "SERAC-MULTI"}:
        return PROJECT_ROOT / "hparams" / "SERAC" / "blip2.yaml"
    if method_key in {"FT", "FT-LLM"}:
        return PROJECT_ROOT / "hparams" / "FT" / "blip2.yaml"
    if method_key == "FT-PROJ":
        return PROJECT_ROOT / "hparams" / "FT" / "blip2_qformer.yaml"
    if method_key == "ASAM-FT":
        return PROJECT_ROOT / "hparams" / "ASAM_FT" / "blip2.yaml"
    raise SystemExit(f"No default VLKEB hparams for method={method}. Pass --hparams explicitly.")


def write_vlkeb_input(args: argparse.Namespace, paths, batch: Dict[str, Any], output_dir: Path) -> Path:
    data_file = Path(batch["source_file"])
    vlkeb_records, report = build_vlkeb_records(
        paths,
        data_file=data_file,
        max_edits=args.max_edits,
        seed=args.seed,
        modality=args.modality,
        department=args.department,
        task=args.task,
        shuffle=args.shuffle,
        stream_type=args.stream_type,
        missing_image_policy=args.missing_image_policy,
    )
    path = output_dir / "medmkeb_vlkeb_input.json"
    write_json(path, vlkeb_records)
    report["path"] = str(path)
    write_json(output_dir / "medmkeb_vlkeb_input_report.json", report)
    return path


def import_local_easyeditor():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import easyeditor  # type: ignore

        return easyeditor
    except Exception as exc:
        raise RuntimeError(
            "Cannot import local VLKEB `easyeditor`. Install the VLKEB/EasyEdit runtime "
            "dependencies first. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def load_vlkeb_hparams(easyeditor, method: str, hparams_path: Path):
    method_key = method.upper().replace("_", "-")
    class_names = {
        "IKE": ["IKEMultimodalHyperParams"],
        "MEND": ["MENDMultimodalHparams"],
        "ASAM-MEND": ["MENDMultimodalHparams"],
        "SERAC": ["SERACMultimodalHparams"],
        "SERAC-MULTI": ["SERACMultimodalHparams"],
        "FT": ["FTMultimodalHparams"],
        "FT-LLM": ["FTMultimodalHparams"],
        "FT-PROJ": ["FTMultimodalHparams"],
        "ASAM-FT": ["FTMultimodalHparams"],
    }.get(method_key)
    if not class_names:
        raise RuntimeError(f"Unsupported VLKEB method in this runner: {method}")
    errors = []
    for class_name in class_names:
        cls = getattr(easyeditor, class_name, None)
        if cls is None:
            errors.append(f"{class_name}: not exported")
            continue
        try:
            return cls.from_hparams(str(hparams_path))
        except Exception as exc:
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not load hparams {hparams_path}: {'; '.join(errors)}")


def apply_vlkeb_overrides(hparams, paths, output_dir: Path, model_name_or_path: Optional[str], device: Optional[int]):
    if model_name_or_path:
        setattr(hparams, "name", model_name_or_path)
        if hasattr(hparams, "tokenizer_name"):
            setattr(hparams, "tokenizer_name", model_name_or_path)
    if device is not None:
        setattr(hparams, "device", device)
    if hasattr(hparams, "results_dir"):
        setattr(hparams, "results_dir", str(output_dir))
    # The processed VLKEB input uses absolute image paths.
    if hasattr(hparams, "coco_image"):
        setattr(hparams, "coco_image", "")
    if hasattr(hparams, "rephrase_image"):
        setattr(hparams, "rephrase_image", "")
    return hparams


def run_vlkeb_real(args: argparse.Namespace, paths, output_dir: Path, vlkeb_json: Path):
    easyeditor = import_local_easyeditor()
    hparams_path = Path(args.hparams) if args.hparams else default_hparams_for_method(args.method)
    hparams = load_vlkeb_hparams(easyeditor, args.method, hparams_path)
    hparams = apply_vlkeb_overrides(hparams, paths, output_dir, args.model_name_or_path, args.device)

    caption_dataset = getattr(easyeditor, "CaptionDataset")
    eval_ds = caption_dataset(str(vlkeb_json), config=hparams, hop=args.hop)
    method_key = args.method.upper().replace("_", "-")

    if method_key in {"FT", "FT-LLM", "FT-PROJ", "ASAM-FT"}:
        trainer_cls = getattr(easyeditor, "MultimodalTrainer")
        trainer = trainer_cls(config=hparams, train_set=eval_ds, val_set=eval_ds)
        trainer.run()
        return [{"case_id": None, "time": None, "post": {"status": "ft_trainer_completed"}}]

    editor_cls = getattr(easyeditor, "MultimodalEditor")
    editor = editor_cls.from_hparams(hparams)
    kwargs: Dict[str, Any] = {}
    if method_key == "IKE":
        train_file = paths.raw / "train_data.json"
        train_records, _ = build_vlkeb_records(
            paths,
            train_file,
            max_edits=args.max_ike_train or None,
            stream_type="random",
            seed=args.seed,
            missing_image_policy=args.missing_image_policy,
        )
        train_json = output_dir / "medmkeb_vlkeb_train_for_ike.json"
        write_json(train_json, train_records)
        kwargs["train_ds"] = caption_dataset(str(train_json), config=hparams, no_image=True)
    metrics, _, _ = editor.edit_dataset(
        ds=eval_ds,
        keep_original_weight=not args.sequential_edit,
        verbose=not args.quiet,
        **kwargs,
    )
    return metrics


def run_once(args: argparse.Namespace) -> int:
    paths = get_paths(args.root)
    ensure_layout(paths)
    set_cache_env(paths)
    batch = build_batch_from_args(args, paths)
    output_dir = Path(args.output_dir) if args.output_dir else make_output_dir(paths, dry_run=args.dry_run)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_artifacts(output_dir, args, paths, batch)
    vlkeb_json = write_vlkeb_input(args, paths, batch, output_dir)

    unresolved = unresolved_image_count(batch)
    image_debug_policy = args.missing_image_policy in {"blank", "m_loc"}
    if unresolved and not (args.dry_run or args.allow_missing_images or args.smoke or args.mock_edit or image_debug_policy):
        raise SystemExit(
            f"{unresolved} selected records have unresolved image fields. "
            "Run check_medmkeb_assets.py and fix/download images, or pass --allow-missing-images for debugging only."
        )

    if args.prepare_ike:
        raise SystemExit(
            "IKE embedding preparation is now VLKEB-native. Use local VLKEB's "
            "`encode_ike_facts_multimodal` path with a real torch/sentence-transformers environment."
        )

    print(json.dumps({
        "output_dir": str(output_dir),
        "source_file": batch["source_file"],
        "selected_records": batch["selected_records"],
        "unresolved_records_with_images": unresolved,
        "dry_run": args.dry_run,
        "smoke": args.smoke,
    }, indent=2))

    if args.dry_run:
        print(f"Dry-run only. Wrote {output_dir / 'dry_run_batch_summary.json'}")
        return 0

    if args.mock_edit:
        metrics = make_mock_metrics(batch, sequential_edit=args.sequential_edit)
        rows = write_metrics_jsonl(output_dir / "metrics.jsonl", metrics, metadata=batch.get("metadata"))
        write_summary_csv(output_dir / "summary.csv", rows)
        write_json(
            output_dir / "mock_run_report.json",
            {
                "status": "mock_completed",
                "records": len(metrics),
                "note": "This validates MedMKEB adapter/output plumbing only. No model was loaded and no benchmark metric was computed.",
            },
        )
        print(f"Mock edit wrote metrics: {output_dir / 'metrics.jsonl'}")
        return 0

    start = time.time()
    try:
        metrics = run_vlkeb_real(args, paths, output_dir, vlkeb_json)
    except Exception as exc:
        write_json(output_dir / "run_error.json", {"status": "failed", "error": str(exc)})
        raise
    rows = write_metrics_jsonl(output_dir / "metrics.jsonl", metrics, metadata=batch.get("metadata"))
    write_summary_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "runtime.json", {"runtime_sec": time.time() - start})
    print(f"Wrote metrics: {output_dir / 'metrics.jsonl'}")
    print(f"Wrote summary: {output_dir / 'summary.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MedMKEB multimodal editing with local VLKEB/easyeditor.")
    parser.add_argument("--root", default="/Volumes/DataP/knowledge_editing")
    parser.add_argument("--data-file")
    parser.add_argument("--image-root", help="Accepted for compatibility; image roots are resolved from --root.")
    parser.add_argument(
        "--method",
        default="IKE",
        choices=["IKE", "MEND", "ASAM-MEND", "ASAM_MEND", "SERAC", "FT", "FT-LLM", "FT-Proj", "ASAM-FT", "ASAM_FT"],
    )
    parser.add_argument("--hparams", help="VLKEB hparams file. Defaults to hparams/<method>/blip2*.yaml.")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--max-edits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modality")
    parser.add_argument("--department")
    parser.add_argument("--task")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--stream-type", default="random", choices=["random", "modality_blocked", "task_blocked"])
    parser.add_argument("--sequential-edit", type=str2bool, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument(
        "--missing-image-policy",
        default="fail",
        choices=["fail", "blank", "m_loc"],
        help="Program-debug policy for unresolved image/image_rephrase. Use fail for benchmark runs.",
    )
    parser.add_argument("--mock-edit", action="store_true", help="Exercise adapter/output flow without loading EasyEdit or a model.")
    parser.add_argument("--prepare-ike", action="store_true")
    parser.add_argument("--max-ike-train", type=int, default=0)
    parser.add_argument("--hop", type=int, choices=[1, 2, 3, 4], help="VLKEB portability hop for CaptionDataset.")
    parser.add_argument("--eval-metric", default="token em")
    parser.add_argument("--file-type", default="image")
    parser.add_argument("--prompt-template", default="Question: {} Short answer:")
    parser.add_argument("--device", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
