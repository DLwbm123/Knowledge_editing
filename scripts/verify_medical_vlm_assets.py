#!/usr/bin/env python3
"""Verify local LLaVA-Med-style VLM assets without loading full weights."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


MODEL_WEIGHT_PATTERNS = (
    "pytorch_model*.bin",
    "model*.safetensors",
    "*.safetensors",
)
TOKENIZER_FILES = (
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
PROJECTOR_PATTERNS = (
    "*mm_projector*",
    "non_lora_trainables.bin",
    "adapter_model.bin",
    "adapter_model.safetensors",
)
VISION_WEIGHT_PATTERNS = (
    "pytorch_model*.bin",
    "model*.safetensors",
    "*.safetensors",
)
VISION_CONFIG_FILES = (
    "config.json",
    "preprocessor_config.json",
    "image_processor_config.json",
)
VISION_KEYS = (
    "vision_tower",
    "mm_vision_tower",
    "image_tower",
)
PROJECTOR_KEYS = (
    "mm_projector_type",
    "projector_type",
    "mm_hidden_size",
)
IMAGE_TOKEN_KEYS = (
    "image_token_index",
    "mm_use_im_start_end",
    "mm_use_im_patch_token",
)


def _offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _glob_any(root: Path, patterns: tuple[str, ...]) -> list[str]:
    if not root.exists():
        return []
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(str(path) for path in found)


def _file_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file() if path.exists() else False,
    }


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload, None
        return None, "json root is not an object"
    except Exception as exc:  # pragma: no cover - exact parser text is env-specific.
        return None, repr(exc)


def _inspect_index_for_projector(root: Path) -> list[str]:
    hits: set[str] = set()
    index_paths = set(root.glob("*.index.json")) | set(root.glob("*index*.json"))
    for index_path in sorted(index_paths):
        payload, error = _read_json(index_path)
        if error or not payload:
            continue
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            for key, value in weight_map.items():
                if "mm_projector" in key or "multi_modal_projector" in key:
                    hits.add(f"{index_path.name}:{key}->{value}")
    return sorted(hits)


def _safe_transformers_loads(model_root: Path, vision_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tokenizer": {"attempted": False, "ok": False, "error": None, "detail": None},
        "model_config": {"attempted": False, "ok": False, "error": None, "detail": None},
        "vision_config": {"attempted": False, "ok": False, "error": None, "detail": None},
    }

    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:
        error = f"transformers import failed: {exc!r}"
        for item in result.values():
            item["error"] = error
        return result

    if model_root.exists():
        result["tokenizer"]["attempted"] = True
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_root),
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False,
            )
            result["tokenizer"]["ok"] = True
            result["tokenizer"]["detail"] = {
                "class": tokenizer.__class__.__name__,
                "vocab_size": len(tokenizer),
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
            }
        except Exception as exc:
            result["tokenizer"]["error"] = repr(exc)

        result["model_config"]["attempted"] = True
        try:
            cfg = AutoConfig.from_pretrained(
                str(model_root),
                local_files_only=True,
                trust_remote_code=True,
            )
            result["model_config"]["ok"] = True
            result["model_config"]["detail"] = {
                "class": cfg.__class__.__name__,
                "model_type": getattr(cfg, "model_type", None),
                "hidden_size": getattr(cfg, "hidden_size", None),
            }
        except Exception as exc:
            result["model_config"]["error"] = repr(exc)

    if vision_root.exists():
        result["vision_config"]["attempted"] = True
        try:
            cfg = AutoConfig.from_pretrained(
                str(vision_root),
                local_files_only=True,
                trust_remote_code=True,
            )
            result["vision_config"]["ok"] = True
            result["vision_config"]["detail"] = {
                "class": cfg.__class__.__name__,
                "model_type": getattr(cfg, "model_type", None),
                "hidden_size": getattr(cfg, "hidden_size", None),
            }
        except Exception as exc:
            result["vision_config"]["error"] = repr(exc)

    return result


def _dry_imports() -> dict[str, Any]:
    imports: dict[str, Any] = {}
    candidates = (
        "easyeditor.trainer.llava.model.builder",
        "easyeditor.trainer.llava.model.language_model.llava_llama",
        "easyeditor.trainer.llava.mm_utils",
    )
    for module in candidates:
        try:
            __import__(module)
            imports[module] = {"ok": True, "error": None}
        except Exception as exc:
            imports[module] = {"ok": False, "error": repr(exc)}
    return imports


def _official_loader_import(source_root: str | None) -> dict[str, Any]:
    result = {
        "attempted": True,
        "ok": False,
        "source_root": source_root,
        "module": "llava.model.builder",
        "symbol": "load_pretrained_model",
        "error": None,
    }
    inserted = False
    if source_root:
        source_path = str(Path(source_root).expanduser().resolve())
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
            inserted = True
        result["source_root"] = source_path
    try:
        module = importlib.import_module("llava.model.builder")
        loader = getattr(module, "load_pretrained_model")
        result["ok"] = callable(loader)
        result["detail"] = repr(loader)
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if inserted:
            try:
                sys.path.remove(str(Path(source_root).expanduser().resolve()))
            except ValueError:
                pass
    return result


def _load_official_smoke(path: str | None) -> dict[str, Any]:
    result = {
        "provided": bool(path),
        "path": path,
        "ok": False,
        "error": None,
        "payload": None,
    }
    if not path:
        return result
    smoke_path = Path(path).expanduser()
    try:
        payload = json.loads(smoke_path.read_text(encoding="utf-8"))
        result["payload"] = payload
        result["ok"] = bool(payload.get("load_ok")) and (
            bool(payload.get("generation_ok")) or bool(payload.get("skip_generation"))
        )
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def _known_auto_config_error(model_type: str | None, error: str | None) -> str | None:
    if model_type == "llava_mistral" and error and "does not recognize this architecture" in error:
        return "known unsupported AutoConfig path for llava_mistral"
    return error


def _infer_vision_tower_root(
    explicit_root: str | None,
    model_config: dict[str, Any] | None,
    model_root: Path,
) -> tuple[Path | None, str]:
    if explicit_root:
        return Path(explicit_root).expanduser(), "explicit"
    if not model_config:
        return None, "not_inferred_no_model_config"
    value = None
    for key in VISION_KEYS:
        if model_config.get(key):
            value = str(model_config[key])
            break
    if not value:
        return None, "not_inferred_no_config_key"
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate, "config_absolute_path"
    local_candidate = model_root / value
    if local_candidate.exists():
        return local_candidate, "config_relative_to_model_root"
    hf_home = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if hf_home:
        cache_candidate = Path(hf_home) / value
        if cache_candidate.exists():
            return cache_candidate, "config_relative_to_cache_root"
    return candidate, "config_reference_not_local_path"


def verify(args: argparse.Namespace) -> dict[str, Any]:
    _offline_env()
    model_root = Path(args.model_root).expanduser()

    model_config_path = model_root / "config.json"
    model_config, model_config_error = _read_json(model_config_path)
    inferred_vision_root, vision_root_source = _infer_vision_tower_root(
        args.vision_tower_root, model_config, model_root
    )
    vision_root = inferred_vision_root or Path("__missing_vision_tower_root__")
    model_weights = _glob_any(model_root, MODEL_WEIGHT_PATTERNS)
    tokenizer_files = [_file_status(model_root / name) for name in TOKENIZER_FILES]
    present_tokenizers = [item["path"] for item in tokenizer_files if item["exists"]]
    projector_files = _glob_any(model_root, PROJECTOR_PATTERNS)
    projector_index_hits = _inspect_index_for_projector(model_root)
    vision_config_files = [_file_status(vision_root / name) for name in VISION_CONFIG_FILES]
    vision_weights = _glob_any(vision_root, VISION_WEIGHT_PATTERNS)

    config_inspection: dict[str, Any] = {
        "loaded_json": model_config is not None,
        "json_error": model_config_error,
        "vision_tower": {},
        "projector": {},
        "image_tokens": {},
    }
    if model_config:
        for key in VISION_KEYS:
            if key in model_config:
                config_inspection["vision_tower"][key] = model_config.get(key)
        for key in PROJECTOR_KEYS:
            if key in model_config:
                config_inspection["projector"][key] = model_config.get(key)
        for key in IMAGE_TOKEN_KEYS:
            if key in model_config:
                config_inspection["image_tokens"][key] = model_config.get(key)

    loads = _safe_transformers_loads(model_root, vision_root)
    imports = _dry_imports()
    official_import = _official_loader_import(args.official_loader_source)
    official_smoke = _load_official_smoke(args.official_loader_smoke_json)

    file_errors: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    if not model_root.exists():
        file_errors.append(f"model root does not exist: {model_root}")
    if not model_config_path.exists():
        file_errors.append(f"model config missing: {model_config_path}")
    if not present_tokenizers:
        file_errors.append(f"tokenizer files missing under: {model_root}")
    if not model_weights:
        file_errors.append(f"language/model checkpoint files missing under: {model_root}")
    if not config_inspection["vision_tower"]:
        file_errors.append("model config does not reference a vision_tower/mm_vision_tower")
    if not config_inspection["projector"]:
        warnings.append("model config has no mm_projector_type/projector metadata")
    if not (projector_files or projector_index_hits):
        file_errors.append(
            "mm_projector/non_lora_trainables/adapter weights were not found as separate files or indexed checkpoint keys"
        )
    if not vision_root.exists():
        file_errors.append(f"vision tower root does not exist: {vision_root}")
    if not any(item["exists"] for item in vision_config_files):
        file_errors.append(f"vision tower config/preprocessor files missing under: {vision_root}")
    if not vision_weights:
        file_errors.append(f"vision tower checkpoint files missing under: {vision_root}")
    if loads["tokenizer"]["attempted"] and not loads["tokenizer"]["ok"]:
        errors.append(f"tokenizer load failed: {loads['tokenizer']['error']}")
    if loads["vision_config"]["attempted"] and not loads["vision_config"]["ok"]:
        errors.append(f"vision config load failed: {loads['vision_config']['error']}")
    if not official_import["ok"]:
        warnings.append(f"official loader import failed: {official_import['error']}")

    model_type = model_config.get("model_type") if model_config else None
    architectures = model_config.get("architectures") if model_config else None
    auto_config_supported = bool(loads["model_config"]["ok"])
    auto_config_error = None
    if loads["model_config"]["attempted"] and not auto_config_supported:
        auto_config_error = _known_auto_config_error(model_type, loads["model_config"]["error"])
    official_loader_model_load = {
        "attempted": official_smoke["provided"],
        "ok": bool(official_smoke["ok"]),
        "error": None,
        "source": "smoke_json" if official_smoke["provided"] else None,
    }
    if official_smoke["payload"]:
        official_loader_model_load["error"] = official_smoke["payload"].get("errors")
    elif official_smoke["error"]:
        official_loader_model_load["error"] = official_smoke["error"]

    files_complete = not file_errors
    official_loader_supported = bool(official_loader_model_load["ok"])
    runnable = files_complete and (auto_config_supported or official_loader_supported)

    return {
        "model_type": model_type,
        "requested_model_type": args.model_type,
        "architectures": architectures,
        "mm_vision_tower": next(iter(config_inspection["vision_tower"].values()), None)
        if config_inspection["vision_tower"]
        else None,
        "mm_projector_type": config_inspection["projector"].get("mm_projector_type")
        or config_inspection["projector"].get("projector_type"),
        "mm_projector_keys_found": projector_index_hits,
        "files_complete": files_complete,
        "auto_config_supported": auto_config_supported,
        "auto_config_error": auto_config_error,
        "official_loader_supported": official_loader_supported,
        "runnable_asset_set": runnable,
        "summary": "runnable" if runnable else "missing or invalid assets",
        "model_root": str(model_root),
        "vision_tower_root": str(vision_root),
        "vision_tower_root_source": vision_root_source,
        "offline_env": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "model_root_exists": model_root.exists(),
        "vision_tower_root_exists": vision_root.exists(),
        "model_config": _file_status(model_config_path),
        "tokenizer_files": tokenizer_files,
        "model_weight_files": model_weights,
        "vision_config_files": vision_config_files,
        "vision_weight_files": vision_weights,
        "config_inspection": config_inspection,
        "projector_files": projector_files,
        "projector_index_hits": projector_index_hits,
        "tokenizer_load": loads["tokenizer"],
        "vision_config_load": loads["vision_config"],
        "official_loader_import": official_import,
        "official_loader_model_load": official_loader_model_load,
        "official_loader_smoke": official_smoke,
        "transformers_loads": loads,
        "dry_imports": imports,
        "file_errors": file_errors,
        "errors": errors,
        "missing": file_errors + errors + ([f"auto_config: {auto_config_error}"] if auto_config_error and not official_loader_supported else []),
        "warnings": warnings,
        "next_steps": (
            ["Official loader assets are runnable; proceed to LLaVA-Med DSCA adapter smoke tests."]
            if runnable
            else [
                "Upload a merged LLaVA-Med model directory or merge the delta with the correct Vicuna/LLaMA base before uploading.",
                "Upload the matching CLIP/EVA/SigLIP vision tower directory with config, preprocessor, and weights.",
                "Include mm_projector weights or a merged checkpoint whose index exposes projector keys.",
                "Re-run this verifier before implementing DSCA LLaVA-Med adapter code.",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--vision-tower-root")
    parser.add_argument("--model-type", default="llava-med")
    parser.add_argument("--official-loader-source", default="third_party/LLaVA-Med")
    parser.add_argument("--official-loader-smoke-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = verify(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["runnable_asset_set"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
