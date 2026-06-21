from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_ROOT = Path(os.environ.get("KNOWLEDGE_EDITING_ROOT", "/Volumes/DataP/knowledge_editing"))
MEDMKEB_JSON_NAMES = (
    "train_data.json",
    "eval_data_attack.json",
    "eval_data_threehop_final.json",
)


@dataclass(frozen=True)
class MedMKEBPaths:
    root: Path
    medmkeb_repo: Path
    data: Path
    raw: Path
    images: Path
    sources: Path
    processed: Path
    reports: Path
    models: Path
    cache: Path
    hf_cache: Path
    torch_cache: Path
    outputs: Path


def get_paths(root: Optional[os.PathLike[str] | str] = None) -> MedMKEBPaths:
    base = Path(root) if root else DEFAULT_ROOT
    data = base / "data" / "medmkeb"
    repos = base / "repos"
    return MedMKEBPaths(
        root=base,
        medmkeb_repo=repos / "MedMKEB",
        data=data,
        raw=data / "raw",
        images=data / "images",
        sources=data / "sources",
        processed=data / "processed",
        reports=data / "reports",
        models=base / "models",
        cache=base / "cache",
        hf_cache=base / "cache" / "huggingface",
        torch_cache=base / "cache" / "torch",
        outputs=base / "outputs" / "medmkeb",
    )


def ensure_layout(paths: MedMKEBPaths) -> None:
    dirs = [
        paths.raw,
        paths.images,
        paths.processed,
        paths.reports,
        paths.models,
        paths.hf_cache,
        paths.torch_cache,
        paths.outputs,
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def set_cache_env(paths: MedMKEBPaths) -> None:
    os.environ.setdefault("HF_HOME", str(paths.hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(paths.hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(paths.hf_cache / "datasets"))
    os.environ.setdefault("TORCH_HOME", str(paths.torch_cache))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_simple_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Small YAML writer to avoid requiring PyYAML for dry-run setup."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def render_value(value: Any, indent: int = 0) -> List[str]:
        prefix = " " * indent
        if isinstance(value, dict):
            lines: List[str] = []
            for key, nested in value.items():
                if isinstance(nested, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(render_value(nested, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {scalar(nested)}")
            return lines
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.extend(render_value(item, indent + 2))
                else:
                    lines.append(f"{prefix}- {scalar(item)}")
            return lines
        return [f"{prefix}{scalar(value)}"]

    def scalar(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value)
        if text == "" or any(ch in text for ch in ":#[]{}&,*>!|%@`\"'\n"):
            return json.dumps(text)
        return text

    path.write_text("\n".join(render_value(payload)) + "\n", encoding="utf-8")

def raw_json_files(paths: MedMKEBPaths) -> List[Path]:
    files = [paths.raw / name for name in MEDMKEB_JSON_NAMES if (paths.raw / name).exists()]
    extras = [
        p
        for p in sorted(paths.raw.glob("*.json"))
        if not p.name.startswith("._") and p not in files
    ]
    return files + extras
