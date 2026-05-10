from __future__ import annotations

import importlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_dir(path: str | Path) -> Path:
    target = resolve_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_parent_dir(path: str | Path) -> Path:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


def load_json(path: str | Path) -> Any:
    with resolve_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return target


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with resolve_path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_number} in {path}: {exc}") from exc


def slugify_model_name(model_name: str) -> str:
    tail = model_name.rsplit("/", maxsplit=1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", tail)
    return slug.strip("-").lower()


def default_baseline_output_dir(
    model_name: str,
    text_column: str,
    seed: int,
    *,
    phase_name: str = "phase1_baselines",
    use_dual_encoder: bool = False,
) -> Path:
    mode = "dual_encoder" if use_dual_encoder else "single_encoder"
    return resolve_path(Path("outputs") / phase_name / slugify_model_name(model_name) / text_column / mode / f"seed_{seed}")


def require_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{module_name}'. Install the cloud environment from requirements-research.txt."
        ) from exc


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
