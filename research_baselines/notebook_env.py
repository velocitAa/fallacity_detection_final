from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _has_marker(path: Path, marker_file: str) -> bool:
    return (path / marker_file).exists()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _find_existing_root(cwd: Path, marker_file: str) -> Path | None:
    for candidate in [cwd, cwd.parent]:
        if _has_marker(candidate, marker_file):
            return candidate
    return None


def _find_kaggle_input_root(kaggle_input: Path, project_name: str, marker_file: str) -> Path | None:
    if not kaggle_input.exists():
        return None

    for dataset_dir in sorted(kaggle_input.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if _has_marker(dataset_dir, marker_file):
            return dataset_dir
        nested = dataset_dir / project_name
        if _has_marker(nested, marker_file):
            return nested
    return None


def prepare_notebook_project_root(
    *,
    project_name: str = "vkr_latest_research",
    marker_file: str = "requirements-research.txt",
    cwd: str | Path | None = None,
    kaggle_input: str | Path = "/kaggle/input",
    kaggle_working: str | Path = "/kaggle/working",
    mutate_process: bool = True,
) -> dict[str, Any]:
    current_dir = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    kaggle_input_path = Path(kaggle_input).expanduser().resolve()
    kaggle_working_path = Path(kaggle_working).expanduser().resolve()
    is_kaggle = kaggle_working_path.exists()

    existing_root = _find_existing_root(current_dir, marker_file)
    copied_from_input = None

    if is_kaggle:
        working_root = kaggle_working_path / project_name
        if _has_marker(working_root, marker_file):
            project_root = working_root
        else:
            if existing_root is not None and not _is_relative_to(existing_root, kaggle_input_path):
                project_root = existing_root
            else:
                input_root = existing_root if existing_root is not None else _find_kaggle_input_root(
                    kaggle_input_path,
                    project_name,
                    marker_file,
                )
                if input_root is None:
                    raise FileNotFoundError(
                        f"Project root with {marker_file!r} not found in {kaggle_input_path}. "
                        "Attach the repository as a Kaggle Dataset first."
                    )
                if not working_root.exists():
                    shutil.copytree(input_root, working_root)
                    copied_from_input = str(input_root)
                project_root = working_root
    else:
        if existing_root is None:
            raise FileNotFoundError(f"Could not locate project root with {marker_file!r} near {current_dir}.")
        project_root = existing_root

    if mutate_process:
        os.chdir(project_root)
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

    return {
        "project_root": str(project_root),
        "is_kaggle": is_kaggle,
        "copied_from_input": copied_from_input,
        "cwd": str(current_dir),
        "kaggle_input": str(kaggle_input_path),
        "kaggle_working": str(kaggle_working_path),
    }
