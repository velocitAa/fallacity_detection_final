from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .common import iter_jsonl, load_json, resolve_path, write_json, write_jsonl


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _group_key(config: dict[str, Any]) -> str:
    prefix = "dual" if config.get("use_dual_encoder") else "single"
    return f"{config['model_slug']}__{config['text_column']}__{prefix}"


def discover_run_dirs(runs_root: str | Path) -> list[Path]:
    root = resolve_path(runs_root)
    return sorted(path.parent for path in root.rglob("metrics_test.json"))


def aggregate_run_directories(runs_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    run_dirs = discover_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No completed run directories found under {runs_root}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    merged_predictions: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        config = load_json(run_dir / "config.json")
        metrics_test = load_json(run_dir / "metrics_test.json")
        confusion = load_json(run_dir / "confusion_matrix_test.json")
        predictions = list(iter_jsonl(run_dir / "predictions_test.jsonl"))

        group_key = _group_key(config)
        groups[group_key].append(
            {
                "run_dir": str(run_dir),
                "config": config,
                "metrics_test": metrics_test,
                "confusion_matrix_test": confusion,
            }
        )

        for prediction in predictions:
            merged_predictions.append(
                {
                    **prediction,
                    "run_dir": str(run_dir),
                    "model_slug": config["model_slug"],
                    "text_column": config["text_column"],
                    "seed": config["seed"],
                    "use_dual_encoder": bool(config.get("use_dual_encoder")),
                }
            )

    summary: dict[str, Any] = {
        "runs_root": str(resolve_path(runs_root)),
        "run_count": len(run_dirs),
        "groups": {},
    }
    confusion_payload: dict[str, Any] = {}

    for group_key, runs in sorted(groups.items()):
        macro_f1 = [run["metrics_test"]["macro_f1"] for run in runs]
        macro_precision = [run["metrics_test"]["macro_precision"] for run in runs]
        macro_recall = [run["metrics_test"]["macro_recall"] for run in runs]
        accuracy = [run["metrics_test"]["accuracy"] for run in runs]

        per_class_values: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            for label, value in run["metrics_test"]["per_class_f1"].items():
                per_class_values[label].append(value)

        per_class_summary = {
            label: {"mean": _safe_mean(values), "std": _safe_std(values)}
            for label, values in sorted(per_class_values.items())
        }

        sum_matrix = None
        normalized_matrices = []
        labels = runs[0]["confusion_matrix_test"]["labels"]
        for run in runs:
            raw = run["confusion_matrix_test"]["matrix"]
            if sum_matrix is None:
                sum_matrix = [[0 for _ in row] for row in raw]
            for i, row in enumerate(raw):
                row_sum = sum(row)
                norm_row = [value / row_sum if row_sum else 0.0 for value in row]
                normalized_matrices.append((i, norm_row))
                for j, value in enumerate(row):
                    sum_matrix[i][j] += value

        average_normalized = []
        for i in range(len(labels)):
            rows = [row for row_idx, row in normalized_matrices if row_idx == i]
            if rows:
                average_normalized.append([_safe_mean([row[j] for row in rows]) for j in range(len(labels))])
            else:
                average_normalized.append([0.0 for _ in range(len(labels))])

        representative = runs[0]["config"]
        summary["groups"][group_key] = {
            "model_name": representative["model_name"],
            "model_slug": representative["model_slug"],
            "text_column": representative["text_column"],
            "use_dual_encoder": bool(representative.get("use_dual_encoder")),
            "seed_count": len(runs),
            "macro_f1": {"mean": _safe_mean(macro_f1), "std": _safe_std(macro_f1)},
            "macro_precision": {"mean": _safe_mean(macro_precision), "std": _safe_std(macro_precision)},
            "macro_recall": {"mean": _safe_mean(macro_recall), "std": _safe_std(macro_recall)},
            "accuracy": {"mean": _safe_mean(accuracy), "std": _safe_std(accuracy)},
            "per_class_f1": per_class_summary,
            "run_dirs": [run["run_dir"] for run in runs],
        }
        confusion_payload[group_key] = {
            "labels": labels,
            "sum_matrix": sum_matrix,
            "average_normalized_matrix": average_normalized,
        }

    out_root = resolve_path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "summary_metrics.json", summary)
    write_json(out_root / "confusion_matrices.json", confusion_payload)
    write_jsonl(out_root / "merged_predictions.jsonl", merged_predictions)

    table_lines = [
        "| group | model | text | dual | seeds | macro_f1 | macro_precision | macro_recall |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for group_key, payload in summary["groups"].items():
        table_lines.append(
            "| {group} | {model} | {text} | {dual} | {seeds} | {f1:.4f}±{f1_std:.4f} | {p:.4f}±{p_std:.4f} | {r:.4f}±{r_std:.4f} |".format(
                group=group_key,
                model=payload["model_slug"],
                text=payload["text_column"],
                dual="yes" if payload["use_dual_encoder"] else "no",
                seeds=payload["seed_count"],
                f1=payload["macro_f1"]["mean"],
                f1_std=payload["macro_f1"]["std"],
                p=payload["macro_precision"]["mean"],
                p_std=payload["macro_precision"]["std"],
                r=payload["macro_recall"]["mean"],
                r_std=payload["macro_recall"]["std"],
            )
        )
    (out_root / "summary_table.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    per_model_dir = out_root / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    for group_key, payload in summary["groups"].items():
        write_json(per_model_dir / f"{group_key}.json", payload)

    return summary
