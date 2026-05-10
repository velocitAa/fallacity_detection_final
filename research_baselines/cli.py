from __future__ import annotations

import argparse
import json
import logging

from .aggregation import aggregate_run_directories
from .masking import prepare_masked_dataset
from .smoke import run_cloud_smoke_test
from .training import BaselineTrainConfig, GenericBaselineTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research_baselines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mask_parser = subparsers.add_parser("prepare-masked", help="Create a masked JSONL companion dataset.")
    mask_parser.add_argument("--input", required=True)
    mask_parser.add_argument("--out", required=True)
    mask_parser.add_argument("--text-column", default="text_ru")

    train_parser = subparsers.add_parser("train-baseline", help="Train one baseline run.")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--model-name", required=True)
    train_parser.add_argument("--text-column", required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--num-epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--eval-batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--max-length", type=int, default=256)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument("--patience", type=int, default=2)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--use-dual-encoder", action="store_true")
    train_parser.add_argument("--explanation-field", default="explanation_json")

    aggregate_parser = subparsers.add_parser("aggregate-baselines", help="Aggregate completed baseline runs.")
    aggregate_parser.add_argument("--runs-root", required=True)
    aggregate_parser.add_argument("--out-dir", required=True)

    smoke_parser = subparsers.add_parser("smoke-test", help="Run one cloud environment smoke test.")
    smoke_parser.add_argument("--dataset", required=True)
    smoke_parser.add_argument("--out-dir", required=True)
    smoke_parser.add_argument("--model-name", default="DeepPavlov/rubert-base-cased")
    smoke_parser.add_argument("--text-column", default="text_ru")
    smoke_parser.add_argument("--max-rows", type=int, default=16)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-masked":
        summary = prepare_masked_dataset(args.input, args.out, text_column=args.text_column)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-baseline":
        config = BaselineTrainConfig(
            dataset_path=args.dataset,
            model_name=args.model_name,
            text_column=args.text_column,
            seed=args.seed,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience,
            use_dual_encoder=args.use_dual_encoder,
            explanation_field=args.explanation_field,
            num_workers=args.num_workers,
        )
        payload = GenericBaselineTrainer(config).train()
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.command == "aggregate-baselines":
        payload = aggregate_run_directories(args.runs_root, args.out_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "smoke-test":
        payload = run_cloud_smoke_test(
            args.dataset,
            args.out_dir,
            model_name=args.model_name,
            text_column=args.text_column,
            max_rows=args.max_rows,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
