from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from cocolofa_loader import COCOLOFALoader
from fallacy_labels import ID_TO_LABEL, LABEL_TO_ID, N_CLASSES

from .common import default_baseline_output_dir, require_module, resolve_path, set_global_seed, slugify_model_name, write_json, write_jsonl

logger = logging.getLogger(__name__)


def _build_structured_explanation_text(payload: dict[str, Any]) -> str:
    fields = [
        ("SCHEME", payload.get("scheme_name", "")),
        ("PREMISE", payload.get("premise_ru", "")),
        ("CONCLUSION", payload.get("conclusion_ru", "")),
        ("IMPLICIT_ASSUMPTION", payload.get("implicit_assumption_ru", "")),
        ("FAILED_CRITICAL_QUESTION", payload.get("failed_critical_question_ru", "")),
        ("EXPLANATION", payload.get("fallacy_explanation_ru", "")),
        ("LOGICAL_SKELETON", payload.get("logical_skeleton", "")),
    ]
    lines = []
    for key, value in fields:
        text = str(value).strip()
        if text:
            lines.append(f"{key}: {text}")
    return "\n".join(lines)


def _extract_explanation_text(payload: Any) -> str:
    if isinstance(payload, dict):
        structured = _build_structured_explanation_text(payload)
        if structured:
            return structured
        return str(payload.get("fallacy_explanation_ru", "")).strip()
    if isinstance(payload, str) and payload.strip():
        try:
            decoded = json.loads(payload)
            if isinstance(decoded, dict):
                structured = _build_structured_explanation_text(decoded)
                if structured:
                    return structured
                return str(decoded.get("fallacy_explanation_ru", "")).strip()
        except json.JSONDecodeError:
            return payload.strip()
    return ""


@dataclass
class BaselineTrainConfig:
    dataset_path: str
    model_name: str
    text_column: str
    seed: int
    output_dir: str | None = None
    num_epochs: int = 5
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    max_length: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.3
    patience: int = 2
    use_dual_encoder: bool = False
    explanation_field: str = "explanation_json"
    phase_name: str = "phase1_baselines"
    num_workers: int = 0

    def materialized_output_dir(self) -> Path:
        if self.output_dir:
            return resolve_path(self.output_dir)
        return default_baseline_output_dir(
            self.model_name,
            self.text_column,
            self.seed,
            phase_name=self.phase_name,
            use_dual_encoder=self.use_dual_encoder,
        )


class FallacyTextDataset:
    def __init__(self, df, tokenizer, config: BaselineTrainConfig):
        require_module("torch")
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.use_dual = config.use_dual_encoder
        self.text_column = config.text_column
        self.explanation_field = config.explanation_field

        self.texts = df[self.text_column].fillna("").astype(str).tolist()
        self.sample_ids = df["sample_id"].astype(int).tolist()
        self.label_ids = df["label_id"].astype(int).tolist()
        self.label_names = df["label_str"].astype(str).tolist()
        self.source_dataset_path = df.get("source_dataset_path")
        self.explanations = [_extract_explanation_text(value) for value in df.get(self.explanation_field, [])] if self.use_dual else [""] * len(df)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch = require_module("torch")
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item: dict[str, Any] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_ids[idx], dtype=torch.long),
            "sample_ids": torch.tensor(self.sample_ids[idx], dtype=torch.long),
            "text": self.texts[idx],
            "label_names": self.label_names[idx],
        }
        if self.use_dual and self.explanations[idx]:
            exp_enc = self.tokenizer(
                self.explanations[idx],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item["exp_input_ids"] = exp_enc["input_ids"].squeeze(0)
            item["exp_attention_mask"] = exp_enc["attention_mask"].squeeze(0)
            item["explanation_text"] = self.explanations[idx]
        elif self.use_dual:
            item["explanation_text"] = ""
        return item


class GenericEncoderClassifier:
    def __init__(self, model_name: str, n_classes: int, dropout: float = 0.3):
        torch_nn = require_module("torch.nn")
        transformers = require_module("transformers")
        self.bert = transformers.AutoModel.from_pretrained(model_name)
        self.dropout = torch_nn.Dropout(dropout)
        self.classifier = torch_nn.Linear(self.bert.config.hidden_size, n_classes)

    def __call__(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(outputs.last_hidden_state[:, 0, :]))

    def parameters(self):
        return list(self.bert.parameters()) + list(self.classifier.parameters()) + list(self.dropout.parameters())

    def to(self, device):
        self.bert.to(device)
        self.dropout.to(device)
        self.classifier.to(device)
        return self

    def train(self):
        self.bert.train()
        self.dropout.train()
        self.classifier.train()

    def eval(self):
        self.bert.eval()
        self.dropout.eval()
        self.classifier.eval()

    def state_dict(self):
        return {
            "bert": self.bert.state_dict(),
            "dropout": self.dropout.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.bert.load_state_dict(state_dict["bert"])
        self.dropout.load_state_dict(state_dict["dropout"])
        self.classifier.load_state_dict(state_dict["classifier"])


class GenericDualEncoderClassifier:
    def __init__(self, model_name: str, n_classes: int, dropout: float = 0.3):
        torch_nn = require_module("torch.nn")
        transformers = require_module("transformers")
        self.encoder1 = transformers.AutoModel.from_pretrained(model_name)
        self.encoder2 = transformers.AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder1.config.hidden_size

        self.cross_attention = torch_nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout, batch_first=True)
        self.layer_norm = torch_nn.LayerNorm(hidden_size)
        self.dropout = torch_nn.Dropout(dropout)
        self.classifier = torch_nn.Sequential(
            torch_nn.Linear(hidden_size * 2, hidden_size),
            torch_nn.ReLU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_size, n_classes),
        )

    def __call__(self, input_ids, attention_mask, exp_input_ids=None, exp_attention_mask=None, **kwargs):
        h1 = self.encoder1(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls1 = h1[:, 0, :]

        if exp_input_ids is not None:
            h2 = self.encoder2(input_ids=exp_input_ids, attention_mask=exp_attention_mask).last_hidden_state
            cls2 = h2[:, 0, :]
            attn_out, _ = self.cross_attention(query=h1[:, :1, :], key=h2, value=h2)
            cross_cls = self.layer_norm(cls1 + attn_out.squeeze(1))
            torch = require_module("torch")
            combined = torch.cat([cross_cls, cls2], dim=-1)
        else:
            torch = require_module("torch")
            combined = torch.cat([cls1, cls1], dim=-1)

        return self.classifier(self.dropout(combined))

    def parameters(self):
        params = list(self.encoder1.parameters()) + list(self.encoder2.parameters())
        params += list(self.cross_attention.parameters()) + list(self.layer_norm.parameters())
        params += list(self.dropout.parameters()) + list(self.classifier.parameters())
        return params

    def to(self, device):
        self.encoder1.to(device)
        self.encoder2.to(device)
        self.cross_attention.to(device)
        self.layer_norm.to(device)
        self.dropout.to(device)
        self.classifier.to(device)
        return self

    def train(self):
        self.encoder1.train()
        self.encoder2.train()
        self.cross_attention.train()
        self.layer_norm.train()
        self.dropout.train()
        self.classifier.train()

    def eval(self):
        self.encoder1.eval()
        self.encoder2.eval()
        self.cross_attention.eval()
        self.layer_norm.eval()
        self.dropout.eval()
        self.classifier.eval()

    def state_dict(self):
        return {
            "encoder1": self.encoder1.state_dict(),
            "encoder2": self.encoder2.state_dict(),
            "cross_attention": self.cross_attention.state_dict(),
            "layer_norm": self.layer_norm.state_dict(),
            "dropout": self.dropout.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.encoder1.load_state_dict(state_dict["encoder1"])
        self.encoder2.load_state_dict(state_dict["encoder2"])
        self.cross_attention.load_state_dict(state_dict["cross_attention"])
        self.layer_norm.load_state_dict(state_dict["layer_norm"])
        self.dropout.load_state_dict(state_dict["dropout"])
        self.classifier.load_state_dict(state_dict["classifier"])


class GenericBaselineTrainer:
    def __init__(self, config: BaselineTrainConfig):
        self.config = config
        self.output_dir = self.config.materialized_output_dir()

    def _setup_runtime(self) -> None:
        torch = require_module("torch")
        require_module("transformers")
        self.torch = torch
        self.device = self._resolve_device(torch)

    @staticmethod
    def _resolve_device(torch) -> Any:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_model(self):
        if self.config.use_dual_encoder:
            return GenericDualEncoderClassifier(self.config.model_name, N_CLASSES, dropout=self.config.dropout)
        return GenericEncoderClassifier(self.config.model_name, N_CLASSES, dropout=self.config.dropout)

    def _build_dataloaders(self):
        transformers = require_module("transformers")
        torch_utils = require_module("torch.utils.data")
        loader = COCOLOFALoader(self.config.dataset_path)
        df = loader.load()
        if self.config.text_column not in df.columns:
            raise KeyError(f"Dataset {self.config.dataset_path} does not contain text column '{self.config.text_column}'")
        if self.config.use_dual_encoder and self.config.explanation_field not in df.columns:
            raise KeyError(
                f"Dual-encoder mode requires explanation field '{self.config.explanation_field}' in dataset {self.config.dataset_path}"
            )

        self.df = df
        self.train_df, self.val_df, self.test_df = loader.get_splits()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_name)

        train_ds = FallacyTextDataset(self.train_df, self.tokenizer, self.config)
        val_ds = FallacyTextDataset(self.val_df, self.tokenizer, self.config)
        test_ds = FallacyTextDataset(self.test_df, self.tokenizer, self.config)

        self.train_loader = torch_utils.DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        self.val_loader = torch_utils.DataLoader(
            val_ds,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        self.test_loader = torch_utils.DataLoader(
            test_ds,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def _move_tensor_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _extract_model_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        allowed = {"input_ids", "attention_mask", "exp_input_ids", "exp_attention_mask"}
        return {key: value for key, value in batch.items() if key in allowed}

    def _compute_metrics(self, labels: list[int], preds: list[int]) -> dict[str, Any]:
        label_order = list(range(N_CLASSES))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            labels=label_order,
            average=None,
            zero_division=0,
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "per_class_precision": {ID_TO_LABEL[idx]: float(value) for idx, value in zip(label_order, precision)},
            "per_class_recall": {ID_TO_LABEL[idx]: float(value) for idx, value in zip(label_order, recall)},
            "per_class_f1": {ID_TO_LABEL[idx]: float(value) for idx, value in zip(label_order, f1)},
            "classification_report": classification_report(
                labels,
                preds,
                labels=label_order,
                target_names=[ID_TO_LABEL[idx] for idx in label_order],
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": {
                "labels": [ID_TO_LABEL[idx] for idx in label_order],
                "matrix": confusion_matrix(labels, preds, labels=label_order).tolist(),
            },
        }

    def _evaluate(self, loader, *, split_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self.model.eval()
        preds: list[int] = []
        labels: list[int] = []
        predictions_rows: list[dict[str, Any]] = []
        with self.torch.no_grad():
            for batch in loader:
                cpu_sample_ids = batch["sample_ids"].cpu().tolist()
                gold_labels = batch["labels"].cpu().tolist()
                texts = list(batch["text"])
                explanation_texts = list(batch["explanation_text"]) if "explanation_text" in batch else [""] * len(texts)

                batch = self._move_tensor_batch(batch)
                logits = self.model(**self._extract_model_inputs(batch))
                batch_preds = logits.argmax(dim=-1).cpu().tolist()

                preds.extend(batch_preds)
                labels.extend(gold_labels)

                for sample_id, gold, pred, text, explanation in zip(cpu_sample_ids, gold_labels, batch_preds, texts, explanation_texts):
                    row = self.df[self.df["sample_id"] == int(sample_id)].iloc[0]
                    predictions_rows.append(
                        {
                            "split": split_name,
                            "sample_id": int(sample_id),
                            "gold_label_id": int(gold),
                            "pred_label_id": int(pred),
                            "gold_label": ID_TO_LABEL[int(gold)],
                            "pred_label": ID_TO_LABEL[int(pred)],
                            "text_column": self.config.text_column,
                            "text": text,
                            "source_dataset_path": str(resolve_path(self.config.dataset_path)),
                            "label_str": row["label_str"],
                            "explanation_text": explanation,
                        }
                    )

        metrics = self._compute_metrics(labels, preds)
        return metrics, predictions_rows

    def train(self) -> dict[str, Any]:
        self._setup_runtime()
        set_global_seed(self.config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._build_dataloaders()
        self.model = self._build_model().to(self.device)

        transformers = require_module("transformers")
        criterion = self.torch.nn.CrossEntropyLoss()
        optimizer = self.torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(len(self.train_loader) * self.config.num_epochs * self.config.warmup_ratio),
            num_training_steps=max(len(self.train_loader) * self.config.num_epochs, 1),
        )

        history = []
        best_macro_f1 = float("-inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            step_count = 0
            for batch in self.train_loader:
                batch = self._move_tensor_batch(batch)
                labels = batch["labels"]
                optimizer.zero_grad()
                logits = self.model(**self._extract_model_inputs(batch))
                loss = criterion(logits, labels)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += float(loss.item())
                step_count += 1

            val_metrics, _ = self._evaluate(self.val_loader, split_name="dev")
            epoch_summary = {
                "epoch": epoch,
                "train_loss": total_loss / max(step_count, 1),
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
            }
            history.append(epoch_summary)
            logger.info(
                "Epoch %s | train_loss=%.4f | val_macro_f1=%.4f",
                epoch,
                epoch_summary["train_loss"],
                epoch_summary["val_macro_f1"],
            )

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                patience_counter = 0
                self.torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
                self.tokenizer.save_pretrained(self.output_dir / "tokenizer")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping triggered at epoch %s", epoch)
                    break

        self.model.load_state_dict(self.torch.load(self.output_dir / "best_model.pt", map_location=self.device))
        val_metrics, _ = self._evaluate(self.val_loader, split_name="dev")
        test_metrics, test_predictions = self._evaluate(self.test_loader, split_name="test")

        config_payload = {
            **asdict(self.config),
            "dataset_path": str(resolve_path(self.config.dataset_path)),
            "output_dir": str(self.output_dir),
            "model_slug": slugify_model_name(self.config.model_name),
            "device": str(self.device),
            "best_epoch": best_epoch,
        }
        dataset_summary = {
            "dataset_path": str(resolve_path(self.config.dataset_path)),
            "accepted_row_count": len(self.df),
            "train_count": len(self.train_df),
            "val_count": len(self.val_df),
            "test_count": len(self.test_df),
            "text_column": self.config.text_column,
            "use_dual_encoder": self.config.use_dual_encoder,
        }

        write_json(self.output_dir / "config.json", config_payload)
        write_json(self.output_dir / "dataset_summary.json", dataset_summary)
        write_json(self.output_dir / "training_history.json", history)
        write_json(self.output_dir / "metrics_val.json", {k: v for k, v in val_metrics.items() if k != "classification_report" and k != "confusion_matrix"})
        write_json(self.output_dir / "metrics_test.json", {k: v for k, v in test_metrics.items() if k != "classification_report" and k != "confusion_matrix"})
        write_json(self.output_dir / "classification_report_val.json", val_metrics["classification_report"])
        write_json(self.output_dir / "classification_report_test.json", test_metrics["classification_report"])
        write_json(self.output_dir / "confusion_matrix_val.json", val_metrics["confusion_matrix"])
        write_json(self.output_dir / "confusion_matrix_test.json", test_metrics["confusion_matrix"])
        write_jsonl(self.output_dir / "predictions_test.jsonl", test_predictions)

        return {
            "config": config_payload,
            "dataset_summary": dataset_summary,
            "metrics_val": val_metrics,
            "metrics_test": test_metrics,
        }
