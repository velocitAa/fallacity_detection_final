from __future__ import annotations

import importlib.metadata
from pathlib import Path

from cocolofa_loader import COCOLOFALoader
from entity_masker import EntityMasker

from .common import require_module, resolve_path, write_json


def run_cloud_smoke_test(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "DeepPavlov/rubert-base-cased",
    text_column: str = "text_ru",
    max_rows: int = 16,
) -> dict:
    transformers = require_module("transformers")
    torch = require_module("torch")
    require_module("datasets")
    require_module("evaluate")
    require_module("natasha")

    loader = COCOLOFALoader(str(dataset_path))
    df = loader.load().head(max_rows).copy()
    masker = EntityMasker(language="ru")
    masked = [masker.mask(text) for text in df[text_column].tolist()]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModel.from_pretrained(model_name)

    enc = tokenizer(
        df[text_column].tolist()[:4],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**enc)

    payload = {
        "dataset_path": str(resolve_path(dataset_path)),
        "output_dir": str(resolve_path(output_dir)),
        "row_count": len(df),
        "masked_preview_count": len(masked),
        "natasha_enabled": bool(getattr(masker, "_natasha_available", False)),
        "model_name": model_name,
        "hidden_size": int(model.config.hidden_size),
        "forward_shape": list(outputs.last_hidden_state.shape),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["transformers", "datasets", "evaluate", "natasha", "scikit-learn", "sentencepiece"]
        },
    }
    write_json(Path(output_dir) / "smoke_report.json", payload)
    return payload
