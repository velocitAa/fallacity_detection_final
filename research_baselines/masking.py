from __future__ import annotations

import logging
from pathlib import Path

from cocolofa_loader import COCOLOFALoader
from entity_masker import EntityMasker, apply_masking_to_dataframe

from .common import resolve_path, write_json, write_jsonl

logger = logging.getLogger(__name__)


def prepare_masked_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_column: str = "text_ru",
    show_progress: bool = True,
) -> dict:
    loader = COCOLOFALoader(str(input_path))
    df = loader.load()

    if text_column not in df.columns:
        raise KeyError(f"Text column '{text_column}' is missing from dataset {input_path}")

    masker = EntityMasker(language="ru")
    df_masked = apply_masking_to_dataframe(df, text_col=text_column, masker=masker)
    rows = df_masked.to_dict(orient="records")
    write_jsonl(output_path, rows)

    summary = {
        "input_path": str(resolve_path(input_path)),
        "output_path": str(resolve_path(output_path)),
        "row_count": len(df_masked),
        "text_column": text_column,
        "natasha_enabled": bool(getattr(masker, "_natasha_available", False)),
        "mean_masked_entity_count": float(df_masked["masked_entity_count"].mean()),
        "max_masked_entity_count": int(df_masked["masked_entity_count"].max()),
    }
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    logger.info("Masked dataset saved to %s", output_path)
    return summary
