"""
Загрузка и анализ оригинального англоязычного датасета COCOLOFA.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from fallacy_labels import ID_TO_LABEL, LABEL_TO_ID, LABEL_TO_RU, normalize_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FALLACY_LABEL_MAP: Dict[str, int] = LABEL_TO_ID
LABEL_ID_MAP: Dict[int, str] = ID_TO_LABEL
FALLACY_RU_NAMES: Dict[str, str] = LABEL_TO_RU
DEFAULT_ACCEPTED_TRANSLATION_STATUSES = ("ok", "repaired_ok")


class COCOLOFALoader:
    def __init__(
        self,
        data_path: str,
        text_col: str = "text",
        label_col: str = "label",
        accepted_translation_statuses: Optional[tuple[str, ...]] = DEFAULT_ACCEPTED_TRANSLATION_STATUSES,
    ):
        self.data_path = Path(data_path)
        self.text_col = text_col
        self.label_col = label_col
        self.accepted_translation_statuses = accepted_translation_statuses
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        suffix = self.data_path.suffix.lower()
        if suffix == ".csv":
            self.df = self._load_csv()
        elif suffix == ".jsonl":
            self.df = self._load_jsonl()
        elif suffix == ".json":
            self.df = self._load_json()
        else:
            raise ValueError(f"Неподдерживаемый формат: {suffix}")

        self.df = self._normalize(self.df)
        logger.info(f"Загружено {len(self.df)} примеров из {self.data_path}")
        return self.df

    def _load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        self._check_columns(df)
        return df

    def _load_jsonl(self) -> pd.DataFrame:
        records = [json.loads(line.strip()) for line in open(self.data_path, encoding="utf-8") if line.strip()]
        df = pd.DataFrame(records)
        self._check_columns(df)
        return df

    def _load_json(self) -> pd.DataFrame:
        data = json.load(open(self.data_path, encoding="utf-8"))
        df = pd.DataFrame(data if isinstance(data, list) else data.get("data", data))
        self._check_columns(df)
        return df

    def _check_columns(self, df: pd.DataFrame) -> None:
        text_alts = ["text", "text_en", "text_ru", "comment", "sentence", "content", "argument", "post"]
        label_alts = ["label", "label_str", "fallacy", "class", "category", "type", "fallacy_type"]

        if self.text_col not in df.columns:
            for alt in text_alts:
                if alt in df.columns:
                    self.text_col = alt
                    break
            else:
                raise KeyError("Не могу найти текстовую колонку.")

        if self.label_col not in df.columns:
            for alt in label_alts:
                if alt in df.columns:
                    self.label_col = alt
                    break
            else:
                raise KeyError("Не могу найти колонку меток.")

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        original_columns = list(df.columns)
        df = df.dropna(subset=[self.text_col, self.label_col]).copy()
        df["text_en"] = df[self.text_col].astype(str).str.strip()
        df = df[df["text_en"].str.len() > 10]

        df["label_str"] = df[self.label_col].astype(str).str.lower().str.strip().map(normalize_label)
        expected_label_ids = df["label_str"].map(FALLACY_LABEL_MAP)

        if "label_id" in df.columns:
            provided_label_ids = pd.to_numeric(df["label_id"], errors="coerce")
            consistent_mask = provided_label_ids.notna() & expected_label_ids.notna()
            mismatches = consistent_mask & (provided_label_ids.astype("Int64") != expected_label_ids.astype("Int64"))
            if mismatches.any():
                logger.warning("Обнаружены несовпадения label_id со схемой, будет использована каноническая карта.")
                df["label_id"] = expected_label_ids
            else:
                df["label_id"] = provided_label_ids.fillna(expected_label_ids)
        else:
            df["label_id"] = expected_label_ids

        na_mask = df["label_id"].isna() | df["label_str"].isna()
        if na_mask.sum() > 0:
            df = df[~na_mask]

        df["label_id"] = df["label_id"].astype(int)
        if "sample_id" in df.columns:
            numeric_sample_ids = pd.to_numeric(df["sample_id"], errors="coerce")
            if numeric_sample_ids.isna().any():
                numeric_sample_ids = pd.Series(range(len(df)), index=df.index)
            df["sample_id"] = numeric_sample_ids.astype(int)
        else:
            df = df.reset_index(drop=True)
            df["sample_id"] = df.index

        if self.accepted_translation_statuses and "translation_status" in df.columns:
            df = df[df["translation_status"].isin(self.accepted_translation_statuses)]

        df = df.reset_index(drop=True)
        preferred_order = ["sample_id", "text_en", "text_ru", "label_str", "label_id", "split", "translation_status"]
        ordered = [column for column in preferred_order if column in df.columns]
        ordered += [column for column in original_columns if column in df.columns and column not in ordered]
        ordered += [column for column in df.columns if column not in ordered]
        return df[ordered]

    def print_stats(self) -> None:
        if self.df is None: return
        print(f"\n📊 COCOLOFA Dataset | Всего примеров: {len(self.df)}")
        counts = self.df["label_str"].value_counts()
        for label, count in counts.items():
            print(f"  {label:<28} {count:>5} ({count/len(self.df)*100:5.1f}%)")

    def get_splits(self, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        if "split" in self.df.columns:
            split_values = {value.lower() for value in self.df["split"].dropna().astype(str)}
            if {"train", "dev", "test"}.issubset(split_values):
                train_df = self.df[self.df["split"].astype(str).str.lower() == "train"].reset_index(drop=True)
                val_df = self.df[self.df["split"].astype(str).str.lower() == "dev"].reset_index(drop=True)
                test_df = self.df[self.df["split"].astype(str).str.lower() == "test"].reset_index(drop=True)
                return train_df, val_df, test_df

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(splitter.split(self.df, self.df["label_id"]))
        df_train_val, df_test = self.df.iloc[train_val_idx].reset_index(drop=True), self.df.iloc[test_idx].reset_index(drop=True)

        val_relative = val_size / (train_size + val_size)
        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=random_state)
        train_idx, val_idx = next(splitter2.split(df_train_val, df_train_val["label_id"]))
        
        return df_train_val.iloc[train_idx].reset_index(drop=True), df_train_val.iloc[val_idx].reset_index(drop=True), df_test
