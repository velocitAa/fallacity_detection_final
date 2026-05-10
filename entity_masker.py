"""
Модуль маскирования сущностей для извлечения «логического скелета» аргумента.
Убирает тематическую предвзятость модели (topic bias).
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MaskedEntity:
    original: str
    mask_token: str
    entity_type: str
    start: int
    end: int


@dataclass
class MaskingResult:
    original_text: str
    masked_text: str
    entities: List[MaskedEntity] = field(default_factory=list)
    entity_map: Dict[str, str] = field(default_factory=dict)

    def restore(self) -> str:
        text = self.masked_text
        for mask, original in self.entity_map.items():
            text = text.replace(mask, original)
        return text


class EntityMasker:
    NATASHA_TO_MASK = {"PER": "Person", "ORG": "Group", "LOC": "Place"}

    REGEX_PATTERNS = [
        (r'\b\d+[.,]?\d*\s*(%|процент[аов]*|млн|млрд|тыс\.?|тысяч|миллион[аов]*|billion|million)\b', "Statistic"),
        (r'\b(19|20)\d{2}\s*(год[уа]?|г\.?)?\b', "Date"),
        (r'\b\d{4,}\b', "Number"),
        (r'https?://\S+|\b\w+\.(ru|com|org|net|info)\b', "Source"),
        (r'«[^»]{3,60}»', "Source"),
    ]

    def __init__(
        self, language: str = "ru",
        mask_persons: bool = True, mask_orgs: bool = True,
        mask_locations: bool = True, mask_numbers: bool = True, mask_dates: bool = True
    ):
        self.language = language
        self.mask_persons = mask_persons
        self.mask_orgs = mask_orgs
        self.mask_locations = mask_locations
        self.mask_numbers = mask_numbers
        self.mask_dates = mask_dates

        self._natasha_available = False
        self._init_natasha()

    def _init_natasha(self) -> None:
        try:
            from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc
            self._segmenter = Segmenter()
            self._emb = NewsEmbedding()
            self._ner_tagger = NewsNERTagger(self._emb)
            self._Doc = Doc
            self._natasha_available = True
            logger.info("Natasha NER инициализирована")
        except ImportError:
            logger.warning("natasha не установлена → только regex-маскирование")

    def mask(self, text: str) -> MaskingResult:
        entities: List[MaskedEntity] = []
        counters: Dict[str, int] = {}
        entity_map: Dict[str, str] = {}
        original_to_mask: Dict[str, str] = {}

        if self._natasha_available and self.language == "ru":
            entities.extend(self._extract_natasha(text))

        existing_spans = {(e.start, e.end) for e in entities}
        entities.extend(self._extract_regex(text, existing_spans))

        # Сортируем справа налево для корректной замены
        entities.sort(key=lambda e: e.start, reverse=True)

        masked_text = text
        for entity in entities:
            original = entity.original.strip()
            if not original:
                continue

            if original in original_to_mask:
                mask_token = original_to_mask[original]
            else:
                etype = entity.entity_type
                idx = counters.get(etype, 0)
                letter = chr(ord('A') + min(idx, 25))
                mask_token = f"[{etype}_{letter}]"
                counters[etype] = idx + 1
                original_to_mask[original] = mask_token
                entity_map[mask_token] = original

            masked_text = masked_text[:entity.start] + mask_token + masked_text[entity.end:]

        return MaskingResult(original_text=text, masked_text=masked_text, entities=entities, entity_map=entity_map)

    def mask_batch(self, texts: List[str], show_progress: bool = True) -> List[MaskingResult]:
        from tqdm import tqdm
        results =[]
        iterator = tqdm(texts, desc="Entity Masking") if show_progress else texts
        for text in iterator:
            try:
                results.append(self.mask(text))
            except Exception as e:
                logger.warning(f"Masking error: {e}")
                results.append(MaskingResult(original_text=text, masked_text=text))
        return results

    def _extract_natasha(self, text: str) -> List[MaskedEntity]:
        entities =[]
        try:
            doc = self._Doc(text)
            doc.segment(self._segmenter)
            doc.tag_ner(self._ner_tagger)
            for span in doc.spans:
                raw_type = span.type
                if (raw_type == "PER" and not self.mask_persons) or \
                   (raw_type == "ORG" and not self.mask_orgs) or \
                   (raw_type == "LOC" and not self.mask_locations):
                    continue
                etype = self.NATASHA_TO_MASK.get(raw_type, raw_type)
                entities.append(MaskedEntity(original=span.text, mask_token="", entity_type=etype, start=span.start, end=span.stop))
        except Exception as e:
            logger.debug(f"Natasha error: {e}")
        return entities

    def _extract_regex(self, text: str, existing_spans: Set[Tuple[int,int]]) -> List[MaskedEntity]:
        entities =[]
        for pattern, etype in self.REGEX_PATTERNS:
            if etype in ("Number", "Statistic") and not self.mask_numbers: continue
            if etype == "Date" and not self.mask_dates: continue
            for match in re.finditer(pattern, text, re.IGNORECASE | re.UNICODE):
                s, e = match.start(), match.end()
                overlap = any(not (e <= es or s >= ee) for es, ee in existing_spans)
                if not overlap:
                    entities.append(MaskedEntity(original=match.group(), mask_token="", entity_type=etype, start=s, end=e))
        return entities


def apply_masking_to_dataframe(df, text_col: str = "text_ru", masker=None):
    import pandas as pd
    if masker is None:
        masker = EntityMasker()
    results = masker.mask_batch(df[text_col].tolist())
    df = df.copy()
    df["text_masked"] =[r.masked_text for r in results]
    df["entity_map_json"] =[json.dumps(r.entity_map, ensure_ascii=False) for r in results]
    df["masked_entity_count"] =[len(r.entities) for r in results]
    logger.info(f"✅ Маскирование завершено. Среднее сущностей/текст: {df['masked_entity_count'].mean():.1f}")
    return df