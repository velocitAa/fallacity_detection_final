"""
Canonical label schema for the current COCOLOFA-derived dataset.
"""

from __future__ import annotations

from typing import Dict

LABEL_TO_ID: Dict[str, int] = {
    "none": 0,
    "appeal to authority": 1,
    "appeal to majority": 2,
    "appeal to nature": 3,
    "appeal to tradition": 4,
    "appeal to worse problems": 5,
    "false dilemma": 6,
    "hasty generalization": 7,
    "slippery slope": 8,
}

ID_TO_LABEL: Dict[int, str] = {value: key for key, value in LABEL_TO_ID.items()}

LABEL_TO_RU: Dict[str, str] = {
    "none": "нет ошибки",
    "appeal to authority": "апелляция к авторитету",
    "appeal to majority": "апелляция к большинству",
    "appeal to nature": "апелляция к природе",
    "appeal to tradition": "апелляция к традиции",
    "appeal to worse problems": "апелляция к худшим проблемам",
    "false dilemma": "ложная дилемма",
    "hasty generalization": "поспешное обобщение",
    "slippery slope": "скользкая дорожка",
}

LABEL_ALIASES: Dict[str, str] = {
    "appeal_to_authority": "appeal to authority",
    "argument from authority": "appeal to authority",
    "appeal_to_majority": "appeal to majority",
    "bandwagon": "appeal to majority",
    "appeal_to_nature": "appeal to nature",
    "naturalistic fallacy": "appeal to nature",
    "appeal_to_tradition": "appeal to tradition",
    "appeal_to_worse_problems": "appeal to worse problems",
    "whataboutism": "appeal to worse problems",
    "false_dilemma": "false dilemma",
    "false dichotomy": "false dilemma",
    "hasty_generalization": "hasty generalization",
    "overgeneralization": "hasty generalization",
    "slippery_slope": "slippery slope",
}

N_CLASSES = len(LABEL_TO_ID)


def normalize_label(raw: str) -> str:
    normalized = str(raw).strip().lower()
    if not normalized:
        return normalized

    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    normalized = LABEL_ALIASES.get(normalized, normalized)

    if normalized in LABEL_TO_ID:
        return normalized

    for canonical in LABEL_TO_ID:
        if canonical in normalized or normalized in canonical:
            return canonical

    return normalized
