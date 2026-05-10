# Translation Quality Analysis Summary

Dataset: `/Users/omnipresence/Desktop/vkr_latest_research/data/cocolofa_ru_v2.jsonl`

## Corpus-level status

- Total rows: 7706
- Accepted rows (`ok` + `repaired_ok`): 7591 (98.51%)
- `repaired_ok`: 22
- `manual_review`: 115

## Automatic sanity checks

- Rows with at least one automatic warning: 1047 (13.59%)
- Mean argument-marker preservation score among rows with source markers: 0.702

## Methodological note

The corpus has no human Russian reference translations, therefore BLEU/chrF are not used
as primary translation-quality metrics. The analysis relies on automatic status filtering,
technical sanity checks, argument-marker preservation proxies, and optional reference-free
quality estimation through COMETKiwi.
