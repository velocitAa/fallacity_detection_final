# Dialog review package

Пакет для доработки статьи по замечаниям рецензентов.

## Dataset

- `data/cocolofa_ru_v2_public.jsonl` — очищенный COCOLOFA-RU v2: оригинальный комментарий, русский перевод, метки, split и `sample_id`.
- `data/cocolofa_ru_v2_public.summary.json` — статистика и список удаленных служебных полей.

Из публичной версии удалены поля с названием используемой модели, провайдера, версии промпта, флагами перевода, статусом перевода, worker/news metadata и generated explanation-полями.

Rows: `7591`

## Notebooks

- `notebooks/kaggle_02_phase1_encoder_baselines_standalone.ipynb`
- `notebooks/kaggle_06_decoder_only_prompting_standalone.ipynb`
- `notebooks/kaggle_07_dspy_prompt_optimization_standalone.ipynb`
- `notebooks/kaggle_09_transformers_dspy_standalone.ipynb`
- `notebooks/kaggle_10_ru_baselines_standalone.ipynb`
- `notebooks/06_translation_quality_analysis_local.ipynb`

Ключевой новый ноутбук для замечаний рецензентов:

- `notebooks/kaggle_10_ru_baselines_standalone.ipynb` — ruRoBERTa fine-tuning и SBERT/RoSBERTa embedding baseline.
