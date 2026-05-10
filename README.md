# COCOLOFA-RU v2: русский корпус логических ошибок и encoder-baselines

Репозиторий содержит репродукционные артефакты по сборке и анализу русскоязычного корпуса **COCOLOFA-RU v2** для задачи детекции логических ошибок в аргументативных текстах. В составе репозитория находятся:

- очищенные публичные датасеты без полей провайдера перевода, модели, версии промпта и служебной метаинформации;
- код `research_baselines`, необходимый для запуска базовых notebook-ов;
- публичные notebooks для маскирования, baseline-экспериментов, оценки качества перевода и экспериментов с ruRoBERTa/RoSBERTa;
- ключевые графики и агрегированные метрики;
- ссылки и публикационные метаданные для интерактивных BertViz attention-артефактов.

## Данные

- Переведённый корпус: [data/cocolofa_ru_v2.jsonl](data/cocolofa_ru_v2.jsonl)
- Маскированный корпус: [data/cocolofa_ru_v2_masked.jsonl](data/cocolofa_ru_v2_masked.jsonl)
- Пакет для рецензии: [release_assets/dialog_review_package](release_assets/dialog_review_package)

Краткие характеристики:

- опубликованный корпус для воспроизведения экспериментов: `7591` записей;
- полный внутренний переведённый корпус до фильтрации: `7706` записей; в публичную версию включен только принятый подкорпус (`ok + repaired_ok`);
- split для обучения: `5300 / 1508 / 783` (`train / dev / test`);
- число классов: `9`.

Публичные JSONL-файлы оставляют только поля, необходимые для воспроизведения экспериментов:

- `sample_id`
- `split`
- `label_str`
- `label_id`
- `text_en`
- `text_ru`
- `text_masked` только для маскированной версии

Инвентарь меток:

`none`, `appeal to authority`, `appeal to majority`, `appeal to nature`, `appeal to tradition`, `appeal to worse problems`, `false dilemma`, `hasty generalization`, `slippery slope`.

## Ноутбуки

- [01_masking.ipynb](notebooks/01_masking.ipynb) — генерация маскированного корпуса; использует модуль [research_baselines](research_baselines).
- [02_phase1_encoder_baselines.ipynb](notebooks/02_phase1_encoder_baselines.ipynb) — запуск Phase 1 encoder-baselines; использует модуль [research_baselines](research_baselines).
- [03_phase1_results_analysis.ipynb](notebooks/03_phase1_results_analysis.ipynb) — анализ уже обученных run-ов и агрегированных метрик.
- [06_translation_quality_analysis_local.ipynb](notebooks/06_translation_quality_analysis_local.ipynb) — проверка качества перевода без LLM-судьи: LaBSE и COMETKiwi.
- [kaggle_10_ru_baselines_standalone.ipynb](notebooks/kaggle_10_ru_baselines_standalone.ipynb) — дополнительные baseline-эксперименты для рецензии: `ai-forever/ruRoberta-large` и `ai-forever/ru-en-RoSBERTa + LogisticRegression`.

Рекомендуемый сценарий воспроизведения:

1. выполнить [01_masking.ipynb](notebooks/01_masking.ipynb) для построения маскированной версии корпуса;
2. выполнить [02_phase1_encoder_baselines.ipynb](notebooks/02_phase1_encoder_baselines.ipynb) для обучения encoder-baselines;
3. выполнить [03_phase1_results_analysis.ipynb](notebooks/03_phase1_results_analysis.ipynb) для агрегации метрик и построения визуальной аналитики;
4. выполнить [06_translation_quality_analysis_local.ipynb](notebooks/06_translation_quality_analysis_local.ipynb) для валидации перевода;
5. выполнить [kaggle_10_ru_baselines_standalone.ipynb](notebooks/kaggle_10_ru_baselines_standalone.ipynb) для проверки дополнительных русскоязычных baseline-моделей.

## Результаты дополнительных baseline-моделей

Эти эксперименты добавлены для расширения набора русскоязычных сравнений.

| Метод | Модель | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| `ruroberta_finetune` | `ai-forever/ruRoberta-large` | `0.819923` | `0.820998` |
| `sbert_logreg` | `ai-forever/ru-en-RoSBERTa` | `0.540230` | `0.559437` |

Интерпретация: дообученная `ruRoBERTa-large` является самым сильным классификационным baseline-ом, а `ru-en-RoSBERTa + LogisticRegression` служит компактной моделью на sentence embeddings без дообучения трансформера.

## Качество перевода

Перевод проверялся без LLM-судьи:

- LaBSE: межъязыковое семантическое сходство, среднее значение на проверенной подвыборке около `0.887`.
- COMETKiwi: reference-free оценка качества машинного перевода, среднее значение около `0.857`.

| LaBSE semantic similarity | COMETKiwi reference-free QE |
| --- | --- |
| ![LaBSE translation quality](docs/assets/img/translation_labse_similarity.png) | ![COMETKiwi translation quality](docs/assets/img/translation_cometkiwi_scores.png) |

Таблицы с численными оценками лежат в [artifacts/translation_quality](artifacts/translation_quality).

## Результаты Phase 1

Агрегированная таблица по трём seed (`42, 52, 62`) на accepted subset:

| Конфигурация | Macro-F1 | ROC-AUC macro OVR |
| --- | ---: | ---: |
| `XLM-R + text_ru` | `0.7676 ± 0.0093` | `0.9481 ± 0.0042` |
| `XLM-R + text_masked` | `0.7666 ± 0.0073` | `0.9480 ± 0.0056` |
| `RuBERT + text_masked` | `0.7422 ± 0.0109` | `0.9392 ± 0.0024` |
| `RuBERT + text_ru` | `0.7411 ± 0.0083` | `0.9435 ± 0.0024` |

Публикационные акценты:

- лучший **mean baseline**: `XLM-R + text_ru`;
- лучший **single run**: `XLM-R + text_masked`, `seed=62`;
- метрики лучшего single run: `macro-F1 = 0.7751`, `macro-precision = 0.7681`, `macro-recall = 0.7878`, `ROC-AUC macro OVR = 0.9542`.

Итоговые агрегаты лежат в:

- [summary_metrics.json](artifacts/summary/summary_metrics.json)
- [summary_table.md](artifacts/summary/summary_table.md)
- [completed_runs.json](artifacts/summary/completed_runs.json)

## Визуальная аналитика

| Распределение классов | Длины текстов |
| --- | --- |
| ![Class distribution](docs/assets/img/class_distribution.png) | ![Token distribution](docs/assets/img/token_distribution.png) |

| Частотные слова по классам | Per-class F1 |
| --- | --- |
| ![Top words](docs/assets/img/top_words_by_class.png) | ![Per-class F1](docs/assets/img/per_class_f1_score.png) |

| Confusion matrix | ROC-AUC |
| --- | --- |
| ![Confusion matrix](docs/assets/img/confusion_matrix_best_models.png) | ![ROC AUC](docs/assets/img/roc_auc.png) |

Дополнительные артефакты:

- ![PCA 2D](docs/assets/img/pca_2d.png)
- ![PCA 3D](docs/assets/img/pca_3d.png)
- ![Attention summary](docs/assets/img/attention_slippery_slope.png)

Примеры маскирования сущностей:

| Пример 1 | Пример 2 |
| --- | --- |
| ![Masking example none](docs/assets/img/masking_example_none.png) | ![Masking example appeal to tradition](docs/assets/img/masking_example_appeal_to_tradition.png) |

## Полные артефакты

Полный каталог `results` с весами моделей, run-артефактами и дополнительными HTML/PNG-материалами опубликован отдельно на Google Drive:

- [Google Drive: full results folder](https://drive.google.com/drive/folders/1yBxYgh-2wmf-Rh3tna9Bm6JNWAO8Qdq5?usp=sharing)

В репозитории сохранены только компактные производные артефакты, необходимые для чтения результатов:

- [summary_metrics.json](artifacts/summary/summary_metrics.json)
- [summary_table.md](artifacts/summary/summary_table.md)
- [completed_runs.json](artifacts/summary/completed_runs.json)
