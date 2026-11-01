# Обучение

Документ описывает модель, функцию потерь, оптимизацию, метрики, подбор порога и то, как
устроен запуск обучения от конфига до артефактов.

## 1. Постановка

Бинарная семантическая сегментация: каждому пикселю патча 4×256×256 присваивается класс
«вода» (1) или «не вода» (0). Модель предсказывает **карту вероятностей**, бинарная маска
получается порогом. Площадь и её изменение считаются отдельно, программно (`inference/`), —
модель никогда не предсказывает площадь напрямую.

## 2. Модель

```
Вход:  float32 (N, 4, 256, 256)                     — нормированный reflectance
U-Net: encoder ResNet34 (ImageNet) + decoder U-Net  — 24 439 505 параметров
Выход: float32 (N, 1, 256, 256)                     — логиты
```

`models/unet.py`:

- `build_model(cfg)` создаёт `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
in_channels=4, classes=1)`;
- адаптация первого свёрточного слоя с 3 на 4 канала выполняется библиотекой
  `segmentation-models-pytorch`: веса RGB усредняются и дублируются для NIR (проверено:
  форма `conv1.weight` — `(64, 4, 7, 7)`);
- `WaterSegmentationModel` — обёртка с явным контрактом (вход/выход, метод
  `predict_probabilities`);
- `ProbabilityModel` — обёртка для экспорта, добавляет сигмоиду внутрь графа (ONNX и TensorRT
  возвращают вероятности, а не логиты);
- `split_parameter_groups` делит параметры на encoder (предобученный) и decoder (случайный) по
  идентичности тензоров, а не по префиксу имени — иначе при обёртке в `self.model` все
  параметры попадали в одну группу.

## 3. Функция потерь

`models/losses.py` — взвешенная сумма BCE и soft Dice:

```
Loss = 0.5 * BCEWithLogits(logits, target) + 0.5 * (1 - Dice)
Dice = (2 * Σ p·t + s) / (Σ p + Σ t + s),  p = sigmoid(logits), s = 1
```

- BCE даёт устойчивый пиксельный градиент;
- Dice напрямую оптимизирует пересечение и не даёт «скатиться» в предсказание одной суши при
  малой доле воды;
- веса 0,5/0,5 (нормируются на сумму), `smooth=1` — сглаживание для пустых масок;
- обе компоненты логируются отдельно (`train/bce`, `train/dice_loss`).

## 4. Оптимизация

| Параметр                | Значение                                | Где                                        |
| ----------------------- | --------------------------------------- | ------------------------------------------ |
| Оптимизатор             | AdamW                                   | `training/module.py::configure_optimizers` |
| Learning rate (decoder) | 3e-4                                    | `train.learning_rate`                      |
| Learning rate (encoder) | 1,5e-4 = 3e-4 × 0,5                     | `train.encoder_learning_rate_scale`        |
| Weight decay            | 1e-4                                    | `train.weight_decay`                       |
| Расписание              | linear warm-up 2 эпохи + cosine до 1e-6 | `train.scheduler`                          |
| Gradient clip           | 1,0                                     | `train.gradient_clip_val`                  |
| Батч                    | 32 (в ноутбуке), 8 (дефолт конфига)     | `train.batch_size`                         |
| Точность                | `16-mixed` на GPU, иначе `32-true`      | `train.precision`                          |
| Эпох                    | 30                                      | `train.max_epochs`                         |

Расписание (`build_warmup_cosine_lambda`):

```
эпоха < warmup:        lr = lr_base * (эпоха + 1) / warmup
иначе:                 lr = lr_min + (lr_base - lr_min) * 0,5 * (1 + cos(π * (эпоха - warmup) / (total - warmup)))
```

Разные LR для encoder и decoder — потому что энкодер уже предобучен, а декодер случаен;
слишком большой LR на энкодере разрушает предобученные признаки.

## 5. Как устроен цикл (PyTorch Lightning)

| Компонент                    | Файл                     | Ответственность                                                          |
| ---------------------------- | ------------------------ | ------------------------------------------------------------------------ |
| `LakeSegmentationModule`     | `training/module.py`     | forward, лосс, метрики, оптимизатор, шедулер, `predict_step`             |
| `LakeSegmentationDataModule` | `training/datamodule.py` | три DataLoader'а из `manifest.csv`, аугментации только для train         |
| `ModelCheckpoint`            | Lightning                | сохраняет лучший по `val/iou` (`save_top_k=1`, `save_weights_only=True`) |
| `EarlyStopping`              | Lightning                | терпение 10 эпох по `val/iou`                                            |
| `LearningRateMonitor`        | Lightning                | LR в логгер                                                              |
| `HistoryCallback`            | `training/callbacks.py`  | собирает метрики по эпохам, строит графики, грузит их в MLflow           |

Аугментации (`data/dataset.py`): только геометрические — отражения по вертикали/горизонтали и
повороты на 90°. Фотометрические искажения запрещены: они меняют физический смысл спектральных
каналов (а именно по соотношению B3/B8 модель отличает воду).

Шаг валидации/теста (`_shared_step`) одинаков для всех стадий: логиты → сигмоида → порог →
`int32`-маски → обновление потоковых метрик → `self.log_dict(...)`. Sanity-check Lightning
исключён из истории (`if trainer.sanity_checking: return`), иначе в графики попадала «нулевая
эпоха» с мусорными значениями.

## 6. Метрики

### 6.1. Определения (общие для всего проекта)

| Метрика             | Формула                 | Краевой случай        |
| ------------------- | ----------------------- | --------------------- | ----------------- | ------------------------------------------------------ |
| IoU                 | `TP / (TP + FP + FN)`   | обе маски пусты → 1,0 |
| Dice                | `2TP / (2TP + FP + FN)` | знаменатель 0 → 1,0   |
| Precision           | `TP / (TP + FP)`        | 0,0                   |
| Recall              | `TP / (TP + FN)`        | 0,0                   |
| Relative Area Error | `                       | S_pred − S_true       | / S_true × 100 %` | `S_true = 0` → не определена, изображение пропускается |

Две реализации, которые обязаны совпадать:

- `metrics.py` — NumPy-версия, эталон; используется в `evaluate`, `baseline`, `infer` и в отчётах;
- `training/metrics.py` — потоковая версия (`BinarySegmentationScore`), считается из TP/FP/FN на
  GPU за эпоху.

Тест `tests/test_imports.py::test_streaming_metrics_match_numpy_definitions` сверяет их на
известном примере. Свои метрики написаны вместо `torchmetrics.classification`, потому что в
torchmetrics 1.9 `Dice` из этого модуля убрали — а определения всё равно должны быть нашими.

### 6.2. Micro и macro

- **micro** — TP/FP/FN суммируются по всем патчам, метрика считается по общей сумме. Крупные
  патчи влияют сильнее;
- **macro** (`*_per_image_mean`) — метрика считается на каждом патче и усредняется. Показывает
  типичное качество на изображение.

Обе версии пишутся в `bundle.json` (блок `metrics.test`), потому что они расходятся: например,
у U-Net micro-IoU 0,9627, а среднее по изображениям 0,9372 — на мелких водоёмах качество ниже.

### 6.3. Откуда берётся «плохая» area error на валидации

Relative Area Error определена через **относительную** ошибку площади, поэтому на патчах, где
эталонной воды почти нет (десятки пикселей), она взрывается: 200 предсказанных пикселей при 30
эталонных — это 566 %. Поэтому в отчётах приводятся и среднее, и медиана (у теста: 4,94 % и
2,14 %; у валидации: 73 % и 1,42 %). Это не ошибка расчёта, а свойство метрики; для сравнения
моделей надёжнее медиана и per-image агрегаты.

## 7. Подбор порога бинаризации

После обучения (`training/train.py`):

1. модель переводится в `eval`, по валидационному сплиту собираются вероятности;
2. перебираются пороги 0,20…0,80 (25 значений), для каждого считается IoU по всему сплиту;
3. выбирается порог с максимальным IoU (в текущем прогоне — **0,55**);
4. этот порог записывается в `bundle.json` и используется инференсом по умолчанию.

Тестовая выборка в подборе не участвует — она оценивается один раз, уже с выбранным порогом.

## 8. Артефакты обучения

| Файл                        | Что это                                                               |
| --------------------------- | --------------------------------------------------------------------- |
| `weights/lake_unet.ckpt`    | только `state_dict` + порог + git commit (93 МБ)                      |
| `weights/best-<epoch>.ckpt` | чекпоинт Lightning, выбранный по `val/iou`                            |
| `weights/bundle.json`       | контракт модели: архитектура, нормализация, порог, tile size, метрики |
| `weights/lake_unet.onnx`    | экспорт (опционально, по `train.export_onnx_after_training`)          |
| `plots/losses.png`          | train/val loss                                                        |
| `plots/metrics.png`         | IoU / Dice / Precision / Recall по эпохам                             |
| `plots/area_error.png`      | Relative Area Error по эпохам                                         |
| `plots/learning_rate.png`   | расписание LR                                                         |
| `plots/history.csv`         | те же числа в табличном виде                                          |
| `plots/val_predictions.png` | image \| reference \| prediction                                      |
| `plots/train.log`           | полный лог обучения                                                   |

`bundle.json` — «карточка модели»:

```json
{
  "bundle_version": 1,
  "model": {"architecture": "Unet", "encoder_name": "resnet34", "in_channels": 4, "classes": 1},
  "preprocessing": {
    "band_order": ["blue", "green", "red", "nir"],
    "reflectance_scale": 10000.0, "clip_min": 0.0, "clip_max": 0.4,
    "normalization": {"mean": [...], "std": [...]}
  },
  "postprocessing": {"threshold": 0.55, "tile_size": 256},
  "artifacts": {"checkpoint": "lake_unet.ckpt"},
  "metrics": {"test": {...}, "validation": {...}, "per_lake_test": {...}}
}
```

## 9. Логирование (MLflow)

`training/loggers.py`:

- адрес сервера — `mlflow.tracking_uri` (по умолчанию `http://127.0.0.1:8080`);
- доступность проверяется запросом `/health` **и** одним реальным вызовом API: на порту может
  отвечать не MLflow (в Colab так делает Jupyter), и тогда обучение не падает, а переключается на
  `CSVLogger` с предупреждением;
- что пишется в эксперимент: 152 гиперпараметра (плоский конфиг), теги `git_commit`, `git_dirty`,
  `project`, полный конфиг как артефакт `config.yaml`, все графики и `bundle.json`;
- метрики пишутся с шагом = эпоха: `train/loss`, `val/loss`, `val/iou`, `val/dice`,
  `val/precision`, `val/recall`, `val/area_error`, `train/learning_rate`, плюс итоговые
  `test/*` и `test/<озеро>/*`;
- откат на CSV (`mlflow.allow_fallback=true`) — на случай офлайн-работы.

## 10. Воспроизводимость

| Что фиксируется   | Как                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| Сид               | `utils.seed_everything(cfg.seed)`: Python, NumPy, PyTorch, `PYTHONHASHSEED`, cuDNN deterministic |
| Сид отбора патчей | `utils.stable_seed(...)` (SHA-256) — не зависит от процесса                                      |
| Версии            | `uv.lock`, все пакеты с точными версиями                                                         |
| Сцены             | `data/raw/scenes_manifest.json` (item_id, дата, облачность)                                      |
| Параметры         | конфиг целиком в `dataset_summary.json` и в MLflow                                               |
| Сплит             | детерминированное перемешивание + запись «озеро → сплит» в `dataset_summary.json`                |
| Веса              | `lake_unet.ckpt` + `bundle.json` в DVC                                                           |

Проверка воспроизводимости, которую легко выполнить: `python commands.py evaluate` на тех же
данных и чекпоинте даёт ровно те же числа, что записаны в `bundle.json` (IoU 0,9627,
area error 4,94 %).

## 11. Ресурсы

| Сценарий                      | Время         | Память        |
| ----------------------------- | ------------- | ------------- |
| 30 эпох, batch 32, T4 (Colab) | ≈ 2,5 мин     | ≈ 3–4 ГБ VRAM |
| 30 эпох, batch 8, T4          | ≈ 6–8 мин     | ≈ 1,5 ГБ VRAM |
| CPU (для проверки пайплайна)  | десятки минут | 4–6 ГБ RAM    |

Влезает в бесплатный T4 с большим запасом; при нехватке памяти уменьшайте `train.batch_size`.

## 12. Результаты текущего прогона

30 эпох, batch 32, `16-mixed`, лосс BCE+Dice, порог 0,55 (лучший чекпоинт — эпоха 22,
`val/iou` = 0,9762).

| Метрика (тест, 66 патчей)                 | U-Net           | NDWI (порог 0,00) |
| ----------------------------------------- | --------------- | ----------------- |
| IoU (micro)                               | 0,9627          | **0,9688**        |
| Dice (micro)                              | 0,9810          | **0,9841**        |
| Precision (micro)                         | 0,9762          | **0,9896**        |
| Recall (micro)                            | **0,9858**      | 0,9788            |
| IoU, среднее по изображениям              | **0,9372**      | 0,9336            |
| Dice, среднее по изображениям             | **0,9615**      | 0,9545            |
| Relative Area Error, среднее / медиана, % | **4,94 / 2,14** | 5,88 / **0,87**   |

По озёрам: Чаны (38 патчей) — IoU 0,9666, Dice 0,9830; Шира (28 патчей) — IoU 0,9487,
Dice 0,9737.

**Интерпретация.** Обе модели сильно превышают целевые ориентиры (IoU ≥ 0,75, Dice ≥ 0,85).
NDWI выигрывает по micro-IoU и precision, потому что разметка WorldCover сама получена
спектральной классификацией: порог по индексу почти оптимален именно для этих меток. Сеть
выигрывает по recall, среднему IoU/Dice на изображение и средней ошибке площади. Чтобы увидеть
реальное преимущество сети, нужна разметка другого происхождения (ручная или JRC Global Surface
Water) — это записано как направление развития.

## 13. Как менять обучение

```bash
# быстрый прогон для проверки
python commands.py train --overrides="train.max_epochs=2,train.batch_size=8,train.limit_train_batches=0.3"

# полный прогон на GPU
python commands.py train --overrides="train.max_epochs=40,train.batch_size=32,train.precision=16-mixed,train.accelerator=gpu"

# без экспорта ONNX в конце
python commands.py train --overrides="train.export_onnx_after_training=false"

# отключить early stopping (например, для отладки расписания)
python commands.py train --overrides="train.early_stopping.enabled=false"
```

Оценка без обучения (по сохранённому bundle и весам):

```bash
python commands.py evaluate
python commands.py baseline
```
