# Развёртывание

Документ описывает, как поднять проект с нуля локально, в Google Colab и в продакшене, а также
что делать с типовыми ошибками.

## 1. Требования

| Компонент                        | Минимум     | Комментарий                                               |
| -------------------------------- | ----------- | --------------------------------------------------------- |
| Python                           | 3.11–3.13   | проверено на 3.12 и 3.13                                  |
| [uv](https://docs.astral.sh/uv/) | 0.4+        | менеджер окружения и зависимостей                         |
| git                              | любая       | нужен для коммитов и версии кода в MLflow                 |
| Диск                             | ~5 ГБ       | окружение 1,4 ГБ + данные 142 МБ + веса 187 МБ + кэши     |
| RAM                              | 4 ГБ        | для обучения на CPU — 6 ГБ                                |
| GPU                              | опционально | обучение: T4 (16 ГБ) достаточно; инференс работает на CPU |

## 2. Локальная установка

```bash
git clone <repo-url> sentinel2-lake-segmentation
cd sentinel2-lake-segmentation

# 1. окружение и зависимости (создаст .venv и поставит всё по uv.lock)
uv sync
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. хуки качества кода
pre-commit install
pre-commit run -a                  # должно быть всё Passed

# 3. проверка, что пакет и тесты живы
python commands.py                 # список команд CLI
pytest -q                          # 84 теста
```

`uv sync` ставит и dev-группу (pre-commit, black, isort, flake8, pytest). Только рантайм:

```bash
uv sync --no-dev
```

## 3. MLflow (для обучения)

Обучение логирует метрики в MLflow. Поднять сервер:

```bash
mlflow server --host 127.0.0.1 --port 8080 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns
```

Проверка: `curl -s http://127.0.0.1:8080/health` → `OK`. UI: <http://127.0.0.1:8080>.

Если сервер не поднят, обучение не падает: в логе будет предупреждение и откат на `CSVLogger`
(`logs/lightning/<run>/`). Управляется `mlflow.allow_fallback`.

> **Важно для Colab:** порт 8080 там занят Jupyter-сервером, поэтому ноутбук поднимает MLflow на
> **5000** и переопределяет `mlflow.tracking_uri`.

## 4. Первый прогон пайплайна

```bash
python commands.py download        # ~2 мин: 24 сцены, манифест сцен
python commands.py preprocess      # 5–8 мин: 441 патч, сплит, статистики, превью
python commands.py train --overrides="train.max_epochs=30,train.batch_size=32,train.precision=16-mixed"
python commands.py evaluate        # метрики на тесте по сохранённому bundle
python commands.py baseline        # сравнение с NDWI
python commands.py export-onnx     # ONNX + проверка против PyTorch
python commands.py infer --image-a data/example/chany_2023-07-15.tif \
                         --image-b data/example/chany_2024-07-20.tif
```

Быстрая проверка «всё ли работает» без долгого обучения:

```bash
python commands.py train --overrides="train.max_epochs=1,train.batch_size=4,train.limit_train_batches=0.1"
```

### Что появится на диске

```
data/raw/scenes_manifest.json     манифест сцен
data/processed/                   441 патч + manifest.csv + norm_stats.json + summary
data/example/                     пример пары GeoTIFF (make-example)
weights/lake_unet.ckpt            93 МБ, state_dict
weights/bundle.json               карточка модели
weights/lake_unet.onnx            93 МБ
plots/*.png, plots/history.csv    графики и история
reports/ndwi_baseline.json        протокол бейзлайна
predictions/                      результаты infer
```

## 5. Данные и DVC

Датасет и веса не лежат в git. По умолчанию удалённое хранилище — локальный каталог
`../sentinel2-lake-dvc-storage` (настраивается в `.dvc/config`).

```bash
dvc pull                       # забрать датасет и веса из хранилища
dvc push                       # отправить после пересборки
dvc status                     # что устарело
dvc repro                      # пересчитать стадии download → preprocess → train
```

Если хранилища нет (например, свежий клон без доступа), ничего настраивать не нужно:
`python commands.py train` увидит отсутствие `data/processed`, попробует `dvc pull`, а затем
сам соберёт датасет из открытых источников (`train.auto_preprocess=true`).

Переключить хранилище на S3:

```bash
dvc remote modify storage url s3://my-bucket/lake-segmentation
dvc remote modify storage access_key_id     $AWS_ACCESS_KEY_ID
dvc remote modify storage secret_access_key $AWS_SECRET_ACCESS_KEY
```

## 6. Google Colab

1. `Среда выполнения → Сменить тип среды выполнения → T4 GPU`;
2. загрузить архив репозитория (или указать `REPO_URL` в первой ячейке);
3. открыть `notebooks/colab_train.ipynb` и выполнить все ячейки.

Ноутбук делает: установку пакета (`%pip install -e .`), запуск MLflow на 5000, `download` +
`preprocess`, обучение с overrides под T4, `evaluate` + `baseline`, `export-onnx`,
`export-tensorrt --dry-run`, показ графиков и сборку `lake_segmentation_artifacts.zip`
(веса + ONNX + датасет + графики + MLflow-база) для скачивания.

Особенности Colab, из-за которых в коде есть специальные проверки:

- Python 3.13 — в `pyproject.toml` нет верхней границы версии;
- порт 8080 занят Jupyter — MLflow на 5000, а доступность сервера проверяется по `/health`, а не
  по открытому порту;
- `sys.executable` в ячейках — `/usr/bin/python3`, поэтому команды запускаются через
  `subprocess` с тем же интерпретатором.

## 7. Продакшен

### 7.1. Комплект поставки модели

Минимально достаточно двух файлов (плюс код `inference/`, у которого нет зависимостей от
обучения):

| Файл                     | Назначение                                                        |
| ------------------------ | ----------------------------------------------------------------- |
| `weights/lake_unet.onnx` | сеть с сигмоидой: вход `(N,4,H,W)`, выход вероятности `(N,1,H,W)` |
| `weights/bundle.json`    | архитектура, порядок каналов, нормализация, порог, метрики        |

`weights/lake_unet.ckpt` нужен только для backend `torch`; `lake_unet.engine` — только для
TensorRT на конкретном GPU.

### 7.2. Порядок подготовки

```bash
python commands.py export-onnx                 # 1. ONNX + валидация
python commands.py export-tensorrt --dry-run   # 2. проверка команды (на GPU — без --dry-run)
python commands.py serve-mlflow                # 3a. вариант MLflow Serving
python commands.py prepare-triton              # 3b. вариант Triton
```

### 7.3. Чек-лист «модель готова к работе»

- [ ] `bundle.json` и веса лежат в DVC, `dvc push` выполнен;
- [ ] ONNX проходит `onnx.checker` и сверку с PyTorch (в логе `max |difference| < 5e-4`);
- [ ] `infer` на двух датах выдаёт маски с корректной геопривязкой (открываются в QGIS);
- [ ] `water_report.json` содержит площади, дельту и порог;
- [ ] сервер (MLflow или Triton) отвечает на health-check и на тестовый запрос;
- [ ] версии зависимостей зафиксированы в `uv.lock`, метрики и git commit — в MLflow.

## 8. Как проверить проект (сценарий приёмки)

Повторяет шаги из задания:

```bash
# 1. клон
git clone <repo-url> && cd sentinel2-lake-segmentation

# 2. чистое окружение по инструкции
uv sync && source .venv/bin/activate

# 3. pre-commit
pre-commit install
pre-commit run -a          # ожидаем все Passed

# 4. обучение (быстрый вариант для проверки: 1–2 эпохи)
python commands.py train --overrides="train.max_epochs=2,train.num_workers=2"

# 5. предсказание
python commands.py make-example
python commands.py infer --image-a data/example/chany_2023-07-15.tif \
                         --image-b data/example/chany_2024-07-20.tif
```

Полный прогон занимает ~10 минут (данные + 30 эпох + экспорт) на T4.

## 9. Типовые проблемы

| Симптом                                                                        | Причина                                             | Решение                                                                                  |
| ------------------------------------------------------------------------------ | --------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `Package 'lake-segmentation' requires a different Python: 3.13 not in '<3.13'` | старая версия `pyproject.toml`                      | обновиться до текущего состояния (`requires-python = ">=3.11"`)                          |
| `cannot import name 'MLflowLogger'`                                            | Lightning ≥ 2.6 переименовал класс в `MLFlowLogger` | код поддерживает оба имени; обновите репозиторий                                         |
| `cannot import name 'Dice' from 'torchmetrics.classification'`                 | torchmetrics 1.9 убрал метрику                      | в проекте свои `BinarySegmentationScore`; обновите код                                   |
| `MlflowException ... 404 ... <html>` при обучении                              | на порту 8080 отвечает Jupyter (Colab)              | поднять MLflow на 5000 и указать `mlflow.tracking_uri`                                   |
| `MisconfigurationException: No GPU found`                                      | `train.accelerator=gpu` без GPU                     | включить GPU в Colab или `train.accelerator=cpu`                                         |
| Обучение «висит» на старте                                                     | скачивание весов ResNet34 с Hugging Face            | нужен доступ к `huggingface.co`; в закрытых сетях — заранее положить кэш в `HF_HOME`     |
| `PermissionError: ... /Library/Caches/dvc`                                     | DVC пишет служебный кэш                             | `dvc config --local core.site_cache_dir $PWD/.dvc/site-cache`                            |
| `dvc pull failed`                                                              | нет доступа к хранилищу                             | проект сам пересоберёт датасет (`train.auto_preprocess`) или настройте remote            |
| CUDA out of memory                                                             | большой батч                                        | `train.batch_size=16` (или 8), при необходимости `train.precision=16-mixed`              |
| `onnxruntime` падает на CUDA                                                   | нет `CUDAExecutionProvider`                         | инференс сам откатится на `CPUExecutionProvider`; проверьте `onnxruntime-gpu`            |
| Долгий `preprocess`                                                            | медленная сеть до S3                                | это чтение ~400 МБ по HTTP; уменьшите AOI или включите `data.download.cache_stacks=true` |
| `RasterioIOError: HTTP response code: 404` для WorldCover                      | неверное имя тайла                                  | тайлы WorldCover — 3°, имя = угол, кратный 3 (`N54E075`)                                 |

## 10. Обновление зависимостей

```bash
uv lock --upgrade          # обновить uv.lock
uv sync                    # применить
pytest -q                  # тесты ловят несовместимости API
pre-commit run -a
```

Тест `tests/test_imports.py` импортирует **все** модули пакета и сверяет потоковые метрики с
NumPy-эталоном — именно он ловит переименования вроде `MLFlowLogger` и `torchmetrics.Dice`,
которые иначе проявляются только в момент запуска обучения.
