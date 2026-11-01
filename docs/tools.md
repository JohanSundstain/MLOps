# Инструменты: что, зачем и как используется

Документ по стеку проекта: зачем взят каждый инструмент, где он в коде, какие команды нужны в
работе и на какие грабли мы уже наступили (все они отмечены явно).

## Карта стека

| Слой                    | Инструменты                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------- |
| Окружение и зависимости | `uv`, `pyproject.toml`, `uv.lock`                                                     |
| Конфигурация            | `Hydra` (+ `OmegaConf`)                                                               |
| CLI                     | `Fire`                                                                                |
| Данные и геометрия      | `pystac-client`, `rasterio`/`GDAL`, `pyproj`, `numpy`, `pandas`                       |
| Версионирование данных  | `DVC`                                                                                 |
| Обучение                | `PyTorch`, `PyTorch Lightning`, `torchmetrics`, `segmentation-models-pytorch`, `timm` |
| Эксперименты            | `MLflow`                                                                              |
| Экспорт и инференс      | `ONNX`, `onnxruntime`, `TensorRT`, `Triton Inference Server`                          |
| Визуализация            | `matplotlib`                                                                          |
| Качество кода           | `pre-commit`, `black`, `isort`, `flake8`, `prettier`, `pytest`                        |

---

## uv

**Что это.** Менеджер Python-проектов и зависимостей на Rust; заменяет одновременно
`pip`, `pip-tools`, `virtualenv` и `poetry`.

**Зачем в проекте.** Воспроизводимость окружения: `uv.lock` фиксирует точные версии всех
транзитивных зависимостей, `uv sync` создаёт `.venv` ровно по этому файлу. Скорость установки в
разы выше `pip`, что важно для Colab и CI.

**Где.** `pyproject.toml` (`[project.dependencies]`, `[dependency-groups].dev`), `uv.lock`,
`.python-version`.

```bash
uv sync                      # окружение + dev-инструменты
uv sync --no-dev             # только рантайм
uv lock --upgrade            # обновить lock
uv run pytest -q             # запуск команды в окружении проекта
```

**Грабли.**

- `requires-python` должен включать версию Python у проверяющего: в Colab это 3.13
  (верхнюю границу `<3.13` мы убрали, иначе `pip install -e .` падал).
- `uv` хранит кэш и скачанные интерпретаторы в `~/.cache/uv` и `~/.local/share/uv`; в
  ограниченных песочницах их нужно перенаправлять (`UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`).
- После изменения `pyproject.toml` всегда `uv lock`, иначе lock разъедется с манифестом.

---

## Hydra (+ OmegaConf)

**Что это.** Библиотека конфигурации от Meta: иерархические YAML-конфиги, группы, интерполяции,
переопределения из командной строки.

**Зачем в проекте.** Требование задания: все гиперпараметры препроцессинга, обучения и
постпроцессинга — в YAML, иерархично, без магических констант в коде. Hydra даёт ещё и
воспроизводимость: конфиг можно целиком записать в MLflow и в `dataset_summary.json`.

**Где.** `configs/` (11 файлов по группам), `config.py` (`load_config`, `find_config_dir`,
`parse_overrides`).

```python
from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["train.max_epochs=5", "data.patch.size=256"])
```

```bash
python commands.py train --overrides="train.max_epochs=40,train.batch_size=32"
python commands.py show-config            # полный резолвнутый конфиг
```

**Почему `compose`, а не `@hydra.main`.** `@hydra.main` перехватывает `sys.argv` и меняет
рабочий каталог — это конфликтует с Fire и с относительными путями к данным. Compose API даёт
обычную функцию от конфига и полный контроль.

**Грабли.**

- `hydra.job.chdir: false` в конфиге: иначе Hydra делает `cd` в каталог запуска и все пути
  ломаются (пути в проекте разрешаются от корня репозитория, а не от CWD).
- Переопределять можно только существующие ключи; для новых — синтаксис `+ключ=значение`
  (`+train.deterministic=true`).
- Списки в overrides: `infer.roi=[77.4,54.85,77.6,54.95]` — без пробелов.
- `OmegaConf.resolve(cfg)` обязателен, чтобы `${...}`-интерполяции превратились в значения.

---

## Fire

**Что это.** Google Fire — превращает любой Python-объект в CLI без описания аргументов.

**Зачем в проекте.** Задание требует не использовать `argparse`; Fire даёт CLI из класса за одну
строку, сам разбирает типы и печатает справку.

**Где.** `cli.py` (`class LakeSegmentationCli`, `fire.Fire(...)`), `commands.py`.

```python
fire.Fire(LakeSegmentationCli)     # команды = методы класса
```

```bash
python commands.py                          # список команд с описаниями
python commands.py train --help             # справка по команде
python commands.py infer --image-a a.tif    # аргументы как флаги
```

**Грабли.**

- Fire нормализует `export_onnx` ↔ `--export-onnx`, поэтому в README можно писать дефисы.
- Значение с запятыми (`--overrides="a=b,c=d"`) нужно брать в кавычки в shell.
- Fire показывает traceback целиком — это удобно для отладки, но в ноутбуке вывод дочернего
  процесса нужно не терять (в Colab `subprocess` без capture скрывает stderr).

---

## PyTorch

**Что это.** Фреймворк глубокого обучения с автодифференцированием.

**Зачем.** Требование «современные библиотеки, не sklearn»; вся модель и инференс на PyTorch.

**Где.** `models/`, `training/`, `export/onnx.py`, `inference/predictor.py` (backend `torch`).

**Грабли.** `torch.load(..., weights_only=False)` нужен для чекпоинтов Lightning (там словарь, а
не только тензоры); для «голых» state_dict достаточно `weights_only=True`.

---

## PyTorch Lightning

**Что это.** Тонкая обёртка над PyTorch: цикл обучения, callbacks, чекпоинты, mixed precision,
логирование.

**Зачем.** Задание требует Lightning (а не свой цикл). Даёт: `Trainer` с AMP и gradient clipping,
`ModelCheckpoint` по метрике, `EarlyStopping`, `LearningRateMonitor`, интеграцию с логгерами,
`MetricCollection` с автоматическим reset по эпохам.

**Где.** `training/module.py` (`LightningModule`), `training/datamodule.py`
(`LightningDataModule`), `training/callbacks.py`, `training/train.py` (`Trainer`).

**Грабли (уже поймали).**

- В Lightning 2.6 класс логгера называется **`MLFlowLogger`** (заглавная `F`), старый
  `MLflowLogger` удалён → в коде импорт с fallback на оба имени.
- `self.trainer` бросает исключение, если модуль ещё не привязан к `Trainer` — в
  `configure_optimizers` берём `getattr(self, "_trainer", None)`.
- В шаблоне имени чекпоинта Lightning ищет метрику по **логируемому** имени (`"val/iou"`), а если
  ключа нет — молча подставляет 0. Поэтому в `filename` больше нет плейсхолдера метрики.
- Sanity-check вызывает `on_validation_epoch_end` до первой эпохи → в историю попадал мусорный
  «epoch 0» (теперь отфильтрован через `trainer.sanity_checking`).
- `precision="16-mixed"` на CPU не работает: в `train.py` есть автопонижение до `32-true`.

---

## torchmetrics

**Что это.** Библиотека метрик с потоковым состоянием (накопление по батчам, распределённый
подсчёт).

**Зачем.** IoU/Dice/Precision/Recall считаются на GPU за эпоху, без хранения всех предсказаний.

**Где.** `training/metrics.py`: базовый класс `Metric` (для `add_state`) и `MetricCollection`;
сами метрики — свои (`BinarySegmentationScore`, `RelativeAreaError`).

**Грабли.** В torchmetrics 1.9 `Dice` убрали из `torchmetrics.classification`, а нам нужны
определения, совпадающие с NumPy-эталоном (`metrics.py`). Поэтому метрики считаются из TP/FP/FN
вручную, а тест `test_streaming_metrics_match_numpy_definitions` следит за совпадением.

---

## segmentation-models-pytorch (+ timm)

**Что это.** Коллекция архитектур сегментации с предобученными энкодерами.

**Зачем.** U-Net + ResNet34 (ImageNet) в одну строку, включая адаптацию первого слоя с 3 на
4 канала — именно это требует постановка задачи.

**Где.** `models/unet.py::build_model`.

```python
smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=4, classes=1)
```

**Грабли.** Веса энкодера теперь тянутся с Hugging Face Hub (`smp-hub/resnet34.imagenet`), а не
из torchvision: нужен доступ к `huggingface.co` (в закрытой сети — заранее прогретый `HF_HOME`).
Адаптация каналов делается библиотекой: формы `conv1.weight` становятся `(64, 4, 7, 7)`.

---

## DVC

**Что это.** «Git для данных»: версионирует большие файлы, хранит их вне git, а в репозитории
держит только хеши и метаданные.

**Зачем.** Требование задания: данные под управлением DVC, скачивание встроено в `train`/`infer`.
Датасет 142 МБ и веса 187 МБ в git класть нельзя.

**Где.** `dvc.yaml` (стадии), `dvc.lock`, `.dvc/config`, `data/dvc_utils.py` (Python API).

```bash
dvc repro          # пересчитать устаревшие стадии
dvc push / pull    # синхронизация с хранилищем
dvc status         # что изменилось
```

```python
from dvc.repo import Repo
Repo(".").pull(targets=["data/processed"], remote="storage")
```

**Грабли.**

- DVC пишет служебный кэш в `~/Library/Caches/dvc` (macOS) — в песочнице это может быть
  запрещено; лечится `dvc config --local core.site_cache_dir <путь>`.
- Стадии сравнивают **деп**-хеши: после правки кода в `src/` `dvc status` покажет «changed deps»,
  хотя данные не менялись — это нормально.
- Для локального remote путь относительный (`.dvc/config`), поэтому на другой машине его надо
  переопределить.

---

## MLflow

**Что это.** Платформа трекинга экспериментов: runs, параметры, метрики, артефакты, реестр
моделей и сервинг.

**Зачем.** Требование задания: логировать метрики и лоссы, гиперпараметры и git commit; плюс
вариант inference-сервера (MLflow Serving).

**Где.** `training/loggers.py` (создание логгера, health-check, гипepпараметры, теги),
`training/callbacks.py` (артефакты), `serving/mlflow_model.py` (публикация ONNX с сигнатурой).

```bash
mlflow server --host 127.0.0.1 --port 8080 \
    --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
mlflow models serve -m runs:/<run-id>/model --port 5001 --env-manager local
```

**Грабли.**

- MLflow 2.x несовместим с SQLAlchemy 2.1 (`FallbackAsyncAdaptedQueuePool`), поэтому в
  зависимостях `sqlalchemy>=2.0,<2.1`.
- Проверять доступность сервера по TCP недостаточно: на порту может отвечать чужой сервис
  (в Colab — Jupyter). Проверяем `/health` и делаем один реальный вызов API.
- У клиента MLflow есть `set_tag(run_id, key, value)`, но **нет** `set_tags(run_id, dict)`.
- `MLflowLogger.save_dir` равен `None` (в отличие от `CSVLogger`) — путь для артефакта конфига
  нужно строить с учётом этого.

---

## ONNX и onnxruntime

**Что это.** Открытый формат графа нейросети и быстрый рантайм для него.

**Зачем.** Требование задания (5 баллов) + независимость инференса от PyTorch и ускорение на CPU.

**Где.** `export/onnx.py` (экспорт, проверка, метаданные), `inference/predictor.py` (backend),
`serving/`.

```python
torch.onnx.export(model, sample, "model.onnx", opset_version=17, dynamic_axes=...)
onnx.checker.check_model("model.onnx")
ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

**Грабли.**

- В torch ≥ 2.9 экспорт по умолчанию идёт через `dynamo=True` (нужен `onnxscript`), поэтому явно
  передаём `dynamo=False` (с fallback на старые версии).
- Динамические оси для U-Net допустимы только при H/W, кратных 32 (проверено на 256 и 512).
- Сигмоида включена в граф: наружу идут вероятности, а имя выхода — `probabilities`, а не
  `logits` (переименовали, чтобы не путать).
- Расхождение с PyTorch проверяется автоматически (в текущем прогоне 2,68e-07 при допуске 5e-4).

---

## TensorRT

**Что это.** Оптимизирующий компилятор NVIDIA: собирает из ONNX движок под конкретный GPU.

**Зачем.** Требование задания (5 баллов): ускорение инференса и FP16.

**Где.** `export/tensorrt.py`, `scripts/build_tensorrt.sh`, `configs/tensorrt/default.yaml`.

```bash
python commands.py export-tensorrt --dry-run   # trtexec --onnx=... --saveEngine=... --fp16
scripts/build_tensorrt.sh                      # то же в контейнере NVIDIA
```

**Грабли.** Движок привязан к GPU и версии TensorRT (переносить между машинами нельзя);
`trtexec` нет в Colab — поэтому в ноутбуке только `--dry-run`.

---

## Triton Inference Server

**Что это.** Сервер инференса NVIDIA: батчинг, несколько моделей, метрики, HTTP/gRPC.

**Зачем.** Второй (более «продакшен») вариант сервера из задания.

**Где.** `serving/triton.py`, `configs/triton/default.yaml`.

Структура репозитория моделей: `triton_repo/lake_unet/1/model.onnx` + `config.pbtxt`
(`platform: onnxruntime_onnx`, `max_batch_size: 8`, dynamic batching, вход `[4,256,256]`).

**Грабли.** В `config.pbtxt` форма фиксирована — для другого размера патча конфиг нужно
перегенерировать.

---

## Rasterio / GDAL

**Что это.** Python-биндинги к GDAL: чтение/запись растров, CRS, перепроецирование, окна.

**Зачем.** Чтение Sentinel-2 и WorldCover как COG (окнами по HTTP), геопривязка масок,
вычисление transform патчей, запись масок в GeoTIFF.

**Где.** `data/scenes.py`, `data/worldcover.py`, `inference/raster.py`.

```python
with rasterio.open(href) as src:                 # https://... открывается через /vsicurl
    window = from_bounds(left, bottom, right, top, transform=src.transform)
    src.read(1, window=window, boundless=True)
reproject(classes, dst, src_transform=..., dst_transform=..., resampling=Resampling.nearest)
```

**Грабли.**

- Для удалённых COG важны переменные окружения (`GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`,
  `GDAL_HTTP_MULTIRANGE=YES`, кэш блоков) — они вынесены в `data/gdal_env.py`.
- SCL имеет разрешение 20 м: его нельзя читать окном 10-метрового канала (получится четверть
  площади и «100 % invalid»).
- `boundless=True` + `fill_value=0` спасают на краях сцены.
- Классы нельзя интерполировать: только `Resampling.nearest`.

---

## pystac-client

**Что это.** Клиент для STAC API (поиск сцен по пространству, времени и свойствам).

**Зачем.** Найти лучшую сцену Sentinel-2 для AOI и даты, получить ссылки на assets.

**Где.** `data/stac.py`.

```python
client = Client.open("https://earth-search.aws.element84.com/v1", timeout=60)
items = client.search(collections=["sentinel-2-l2a"], bbox=aoi, datetime=window,
                      query={"eo:cloud_cover": {"lt": 20}}).items()
```

**Грабли.** `item.datetime` — объект `datetime`, а не строка; фильтр по облачности
чувствителен к типу (`< 20`, а не `"lt": "20"`); часть сцен покрывает AOI не полностью —
нужна проверка bbox.

---

## pyproj

**Что это.** Обёртка над PROJ: преобразования координат и геодезические расчёты.

**Зачем.** Перевод bbox из EPSG:4326 в CRS сцены и обратно (для ROI), площадь пикселя на
эллипсоиде для растров в градусах.

**Где.** `geo.py`, `data/scenes.py`, `data/worldcover.py`.

**Грабли.** `Geod.polygon_area_perimeter` считает площадь **замкнутого** полигона: если передать
точки в неправильном порядке (самопересечение), получится почти ноль. Именно на этом мы
поймали баг в площади географического пикселя.

---

## NumPy и pandas

**Зачем.** Массивы — вся арифметика с растрами и масками; pandas — манифест датасета
(`manifest.csv`), сводки, группировки по озёрам.

**Где.** `patches.py`, `normalization.py`, `metrics.py`, `pipeline.py`.

**Грабли.** В `groupby(...).size()` и агрегатах важно не терять пустые сплиты; метрики считаем
в `float64`, иначе на больших масках накапливается ошибка.

---

## matplotlib

**Зачем.** Все графики: превью датасета, кривые обучения, предсказания на валидации, отчёт
инференса.

**Где.** `plots.py`, `inference/report.py`.

**Грабли.** Серверный бэкенд обязателен (`matplotlib.use("Agg")` до импорта pyplot), фигуры надо
закрывать (`plt.close`), а «настоящий цвет» лучше рисовать фиксированным растяжением reflectance,
а не перцентилями — иначе один и тот же участок на разных патчах выглядит по-разному.

---

## pre-commit + black + isort + flake8 + prettier

**Зачем.** Требование задания: хуки должны проходить «в зелёном». Плюс это защита от мусора в
коммитах.

**Где.** `.pre-commit-config.yaml`, `.flake8`, `.prettierrc.json`, `.prettierignore`.

```bash
pre-commit install
pre-commit run -a
```

**Грабли.**

- Хукам нужны свои окружения: первый запуск качает их (нужен интернет) и складывает в
  `~/.cache/pre-commit`.
- `mirrors-prettier` требует `node`.
- `check-added-large-files` по умолчанию 500 КБ — подняли до 4 МБ, потому что графики
  (`plots/*.png`) лежат в репозитории осознанно.
- `black`/`isort` правят файлы, поэтому после `pre-commit run -a` нужно переиндексировать их
  (`git add -A`), иначе в коммит попадёт неотформатированная версия.

---

## pytest

**Зачем.** 84 теста: геометрия площадей, метрики, тайлинг, сплит, стабильность сидов,
импортируемость всех модулей.

**Где.** `tests/`, настройки в `pyproject.toml` (`pythonpath = ["src"]`).

```bash
pytest -q                     # всё
pytest -q tests/test_geo.py   # один файл
```

**Грабли.** Тест импортов импортирует **каждый** модуль пакета — это ловит переименования API
(`MLFlowLogger`, `torchmetrics.Dice`), которые иначе всплывают только при запуске обучения.

---

## tqdm

**Зачем.** Прогресс долгих операций: препроцессинг сцен (24 шага), тайловый инференс.

**Грабли.** В логах `\r`-обновления выглядят «склеенными» — для читаемых логов лучше печатать
итоговую строку на сцену (так и сделано в `pipeline.py`).
