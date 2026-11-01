# Автоматическая сегментация озёр на спутниковых снимках и анализ изменения их площади

Копылов Матвей, Тишкин Андрей, Бакумов Алексей

Репозиторий содержит полный пайплайн бинарной семантической сегментации водной поверхности
озёр на снимках Sentinel-2: загрузку данных, препроцессинг, обучение, подготовку к
продакшену (ONNX, TensorRT) и инференс с расчётом площади водоёма и её изменения между двумя
датами.

---

## Постановка задачи

Цель проекта — разработка системы автоматической сегментации водной поверхности озёр на
спутниковых снимках с последующим сравнением их площади для двух дат наблюдения. Система
должна выделять водные пиксели на изображениях, полученных в разные даты, вычислять площадь
выделенной водной поверхности и определять абсолютное и относительное изменение площади.

Результаты могут использоваться для экологического мониторинга, наблюдения за сезонными и
долгосрочными изменениями водоёмов, а также для предварительного анализа последствий засух и
других природных процессов.

Основная задача машинного обучения — бинарная семантическая сегментация: каждому пикселю
изображения присваивается один из двух классов: «вода» или «не вода». Модель не предсказывает
изменение площади напрямую: она независимо обрабатывает каждый снимок, после чего площади
сравниваются программно.

### Формат входных и выходных данных

Входные данные — геопривязанные спутниковые изображения Sentinel-2 в формате GeoTIFF. Для
модели используются четыре спектральных канала с разрешением 10 м: B2 (синий), B3 (зелёный),
B4 (красный) и B8 (ближний инфракрасный). Перед подачей в модель снимки приводятся к единому
масштабу значений, проверяются на совпадение географической привязки и разбиваются на
фрагменты размером 256 × 256 пикселей.

Тензор одного фрагмента имеет размерность 4 × 256 × 256. Эталонная маска имеет размерность
256 × 256 и содержит значения 0 («не вода») и 1 («вода»).

Выход модели — карта вероятностей принадлежности пикселей к классу «вода» и бинарная маска,
полученная пороговой обработкой вероятностей. Для пары дат система формирует две маски, оценки
площади в м² или км², абсолютное изменение площади и относительное изменение в процентах.

Относительное изменение площади рассчитывается по формуле:

```
ΔS = ((S₂ − S₁) / S₁) × 100 %
```

где S₁ и S₂ — площади водной поверхности на первом и втором снимках. Если S₁ = 0,
относительное изменение не вычисляется и обозначается отдельно (`"delta_percent": null` и
пояснение в отчёте).

Для корректного подсчёта площади учитываются геопривязка и фактический размер пикселя: для
проекционных систем координат (UTM) площадь пикселя постоянна, для географических (EPSG:4326)
используется геодезический расчёт площади пикселя по эллипсоиду WGS84
(`src/lake_segmentation/geo.py`).

### Метрики

Качество сегментации оценивается на пиксельном уровне: IoU, Dice, Precision, Recall и Relative
Area Error. Все метрики считаются как по всему набору (micro), так и усреднением по
изображениям (macro), а также в разрезе отдельных озёр.

```
IoU                    = TP / (TP + FP + FN)
Dice                   = 2TP / (2TP + FP + FN)
Precision              = TP / (TP + FP)
Recall                 = TP / (TP + FN)
Relative Area Error    = |S_pred − S_true| / S_true × 100 %
```

Целевые ориентиры для основной модели: IoU ≥ 0,75; Dice ≥ 0,85; Precision ≥ 0,85;
Recall ≥ 0,85; Relative Area Error ≤ 10 %. Это гипотезы для планирования эксперимента,
итоговые значения см. в разделе [Результаты](#результаты).

### Валидация

Данные делятся на обучающую, валидационную и тестовую выборки в пропорции 70 % / 15 % / 15 %.
Разделение выполняется **по целым озёрам (AOI)**, а не случайно по фрагментам: все фрагменты
одного озера попадают только в одну выборку. Это исключает утечку данных из-за сходства
соседних фрагментов (`src/lake_segmentation/data/splits.py`).

Валидационная выборка используется для выбора гиперпараметров, порога бинаризации и контроля
во время обучения. Тестовая выборка применяется только для итоговой оценки.

Для воспроизводимости фиксируются random seed, версии библиотек (`uv.lock`), перечень сцен
(`data/raw/scenes_manifest.json`), параметры препроцессинга (Hydra-конфиги), конфигурация
обучения и веса модели.

### Данные

Основной источник спутниковых изображений — миссия Sentinel-2 программы Copernicus. Каналы
B2, B3, B4 и B8 имеют пространственное разрешение 10 м. Сцены читаются по протоколу STAC из
публичного каталога [Element84 Earth Search](https://earth-search.aws.element84.com/v1)
(данные лежат в открытом AWS-бакете `sentinel-cogs`, авторизация не требуется). Альтернативные
точки доступа: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).

В качестве источника эталонной разметки используется
[ESA WorldCover 2021 v200](https://esa-worldcover.org/en/data-access) — глобальная карта
земного покрова с разрешением 10 м. Для задачи выделяется класс постоянных водоёмов (класс
`80`). Данные читаются напрямую из открытого бакета
(`https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/...`) — тайлы 3° × 3°,
которые при необходимости мозаицируются.

Дополнительный источник для проверки и контекстного анализа —
[JRC Global Surface Water](https://global-surface-water.appspot.com/).

Области интереса (12 озёр на юге Западной Сибири: Чаны, Сартлан, Убинское, Телецкое,
Новосибирское водохранилище, Большое Яровое, Бурлинское, Кулундинская группа, озёра Хакасии и
Омской области) перечислены в `configs/data/default.yaml`. Для каждого озера берутся две даты
(лето 2023 и лето 2024), сцена выбирается автоматически: минимальная облачность в окне ±45
дней и полное покрытие AOI.

Особенности и потенциальные проблемы данных:

- облачность, тени облаков, дымка и атмосферные эффекты затрудняют определение границ водоёмов;
  пиксели классов SCL «облако», «тень», «снег», «нет данных» исключаются из обучающих
  фрагментов (требуется ≥ 95 % валидных пикселей в фрагменте);
- разметка WorldCover относится к 2021 году и описывает постоянные водоёмы, поэтому может не
  совпадать с фактическим состоянием воды на снимках других дат и хуже отражать сезонные или
  временно затопленные участки (для пересыхающих солёных озёр это заметно);
- класс воды включает реки, водохранилища и другие водные объекты, а не только озёра — для
  анализа конкретного озера используется ROI (`infer.roi`);
- снимки разных дат отличаются по геопривязке, сезону, уровню воды и условиям съёмки;
- перед обучением пары «снимок — маска» проверяются визуально
  (`plots/dataset_preview.png`), а совпадение маски с водой на снимке контролируется
  количественно (IoU WorldCover против NDWI на тестовом AOI ≈ 0,95 при нулевом сдвиге).

## Моделирование

### Бейзлайн

В качестве простого бейзлайна используется пороговая сегментация по индексу NDWI
([McFeeters, 1996](https://doi.org/10.1080/01431169608948714)), вычисляемому по зелёному и
ближнему инфракрасному каналам Sentinel-2:

```
NDWI = (B3 − B8) / (B3 + B8)
```

Пиксели, для которых NDWI превышает выбранный порог, классифицируются как вода. Порог
подбирается на валидационной выборке (25 значений в диапазоне 0…0,6), а не на тестовых данных,
и оценивается по тем же метрикам. Метод не требует обучения нейронной сети и служит точкой
отсчёта для измерения прироста качества основной модели
(`src/lake_segmentation/baseline/ndwi.py`, запуск — `python commands.py baseline`).

### Основная модель

Основная архитектура — U-Net с encoder ResNet34
([U-Net](https://arxiv.org/abs/1505.04597), [ResNet](https://arxiv.org/abs/1512.03385)) из
[Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch)
([документация](https://segmentation-models-pytorch.readthedocs.io/)). Encoder инициализируется
весами ImageNet; библиотека адаптирует первый свёрточный слой с трёх каналов на четыре (веса
усредняются по RGB и дублируются для NIR), после чего вся сеть дообучается на снимках
Sentinel-2.

Схема обучения:

1. входной слой адаптируется под 4 канала Sentinel-2 (B2, B3, B4, B8);
2. выход модели — один канал логитов класса «вода»;
3. нормализация: reflectance `uint16 / 10 000`, клиппинг в `[0, 0.4]`, z-нормализация по
   статистикам обучающей выборки (`norm_stats.json`);
4. аугментации только геометрические (отражения, повороты на 90°), чтобы не искажать
   спектральные каналы;
5. функция потерь — комбинация Binary Cross Entropy и soft Dice Loss (веса 0,5 / 0,5);
6. гиперпараметры и порог бинаризации подбираются по валидационной выборке;
7. итоговая оценка на отложенной тестовой выборке и сравнение с NDWI-бейзлайном.

Фреймворк обучения — [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/): модуль
`LakeSegmentationModule` использует готовые механизмы Trainer, callbacks, логирования и
`torchmetrics`.

## Внедрение

Результат внедрён в виде CLI-инструмента (этот репозиторий) и inference-серверов. Пользователь
передаёт два геопривязанных снимка одной области за разные даты и получает отчёт: маски
водной поверхности, площади в м² / км², абсолютное и относительное изменение, картинку с
визуализацией и JSON-отчёт. Дополнительно модель упакована в ONNX и TensorRT, а также может
быть поднята как MLflow Serving или Triton Inference Server.

---

## Структура репозитория

```
configs/                     Hydra-конфиги (по стадиям: data, preprocess, model, train, infer, export, ...)
commands.py                  единственная точка входа CLI
src/lake_segmentation/
├── cli.py                   команды Fire + compose API Hydra
├── config.py                загрузка конфигов, разрешение путей от корня репозитория
├── constants.py             константы предметной области (без «магических чисел» в коде)
├── geo.py                   площадь пикселя/маски, изменение площади (WGS84 и UTM)
├── metrics.py               IoU, Dice, Precision, Recall, Relative Area Error (NumPy)
├── tiling.py                нарезка растра на тайлы и склейка вероятностей
├── plots.py                 графики: превью датасета, кривые обучения, предсказания
├── data/
│   ├── stac.py              поиск сцен в STAC, выбор лучшей сцены
│   ├── scenes.py            чтение 4 каналов + SCL окном из COG
│   ├── worldcover.py        эталонная маска воды (мозаика тайлов 3°×3°)
│   ├── patches.py           нарезка фрагментов и фильтрация по доле воды
│   ├── splits.py            групповое разбиение train/val/test по озёрам
│   ├── normalization.py     статистики нормализации
│   ├── dataset.py           torch Dataset и аугментации
│   ├── dvc_utils.py         dvc pull/push через Python API
│   ├── example.py           генерация примера пары GeoTIFF для инференса
│   └── pipeline.py          команды download и preprocess
├── models/                  фабрика U-Net, лоссы, bundle модели
├── training/                LightningModule, DataModule, callbacks, логгеры, train, evaluate
├── export/                  ONNX и TensorRT
├── inference/               чтение растров, предсказание, отчёт
├── serving/                 MLflow Serving и Triton
└── baseline/                NDWI-бейзлайн
scripts/                     shell-обёртки (TensorRT, загрузка данных)
tests/                       unit-тесты (pytest)
notebooks/colab_train.ipynb  обучение в Google Colab
dvc.yaml                     стадии DVC (download, preprocess, train)
```

Данные и веса в git не хранятся: `data/` и `weights/` управляются DVC (см.
[Data management](#data-management-dvc)).

---

## Setup

Требования: Python 3.11–3.13 (обучение проверялось на 3.12 и 3.13), [uv](https://docs.astral.sh/uv/), git. GPU не обязателен для
инференса, но нужен для обучения (CPU-обучение поддерживается, но медленное).

```bash
git clone https://github.com/<your-account>/sentinel2-lake-segmentation.git
cd sentinel2-lake-segmentation

# установка uv (если ещё не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# создание виртуального окружения и установка всех зависимостей (включая dev-группу)
uv sync
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# хуки качества кода
pre-commit install
pre-commit run -a              # должен завершиться без ошибок

# проверка, что всё работает
pytest -q
python commands.py --help
```

Поднять MLflow (нужен для логирования обучения; адрес по умолчанию
`http://127.0.0.1:8080`):

```bash
mlflow server --host 127.0.0.1 --port 8080 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns
```

Если сервер недоступен, обучение не падает: включается `CSVLogger`
(`mlflow.allow_fallback: true`).

## Train

Обучение состоит из трёх шагов.

### 1. Загрузка данных (поиск сцен)

```bash
python commands.py download
```

Команда ищет для каждого AOI и каждой даты лучшую сцену Sentinel-2 (минимальная облачность в
окне ±45 дней, полное покрытие AOI), проверяет доступность всех каналов и сохраняет манифест
сцен в `data/raw/scenes_manifest.json`. Сами сцены не скачиваются целиком: они читаются окном
из COG по HTTP при препроцессинге.

Полезные overrides:

```bash
python commands.py download --overrides="data.stac.max_cloud_cover=10"
python commands.py download --overrides="data.stac.date_window_days=20"
```

### 2. Препроцессинг (датасет фрагментов)

```bash
python commands.py preprocess
```

Шаги: чтение 4 каналов и SCL для каждого AOI → маска воды из WorldCover, перепроецированная на
сетку сцены → нарезка на фрагменты 256 × 256 → отбор фрагментов с береговой линией (доля воды
2…98 %, ≥ 95 % валидных пикселей) и доли негативных примеров (30 %) → групповое разбиение
train/val/test по озёрам → статистики нормализации по train.

Результат:

```
data/processed/
├── patches/<aoi>__<date>__r<row>_c<col>.npz   image (4,256,256) uint16 + mask (256,256) uint8
├── manifest.csv                               patch_id, split, aoi, сцена, water_fraction, transform, crs
├── norm_stats.json                            mean/std по каналам (только train)
└── dataset_summary.json                       сводка: количество патчей, разбиение, сцены
plots/dataset_preview.png                      визуальная проверка пар «снимок — маска»
```

Полезные overrides:

```bash
python commands.py preprocess --overrides="data.patch.negative_ratio=0.2"
python commands.py preprocess --overrides="data.patch.max_patches_per_scene=30"
python commands.py dataset-info            # сводка по готовому датасету
```

### 3. Обучение

```bash
python commands.py train
```

Внутри: `dvc pull` (если датасета нет), проверка/автосборка датасета, обучение U-Net ResNet34
с AdamW (cosine-расписание с warm-up, разные learning rate для encoder и decoder), контроль
`val/iou`, early stopping, подбор порога бинаризации на валидации, оценка на тесте, сохранение
артефактов и (опционально) экспорт в ONNX.

Типичные overrides:

```bash
python commands.py train --overrides="train.max_epochs=40,train.batch_size=32,train.precision=16-mixed"
python commands.py train --overrides="train.accelerator=gpu,train.devices=1"
```

Артефакты обучения:

```
weights/lake_unet.ckpt        state_dict + порог + git commit
weights/bundle.json           «карточка модели»: архитектура, нормализация, порог, метрики
weights/<checkpoint>.ckpt     чекпоинт Lightning по val/iou (save_top_k=1)
plots/losses.png              train/val loss
plots/metrics.png             IoU / Dice / Precision / Recall
plots/area_error.png          Relative Area Error
plots/learning_rate.png       расписание learning rate
plots/history.csv             сырые значения по эпохам
plots/val_predictions.png     image | reference | prediction на валидации
```

### Оценка и сравнение с бейзлайном

```bash
python commands.py evaluate    # метрики обученной модели на тесте (по сохранённому bundle)
python commands.py baseline    # NDWI-бейзлайн: порог по валидации, метрики на тесте
```

Отчёт бейзлайна сохраняется в `reports/ndwi_baseline.json`.

## Production preparation

### ONNX

```bash
python commands.py export-onnx
```

- читает `weights/bundle.json` + `weights/lake_unet.ckpt`, собирает сеть заново и экспортирует
  её вместе с сигмоидой (на выходе — вероятности в `[0, 1]`, имя выхода `probabilities`);
- динамические оси: batch (и `height`/`width`, если `export.dynamic_spatial=true`);
- `onnx.checker` + сверка с PyTorch через `onnxruntime` (порог расхождения `export.atol`);
- в метаданные ONNX записываются порог, порядок каналов, масштаб reflectance и git commit.

Результат: `weights/lake_unet.onnx`. Эту же операцию обучение выполняет автоматически
(`train.export_onnx_after_training=true`).

### TensorRT

```bash
python commands.py export-tensorrt --dry-run   # показать команду trtexec
python commands.py export-tensorrt             # собрать движок (нужен GPU NVIDIA)
scripts/build_tensorrt.sh                      # то же в виде shell-скрипта
```

Сборка идёт из готового ONNX через `trtexec` (FP16 по умолчанию, `workspace_gb`, форма
`image:1x4x256x256`). Без TensorRT команда печатает понятную инструкцию с запуском контейнера
`nvcr.io/nvidia/tensorrt`. Результат: `weights/lake_unet.engine`.

### Комплектация поставки модели

Для запуска инференса нужны ровно два файла (плюс код инференса, у которого нет зависимостей
от обучения):

| Файл                     | Назначение                                                            |
| ------------------------ | --------------------------------------------------------------------- |
| `weights/lake_unet.onnx` | сеть с сигмоидой, вход `(N,4,H,W)`, выход вероятности `(N,1,H,W)`     |
| `weights/bundle.json`    | архитектура, порядок каналов, статистики нормализации, порог, метрики |

`weights/lake_unet.ckpt` нужен только для backend `torch` (без ONNX). Данные для инференса
(`data/example/`) генерируются командой `make-example` и/или версионируются в DVC.

## Infer

### Формат входных данных

- GeoTIFF, **4 канала** в порядке **B2, B3, B4, B8** (blue, green, red, NIR), 10 м;
- значения: `uint16` reflectance (масштаб 10 000) **или** `float32` reflectance в `[0, 1]`;
- обязательна геопривязка (CRS + transform);
- для сравнения двух дат снимки должны лежать в одной сетке (одинаковые CRS, размер и
  geotransform) — иначе команда завершится понятной ошибкой;
- при необходимости область интереса ограничивается `infer.roi` (bbox в EPSG:4326).

Пример данных и его формат:

```bash
python commands.py make-example            # data/example/<aoi>_<date>.tif + example_manifest.json
```

### Запуск

```bash
# одна дата
python commands.py infer --image-a data/example/chany_2023-07-15.tif

# две даты: площадь, абсолютное и относительное изменение
python commands.py infer \
    --image-a data/example/chany_2023-07-15.tif \
    --image-b data/example/chany_2024-07-20.tif \
    --date-a 2023-07-15 --date-b 2024-07-20
```

Результаты в `predictions/`:

```
<patch>_mask.tif           бинарная маска воды (uint8, с геопривязкой)
<patch>_probabilities.tif  карта вероятностей (включается infer.save_probabilities)
water_report.png           true colour | маска | overlay по каждой дате
water_report.json          площади (м², км²), дельта, относительное изменение, параметры модели
```

Пример отчёта:

```json
{
  "first": { "date": "2023-07-15", "area_km2": 1.9143, "water_fraction": 0.731 },
  "second": { "date": "2024-07-20", "area_km2": 1.6402, "water_fraction": 0.626 },
  "change": { "delta_km2": -0.2741, "delta_percent": -14.32 }
}
```

Полезные overrides:

```bash
python commands.py infer --image-a a.tif --overrides="infer.backend=torch,infer.tile_overlap=32"
python commands.py infer --image-a a.tif --overrides="infer.roi=[77.4,54.85,77.6,54.95]"
```

### Inference server

**MLflow Serving** (модель регистрируется в MLflow вместе с сигнатурой):

```bash
python commands.py serve-mlflow
# команда печатает model URI и готовый запрос request.json, затем:
mlflow models serve -m runs:/<run-id>/model --host 127.0.0.1 --port 5001 --env-manager local
curl -X POST http://127.0.0.1:5001/invocations \
     -H 'Content-Type: application/json' -d @weights/request.json
```

**Triton Inference Server** (ONNX напрямую, батчинг на стороне сервера):

```bash
python commands.py prepare-triton
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "$PWD/triton_repo:/models" --name triton-lake \
    nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models
```

Клиентский код печатает команда `prepare-triton`; модель отдаёт вероятности, порог берётся из
`bundle.json`. Конфиги серверов разведены по группам Hydra: `configs/serving/mlflow.yaml`
(`serving.*`) и `configs/triton/default.yaml` (`triton.*`), поэтому оба варианта описаны в одном
конфиге без переключения групп.

## Dependencies

- управление зависимостями — [uv](https://docs.astral.sh/uv/), декларация в `pyproject.toml`,
  lock-файл `uv.lock` (`uv sync` ставит ровно зафиксированные версии);
- лишних зависимостей нет: каждый пакет используется в коде (PyTorch, Lightning, torchmetrics,
  segmentation-models-pytorch, Hydra, Fire, MLflow, DVC, Rasterio, pystac-client, PyProj, NumPy,
  pandas, Matplotlib, tqdm, ONNX, onnxruntime);
- dev-группа: pre-commit, black, isort, flake8, pytest.

```bash
uv sync                    # окружение + dev-инструменты
uv sync --no-dev           # только рантайм
uv lock --upgrade          # обновить lock-файл
```

## Code quality

Настроены хуки pre-commit (`pre-commit run -a` проходит без ошибок):

- `pre-commit-hooks`: trailing-whitespace, end-of-file-fixer, check-yaml, check-toml,
  check-json, check-added-large-files, check-merge-conflict, debug-statements;
- `black` (line-length 100), `isort` (профиль black), `flake8` (`.flake8`);
- `prettier` для markdown / yaml / json.

```bash
pre-commit install
pre-commit run -a
```

Дополнительно: `pytest -q` (unit-тесты на геометрию площадей, метрики, тайлинг, разбиение).

## Data management (DVC)

Данные и артефакты модели версионируются [DVC](https://dvc.org/), в git попадают только
`.dvc`-метаданные и `dvc.yaml`:

```
dvc.yaml                      стадии download -> preprocess -> train
data/raw/scenes_manifest.json манифест сцен (аут стадии download)
data/processed/               датасет фрагментов (аут стадии preprocess)
weights/, plots/              артефакты обучения (ауты стадии train)
```

Удалённое хранилище — локальная директория (по умолчанию `../sentinel2-lake-dvc-storage`,
настраивается в `.dvc/config`); можно переключить на S3 / Google Drive:

```bash
dvc remote modify storage url /path/to/shared/storage
dvc repro                     # пересчитать стадии
dvc push                      # отправить данные и веса в хранилище
```

`commands.py train` и `commands.py infer` вызывают `dvc pull` через Python API
(`src/lake_segmentation/data/dvc_utils.py`), когда датасета нет локально. Если хранилище
недоступно, `train` автоматически собирает датасет из открытых источников
(`train.auto_preprocess=true`) — то есть загрузка данных из открытых источников встроена в
пайплайн и не требует доступа к приватному storage.

## Logging

- логирование в [MLflow](https://mlflow.org/) по адресу из `configs/mlflow/default.yaml`
  (`http://127.0.0.1:8080`): функции потерь, метрики, learning rate, гиперпараметры, git commit
  и полный конфиг как артефакт;
- графики и логи складываются в `plots/` (`losses.png`, `metrics.png`, `area_error.png`,
  `learning_rate.png`, `history.csv`, `train.log`, `val_predictions.png`) и загружаются в MLflow;
- если tracking-сервер недоступен, включается `CSVLogger` (`mlflow.allow_fallback=true`).

## Результаты

Заполняется после запуска обучения в Google Colab (ноутбук `notebooks/colab_train.ipynb`).

| Модель              | IoU | Dice | Precision | Recall | Relative Area Error, % |
| ------------------- | --- | ---- | --------- | ------ | ---------------------- |
| NDWI (порог по val) | —   | —    | —         | —      | —                      |
| U-Net ResNet34      | —   | —    | —         | —      | —                      |

Метрики по каждому озеру и графики обучения — в `plots/` и в MLflow-эксперименте
`lake-water-segmentation`.

## Лицензия

MIT.
