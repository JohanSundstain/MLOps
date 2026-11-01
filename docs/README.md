# Документация проекта

`sentinel2-lake-segmentation` — система автоматической сегментации водной поверхности озёр на
снимках Sentinel-2 и анализа изменения их площади между двумя датами.

Здесь описано, **что проект умеет**, **как он устроен внутри**, **как его развернуть** и **какие
инструменты** для чего используются. Основной README репозитория (описание задачи, метрики,
результаты) лежит в корне: [../README.md](../README.md).

## Что умеет проект

| Возможность                                           | Команда                                                | Что получается                                                      |
| ----------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------- |
| Найти лучшие снимки Sentinel-2 для 12 озёр            | `python commands.py download`                          | `data/raw/scenes_manifest.json`                                     |
| Собрать датасет фрагментов 256×256 с масками воды     | `python commands.py preprocess`                        | `data/processed/` (441 патч, сплит 70/15/15)                        |
| Обучить U-Net (ResNet34)                              | `python commands.py train`                             | `weights/lake_unet.ckpt`, `weights/bundle.json`, графики в `plots/` |
| Оценить модель на отложенном тесте                    | `python commands.py evaluate`                          | метрики по всему тесту и по каждому озеру                           |
| Сравнить с индексным бейзлайном NDWI                  | `python commands.py baseline`                          | `reports/ndwi_baseline.json`                                        |
| Экспортировать модель в ONNX и проверить              | `python commands.py export-onnx`                       | `weights/lake_unet.onnx`                                            |
| Собрать TensorRT-движок                               | `python commands.py export-tensorrt`                   | `weights/lake_unet.engine` (нужен GPU NVIDIA)                       |
| Посчитать площадь воды и её изменение по двум снимкам | `python commands.py infer --image-a ... --image-b ...` | маски `.tif`, `water_report.json`, `water_report.png`               |
| Поднять MLflow-сервер с моделью                       | `python commands.py serve-mlflow`                      | готовый `mlflow models serve` + `request.json`                      |
| Подготовить Triton model repository                   | `python commands.py prepare-triton`                    | `triton_repo/` + `config.pbtxt`                                     |
| Сгенерировать пример пары снимков для демо            | `python commands.py make-example`                      | `data/example/*.tif`                                                |
| Посмотреть итоговый конфиг / пути / сводку данных     | `show-config`, `paths`, `dataset-info`                 | печать в консоль                                                    |

Итоговые метрики обученной модели на тесте (подробности — в разделе
[Результаты](../README.md#результаты) и в [`training.md`](training.md)):

| Метрика                | U-Net ResNet34  | NDWI-бейзлайн   | Ориентир |
| ---------------------- | --------------- | --------------- | -------- |
| IoU (micro)            | 0,9627          | 0,9688          | ≥ 0,75   |
| Dice                   | 0,9810          | 0,9841          | ≥ 0,85   |
| Precision / Recall     | 0,9762 / 0,9858 | 0,9896 / 0,9788 | ≥ 0,85   |
| Relative Area Error, % | 4,94            | 5,88            | ≤ 10     |

## Быстрый старт

```bash
git clone <repo-url> && cd sentinel2-lake-segmentation
uv sync                                  # окружение + dev-инструменты
python commands.py download              # ~2 мин: поиск сцен
python commands.py preprocess            # ~7 мин: датасет фрагментов
python commands.py train --overrides="train.max_epochs=30,train.batch_size=32,train.precision=16-mixed"
python commands.py infer \
    --image-a data/example/chany_2023-07-15.tif \
    --image-b data/example/chany_2024-07-20.tif
```

## Карта документации

| Документ                             | О чём                                                                    | Когда читать                          |
| ------------------------------------ | ------------------------------------------------------------------------ | ------------------------------------- |
| [`architecture.md`](architecture.md) | Устройство кода: слои, модули, поток данных, конфиги Hydra, артефакты    | Понять, где что лежит и как расширять |
| [`data.md`](data.md)                 | Данные: Sentinel-2, STAC, WorldCover, нарезка, сплит, нормализация, DVC  | Разобраться, откуда берётся датасет   |
| [`training.md`](training.md)         | Модель, лоссы, оптимизация, метрики, подбор порога, MLflow               | Понять качество и как его повторить   |
| [`inference.md`](inference.md)       | Инференс, площади, изменение площади, ONNX/TensorRT, серверы             | Запускать модель на своих снимках     |
| [`deployment.md`](deployment.md)     | Развёртывание локально, в Colab, в продакшене, чек-листы, разбор ошибок  | Поднять проект с нуля                 |
| [`tools.md`](tools.md)               | Подробно про каждый инструмент: зачем, где используется, подводные камни | Понять стек и защитить проект         |

## Структура репозитория кратко

```
commands.py               единственная точка входа CLI
configs/                  Hydra-конфиги по стадиям (data, preprocess, model, train, infer, export, ...)
src/lake_segmentation/    сам пакет (данные, модель, обучение, инференс, экспорт, сервинг, бейзлайн)
dvc.yaml                  стадии DVC: download -> preprocess -> train
notebooks/                обучение в Google Colab
scripts/                  shell-обёртки для TensorRT и загрузки данных
tests/                    unit-тесты (геометрия, метрики, тайлинг, сплит, импорты)
docs/                     эта документация
```

## Глоссарий

| Термин                     | Значение                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------ |
| **AOI** (area of interest) | Прямоугольная область вокруг озера, заданная bbox в EPSG:4326                        |
| **Патч (фрагмент)**        | Кусок снимка 256×256 пикселей (6,5 км × 6,5 км при 10 м) с маской 0/1                |
| **SCL**                    | Scene Classification Layer Sentinel-2: карта классов пикселей (облака, тени, вода)   |
| **WorldCover**             | Глобальная карта земного покрова ESA, 10 м, класс `80` — постоянные водоёмы          |
| **NDWI**                   | `(B3 − B8) / (B3 + B8)` — водный индекс, бейзлайн проекта                            |
| **bundle**                 | `weights/bundle.json` — «карточка модели»: архитектура, нормализация, порог, метрики |
| **Сцена**                  | Один снимок Sentinel-2 L2A (тайл 110×110 км), выбранный для AOI и даты               |
| **Порог бинаризации**      | Значение вероятности, выше которого пиксель считается водой (подобран на валидации)  |
| **micro / macro**          | Метрики по сумме пикселей всего набора / среднее по изображениям                     |

## Версии основных библиотек

Зафиксированы в `uv.lock`, проверены на Python 3.11–3.13:

| Пакет                       | Версия              | Роль                                 |
| --------------------------- | ------------------- | ------------------------------------ |
| torch                       | 2.15.0              | нейросети                            |
| lightning                   | 2.6.6               | фреймворк обучения                   |
| torchmetrics                | 1.9.0               | базовый класс streaming-метрик       |
| segmentation-models-pytorch | 0.6.0               | архитектура U-Net + encoder ResNet34 |
| hydra-core                  | 1.3.8               | конфигурация                         |
| fire                        | 0.7.1               | CLI                                  |
| mlflow                      | 2.22.5              | трекинг экспериментов и сервинг      |
| dvc                         | 3.67.1              | версионирование данных и весов       |
| rasterio                    | 1.5.2 (GDAL 3.12.2) | чтение/запись GeoTIFF и COG          |
| pystac-client               | 0.9.1               | поиск сцен в STAC                    |
| pyproj                      | 3.8.0               | геодезические расчёты                |
| onnx / onnxruntime          | 1.23.2 / 1.31.1     | экспорт и быстрый инференс           |
| numpy / pandas              | 2.5.4 / 2.3.3       | массивы и манифест                   |
| matplotlib                  | 3.11.2              | графики                              |
