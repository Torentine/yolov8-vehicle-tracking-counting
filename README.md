# Vehicle Detection, Tracking and Counting

Проект решает задачу детекции, трекинга и подсчета автомобилей на видео из `drivingDataset`.

Основной файл решения: `vehicle_tracking_counting.ipynb`.

## Возможности

- Детекция автомобилей YOLOv8 Ultralytics, класс COCO `car`.
- Трекинг объектов через встроенный BoT-SORT.
- Отображение bounding boxes, класса, confidence и уникального `track_id`.
- Подсчет пересечения заданной линии по центру bounding box.
- Учет пересечения только в одном направлении.
- Защита от повторного подсчета одного и того же `track_id`.
- Визуализация линии, центров объектов и общего счетчика на кадре.
- CSV-отчет с `track_id`, временем пересечения, кадром, координатами центра и распознанным номером.
- Дополнительный модуль детекции номерных знаков YOLOv8 + распознавание текста через EasyOCR.

## Структура проекта

```text
.
├── vehicle_tracking_counting.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
├── dataset_splits/
│   ├── train.txt
│   ├── test.txt
│   └── dataset_split.csv
└── drivingDataset/
    ├── normalDay/
    ├── normalNight/
    └── rainyDay/
```

`drivingDataset` входит в проект и нужен для запуска ноутбука. Так как видеофайлы тяжелые, они должны храниться через Git LFS. Для этого в проект добавлен файл `.gitattributes`, где `drivingDataset/**/*.mp4` отмечены как LFS-файлы.

Текущий размер датасета: около `1.125 GB`, 60 видео.

## Разделение датасета

Датасет разделен на train/test через manifest-файлы без перемещения видео:

- `dataset_splits/train.txt` - 48 видео.
- `dataset_splits/test.txt` - 12 видео.
- `dataset_splits/dataset_split.csv` - полный список всех 60 видео с колонками `path`, `split`, `category`.

Разделение стратифицировано по категориям:

| Категория | Train | Test | Всего |
|---|---:|---:|---:|
| `normalDay` | 16 | 4 | 20 |
| `normalNight` | 16 | 4 | 20 |
| `rainyDay` | 16 | 4 | 20 |
| Всего | 48 | 12 | 60 |

## Установка

Рекомендуется использовать Python 3.10-3.13.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Запуск ноутбука

```bash
jupyter notebook vehicle_tracking_counting.ipynb
```

После открытия ноутбука запустите все ячейки сверху вниз.

По умолчанию используется видео:

```python
VIDEO_PATH = Path('drivingDataset/normalDay/nD_5.mp4')
```

Для обработки другого видео измените `VIDEO_PATH` в ячейке настроек.

Для обработки всего видео поставьте:

```python
MAX_FRAMES = None
```

Для быстрой проверки можно оставить ограничение, например:

```python
MAX_FRAMES = 600
```

## CUDA/GPU

Устройство выбирается автоматически:

```python
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
```

Если CUDA доступна, Ultralytics и EasyOCR будут использовать GPU. Если CUDA недоступна, код работает на CPU, но медленнее.

## Модель детекции номеров

Ноутбук автоматически скачивает специализированную YOLOv8-модель номерных знаков с Hugging Face:

```python
PLATE_MODEL_REPO_ID = 'Koushim/yolov8-license-plate-detection'
PLATE_MODEL_FILENAME = 'best.pt'
PLATE_DETECTOR_MODEL = Path('models/license_plate_detector.pt')
```

После скачивания модель хранится локально:

```text
models/license_plate_detector.pt
```

Если интернет недоступен, файл можно скачать вручную:

```text
https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt
```

Сохраните его как:

```text
models/license_plate_detector.pt
```

## Результаты работы

После запуска ноутбука создается папка `outputs`.

В нее сохраняются:

- размеченное видео `*_tracked_counted.mp4`;
- CSV-отчет `*_report.csv`.

CSV содержит основные поля:

- `track_id` - ID трека;
- `class_id` - ID класса COCO;
- `class_name` - имя класса;
- `frame` - номер кадра пересечения;
- `time_sec` - время пересечения в секундах;
- `center_x`, `center_y` - центр bounding box;
- `plate_text` - распознанный номер, если OCR сработал.

## Проверка проекта

Быстрая проверка структуры ноутбука:

```bash
python -c "import json, ast, pathlib; p=pathlib.Path('vehicle_tracking_counting.ipynb'); nb=json.loads(p.read_text(encoding='utf-8')); cells=nb['cells']; src=[]; ok=[]; [src.extend(c.get('source',[])+['\\n']) for c in cells if c.get('cell_type')=='code']; ast.parse(''.join(src)); [ok.append(i>0 and cells[i-1].get('cell_type')=='markdown' and ''.join(cells[i-1].get('source',[])).strip()) for i,c in enumerate(cells) if c.get('cell_type')=='code']; assert all(ok); print('Notebook OK')"
```

Проверка split-файлов датасета:

```bash
python -c "from pathlib import Path; import csv, collections; all_videos={p.as_posix() for p in Path('drivingDataset').rglob('*.mp4')}; train=[x.strip() for x in Path('dataset_splits/train.txt').read_text().splitlines() if x.strip()]; test=[x.strip() for x in Path('dataset_splits/test.txt').read_text().splitlines() if x.strip()]; assert len(train)==48 and len(test)==12; assert len(train+test)==len(set(train+test))==60; assert set(train+test)==all_videos; assert not (set(train)&set(test)); rows=list(csv.DictReader(Path('dataset_splits/dataset_split.csv').open(newline='', encoding='utf-8'))); assert len(rows)==60; print('Dataset split OK')"
```

## Что заливать на Git

Заливать:

- `vehicle_tracking_counting.ipynb`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `.gitattributes`
- `dataset_splits/train.txt`
- `dataset_splits/test.txt`
- `dataset_splits/dataset_split.csv`
- `drivingDataset/normalDay/*.mp4`
- `drivingDataset/normalNight/*.mp4`
- `drivingDataset/rainyDay/*.mp4`

Не заливать:

- `outputs/`
- `models/`
- `*.pt`, включая `yolov8n.pt` и `models/license_plate_detector.pt`
- `.idea/`
- `.ipynb_checkpoints/`

Видео из `drivingDataset` заливать нужно, но именно через Git LFS, а не как обычные Git blob-файлы.

## Команды для Git

Сначала установите Git LFS, если он еще не установлен:

```bash
git lfs install
```

Инициализация репозитория и настройка LFS:

```bash
git init
git lfs install
git lfs track "drivingDataset/**/*.mp4"
git lfs track "*.pt"
git lfs track "*.onnx"
git lfs track "*.engine"
git add .gitattributes .gitignore README.md requirements.txt vehicle_tracking_counting.ipynb dataset_splits/ drivingDataset/
git commit -m "Add vehicle tracking notebook"
```

Подключение удаленного репозитория:

```bash
git remote add origin <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
git branch -M main
git push -u origin main
```

Перед коммитом проверьте, что тяжелые файлы не попали в индекс:

```bash
git status
git lfs ls-files
```

В `git lfs ls-files` должны отображаться `.mp4` файлы из `drivingDataset`. В `git status` не должно быть `outputs`, `models` и локальных `.pt` файлов.

После клонирования другим пользователям нужно загрузить LFS-файлы:

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd <ИМЯ_РЕПОЗИТОРИЯ>
git lfs pull
python -m pip install -r requirements.txt
jupyter notebook vehicle_tracking_counting.ipynb
```
