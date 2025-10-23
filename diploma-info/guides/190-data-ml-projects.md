# Организация проектов с данными и ML-моделями

> **Для кого:** Студенты, работающие над проектами с датасетами, ML-моделями, IoT-данными или аналитическими системами.

> **Дополняет:** [160-project-organization.md](160-project-organization.md) — специфика для data-heavy проектов.

## 1. Структура репозитория для ML-проекта

### 1.1 Расширенная структура

```text
ml-project/
├── README.md
├── .gitignore              # Специальный для ML-проектов
├── .env.example
├── requirements.txt
│
├── data/                   # Данные (НЕ коммитим в Git!)
│   ├── .gitkeep
│   ├── README.md           # Описание источников данных
│   ├── raw/                # Исходные данные
│   ├── processed/          # Обработанные данные
│   ├── external/           # Внешние источники
│   └── interim/            # Промежуточные результаты
│
├── models/                 # Обученные модели (НЕ коммитим!)
│   ├── .gitkeep
│   ├── README.md           # Описание моделей и версий
│   ├── baseline/           # Baseline модели
│   ├── experiments/        # Экспериментальные модели
│   └── production/         # Production-ready модели
│
├── notebooks/              # Jupyter notebooks для исследований
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_experiments.ipynb
│
├── src/                    # Production код
│   ├── data/               # Загрузка и обработка данных
│   │   ├── loaders.py
│   │   └── preprocessing.py
│   ├── features/           # Feature engineering
│   │   └── build_features.py
│   ├── models/             # Обучение и инференс
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/              # Вспомогательные функции
│
├── tests/                  # Тесты
│   └── test_preprocessing.py
│
├── configs/                # Конфигурации моделей
│   ├── model_config.yaml
│   └── data_config.yaml
│
├── reports/                # Результаты и отчёты
│   ├── figures/            # Графики для диплома
│   ├── metrics/            # Метрики экспериментов
│   └── experiments.json    # Лог экспериментов
│
└── scripts/                # Скрипты запуска
    ├── train.sh
    └── evaluate.sh
```

### 1.2 Ключевые принципы

**Что НЕ коммитим в Git:**
- ❌ Датасеты (`data/`)
- ❌ Обученные модели (`models/*.pkl`, `*.h5`, `*.pt`)
- ❌ Результаты экспериментов (большие файлы)
- ❌ Кэш и временные файлы

**Что ОБЯЗАТЕЛЬНО коммитим:**
- ✅ Код обработки данных (`src/`)
- ✅ Конфигурации моделей (`configs/`)
- ✅ Notebooks с результатами (очищенные outputs)
- ✅ Метаданные о данных и моделях (`README.md` в папках)
- ✅ Скрипты воспроизведения экспериментов

## 2. Специальный .gitignore для ML-проектов

```gitignore
# ============================================
# Данные
# ============================================
data/raw/*
data/processed/*
data/interim/*
data/external/*
!data/**/.gitkeep
!data/**/README.md

# Форматы данных
*.csv
*.tsv
*.xlsx
*.parquet
*.feather
*.h5
*.hdf5
*.db
*.sqlite

# ============================================
# Модели и веса
# ============================================
models/**/*.pkl
models/**/*.joblib
models/**/*.h5
models/**/*.hdf5
models/**/*.pt
models/**/*.pth
models/**/*.ckpt
models/**/*.pb
models/**/*.onnx
!models/**/README.md

# Checkpoints
checkpoints/
*.ckpt
*.checkpoint

# ============================================
# Эксперименты и логи
# ============================================
mlruns/
runs/
logs/
wandb/
.neptune/
lightning_logs/

# Кэш
__pycache__/
*.pyc
.ipynb_checkpoints/
.cache/

# ============================================
# Окружения
# ============================================
venv/
env/
.venv/
.conda/

# ============================================
# IDE и системные
# ============================================
.vscode/
.idea/
.DS_Store
*.swp

# ============================================
# Результаты (большие файлы)
# ============================================
reports/experiments/*
!reports/experiments/.gitkeep
reports/figures/*.png
reports/figures/*.jpg
# Но оставляем итоговые графики для диплома:
!reports/figures/final_*.png
```

## 3. Хранение больших файлов

### 3.1 Проблема

Git не предназначен для больших файлов (>100 MB). Датасеты и модели нужно хранить отдельно.

### 3.2 Решения

**Вариант 1: DVC (Data Version Control)** — рекомендуется

**Установка:**
```bash
pip install dvc
dvc init
```

**Добавление данных:**
```bash
# Добавить данные под версионирование DVC
dvc add data/raw/dataset.csv

# Это создаст data/raw/dataset.csv.dvc (его коммитим в Git)
# Сам файл добавится в .gitignore

# Закоммитить метаданные
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Add dataset to DVC"
```

**Настройка удалённого хранилища:**
```bash
# Google Drive (бесплатно)
dvc remote add -d storage gdrive://your-folder-id

# Или локальная папка
dvc remote add -d storage /path/to/external/drive

# Загрузить данные
dvc push
```

**Использование другим человеком:**
```bash
git clone <repo>
dvc pull  # Скачает данные из remote storage
```

**Вариант 2: Git LFS** — для файлов до 2GB

```bash
git lfs install
git lfs track "*.csv"
git lfs track "*.pkl"
git add .gitattributes
```

**Вариант 3: Облачное хранилище + README**

Если DVC сложно:
1. Загрузить данные на Google Drive / Яндекс.Диск
2. Получить публичную ссылку
3. Добавить в `data/README.md`:

```markdown
# Данные

## Где скачать

Датасет находится по ссылке: [Google Drive](https://drive.google.com/...)

## Как использовать

1. Скачать файл `dataset.csv`
2. Положить в `data/raw/dataset.csv`
3. Запустить обработку: `python src/data/preprocessing.py`
```

### 3.3 Что выбрать

| Ситуация | Решение |
|----------|---------|
| Датасет < 100 MB | Можно закоммитить в Git (но лучше DVC) |
| Датасет 100 MB - 2 GB | Git LFS или DVC |
| Датасет > 2 GB | DVC + облако |
| Работаете один | Облако + README |
| Работаете в команде | DVC обязательно |

## 4. Управление моделями

### 4.1 Версионирование моделей

**Структура:**
```
models/
├── README.md               # Описание всех моделей
├── baseline/
│   ├── rf_v1.pkl          # Baseline Random Forest
│   └── metadata.json      # Метаданные модели
├── experiments/
│   ├── xgb_exp1.pkl
│   ├── lstm_exp2.h5
│   └── ...
└── production/
    ├── best_model.pkl     # Лучшая модель
    └── metadata.json
```

**metadata.json (пример):**
```json
{
  "model_name": "XGBoost Classifier",
  "version": "v2.1",
  "date_trained": "2025-10-20",
  "data_version": "v1.2",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1
  },
  "metrics": {
    "accuracy": 0.89,
    "f1_score": 0.87,
    "precision": 0.88,
    "recall": 0.86
  },
  "features_used": ["age", "income", "credit_score", "..."],
  "training_time_minutes": 12,
  "notes": "Добавлены географические признаки"
}
```

**models/README.md:**
```markdown
# Модели

## Production модель

- **Файл:** `production/best_model.pkl`
- **Тип:** XGBoost Classifier v2.1
- **Дата:** 2025-10-20
- **Метрики:** F1=0.87, Accuracy=0.89
- **Описание:** Лучшая модель после экспериментов с географическими признаками

## История экспериментов

| Версия | Дата | Тип | F1 | Примечание |
|--------|------|-----|----|-----------| 
| v1.0 | 2025-09-15 | Random Forest | 0.82 | Baseline |
| v1.5 | 2025-10-01 | XGBoost | 0.85 | Тюнинг параметров |
| v2.0 | 2025-10-15 | XGBoost | 0.86 | Добавлены временные признаки |
| v2.1 | 2025-10-20 | XGBoost | 0.87 | Географические признаки ✓ |

## Как использовать

```python
import joblib
model = joblib.load('models/production/best_model.pkl')
predictions = model.predict(X_test)
```
```

### 4.2 Сохранение моделей с метаданными

```python
import joblib
import json
from datetime import datetime

def save_model_with_metadata(model, filepath, metadata):
    """Сохранить модель с метаданными."""
    # Сохранить модель
    joblib.dump(model, filepath)
    
    # Добавить timestamp
    metadata['saved_at'] = datetime.now().isoformat()
    
    # Сохранить метаданные
    metadata_path = filepath.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Модель сохранена: {filepath}")
    print(f"✓ Метаданные: {metadata_path}")

# Использование:
metadata = {
    'model_name': 'XGBoost',
    'version': 'v2.1',
    'hyperparameters': {'n_estimators': 200, 'max_depth': 5},
    'metrics': {'f1': 0.87, 'accuracy': 0.89}
}

save_model_with_metadata(
    model=xgb_model,
    filepath='models/experiments/xgb_v2.1.pkl',
    metadata=metadata
)
```

## 5. Логирование экспериментов

### 5.1 Простой способ: JSON-файл

```python
import json
from datetime import datetime

def log_experiment(experiment_data, log_file='reports/experiments.json'):
    """Добавить эксперимент в лог."""
    experiment_data['timestamp'] = datetime.now().isoformat()
    
    # Прочитать существующие эксперименты
    try:
        with open(log_file, 'r') as f:
            experiments = json.load(f)
    except FileNotFoundError:
        experiments = []
    
    # Добавить новый
    experiments.append(experiment_data)
    
    # Сохранить
    with open(log_file, 'w') as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Эксперимент записан: {log_file}")

# Использование:
log_experiment({
    'model': 'XGBoost',
    'version': 'v2.1',
    'hyperparameters': {'n_estimators': 200, 'max_depth': 5},
    'data_version': 'v1.2',
    'metrics': {'f1': 0.87, 'accuracy': 0.89},
    'notes': 'Добавлены географические признаки'
})
```

### 5.2 Продвинутый способ: MLflow (опционально)

Если экспериментов много:

```bash
pip install mlflow
```

```python
import mlflow

mlflow.set_experiment("fraud_detection")

with mlflow.start_run():
    # Логировать параметры
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    
    # Обучить модель
    model.fit(X_train, y_train)
    
    # Логировать метрики
    mlflow.log_metric("f1", 0.87)
    mlflow.log_metric("accuracy", 0.89)
    
    # Сохранить модель
    mlflow.sklearn.log_model(model, "model")
```

Просмотр экспериментов:
```bash
mlflow ui
# Откроется веб-интерфейс на http://localhost:5000
```

## 6. Конфигурации для ML-проектов

### 6.1 configs/model_config.yaml

```yaml
# Конфигурация модели
model:
  type: "XGBoost"
  hyperparameters:
    n_estimators: 200
    max_depth: 5
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
  random_state: 42

# Конфигурация данных
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  train_file: "train.csv"
  test_file: "test.csv"
  target_column: "is_fraud"
  
# Разделение данных
split:
  test_size: 0.2
  val_size: 0.1
  stratify: true

# Feature engineering
features:
  categorical: ["region", "category", "payment_method"]
  numerical: ["amount", "age", "income"]
  create_interactions: true

# Обучение
training:
  batch_size: 1024
  early_stopping_rounds: 50
  verbose: 100

# Сохранение
paths:
  model_save: "models/experiments/xgb_latest.pkl"
  metrics_save: "reports/metrics/latest.json"
  figures_save: "reports/figures"
```

### 6.2 Чтение конфига в коде

```python
import yaml

def load_config(config_path='configs/model_config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Использование
config = load_config()

model = XGBClassifier(
    n_estimators=config['model']['hyperparameters']['n_estimators'],
    max_depth=config['model']['hyperparameters']['max_depth'],
    random_state=config['model']['random_state']
)
```

## 7. data/README.md — документация данных

Каждый проект должен иметь описание данных:

```markdown
# Данные проекта

## Источник

- **Название:** Fraud Detection Dataset
- **Источник:** [Kaggle](https://www.kaggle.com/...)
- **Дата получения:** 2025-10-01
- **Лицензия:** CC BY 4.0

## Описание

Датасет содержит 100,000 записей заявок на социальные выплаты с метками мошенничества.

## Структура

### raw/
- `applications.csv` — исходные заявки (50,000 записей)
- `users.csv` — информация о пользователях (25,000 записей)
- `payments.csv` — история выплат (100,000 записей)

### processed/
- `train.csv` — обучающая выборка (70%)
- `test.csv` — тестовая выборка (30%)
- `features_v1.csv` — с feature engineering

## Размеры

- `raw/` — 250 MB
- `processed/` — 180 MB

## Как получить данные

### Вариант 1: Автоматически (DVC)
```bash
dvc pull
```

### Вариант 2: Вручную
1. Скачать с [Google Drive](https://drive.google.com/...)
2. Распаковать в `data/raw/`
3. Запустить обработку: `python src/data/make_dataset.py`

## Предобработка

```bash
# Обработать данные
python src/data/preprocessing.py

# Создать признаки
python src/features/build_features.py
```

## Статистика

- **Всего записей:** 100,000
- **Пропуски:** 
  - `income`: 2% (2,000 записей)
  - `age`: 0.5% (500 записей)
- **Дисбаланс классов:** 95% норма / 5% мошенничество
- **Временной период:** 2020-2024

## Версии

- **v1.0** (2025-09-01): исходная версия
- **v1.1** (2025-09-15): добавлены временные признаки
- **v1.2** (2025-10-01): добавлены географические признаки ← текущая
```

## 8. Воспроизводимость экспериментов

### 8.1 Фиксация random seed

**Во всех скриптах:**
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Фиксировать random seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# В начале каждого скрипта
set_seed(42)
```

### 8.2 Скрипт воспроизведения

**scripts/reproduce.sh:**
```bash
#!/bin/bash
# Скрипт для полного воспроизведения результатов

echo "========================================="
echo "Воспроизведение эксперимента"
echo "========================================="

# 1. Обработка данных
echo "[1/4] Обработка данных..."
python src/data/preprocessing.py

# 2. Feature engineering
echo "[2/4] Feature engineering..."
python src/features/build_features.py

# 3. Обучение модели
echo "[3/4] Обучение модели..."
python src/models/train.py --config configs/model_config.yaml

# 4. Оценка
echo "[4/4] Оценка модели..."
python src/models/evaluate.py

echo "========================================="
echo "Готово! Результаты в reports/"
echo "========================================="
```

## 9. Чек-листы

### 9.1 Перед началом проекта

- [ ] Создана структура папок (data/, models/, notebooks/, src/)
- [ ] Настроен `.gitignore` для ML-проекта
- [ ] Выбран способ хранения больших файлов (DVC / облако)
- [ ] Создан `data/README.md` с описанием источников
- [ ] Создан `models/README.md` для версионирования моделей
- [ ] Зафиксирован random seed в конфигах

### 9.2 После каждого эксперимента

- [ ] Эксперимент залогирован (JSON / MLflow)
- [ ] Модель сохранена с метаданными
- [ ] Метрики записаны
- [ ] Обновлен `models/README.md`
- [ ] Графики сохранены в `reports/figures/`

### 9.3 Перед коммитом

- [ ] Очищены outputs в notebooks
- [ ] Данные и модели НЕ в коммите (проверить `git status`)
- [ ] Обновлены README файлы
- [ ] Конфиги актуальны
- [ ] Скрипты воспроизведения работают

### 9.4 Перед защитой

- [ ] Результаты воспроизводимы (протестировано на чистом окружении)
- [ ] Все данные доступны (DVC / облако с инструкциями)
- [ ] Финальная модель в `models/production/`
- [ ] Ключевые графики в `reports/figures/final_*.png`
- [ ] Документация полная и понятная

## 10. Типичные ошибки

### ❌ Что НЕ делать

1. **Коммитить данные и модели в Git**
   - Репозиторий раздуется до GB
   - Git станет медленным
   - Решение: DVC или облако

2. **Не документировать эксперименты**
   - Через месяц забудете, что делали
   - Решение: вести лог экспериментов

3. **Хардкодить пути**
   ```python
   # Плохо
   df = pd.read_csv('/Users/myname/data/dataset.csv')
   
   # Хорошо
   df = pd.read_csv(config['data']['raw_dir'] + '/dataset.csv')
   ```

4. **Не фиксировать random seed**
   - Результаты не воспроизводятся
   - Решение: `set_seed(42)` везде

5. **Удалять неудачные эксперименты**
   - Теряется опыт "что не работает"
   - Решение: логировать всё, отмечать неудачное

## 11. Связь с другими гайдами

- **Общая организация:** [160-project-organization.md](160-project-organization.md)
- **Jupyter Notebooks:** [200-jupyter-notebooks.md](200-jupyter-notebooks.md)
- **Работа с данными:** [180-data-work.md](180-data-work.md)
- **Примеры кода:** [210-code-examples.md](210-code-examples.md)

---

**Помните:** Главная цель — чтобы через месяц (или на защите) вы могли воспроизвести все результаты. Документируйте, версионируйте, автоматизируйте.
