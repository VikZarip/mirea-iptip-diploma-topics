# Организация проектного репозитория

> **Цель:** Быстро и грамотно организовать репозиторий дипломного проекта так, чтобы работа была воспроизводимой, понятной и легко демонстрируемой.

## 1. Структура репозитория

### 1.1 Минимальная структура (подходит для любого стека)

```text
diploma-project/
  README.md              # главный документ: установка, запуск, тестирование
  .gitignore            # что не коммитить в Git
  .env.example          # шаблон переменных окружения (без секретов!)
  
  src/                  # исходный код по логическим слоям
  tests/                # автоматические тесты
  data/                 # данные (обычно в .gitignore!)
  docs/                 # документация, диаграммы, схемы
  
  [файл зависимостей]   # requirements.txt / package.json / pom.xml
  [файл сборки]         # Makefile / package.json scripts
```

**Принципы:**
- Разделение: код ≠ данные ≠ документация
- Самодокументирование: из имён папок понятно их назначение
- Воспроизводимость: любой может склонировать и запустить

### 1.2 Примеры для разных типов проектов

**ML/Data Science:**
```text
project/
  notebooks/            # Jupyter notebooks для исследований
  src/
    data/              # загрузка и обработка данных
    models/            # обучение моделей
    features/          # feature engineering
  data/
    raw/               # исходные данные
    processed/         # обработанные данные
  models/              # сохранённые модели (.pkl, .h5)
  reports/             # результаты экспериментов
```

**Backend API:**
```text
project/
  src/
    api/               # контроллеры, эндпоинты
    domain/            # бизнес-логика, сервисы
    data/              # работа с БД, репозитории
  tests/
    unit/
    integration/
  migrations/          # миграции БД
```

**Full-stack приложение:**
```text
project/
  frontend/            # React/Vue/Angular
    src/
    public/
    package.json
  backend/             # Express/Django/FastAPI
    src/
    requirements.txt
  docker-compose.yml   # локальная инфраструктура
```

## 2. Ключевые файлы проекта

### 2.1 README.md — лицо проекта

**Обязательные разделы:**

```markdown
# Название проекта

Краткое описание (1-2 предложения): что делает и зачем.

## Требования
- Язык и версия (Python 3.9+, Node.js 18+)
- Основные зависимости

## Установка

### Клонирование
```bash
git clone https://github.com/username/project.git
cd project
```

### Настройка окружения
[команды для создания venv, установки зависимостей]

### Конфигурация
Скопируйте `.env.example` в `.env` и заполните переменные.

## Запуск

### Локальный запуск
```bash
make run
# или: npm start, python main.py
```

### Тестирование
```bash
make test
```

## Структура проекта
Краткое описание основных папок.

## Результаты
- Ключевые метрики
- Ссылки на визуализации/отчёты

## Автор
Ваше имя, контакты, группа
```

**Правило:** README должен позволять запустить проект за 5-10 минут без дополнительных вопросов.

### 2.2 .gitignore — что не коммитить

**Универсальный шаблон:**

```gitignore
# Данные
data/raw/*
data/processed/*
*.csv
*.db
*.sqlite

# Модели и веса
models/*.pkl
models/*.h5
*.pt
*.pth

# Окружения
venv/
env/
.venv/
node_modules/

# Конфиги и секреты
.env
secrets/
*.key
config.local.*

# IDE
.vscode/
.idea/
*.swp

# Системные
.DS_Store
Thumbs.db
__pycache__/
*.pyc

# Результаты экспериментов (тяжёлые файлы)
reports/experiments/*
!reports/experiments/.gitkeep
```

**Почему это важно:**
- Данные могут быть большими (GB) или конфиденциальными
- Модели занимают много места
- Секреты в Git = угроза безопасности

**Что ОБЯЗАТЕЛЬНО коммитить:**
- Исходный код (`src/`, `notebooks/`)
- Конфигурационные файлы (`.env.example`, `config.yaml.example`)
- Тесты
- Документацию
- Примеры данных (маленькие, для демонстрации)

### 2.3 .env и конфигурация

**Плохо (хардкод):**
```python
DB_URL = "postgresql://user:password@localhost/db"
API_KEY = "sk-abc123xyz"
```

**Хорошо:**

**.env** (локально, в .gitignore):
```bash
DATABASE_URL=postgresql://user:password@localhost/db
API_KEY=sk-abc123xyz
MODEL_PATH=models/model_v1.pkl
RANDOM_SEED=42
```

**.env.example** (в Git):
```bash
DATABASE_URL=postgresql://user:pass@host/db
API_KEY=your_api_key_here
MODEL_PATH=models/model_v1.pkl
RANDOM_SEED=42
```

**Чтение в коде (Python):**
```python
from dotenv import load_dotenv
import os

load_dotenv()
db_url = os.getenv('DATABASE_URL')
```

## 3. Слоистая архитектура кода

### 3.1 Зачем нужны слои

**Проблема:** Весь код в одном файле → сложно тестировать, переиспользовать, понимать.

**Решение:** Разделить по ответственности.

```
┌─────────────────────────┐
│  Presentation (UI/API)  │  ← точка входа, интерфейс
├─────────────────────────┤
│  Application (Логика)   │  ← сценарии, пайплайны
├─────────────────────────┤
│  Domain (Модели)        │  ← бизнес-логика, алгоритмы
├─────────────────────────┤
│  Data (Хранение)        │  ← работа с данными, БД
└─────────────────────────┘
```

**Преимущества:**
- Изменения в одном слое не ломают другие
- Легко тестировать каждый слой отдельно
- Можно заменить реализацию (например, сменить БД)

### 3.2 Примеры организации слоёв

**Python (ML-проект):**
```
src/
  data/
    loader.py         # загрузка данных
    preprocessing.py  # очистка, feature engineering
  models/
    baseline.py       # простая модель
    advanced.py       # продвинутая модель
  pipeline.py         # полный пайплайн: данные → модель → результат
```

**JavaScript (Backend):**
```
src/
  api/
    routes/           # маршруты
    controllers/      # обработчики запросов
  domain/
    services/         # бизнес-логика
  data/
    repositories/     # работа с БД
    models/           # схемы данных
```

**См. примеры кода:** [210-code-examples.md](210-code-examples.md)

## 4. Работа с Git

### 4.1 Базовый workflow для одного человека

**Инициализация:**
```bash
git init
git add .
git commit -m "Initial commit: project structure"
```

**Регулярные коммиты:**
```bash
# После каждого логического изменения
git add src/preprocessing.py
git commit -m "feat: add data cleaning function"
```

**Частота:** Минимум 1-2 коммита в день активной работы.

### 4.2 Conventional Commits

**Стандарт оформления коммитов:**

```text
<тип>: <описание>

Типы:
  feat     - новая функциональность
  fix      - исправление бага
  docs     - документация
  refactor - рефакторинг без изменения функциональности
  test     - добавление тестов
  chore    - рутинные задачи (зависимости, конфиги)
```

**Примеры:**
```bash
feat: implement baseline Random Forest model
fix: correct data splitting bug in preprocessing
docs: add installation instructions to README
refactor: extract feature engineering into separate module
test: add unit tests for data loader
chore: update dependencies to latest versions
```

**Зачем это нужно:**
- История изменений читаема
- Легко найти, когда добавили/сломали функцию
- Можно автоматически генерировать changelog

### 4.3 Ветвление (опционально)

**Для экспериментов:**
```bash
# Создать ветку
git checkout -b experiment/xgboost-model

# Работать в ветке
# ...код, коммиты...

# Если успешно — слить в main
git checkout main
git merge experiment/xgboost-model

# Если нет — просто удалить
git branch -d experiment/xgboost-model
```

**Когда использовать:**
- Рискованные эксперименты
- Крупный рефакторинг
- Параллельная работа над несколькими фичами

**Для студента-одиночки:** Можно работать в main, ветки опциональны.

### 4.4 Работа с GitHub/GitLab

**Создание удалённого репозитория:**
```bash
# На GitHub: создать новый репозиторий (без README)
# Локально:
git remote add origin https://github.com/username/project.git
git branch -M main
git push -u origin main
```

**Регулярный push:**
```bash
git push origin main
```

**Преимущества:**
- Backup кода
- Доступ с любого устройства
- История изменений
- Можно показать научруку/работодателю

## 5. Jupyter Notebooks

> **Подробный гайд:** См. [200-jupyter-notebooks.md](200-jupyter-notebooks.md)

### 5.1 Краткое описание

**Jupyter Notebook** — интерактивная среда для экспериментов, где код, результаты и графики объединены в одном документе.

**Используйте для:**
- ✅ Исследовательского анализа данных (EDA)
- ✅ Экспериментов с моделями
- ✅ Визуализации результатов

**НЕ используйте для:**
- ❌ Production кода
- ❌ Модулей для импорта

### 5.2 Организация

**Нумеруйте notebooks по порядку:**
```
notebooks/
├── 01_data_exploration.ipynb
├── 02_baseline_model.ipynb
├── 03_model_comparison.ipynb
└── 04_final_results.ipynb
```

**Переиспользуемый код → выносите в `src/`**

## 6. Документирование

### 6.1 Docstrings для функций

**Используйте docstrings для сложных функций:**

```python
def train_model(X_train, y_train, model_type='rf'):
    """
    Обучить модель классификации.
    
    Args:
        X_train: Обучающие признаки
        y_train: Целевая переменная
        model_type: Тип модели ('rf', 'xgb', 'lr')
    
    Returns:
        model: Обученная модель
    
    Example:
        >>> model = train_model(X_train, y_train, model_type='rf')
    """
    # код
```

### 6.2 Комментарии

**Хорошие комментарии — объясняют ПОЧЕМУ:**
```python
# Удаляем выбросы по правилу 3σ, так как они искажают обучение
df = df[(df['value'] - df['value'].mean()).abs() < 3 * df['value'].std()]

# FIXME: метод работает медленно на больших данных
# TODO: переписать с использованием vectorization
```

**Плохие комментарии — очевидны из кода:**
```python
# Очистить данные (очевидно!)
df = clean_data(df)
```

### 6.3 README в подпапках

Для сложных модулей добавьте `README.md` в папку:

```markdown
# src/models/

## Структура
- `baseline.py` — Random Forest для сравнения
- `advanced.py` — XGBoost с тюнингом

## Использование
[примеры кода]

## Результаты
| Модель | F1-score |
|--------|----------|
| Baseline | 0.82 |
| Advanced | 0.89 |
```

## 7. Практические советы

### 7.1 Начните с минимального рабочего примера

**День 1-2: Proof of Concept**
- Один файл, весь код внутри
- Простейшая обработка данных
- Baseline модель
- Работает? ✅

**Затем: постепенный рефакторинг**
1. Разделение на функции
2. Разделение на файлы
3. Создание структуры папок
4. Добавление классов и слоёв

**Не пытайтесь сразу создать идеальную архитектуру!**

### 7.2 Фиксируйте эксперименты

**Ведите лог экспериментов (простой JSON/CSV):**

```json
{
  "date": "2025-10-20",
  "model": "XGBoost",
  "hyperparameters": {"n_estimators": 200, "max_depth": 5},
  "data_version": "v1.2",
  "metrics": {"f1": 0.87, "accuracy": 0.89},
  "notes": "Added geo features"
}
```

**Зачем:**
- Отследить, что работает, а что нет
- Материал для раздела "Эксперименты" в дипломе
- Избежать повторения неудачных попыток

### 7.3 Воспроизводимость

**Чек-лист:**
- [ ] Зафиксированы версии зависимостей
- [ ] Зафиксирован random seed
- [ ] Инструкции в README понятны
- [ ] Можно запустить на чистом окружении
- [ ] Нет хардкоженных путей (`/Users/myname/...`)

## 8. Чек-листы

### 8.1 Перед стартом проекта

- [ ] Создана структура репозитория
- [ ] README с установкой, запуском, тестированием
- [ ] `.env.example` заполнен; `.env` локально
- [ ] `.gitignore` настроен правильно
- [ ] Инициализирован Git, сделан первый коммит
- [ ] Создан удалённый репозиторий на GitHub/GitLab
- [ ] Базовые тесты в `tests/` (хотя бы smoke test)

### 8.2 Перед демо/защитой

- [ ] Код запускается из коробки (проверено на чистом окружении)
- [ ] README актуален и самодостаточен
- [ ] Есть примеры использования
- [ ] Все секреты перенесены в `.env`
- [ ] Notebooks очищены от лишнего вывода
- [ ] Нет хардкоженных путей
- [ ] Тесты проходят (если есть)
- [ ] Результаты воспроизводимы
- [ ] Создан тег версии (`git tag v1.0.0`)

### 8.3 Минимальный проект для диплома

```
diploma-project/
├── README.md           ✅ С полными инструкциями
├── requirements.txt    ✅ Все зависимости
├── .env.example        ✅ Шаблон конфигурации
├── .gitignore          ✅ Настроен правильно
├── data/
│   └── README.md       ✅ Откуда данные, как получить
├── src/                ✅ Переиспользуемый код
│   ├── data/
│   └── models/
├── notebooks/          ✅ Исследования и эксперименты
│   ├── 01_eda.ipynb
│   └── 02_model.ipynb
├── tests/              ✅ Хотя бы базовые тесты
└── reports/            ✅ Результаты, графики, метрики
    ├── figures/
    └── metrics.txt
```

**Этого достаточно для успешной защиты!** Остальное — для улучшения оценки.

## 9. Связь с другими гайдами

- **Примеры кода:** См. [210-code-examples.md](210-code-examples.md)
- **Проекты с данными/ML:** См. [190-data-ml-projects.md](190-data-ml-projects.md)
- **Jupyter Notebooks:** См. [200-jupyter-notebooks.md](200-jupyter-notebooks.md)
- **Выбор стека:** См. [150-architecture-choice.md](150-architecture-choice.md)
- **Тайм-менеджмент:** См. [120-timemanagement.md](120-timemanagement.md)
- **Методология разработки:** См. [140-system-dev.md](140-system-dev.md)

---

**Помните:** Цель — не идеальная архитектура, а работающий проект, который легко понять и воспроизвести.
