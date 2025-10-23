# Работа с данными: сбор, предобработка, управление

## 1. Источники данных

### 1.1 Открытые датасеты

> **Подробный справочник источников данных:** См. [170-sources.md](170-sources.md) — российские и международные порталы, тематические реестры, API для интеграции.

**Краткий список для быстрого старта:**
- **Российские:** data.gov.ru, data.mos.ru, Росстат
- **Международные:** Kaggle, UCI ML Repository, Google Dataset Search
- **Тематические:** PhysioNet (медицина), Open Power System Data (IoT/энергетика)

### 1.2 Web Scraping

**Когда использовать:** нужные данные есть на сайтах, но нет API

**Проверьте законность:**
```python
import requests
from bs4 import BeautifulSoup

# Проверить robots.txt
response = requests.get('https://example.com/robots.txt')
print(response.text)
```

**Базовый scraping:**
```python
from bs4 import BeautifulSoup
import requests

url = 'https://example.com/data'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Найти все таблицы
tables = soup.find_all('table')

# Извлечь данные
data = []
for row in tables[0].find_all('tr'):
    cols = [col.text.strip() for col in row.find_all('td')]
    data.append(cols)

import pandas as pd
df = pd.DataFrame(data, columns=['col1', 'col2', 'col3'])
```

**Динамические сайты (JavaScript):**
```python
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get('https://example.com')

# Подождать загрузки
driver.implicitly_wait(10)

# Извлечь данные
element = driver.find_element(By.CLASS_NAME, 'data-table')
data = element.text

driver.quit()
```

**Этические принципы:**
- Соблюдайте robots.txt
- Добавляйте задержки между запросами (time.sleep)
- Не перегружайте серверы
- Указывайте User-Agent

### 1.3 Работа с API

**REST API:**
```python
import requests

# GET запрос
response = requests.get(
    'https://api.example.com/data',
    headers={'Authorization': 'Bearer YOUR_TOKEN'},
    params={'limit': 100, 'offset': 0}
)

data = response.json()
```

**Пагинация:**
```python
def fetch_all_data(base_url, token):
    all_data = []
    offset = 0
    limit = 100
    
    while True:
        response = requests.get(
            base_url,
            headers={'Authorization': f'Bearer {token}'},
            params={'limit': limit, 'offset': offset}
        )
        batch = response.json()
        
        if not batch:
            break
        
        all_data.extend(batch)
        offset += limit
    
    return all_data
```

**Rate limiting:**
```python
import time
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=100, period=60)  # 100 запросов в минуту
def fetch_data(url):
    return requests.get(url).json()
```

## 2. Предобработка данных

### 2.1 Очистка данных

**Обработка пропусков:**

```python
import pandas as pd

# Проверить пропуски
print(df.isnull().sum())

# Стратегии обработки:
# 1. Удалить строки с пропусками
df_clean = df.dropna()

# 2. Удалить столбцы с >50% пропусков
df_clean = df.dropna(thresh=len(df)*0.5, axis=1)

# 3. Заполнить средним/медианой
df['age'].fillna(df['age'].mean(), inplace=True)

# 4. Заполнить модой (для категорий)
df['category'].fillna(df['category'].mode()[0], inplace=True)

# 5. Интерполяция (для временных рядов)
df['value'].interpolate(method='linear', inplace=True)
```

**Обработка выбросов:**

```python
# IQR метод
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Удалить выбросы
df_clean = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

# Или заменить на границы
df['value'] = df['value'].clip(lower_bound, upper_bound)
```

**Удаление дубликатов:**

```python
# Проверить дубликаты
print(f"Дубликаты: {df.duplicated().sum()}")

# Удалить полные дубликаты
df_clean = df.drop_duplicates()

# Удалить дубликаты по определённым колонкам
df_clean = df.drop_duplicates(subset=['user_id', 'date'], keep='first')
```

### 2.2 Feature Engineering

**Создание признаков:**

```python
# Из даты
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# Математические операции
df['bmi'] = df['weight'] / (df['height'] ** 2)
df['income_per_capita'] = df['income'] / df['family_size']

# Категоризация
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], 
                         labels=['child', 'young', 'middle', 'senior'])

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

# Агрегации
user_stats = df.groupby('user_id').agg({
    'transaction': 'count',
    'amount': ['sum', 'mean', 'std']
}).reset_index()
```

**Временные признаки (лаги, окна):**

```python
# Лаги
df['value_lag1'] = df['value'].shift(1)  # значение предыдущего дня
df['value_lag7'] = df['value'].shift(7)  # неделю назад

# Скользящее среднее
df['value_ma7'] = df['value'].rolling(window=7).mean()

# Разность
df['value_diff'] = df['value'].diff()
```

### 2.3 Нормализация и кодирование

**Числовые признаки:**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: (x - μ) / σ
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# MinMaxScaler: (x - min) / (max - min) → [0, 1]
scaler = MinMaxScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# RobustScaler: устойчив к выбросам, использует медиану и IQR
scaler = RobustScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])
```

**Категориальные признаки:**

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding (для порядковых категорий)
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
# high_school=0, bachelor=1, master=2

# One-Hot Encoding (для номинальных категорий)
df = pd.get_dummies(df, columns=['city'], prefix='city')
# city_Moscow, city_SPb, city_Kazan

# Target Encoding (среднее target по категории)
target_means = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(target_means)
```

### 2.4 Разделение данных

**Простое разделение:**

```python
from sklearn.model_selection import train_test_split

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 60% train, 20% val, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

**Стратифицированное разделение** (сохранить пропорции классов):

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**Временные ряды** (не перемешивать!):

```python
# Взять последние 20% как test
split_idx = int(len(df) * 0.8)
train = df[:split_idx]
test = df[split_idx:]
```

## 3. Управление данными

### 3.1 Версионирование данных

**DVC (Data Version Control):**

```bash
# Установка
pip install dvc

# Инициализация
dvc init

# Добавить файл данных под версионирование
dvc add data/dataset.csv

# Зафиксировать в Git
git add data/dataset.csv.dvc data/.gitignore
git commit -m "Add dataset v1"

# Изменили данные
dvc add data/dataset.csv
git add data/dataset.csv.dvc
git commit -m "Update dataset v2"

# Вернуться к предыдущей версии
git checkout HEAD~1 data/dataset.csv.dvc
dvc checkout
```

**Хранение больших файлов:**

```bash
# Настроить remote storage (S3, Google Drive, SSH)
dvc remote add -d storage s3://mybucket/dvcstore

# Загрузить данные в remote
dvc push

# Скачать данные из remote
dvc pull
```

### 3.2 Документирование данных

**Data Card / Dataset Card:**

```markdown
# Dataset: Fraud Detection Dataset

## Описание
Датасет содержит 100K записей заявок на социальные выплаты, 
из которых 5% помечены как мошеннические.

## Источник
Синтетические данные на основе реальных паттернов из 
открытых источников социальных служб РФ (2020-2023).

## Структура
- `application_id`: уникальный ID заявки
- `user_id`: ID пользователя
- `income`: декларируемый доход (руб/мес)
- `family_size`: размер семьи
- `is_fraud`: метка (0=легально, 1=мошенничество)

## Статистика
- Размер: 100,000 строк × 15 столбцов
- Пропуски: 2% в `income`, 0.5% в `family_size`
- Дисбаланс классов: 95% / 5%

## Лицензия
CC BY 4.0 — можно использовать в исследовательских целях

## Версии
- v1.0 (2024-01-15): начальная версия
- v1.1 (2024-03-10): добавлены географические признаки
```

### 3.3 Воспроизводимость

**Фиксация random seed:**

```python
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Для PyTorch:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

set_seed(42)
```

**Запись конфигурации:**

```python
config = {
    'data_version': 'v1.1',
    'train_test_split': 0.8,
    'random_state': 42,
    'scaler': 'StandardScaler',
    'model': 'XGBoost',
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1
    }
}

import json
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

**Docker для воспроизводимости:**

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data/ data/
COPY src/ src/

CMD ["python", "src/train.py"]
```

## 4. Практические советы

### Чек-лист работы с данными:

- [ ] Данные задокументированы (источник, структура, лицензия)
- [ ] Проверены на пропуски, выбросы, дубликаты
- [ ] Созданы осмысленные новые признаки
- [ ] Данные нормализованы/стандартизованы
- [ ] Правильно разделены train/val/test
- [ ] Зафиксирован random_state для воспроизводимости
- [ ] Данные версионированы (DVC или Git LFS)

### Типичные ошибки:

❌ Нормализовать данные до split (утечка информации из test в train)  
❌ Использовать test данные для выбора признаков  
❌ Не проверить дисбаланс классов  
❌ Удалить все строки с пропусками (потеря данных)  
❌ Не документировать источники и версии данных  

### Правильный конвейер обработки данных:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Pipeline автоматически применяет преобразования
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# fit применяет все шаги к train
pipeline.fit(X_train, y_train)

# predict применяет те же преобразования к test
y_pred = pipeline.predict(X_test)
```

## 5. Инструменты

**Анализ и обработка:**
- pandas — табличные данные
- numpy — числовые вычисления
- polars — быстрая альтернатива pandas

**Визуализация:**
- matplotlib, seaborn — графики
- pandas-profiling — автоматический EDA отчёт

**Версионирование:**
- DVC — version control для данных
- Git LFS — для больших файлов в Git

**Валидация:**
- Great Expectations — проверка качества данных
- pandera — валидация схемы DataFrame

## 6. Чек-лист перед началом моделирования

- [ ] Провели EDA (exploratory data analysis)
- [ ] Поняли распределение признаков и target
- [ ] Выявили корреляции и зависимости
- [ ] Обработали пропуски и выбросы
- [ ] Создали baseline модель на чистых данных
- [ ] Проверили, что нет утечки данных (data leakage)
