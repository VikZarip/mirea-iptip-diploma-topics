# Математический аппарат и алгоритмы

> **Для кого:** Студенты, работающие над проектами с алгоритмическими задачами — ML/Data Science, оптимизация, временные ряды, анализ графов. Если ваш проект не использует эти методы, пропускайте данный гайд.

## 1. Машинное обучение

### 1.1 Выбор алгоритма

**Классификация** (предсказываем категорию):

| Алгоритм | Когда использовать | Плюсы | Минусы | Библиотека |
|----------|-------------------|-------|--------|-----------|
| **Логистическая регрессия** | Baseline, линейно разделимые данные | Простота, интерпретируемость | Низкая точность на сложных данных | sklearn.linear_model |
| **Random Forest** | Табличные данные, средняя точность | Устойчив к переобучению | Долгое обучение | sklearn.ensemble |
| **XGBoost/LightGBM** | Высокая точность на табличных данных | SOTA для табличных | Долгая настройка | xgboost, lightgbm |
| **SVM** | Малые выборки, высокая размерность | Хорошая обобщаемость | Долгое обучение | sklearn.svm |
| **Нейросети (MLP)** | Сложные нелинейные зависимости | Универсальность | Требует много данных | keras, pytorch |

**Регрессия** (предсказываем число):

| Алгоритм | Применение | Пример |
|----------|-----------|--------|
| **Линейная регрессия** | Простые зависимости | Прогноз потребления энергии |
| **Ridge/Lasso** | Много признаков, регуляризация | Прогноз цен |
| **Gradient Boosting** | Высокая точность | Прогноз спроса |

**Пример выбора:**
> Задача: классификация мошеннических заявок (2 класса)  
> Данные: 10K записей, 50 признаков, несбалансированные (5% мошенников)  
> **Выбор:** XGBoost + обработка дисбаланса (SMOTE)  
> **Обоснование:** SOTA для табличных данных, устойчив к выбросам

### 1.2 Метрики качества

**Для классификации:**

| Метрика | Формула | Когда использовать |
|---------|---------|-------------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Сбалансированные данные |
| **Precision** | TP/(TP+FP) | Важно избегать ложных срабатываний |
| **Recall** | TP/(TP+FN) | Важно не пропустить позитивные случаи |
| **F1-score** | 2·P·R/(P+R) | Баланс между precision и recall |
| **ROC-AUC** | Площадь под ROC-кривой | Общая способность различать классы |

**Confusion Matrix:**
```
                Predicted
              Fraud   Normal
Actual Fraud   TP      FN
       Normal  FP      TN
```

**Для регрессии:**

| Метрика | Формула | Интерпретация |
|---------|---------|---------------|
| **MAE** | mean(\|y - ŷ\|) | Средняя абсолютная ошибка |
| **RMSE** | √(mean((y-ŷ)²)) | Корень из средней квадратичной ошибки |
| **MAPE** | mean(\|y-ŷ\|/y)·100% | Средняя процентная ошибка |
| **R²** | 1 - SS_res/SS_tot | Доля объяснённой дисперсии (0-1) |

### 1.3 Работа с данными

**Feature Engineering** — создание признаков:

```python
# Исходные данные
df['income'] = 50000
df['family_size'] = 4

# Новые признаки
df['income_per_person'] = df['income'] / df['family_size']
df['has_children'] = df['family_size'] > 2
df['income_category'] = pd.cut(df['income'], bins=[0, 20k, 50k, 100k, inf])
```

**Обработка несбалансированных данных:**

1. **Undersampling** — уменьшить мажоритарный класс
2. **Oversampling** — дублировать миноритарный класс
3. **SMOTE** — синтетические примеры миноритарного класса
4. **Cost-sensitive learning** — штрафовать ошибки на миноритарном классе

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5)  # 50% мошенников
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**Нормализация:**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: (x - mean) / std
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler: (x - min) / (max - min)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### 1.4 Валидация моделей

**Train/Validation/Test split:**
```python
from sklearn.model_selection import train_test_split

# 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

**Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score

# 5-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Hyperparameter Tuning:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

## 2. Методы оптимизации

### 2.1 Линейное программирование

**Постановка задачи:**
```
Минимизировать: c₁·x₁ + c₂·x₂ + ... + cₙ·xₙ
При ограничениях:
  a₁₁·x₁ + a₁₂·x₂ + ... ≤ b₁
  a₂₁·x₁ + a₂₂·x₂ + ... ≤ b₂
  x₁, x₂, ... ≥ 0
```

**Пример: распределение персонала**
```python
from scipy.optimize import linprog

# Минимизировать затраты: 1000·x₁ + 1500·x₂ (где x₁, x₂ - кол-во врачей)
c = [1000, 1500]

# Ограничения: x₁ + x₂ >= 10 (минимум 10 врачей)
A_ub = [[-1, -1]]
b_ub = [-10]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
print(f"Оптимальное решение: {result.x}")
```

**Библиотеки:**
- `scipy.optimize.linprog` — базовый LP
- `pulp`, `cvxpy` — более удобный синтаксис
- `ortools` — от Google, поддерживает constraint programming

### 2.2 Целочисленное программирование

**Когда использовать:** переменные должны быть целыми (кол-во людей, маршруты)

```python
from pulp import *

# Задача: назначить N врачей на M смен
prob = LpProblem("Scheduling", LpMinimize)

# Переменные: x[i][j] = 1, если врач i на смене j
doctors = range(5)
shifts = range(7)
x = LpVariable.dicts("assign", (doctors, shifts), cat='Binary')

# Целевая функция: минимизировать переработки
prob += lpSum([x[i][j] * cost[i][j] for i in doctors for j in shifts])

# Ограничения: каждый врач работает не более 5 смен
for i in doctors:
    prob += lpSum([x[i][j] for j in shifts]) <= 5

prob.solve()
```

### 2.3 Генетические алгоритмы

**Когда использовать:** сложные комбинаторные задачи, NP-hard проблемы

**Алгоритм:**
1. Создать популяцию решений
2. Оценить fitness каждого решения
3. Селекция лучших
4. Crossover (скрещивание)
5. Mutation (мутация)
6. Повторить 2-5

```python
import numpy as np

def genetic_algorithm(fitness_func, n_vars, n_pop=100, n_gen=100):
    # Инициализация
    population = np.random.rand(n_pop, n_vars)
    
    for gen in range(n_gen):
        # Оценка
        scores = [fitness_func(ind) for ind in population]
        
        # Селекция (выбрать лучших 50%)
        sorted_idx = np.argsort(scores)[:n_pop//2]
        parents = population[sorted_idx]
        
        # Crossover
        offspring = []
        for i in range(n_pop//2):
            p1, p2 = parents[np.random.choice(len(parents), 2)]
            child = 0.5 * p1 + 0.5 * p2
            offspring.append(child)
        
        # Mutation
        offspring = np.array(offspring)
        offspring += np.random.normal(0, 0.1, offspring.shape)
        
        # Новая популяция
        population = np.vstack([parents, offspring])
    
    return population[np.argmax(scores)]
```

## 3. Временные ряды

### 3.1 Классические методы

**ARIMA (AutoRegressive Integrated Moving Average):**

```python
from statsmodels.tsa.arima.model import ARIMA

# Загрузить данные
data = pd.read_csv('energy_consumption.csv', index_col='date', parse_dates=True)

# Обучить модель ARIMA(p,d,q)
model = ARIMA(data['consumption'], order=(1,1,1))
fitted = model.fit()

# Прогноз на 30 дней вперёд
forecast = fitted.forecast(steps=30)
```

**Параметры ARIMA:**
- **p** (AR) — авторегрессия (зависимость от прошлых значений)
- **d** — разность (сколько раз дифференцировать для стационарности)
- **q** (MA) — скользящее среднее (зависимость от прошлых ошибок)

**Как выбрать p, d, q:**
1. **ACF/PACF графики** — автокорреляция
2. **ADF тест** — проверка стационарности
3. **Auto ARIMA** — автоматический подбор параметров

```python
from pmdarima import auto_arima

model = auto_arima(data, seasonal=True, m=7)  # m=7 для недельной сезонности
print(model.summary())
```

### 3.2 Deep Learning для временных рядов

**LSTM (Long Short-Term Memory):**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Подготовка данных: окно 30 дней → прогноз 1 день
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# Модель
model = Sequential([
    LSTM(50, activation='relu', input_shape=(30, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=32)

# Прогноз
prediction = model.predict(X_test)
```

## 4. Теория графов

### 4.1 Основные алгоритмы

**Построение графа:**
```python
import networkx as nx

G = nx.Graph()
G.add_edges_from([
    ('person1', 'person2', {'relation': 'family'}),
    ('person2', 'person3', {'relation': 'business'}),
    ('person1', 'company1', {'relation': 'owner'})
])
```

**Поиск кратчайшего пути:**
```python
path = nx.shortest_path(G, 'person1', 'person3')
length = nx.shortest_path_length(G, 'person1', 'person3')
```

**Выявление сообществ:**
```python
from networkx.algorithms import community

communities = community.greedy_modularity_communities(G)
print(f"Найдено {len(communities)} сообществ")
```

**Меры центральности:**
```python
# Degree centrality: кол-во связей
degree = nx.degree_centrality(G)

# Betweenness: важность как посредника
betweenness = nx.betweenness_centrality(G)

# PageRank: важность узла
pagerank = nx.pagerank(G)
```

## 5. Практические советы

### Чек-лист выбора метода:

- [ ] Определён тип задачи (классификация/регрессия/кластеризация)
- [ ] Оценён объём данных и кол-во признаков
- [ ] Выбрана baseline модель (простейшая)
- [ ] Определены метрики успеха
- [ ] Настроен pipeline предобработки
- [ ] Проведена валидация на отложенной выборке

### Типичные ошибки:

❌ Использовать accuracy на несбалансированных данных  
❌ Обучать модель на всех данных без test set  
❌ Не нормализовать признаки  
❌ Использовать сложную модель без проверки простой baseline  
❌ Не фиксировать random_state (результаты не воспроизводимы)  

### Рекомендуемые курсы:

- **Машинное обучение:** [ВШЭ на Coursera](https://www.coursera.org/specializations/machine-learning-data-analysis)
- **Deep Learning:** [Stanford CS231n](http://cs231n.stanford.edu/)
- **Временные ряды:** [Kaggle Time Series](https://www.kaggle.com/learn/time-series)
- **Оптимизация:** [MIT 6.079](https://ocw.mit.edu/courses/15-079j-introduction-to-mathematical-programming-fall-2009/)
