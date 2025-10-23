# Оценка эффективности и валидация решений

> **Для кого:** Студенты, работающие над проектами с ML/Data Science компонентами. Для других типов проектов используйте соответствующие метрики (производительность, доступность, масштабируемость).

## 1. Метрики качества

### 1.1 Метрики для классификации

**Базовые метрики:**

| Метрика | Формула | Интерпретация | Когда использовать |
|---------|---------|---------------|-------------------|
| **Accuracy** | \(\frac{TP + TN}{TP + TN + FP + FN}\) | Доля правильных ответов | Сбалансированные классы |
| **Precision** | \(\frac{TP}{TP + FP}\) | Точность позитивных предсказаний | Важно минимизировать FP |
| **Recall** | \(\frac{TP}{TP + FN}\) | Полнота выявления позитивных | Важно не пропустить позитивные |
| **F1-score** | \(\frac{2 \cdot P \cdot R}{P + R}\) | Гармоническое среднее P и R | Баланс точности и полноты |

**Пример интерпретации** (детекция мошенничества):
- High Precision: мало ложных обвинений
- High Recall: не пропускаем мошенников
- F1-score: баланс между ними

**ROC-AUC:**
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Вычислить ROC-AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

# Построить ROC-кривую
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

**Интерпретация AUC:**
- 0.5 — случайное угадывание
- 0.7-0.8 — приемлемо
- 0.8-0.9 — хорошо
- 0.9+ — отлично

### 1.2 Метрики для регрессии

| Метрика | Формула | Интерпретация | Единицы |
|---------|---------|---------------|---------|
| **MAE** | \(\frac{1}{n} \sum \|y_i - \hat{y}_i\|\) | Средняя абсолютная ошибка | Те же, что y |
| **RMSE** | \(\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}\) | Корень из средней квадратичной ошибки | Те же, что y |
| **MAPE** | \(\frac{100\%}{n} \sum \frac{\|y_i - \hat{y}_i\|}{y_i}\) | Средняя процентная ошибка | Проценты |
| **R²** | \(1 - \frac{SS_{res}}{SS_{tot}}\) | Доля объяснённой дисперсии | 0-1 |

**Пример:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
```

**Выбор метрики:**
- MAE — устойчив к выбросам, интерпретируем
- RMSE — сильнее штрафует большие ошибки
- MAPE — для сравнения моделей с разными масштабами
- R² — общая объясняющая способность модели

### 1.3 Бизнес-метрики

**Экономический эффект:**

```python
# Допустим, мошенничество стоит 50k, проверка 1k
cost_fraud = 50000
cost_check = 1000

# Без модели: проверяем всех
cost_baseline = len(y_test) * cost_check

# С моделью: проверяем только подозрительных
n_flagged = (y_pred == 1).sum()
n_caught = ((y_pred == 1) & (y_test == 1)).sum()
n_missed = ((y_pred == 0) & (y_test == 1)).sum()

cost_model = n_flagged * cost_check + n_missed * cost_fraud
savings = cost_baseline - cost_model

print(f"Экономия: {savings/1e6:.1f}M руб")
```

**ROI (Return on Investment):**
```python
investment = 2_000_000  # стоимость разработки
roi = (savings - investment) / investment * 100
print(f"ROI: {roi:.1f}%")
```

## 2. Методы валидации

### 2.1 Hold-out валидация

**Простое разделение:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

**Когда использовать:** большие датасеты (>10K примеров)

**Минусы:** 
- Результат зависит от конкретного split
- Теряем часть данных для обучения

### 2.2 Cross-Validation

**K-Fold CV:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, 
    cv=5,  # 5 фолдов
    scoring='f1'
)

print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Stratified K-Fold** (сохраняет пропорции классов):
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

**Time Series Split** (для временных рядов):
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Score: {score:.3f}")
```

### 2.3 Hyperparameter Tuning

**Grid Search:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")

# Использовать лучшую модель
best_model = grid.best_estimator_
```

**Random Search** (быстрее для большого пространства параметров):
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    XGBClassifier(),
    param_dist,
    n_iter=50,  # 50 случайных комбинаций
    cv=5,
    scoring='f1',
    random_state=42
)

random_search.fit(X_train, y_train)
```

## 3. Диагностика проблем

### 3.1 Переобучение (Overfitting)

**Признаки:**
- Train score >> Test score
- Модель хорошо работает на обучающей выборке, плохо на тестовой

**Learning Curve:**
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, 
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.show()
```

**Как бороться:**
- Регуляризация (L1/L2)
- Уменьшить сложность модели
- Добавить данных
- Early stopping

### 3.2 Недообучение (Underfitting)

**Признаки:**
- Train score ≈ Test score, но оба низкие
- Модель слишком простая

**Как бороться:**
- Увеличить сложность модели
- Добавить признаков
- Уменьшить регуляризацию

### 3.3 Data Leakage

**Типичные источники утечки:**

1. **Использование future data:**
```python
# ❌ Неправильно: нормализация до split
scaler.fit(X)  # видит все данные!
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ Правильно: нормализация после split
X_train, X_test = train_test_split(X)
scaler.fit(X_train)  # видит только train
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

2. **Target в признаках:**
```python
# ❌ Признак коррелирует с target по определению
df['fraud_rate_by_city'] = df.groupby('city')['is_fraud'].transform('mean')

# ✅ Использовать только исторические данные
df['fraud_rate_by_city'] = df.groupby('city')['is_fraud'].shift(1).transform('mean')
```

## 4. Сравнение моделей

### 4.1 Baseline модель

**Всегда начинайте с простой модели:**

```python
from sklearn.dummy import DummyClassifier

# Baseline: предсказывать самый частый класс
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)

print(f"Baseline accuracy: {baseline_score:.3f}")

# Ваша модель должна быть лучше!
model.fit(X_train, y_train)
model_score = model.score(X_test, y_test)

print(f"Model accuracy: {model_score:.3f}")
print(f"Improvement: {(model_score - baseline_score) / baseline_score * 100:.1f}%")
```

### 4.2 Статистическое сравнение

**Парный t-test:**
```python
from scipy.stats import ttest_rel

# CV scores для двух моделей
scores_model1 = cross_val_score(model1, X, y, cv=10)
scores_model2 = cross_val_score(model2, X, y, cv=10)

t_stat, p_value = ttest_rel(scores_model1, scores_model2)

if p_value < 0.05:
    print("Модели статистически различаются")
else:
    print("Нет значимого различия")
```

### 4.3 Таблица сравнения

```python
import pandas as pd

results = []
for name, model in [('Baseline', baseline), ('LogReg', logreg), ('XGBoost', xgb)]:
    scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    results.append({
        'Model': name,
        'F1 Mean': scores.mean(),
        'F1 Std': scores.std(),
        'Train Time': train_time  # замерить через time.time()
    })

df_results = pd.DataFrame(results).sort_values('F1 Mean', ascending=False)
print(df_results)
```

## 5. Интерпретируемость

### 5.1 Feature Importance

**Для tree-based моделей:**
```python
import matplotlib.pyplot as plt

importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]  # топ-10

plt.barh(range(10), importances[indices])
plt.yticks(range(10), [feature_names[i] for i in indices])
plt.xlabel('Importance')
plt.show()
```

### 5.2 SHAP values

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Важность признаков
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Объяснение конкретного примера
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
```

## 6. Документирование результатов

### 6.1 Отчёт об эксперименте

```markdown
## Эксперимент: XGBoost для детекции мошенничества

**Дата:** 2024-10-23
**Данные:** fraud_dataset_v1.1 (100K записей)
**Цель:** Улучшить F1-score >0.85

### Конфигурация
- Train/Test split: 80/20
- Cross-validation: 5-fold stratified
- Hyperparameters: n_estimators=200, max_depth=5, lr=0.1

### Результаты
| Метрика | Train | Test |
|---------|-------|------|
| Accuracy | 0.92 | 0.89 |
| Precision | 0.88 | 0.85 |
| Recall | 0.91 | 0.87 |
| F1 | 0.89 | 0.86 |

### Выводы
- Цель достигнута: F1=0.86 > 0.85
- Модель не переобучена (train/test близки)
- Топ-3 признака: income, family_size, previous_claims

### Следующие шаги
- Попробовать ансамбль с LightGBM
- Добавить графовые признаки
```

### 6.2 MLflow для tracking

```python
import mlflow

mlflow.set_experiment("fraud_detection")

with mlflow.start_run():
    # Логировать параметры
    mlflow.log_params({
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1
    })
    
    # Обучить модель
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Логировать метрики
    mlflow.log_metrics({
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred)
    })
    
    # Сохранить модель
    mlflow.sklearn.log_model(model, "model")
```

## 7. Чек-лист валидации

- [ ] Определены метрики успеха до начала экспериментов
- [ ] Создана baseline модель
- [ ] Данные правильно разделены (нет data leakage)
- [ ] Проведена cross-validation
- [ ] Подобраны гиперпараметры
- [ ] Проверено на переобучение
- [ ] Сравнено несколько моделей статистически
- [ ] Результаты задокументированы и воспроизводимы
- [ ] Модель интерпретируема (SHAP/feature importance)
- [ ] Оценён бизнес-эффект (ROI, экономия)
