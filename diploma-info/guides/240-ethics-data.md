# Этика, безопасность данных и академическая честность

## 1. Работа с персональными данными

### 1.1 Законодательство РФ

**Закон 152-ФЗ "О персональных данных":**

**Персональные данные** — любая информация, относящаяся к прямо или косвенно определённому физическому лицу:
- ФИО, паспортные данные
- Адрес, телефон, email
- Фотографии, биометрия
- IP-адреса, cookies
- Медицинские, финансовые данные

**Основные требования:**
1. **Согласие субъекта** на обработку (за исключениями)
2. **Цель обработки** должна быть конкретной и законной
3. **Минимизация** — собирать только необходимые данные
4. **Безопасность** — защита от несанкционированного доступа
5. **Уведомление Роскомнадзора** (для операторов ПД)

**Для дипломной работы:**
- Если используете реальные ПД — получите письменное согласие
- Лучше использовать **анонимизированные** или **синтетические** данные
- Укажите в работе, как обеспечивается защита ПД

### 1.2 GDPR (для международных исследований)

**General Data Protection Regulation (ЕС):**

Основные принципы (применимы везде):
1. **Lawfulness** — законное основание для обработки
2. **Purpose limitation** — строго по заявленной цели
3. **Data minimization** — только необходимое
4. **Accuracy** — данные должны быть актуальными
5. **Storage limitation** — хранить не дольше необходимого
6. **Security** — обеспечить защиту

**Права субъектов:**
- Право на доступ к своим данным
- Право на исправление
- Право на удаление ("right to be forgotten")
- Право на переносимость данных

### 1.3 Анонимизация данных

**Методы анонимизации:**

**1. Удаление прямых идентификаторов:**
```python
# Удалить ФИО, паспорт, телефон
df_anon = df.drop(columns=['full_name', 'passport', 'phone'])
```

**2. Генерализация (обобщение):**
```python
# Возраст → возрастная группа
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100])
df = df.drop(columns=['age'])

# Точный адрес → город
df['location'] = df['address'].apply(lambda x: x.split(',')[0])
df = df.drop(columns=['address'])
```

**3. Маскирование:**
```python
# Email → маскированный email
df['email_masked'] = df['email'].str.replace(r'(.{2}).*(@.*)', r'\1***\2', regex=True)
# ivan.petrov@mail.ru → iv***@mail.ru
```

**4. Добавление шума:**
```python
# Доход + случайный шум ±5%
df['income_noisy'] = df['income'] * (1 + np.random.uniform(-0.05, 0.05, len(df)))
```

**5. Псевдонимизация:**
```python
import hashlib

def pseudonymize(value):
    return hashlib.sha256(value.encode()).hexdigest()[:16]

df['user_id_pseudo'] = df['user_id'].apply(pseudonymize)
```

**K-anonymity** — каждая запись неотличима минимум от K-1 других:
```python
# Пример: группировка по возрастной группе и городу
# Каждая группа должна содержать минимум k=5 записей
grouped = df.groupby(['age_group', 'city']).filter(lambda x: len(x) >= 5)
```

### 1.4 Безопасное хранение

**Шифрование данных:**
```python
from cryptography.fernet import Fernet

# Сгенерировать ключ
key = Fernet.generate_key()
cipher = Fernet(key)

# Зашифровать
encrypted = cipher.encrypt(b"sensitive data")

# Расшифровать
decrypted = cipher.decrypt(encrypted)
```

**Переменные окружения для секретов:**
```python
import os
from dotenv import load_dotenv

load_dotenv()  # Загрузить из .env файла

DATABASE_PASSWORD = os.getenv('DB_PASSWORD')
API_KEY = os.getenv('API_KEY')

# ❌ Никогда не храните пароли в коде!
# password = "mypassword123"
```

**.env файл** (добавить в .gitignore!):
```
DB_PASSWORD=your_secure_password
API_KEY=your_api_key
SECRET_KEY=random_secret_string
```

## 2. Этика искусственного интеллекта

### 2.1 Справедливость (Fairness)

**Проблема bias (предвзятости):**

Модели могут дискриминировать по:
- Полу, возрасту
- Расе, национальности
- Социально-экономическому статусу

**Пример:**
```python
# Проверить различие в метриках по группам
for gender in ['M', 'F']:
    subset = df[df['gender'] == gender]
    score = model.score(subset[features], subset['target'])
    print(f"Accuracy for {gender}: {score:.3f}")

# Если accuracy сильно различается — есть bias!
```

**Как бороться:**
1. **Сбалансировать обучающие данные** по группам
2. **Удалить** чувствительные признаки (пол, возраст) — но может быть недостаточно
3. **Fair ML алгоритмы:**
```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

mitigator = ExponentiatedGradient(
    estimator=model,
    constraints=DemographicParity()
)

mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
```

4. **Post-processing:** корректировать предсказания для справедливости

### 2.2 Прозрачность (Transparency)

**Explainable AI (XAI):**

Пользователь должен понимать, почему модель приняла решение.

**SHAP (SHapley Additive exPlanations):**
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Объяснение для конкретного случая
shap.force_plot(
    explainer.expected_value, 
    shap_values[0], 
    X_test.iloc[0]
)
```

**LIME (Local Interpretable Model-agnostic Explanations):**
```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Normal', 'Fraud'],
    mode='classification'
)

exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
exp.show_in_notebook()
```

### 2.3 Ответственность (Accountability)

**Документируйте:**
- Какие данные использовались
- Как модель обучалась
- Какие ограничения у модели
- Кто отвечает за решения модели

**Model Card** (пример):
```markdown
## Model Card: Fraud Detection System

### Назначение
Выявление мошеннических заявок на социальные выплаты

### Ограничения
- Точность снижается для редких категорий заявителей
- Не учитывает контекст семейной ситуации
- Требует регулярной переобучения (drift detection)

### Риски
- Возможны ложноположительные срабатывания (2-3%)
- Финальное решение должен принимать человек

### Справедливость
- Проверено на отсутствие дискриминации по полу и возрасту
- F1-score различается <5% между группами

### Обслуживание
- Модель обучена на данных 2020-2024
- Рекомендуется переобучение раз в квартал
```

## 3. Академическая честность

### 3.1 Плагиат

**Что считается плагиатом:**
- Копирование чужого текста без цитирования
- Перефразирование без ссылки на источник
- Использование чужого кода без указания автора

**Проверка на плагиат:**
- [Antiplagiat.ru](https://www.antiplagiat.ru/) — основная система в РФ
- Обычно требуется уникальность >70-80%

**Как избежать:**
```markdown
❌ Неправильно:
"Машинное обучение позволяет компьютерам учиться без явного программирования."

✅ Правильно:
Машинное обучение, по определению А. Самуэля, позволяет 
компьютерам учиться без явного программирования [1].

[1] Samuel, A. L. (1959). Some studies in machine learning using the game of checkers.
```

**Самоплагиат:**
- Использование своих предыдущих работ без ссылки
- Тоже считается нарушением!

### 3.2 Корректное цитирование кода

**Open Source код:**
```python
# ✅ Указать источник
# Adapted from: https://github.com/author/repo
# License: MIT

def process_data(data):
    # ... ваш код с модификациями
    pass
```

**Библиотеки:**
```python
# ✅ Указать в requirements.txt и в разделе "Используемые инструменты"
import pandas as pd
import scikit-learn
```

**StackOverflow snippets:**
```python
# ✅ Добавить комментарий
# Source: https://stackoverflow.com/questions/12345/...
# Modified for current use case
```

### 3.3 Честность в исследовании

**Фабрикация данных — недопустима!**
- ❌ Придумывать результаты экспериментов
- ❌ Выборочно публиковать только "хорошие" результаты
- ❌ Манипулировать данными для нужного результата

**Cherry-picking:**
```python
# ❌ Неправильно:
best_score = 0
for seed in range(100):
    score = train_and_evaluate(seed=seed)
    if score > best_score:
        best_score = score

# Публикуем только best_score — это обман!

# ✅ Правильно:
scores = []
for seed in [42, 123, 456, 789, 999]:  # фиксированные seed
    score = train_and_evaluate(seed=seed)
    scores.append(score)

print(f"Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

### 3.4 Конфликт интересов

**Раскрывайте:**
- Финансовую поддержку от компаний
- Личную заинтересованность в результате
- Использование коммерческих данных

**Пример раскрытия:**
```markdown
## Конфликт интересов
Автор работает в компании X, предоставившей данные для исследования.
Компания не влияла на результаты и выводы работы.
```

## 4. Практические рекомендации

### Чек-лист этики и безопасности:

- [ ] Персональные данные анонимизированы или получено согласие
- [ ] Данные хранятся безопасно (шифрование, access control)
- [ ] Секреты (пароли, ключи) в .env, не в коде
- [ ] Проверено на bias по защищённым группам
- [ ] Модель интерпретируема (SHAP/LIME)
- [ ] Создан Model Card с ограничениями и рисками
- [ ] Проверка на плагиат >70% уникальности
- [ ] Все источники процитированы
- [ ] Код из интернета атрибутирован
- [ ] Результаты воспроизводимы и честны
- [ ] Раскрыт конфликт интересов (если есть)

### Типичные ошибки:

❌ Использовать реальные ПД без согласия  
❌ Хранить пароли в Git репозитории  
❌ Копировать код без ссылки на источник  
❌ Публиковать только лучшие результаты  
❌ Игнорировать bias в модели  

### Полезные ресурсы:

**Законодательство:**
- [152-ФЗ о персональных данных](http://www.consultant.ru/document/cons_doc_LAW_61801/)
- [GDPR official site](https://gdpr.eu/)

**Этика ИИ:**
- [Google AI Principles](https://ai.google/responsibility/principles/)
- [Microsoft Responsible AI](https://www.microsoft.com/en-us/ai/responsible-ai)
- [IEEE Ethics in Action](https://ethicsinaction.ieee.org/)

**Fairness tools:**
- [Fairlearn](https://fairlearn.org/) — библиотека для fair ML
- [AI Fairness 360](https://aif360.mybluemix.net/) — IBM toolkit

**Explainability:**
- [SHAP](https://github.com/slundberg/shap)
- [LIME](https://github.com/marcotcr/lime)
- [InterpretML](https://interpret.ml/)

## 5. Шаблон заявления о согласии

```markdown
СОГЛАСИЕ
на обработку персональных данных

Я, _______________________ (ФИО),
даю согласие на обработку моих персональных данных
в рамках исследования "___________________" (название)
для целей: _______________________ (цель).

Обрабатываемые данные: возраст, пол, доход, история обращений.

Я проинформирован о:
- Целях обработки данных
- Методах анонимизации и защиты
- Праве отозвать согласие в любой момент

Дата: __________
Подпись: __________
```
