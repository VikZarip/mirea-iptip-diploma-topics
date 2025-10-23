# Примеры организации кода

> **Справочник:** Конкретные примеры реализации слоистой архитектуры для разных типов проектов.

⚠️ **Важно:** Примеры используют Python и TypeScript, но **принципы организации кода** (слои, модульность, разделение ответственности) **применимы к любому стеку**. Адаптируйте примеры под свой выбранный язык и фреймворк.

## 1. Python: ML/Data Science проект

### 1.1 Data Layer — работа с данными

**`src/data/loader.py`:**
```python
import pandas as pd
from pathlib import Path
from typing import Optional

class DataLoader:
    """Загрузка данных из разных источников."""
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = Path(data_dir)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Загрузить CSV файл."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        return pd.read_csv(filepath)
    
    def load_from_url(self, url: str) -> pd.DataFrame:
        """Загрузить данные из URL."""
        return pd.read_csv(url)
```

**`src/data/preprocessing.py`:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

class DataPreprocessor:
    """Предобработка данных."""
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка: пропуски, дубликаты."""
        df = df.dropna()
        df = df.drop_duplicates()
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание новых признаков."""
        # Пример: создание нового признака
        if 'col_a' in df.columns and 'col_b' in df.columns:
            df['feature_ratio'] = df['col_a'] / (df['col_b'] + 1e-8)
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'target',
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Разделить на train/test."""
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        return train_test_split(X, y, test_size=test_size, random_state=42)
```

### 1.2 Domain Layer — модели

**`src/models/baseline.py`:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path
from typing import Dict, Any

class BaselineModel:
    """Baseline модель для сравнения."""
    
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
        self.metrics = {}
    
    def train(self, X_train, y_train):
        """Обучить модель."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Предсказание."""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Оценить качество модели."""
        y_pred = self.predict(X_test)
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        return self.metrics
    
    def save(self, filepath: str):
        """Сохранить модель."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
    
    def load(self, filepath: str):
        """Загрузить модель."""
        self.model = joblib.load(filepath)
```

### 1.3 Application Layer — пайплайн

**`src/pipeline.py`:**
```python
from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.models.baseline import BaselineModel
from typing import Dict, Any
import json

class MLPipeline:
    """Полный пайплайн: данные → модель → результат."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loader = DataLoader(config['data_dir'])
        self.preprocessor = DataPreprocessor()
        self.model = BaselineModel(**config.get('model_params', {}))
    
    def run(self):
        """Запустить весь процесс."""
        print("=" * 50)
        print("Запуск ML Pipeline")
        print("=" * 50)
        
        # 1. Загрузка
        print("\n[1/5] Загрузка данных...")
        df = self.loader.load_csv(self.config['data_file'])
        print(f"✓ Загружено {len(df)} записей")
        
        # 2. Предобработка
        print("\n[2/5] Предобработка данных...")
        df = self.preprocessor.clean(df)
        df = self.preprocessor.engineer_features(df)
        print(f"✓ После очистки: {len(df)} записей")
        
        # 3. Разделение
        print("\n[3/5] Разделение на train/test...")
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            df, 
            target_col=self.config['target_col'],
            test_size=self.config.get('test_size', 0.2)
        )
        print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
        
        # 4. Обучение
        print("\n[4/5] Обучение модели...")
        self.model.train(X_train, y_train)
        print("✓ Модель обучена")
        
        # 5. Оценка
        print("\n[5/5] Оценка качества...")
        metrics = self.model.evaluate(X_test, y_test)
        print("✓ Результаты:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 6. Сохранение
        model_path = self.config.get('model_save_path', 'models/model.pkl')
        self.model.save(model_path)
        print(f"\n✓ Модель сохранена: {model_path}")
        
        # 7. Сохранение метрик
        metrics_path = self.config.get('metrics_path', 'reports/metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Метрики сохранены: {metrics_path}")
        
        return metrics
```

### 1.4 Presentation Layer — точка входа

**`main.py`:**
```python
from src.pipeline import MLPipeline
from src.utils.config import load_config
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Загрузить конфигурацию
    config = load_config(args.config)
    
    # Запустить пайплайн
    pipeline = MLPipeline(config)
    metrics = pipeline.run()
    
    print("\n" + "=" * 50)
    print("Pipeline завершён успешно!")
    print("=" * 50)

if __name__ == '__main__':
    main()
```

### 1.5 Utilities — вспомогательные функции

**`src/utils/config.py`:**
```python
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Загрузить конфигурацию из YAML файла."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config
```

**`config/config.yaml`:**
```yaml
# Пути к данным
data_dir: "data/raw"
data_file: "dataset.csv"
target_col: "target"

# Параметры разделения
test_size: 0.2

# Параметры модели
model_params:
  n_estimators: 100
  max_depth: 10
  random_state: 42

# Пути сохранения
model_save_path: "models/baseline_model.pkl"
metrics_path: "reports/metrics.json"
```

---

## 2. JavaScript/TypeScript: Backend API

### 2.1 Data Layer

**`src/data/repositories/userRepository.ts`:**
```typescript
import { User } from '../models/User';
import { Database } from '../database';

export class UserRepository {
  constructor(private db: Database) {}

  async findById(id: string): Promise<User | null> {
    const result = await this.db.query(
      'SELECT * FROM users WHERE id = $1',
      [id]
    );
    return result.rows[0] || null;
  }

  async create(userData: Partial<User>): Promise<User> {
    const result = await this.db.query(
      'INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *',
      [userData.name, userData.email]
    );
    return result.rows[0];
  }

  async update(id: string, userData: Partial<User>): Promise<User> {
    const result = await this.db.query(
      'UPDATE users SET name = $1, email = $2 WHERE id = $3 RETURNING *',
      [userData.name, userData.email, id]
    );
    return result.rows[0];
  }

  async delete(id: string): Promise<void> {
    await this.db.query('DELETE FROM users WHERE id = $1', [id]);
  }
}
```

### 2.2 Domain Layer

**`src/domain/services/userService.ts`:**
```typescript
import { UserRepository } from '../../data/repositories/userRepository';
import { User } from '../../data/models/User';

export class UserService {
  constructor(private userRepo: UserRepository) {}

  async getUser(id: string): Promise<User> {
    const user = await this.userRepo.findById(id);
    if (!user) {
      throw new Error('User not found');
    }
    return user;
  }

  async createUser(userData: Partial<User>): Promise<User> {
    // Бизнес-логика валидации
    if (!userData.email || !userData.email.includes('@')) {
      throw new Error('Invalid email');
    }

    return await this.userRepo.create(userData);
  }

  async updateUser(id: string, userData: Partial<User>): Promise<User> {
    // Проверка существования
    await this.getUser(id);
    
    return await this.userRepo.update(id, userData);
  }

  async deleteUser(id: string): Promise<void> {
    // Проверка существования
    await this.getUser(id);
    
    await this.userRepo.delete(id);
  }
}
```

### 2.3 API Layer

**`src/api/controllers/userController.ts`:**
```typescript
import { Request, Response } from 'express';
import { UserService } from '../../domain/services/userService';

export class UserController {
  constructor(private userService: UserService) {}

  async getUser(req: Request, res: Response) {
    try {
      const user = await this.userService.getUser(req.params.id);
      res.json(user);
    } catch (error) {
      res.status(404).json({ error: error.message });
    }
  }

  async createUser(req: Request, res: Response) {
    try {
      const user = await this.userService.createUser(req.body);
      res.status(201).json(user);
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async updateUser(req: Request, res: Response) {
    try {
      const user = await this.userService.updateUser(req.params.id, req.body);
      res.json(user);
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async deleteUser(req: Request, res: Response) {
    try {
      await this.userService.deleteUser(req.params.id);
      res.status(204).send();
    } catch (error) {
      res.status(404).json({ error: error.message });
    }
  }
}
```

**`src/api/routes/userRoutes.ts`:**
```typescript
import { Router } from 'express';
import { UserController } from '../controllers/userController';

export function createUserRoutes(userController: UserController): Router {
  const router = Router();

  router.get('/users/:id', (req, res) => userController.getUser(req, res));
  router.post('/users', (req, res) => userController.createUser(req, res));
  router.put('/users/:id', (req, res) => userController.updateUser(req, res));
  router.delete('/users/:id', (req, res) => userController.deleteUser(req, res));

  return router;
}
```

### 2.4 Presentation Layer

**`src/app.ts`:**
```typescript
import express from 'express';
import { Database } from './data/database';
import { UserRepository } from './data/repositories/userRepository';
import { UserService } from './domain/services/userService';
import { UserController } from './api/controllers/userController';
import { createUserRoutes } from './api/routes/userRoutes';

// Инициализация зависимостей
const db = new Database(process.env.DATABASE_URL);
const userRepo = new UserRepository(db);
const userService = new UserService(userRepo);
const userController = new UserController(userService);

// Создание Express приложения
const app = express();
app.use(express.json());

// Подключение маршрутов
app.use('/api', createUserRoutes(userController));

// Обработка ошибок
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Internal server error' });
});

export default app;
```

**`src/index.ts`:**
```typescript
import app from './app';
import dotenv from 'dotenv';

dotenv.config();

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

---

## 3. Makefile / Task Runner

### 3.1 Makefile (универсальный)

**`Makefile`:**
```makefile
.PHONY: help install run test lint format clean

help:  ## Показать эту справку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Установить зависимости
	pip install -r requirements.txt

run:  ## Запустить приложение
	python main.py

test:  ## Запустить тесты
	pytest tests/ -v

lint:  ## Проверить код линтерами
	flake8 src/ tests/
	mypy src/

format:  ## Форматировать код
	black src/ tests/
	isort src/ tests/

clean:  ## Очистить временные файлы
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache
```

### 3.2 package.json scripts (Node.js)

**`package.json`:**
```json
{
  "name": "diploma-project",
  "version": "1.0.0",
  "scripts": {
    "start": "node dist/index.js",
    "dev": "nodemon src/index.ts",
    "build": "tsc",
    "test": "jest --coverage",
    "test:watch": "jest --watch",
    "lint": "eslint src/**/*.ts",
    "format": "prettier --write src/**/*.ts",
    "clean": "rm -rf dist"
  }
}
```

---

## 4. Конфигурация окружения

### 4.1 Python: requirements.txt

**`requirements.txt`:**
```text
# Core
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.0

# Visualization
matplotlib==3.7.0
seaborn==0.12.0

# Configuration
python-dotenv==1.0.0
pyyaml==6.0

# ML (optional)
xgboost==1.7.0
lightgbm==3.3.0

# Development
pytest==7.2.0
black==23.1.0
flake8==6.0.0
```

### 4.2 Node.js: package.json dependencies

```json
{
  "dependencies": {
    "express": "^4.18.0",
    "dotenv": "^16.0.0",
    "pg": "^8.10.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "ts-node": "^10.9.0",
    "nodemon": "^2.0.0",
    "jest": "^29.0.0",
    "@types/express": "^4.17.0",
    "@types/node": "^18.0.0",
    "eslint": "^8.0.0",
    "prettier": "^2.8.0"
  }
}
```

---

## 5. Тестирование

### 5.1 Python: pytest

**`tests/test_preprocessing.py`:**
```python
import pytest
import pandas as pd
from src.data.preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Тестовые данные."""
    return pd.DataFrame({
        'col_a': [1, 2, None, 4],
        'col_b': [10, None, 30, 40],
        'target': [0, 1, 0, 1]
    })

def test_clean_removes_na(sample_data):
    """Тест: clean() удаляет пропуски."""
    preprocessor = DataPreprocessor()
    cleaned = preprocessor.clean(sample_data)
    
    assert len(cleaned) == 2  # только 2 строки без NA
    assert cleaned.isna().sum().sum() == 0

def test_engineer_features_creates_ratio(sample_data):
    """Тест: engineer_features() создаёт новый признак."""
    preprocessor = DataPreprocessor()
    df = preprocessor.clean(sample_data)
    df = preprocessor.engineer_features(df)
    
    assert 'feature_ratio' in df.columns
```

### 5.2 JavaScript: Jest

**`tests/userService.test.ts`:**
```typescript
import { UserService } from '../src/domain/services/userService';
import { UserRepository } from '../src/data/repositories/userRepository';

// Mock repository
const mockUserRepo = {
  findById: jest.fn(),
  create: jest.fn(),
  update: jest.fn(),
  delete: jest.fn(),
};

describe('UserService', () => {
  let userService: UserService;

  beforeEach(() => {
    userService = new UserService(mockUserRepo as any);
    jest.clearAllMocks();
  });

  describe('getUser', () => {
    it('should return user when found', async () => {
      const mockUser = { id: '1', name: 'John', email: 'john@example.com' };
      mockUserRepo.findById.mockResolvedValue(mockUser);

      const result = await userService.getUser('1');

      expect(result).toEqual(mockUser);
      expect(mockUserRepo.findById).toHaveBeenCalledWith('1');
    });

    it('should throw error when user not found', async () => {
      mockUserRepo.findById.mockResolvedValue(null);

      await expect(userService.getUser('1')).rejects.toThrow('User not found');
    });
  });

  describe('createUser', () => {
    it('should throw error for invalid email', async () => {
      const userData = { name: 'John', email: 'invalid' };

      await expect(userService.createUser(userData)).rejects.toThrow('Invalid email');
    });
  });
});
```

---

## 6. Docker (опционально)

### 6.1 Dockerfile (Python)

**`Dockerfile`:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Запуск
CMD ["python", "main.py"]
```

### 6.2 docker-compose.yml

**`docker-compose.yml`:**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
    depends_on:
      - db
    volumes:
      - ./data:/app/data

  db:
    image: postgres:14
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

**Примечание:** Это только примеры. Адаптируйте под свой проект и стек.
