# Система классификации документов с применением ИИ

Система для автоматической классификации договоров и документов по шаблонам с использованием SBERT эмбеддингов и анализа стандартности.

## 📋 Архитектура системы

### 🏗️ Основные компоненты

```
├── app_nw.py                    # 🟢 ОСНОВНОЕ WEB-ПРИЛОЖЕНИЕ (Streamlit)
├── classification.py            # 🟢 СКРИПТ КЛАССИФИКАЦИИ ДОКУМЕНТОВ
├── index_templates.py           # 🟢 ИНДЕКСАЦИЯ ШАБЛОНОВ В БД
├── doc_rebuild_app.py          # 🔴 НЕ ИСПОЛЬЗУЕТСЯ
├── run_apps.py                 # 🔴 НЕ ИСПОЛЬЗУЕТСЯ
└── schema.sql                  # 🟢 СХЕМА БАЗЫ ДАННЫХ
```

### 📁 Структура модулей

```
modules/
├── __init__.py                 # 🟢 Инициализация пакета
├── db.py                       # 🟢 КРИТИЧЕСКИЙ - работа с PostgreSQL
├── embeddings.py               # 🟢 КРИТИЧЕСКИЙ - SBERT эмбеддинги
├── parsers.py                  # 🟢 КРИТИЧЕСКИЙ - парсинг документов
├── docx_parser.py              # 🟢 Парсинг DOCX файлов
├── pdf_parser.py               # 🟢 Парсинг PDF файлов
├── fast_adaptive_chunker.py    # 🟢 Умное разбиение на чанки
├── extractor_names.py          # 🟢 Извлечение названий документов
├── audit_standart.py           # 🟢 КРИТИЧЕСКИЙ - анализ стандартности
└── models_data.py              # 🔴 НЕ ИСПОЛЬЗУЕТСЯ
```

### 🗄️ Структура данных

```
data/
├── templates/                  # 🟢 ШАБЛОНЫ ДОКУМЕНТОВ (48 DOCX, 3 PDF)
└── templates_2/                # 🟢 ДОПОЛНИТЕЛЬНЫЕ ШАБЛОНЫ (1 DOCX)

input/
├── insert_doc/                 # 🟢 Новые тестовые документы
│   ├── standart/              # 🟢 Стандартные договоры (5 файлов)
│   └── no_standart/           # 🟢 Нестандартные договоры (5 файлов)
├── isert1/                    # 🟢 Тестовые документы набор 1 (7 файлов)
├── isert2/                    # 🟢 Тестовые документы набор 2 (8 файлов)
├── isert3/                    # 🟢 Тестовые документы набор 3 (4 файла)
└── isert4/                    # 🟢 Тестовые документы набор 4 (4 файла)

outputs/
├── cache/                     # 🟢 Кеш обработанных файлов (7831 PKL)
├── index/                     # 🟢 Индексы эмбеддингов
└── media/                     # 🟢 Извлеченные изображения (50 PNG)

logo/
└── logo_w.svg                 # 🟢 Логотип приложения

sours/                         # 🟢 Исходные файлы
├── requirements.txt           # 🟢 Зависимости Python
├── example.txt                # 🟢 Пример текста
└── Договор по стандартной форме.txt  # 🟢 Образец договора
```

### 🤖 Модели ИИ

```
models/
├── sbert_large_nlu_ru/         # 🟢 ОСНОВНАЯ - русская SBERT модель
├── reranker_ru/                # 🟡 ДОСТУПНА - реранкер для улучшения поиска
└── llm/gguf/                   # 🟡 ДОСТУПНА - Phi-3-mini локальная LLM
    └── Phi-3-mini-4k-instruct-q4.gguf
```

### ⚙️ Конфигурация

```
configs/
└── app_config.yaml             # 🟢 Конфигурация приложения

schema.sql                      # 🟢 КРИТИЧЕСКИЙ - схема PostgreSQL БД
```

**Примечание**: Правила анализа стандартности жестко закодированы в модуле `modules/audit_standart.py`

## 🔄 Поток данных

```mermaid
graph TD
    A[Загрузка документа] --> B[Парсинг parsers.py]
    B --> C[Извлечение названия extractor_names.py]
    C --> D[Разбиение на чанки fast_adaptive_chunker.py]
    D --> E[Создание эмбеддингов embeddings.py]
    E --> F[Поиск похожих шаблонов db.py]
    F --> G[Анализ стандартности audit_standart.py]
    G --> H[Отображение результатов app_nw.py]
```

## 📊 База данных PostgreSQL

### Схема таблиц:

```sql
-- ШАБЛОНЫ
templates (id, name, version, created_at, is_active, embedding, title, title_emb)
template_chunks (chunk_id, template_id, ord, heading, text, embedding)

-- ВХОДЯЩИЕ ДОКУМЕНТЫ
docs_inserted (id, name, version, created_at, embedding, similar_id, 
               similarity_score, session_id, user_choice_doc_id, title, title_emb)
docs_inserted_chunks (chunk_id, doc_id, ord, heading, text, embedding)
```

## 📝 Инструкции по использованию

### 1️⃣ Добавление новых шаблонов

#### Шаг 1: Подготовка файлов шаблонов
```bash
# Скопируйте шаблоны в основную директорию
cp ваш_шаблон.docx data/templates/
# или в дополнительную директорию
cp ваш_шаблон.docx data/templates_2/
# Поддерживаемые форматы: DOCX, PDF
cp ваш_шаблон.pdf data/templates/
```

#### Шаг 2: Индексация шаблонов
```bash
# Активируйте conda окружение
conda activate corp_pp_doc

# Запустите индексацию основных шаблонов
python index_templates.py \
    --templates-dir ./data/templates \
    --sbert-path ./models/sbert_large_nlu_ru \
    --embedding-dim 1024 \
    --chunk-size 350 \
    --batch-size 16 \
    --device cpu \
    --use-db

# Если нужно, проиндексируйте дополнительные шаблоны
python index_templates.py \
    --templates-dir ./data/templates_2 \
    --sbert-path ./models/sbert_large_nlu_ru \
    --embedding-dim 1024 \
    --chunk-size 350 \
    --batch-size 16 \
    --device cpu \
    --use-db
```

#### Параметры индексации:
- `--templates-dir`: Путь к директории с шаблонами
- `--sbert-path`: Путь к модели SBERT
- `--embedding-dim`: Размерность эмбеддингов (1024)
- `--chunk-size`: Размер чанка в токенах (350)
- `--batch-size`: Размер батча (16)
- `--device`: Устройство (cpu/cuda)
- `--use-db`: Сохранение в PostgreSQL

#### Шаг 3: Проверка индексации
```bash
# Проверьте, что шаблоны добавлены в БД
python -c "
from modules.db import DB
db = DB()
templates = db.get_all_templates()
print(f'Всего шаблонов: {len(templates)}')
for t in templates[-5:]:
    print(f'- {t[\"name\"]} (ID: {t[\"id\"][:8]}...)')
"
```

### 2️⃣ Путь документа при загрузке

#### Поток обработки документа:

```
1. ЗАГРУЗКА (app_nw.py)
   ↓
   📁 Временная директория tempfile.TemporaryDirectory()
   ↓
   💾 Сохранение файла: temp_dir/filename.docx

2. КЛАССИФИКАЦИЯ (classification.py)
   ↓
   📖 Парсинг: parsers.py → parse_file_to_elements()
   ↓
   🏷️ Извлечение названия: extractor_names.py → extract_title_universal()
   ↓
   ✂️ Чанкинг: fast_adaptive_chunker.py → chunk_elements()
   ↓
   🧠 Эмбеддинги: embeddings.py → EmbeddingBackend.encode()
   ↓
   🔍 Поиск похожих: db.py → find_similar_documents_enhanced()

3. СОХРАНЕНИЕ (db.py)
   ↓
   💾 docs_inserted: основная информация о документе
   ↓
   💾 docs_inserted_chunks: чанки документа с эмбеддингами

4. АНАЛИЗ СТАНДАРТНОСТИ (audit_standart.py)
   ↓
   📋 check_contract_standardness(): анализ по правилам
   ↓
   📊 Результат: STANDARD/NONSTANDARD/UNKNOWN + нарушения

5. ОТОБРАЖЕНИЕ (app_nw.py)
   ↓
   📈 Таблица результатов
   ↓
   🎯 Детальная информация
   ↓
   ⚖️ Анализ стандартности
```

#### Временные файлы:
- **Входящие документы**: `input/` (постоянное хранение)
- **Загруженные через веб**: `tempfile.TemporaryDirectory()` (автоудаление)
- **Кеш парсинга**: `outputs/cache/*.pkl` (постоянный кеш)

## 🚀 Рекомендации по улучшению

### 1️⃣ Краткосрочные улучшения (1-2 недели)

#### A. Использование Reranker модели
```python
# В modules/embeddings.py добавить:
class RerankerBackend:
    def __init__(self, model_path="./models/reranker_ru"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def rerank(self, query: str, candidates: List[str], top_k: int = 5):
        # Реализация реранкинга для улучшения точности
        pass
```

#### B. Улучшение интерфейса
- ✅ Добавить прогресс-бар для обработки
- ✅ Группировка документов по типам
- ✅ Экспорт результатов в Excel/PDF
- ✅ Фильтрация по дате/типу/схожести

#### C. Кеширование и производительность
```python
# Добавить Redis для кеширования эмбеддингов
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_embedding(text_hash):
    return r.get(f"emb:{text_hash}")

def cache_embedding(text_hash, embedding):
    r.set(f"emb:{text_hash}", embedding.tobytes())
```

### 2️⃣ Среднесрочные улучшения (1-2 месяца)

#### A. Интеграция LLM модели (Phi-3-mini)
```python
# Новый модуль: modules/llm_analyzer.py
class LocalLLMAnalyzer:
    def __init__(self, model_path="./models/llm/gguf/Phi-3-mini-4k-instruct-q4.gguf"):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=4096)
    
    def analyze_contract_differences(self, doc_text: str, template_text: str):
        prompt = f"""
        Сравни договор с шаблоном и найди отличия:
        
        ШАБЛОН:
        {template_text[:2000]}
        
        ДОКУМЕНТ:
        {doc_text[:2000]}
        
        Найди ключевые отличия:
        """
        return self.llm(prompt, max_tokens=512)
```

#### B. Автоматическое обучение
- ✅ Сбор обратной связи пользователей
- ✅ Переобучение модели на новых данных
- ✅ A/B тестирование разных порогов схожести

#### C. Расширенная аналитика
```python
# modules/analytics.py
class DocumentAnalytics:
    def analyze_trends(self):
        # Тренды по типам документов
        # Частота использования шаблонов
        # Статистика нарушений стандартности
        pass
    
    def generate_report(self, period: str):
        # Автоматические отчеты
        pass
```

### 3️⃣ Долгосрочные улучшения (3-6 месяцев)

#### A. Микросервисная архитектура
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │   Auth Service  │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   (JWT/OAuth)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
        │ Classification│ │  Templates  │ │  Analytics │
        │   Service     │ │   Service   │ │   Service  │
        └───────────────┘ └─────────────┘ └────────────┘
                                │
                        ┌───────▼───────┐
                        │   Database    │
                        │ (PostgreSQL + │
                        │    Redis)     │
                        └───────────────┘
```

#### B. Продвинутый ML Pipeline
```python
# ml_pipeline/
├── feature_engineering.py     # Извлечение признаков
├── model_training.py          # Обучение моделей
├── model_evaluation.py        # Оценка качества
├── hyperparameter_tuning.py   # Настройка параметров
└── deployment.py              # Деплой моделей
```

#### C. Интеграция с внешними системами
- ✅ API для интеграции с CRM/ERP
- ✅ Webhook для автоматической обработки
- ✅ Интеграция с электронной почтой
- ✅ Синхронизация с облачными хранилищами

### 4️⃣ Технические улучшения

#### A. Мониторинг и логирование
```python
# Добавить в requirements.txt:
# prometheus-client
# grafana-api
# structlog

import structlog
logger = structlog.get_logger()

# Метрики для мониторинга
from prometheus_client import Counter, Histogram
classification_requests = Counter('classification_requests_total')
processing_time = Histogram('document_processing_seconds')
```

#### B. Тестирование
```python
# tests/
├── test_classification.py     # Тесты классификации
├── test_embeddings.py         # Тесты эмбеддингов
├── test_db.py                 # Тесты базы данных
├── test_parsers.py            # Тесты парсеров
└── test_integration.py        # Интеграционные тесты
```

#### C. Docker контейнеризация
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app_nw.py", "--server.address", "0.0.0.0"]
```

## 🔧 Настройка окружения

### Системные требования:
- **Python**: 3.9+
- **PostgreSQL**: 12+ с расширением vector
- **RAM**: 8GB+ (для SBERT модели)
- **Storage**: 10GB+ (для моделей и кеша)

### Установка зависимостей:
```bash
# Создание conda окружения
conda create -n corp_pp_doc python=3.9
conda activate corp_pp_doc

# Установка зависимостей
pip install -r sours/requirements.txt

# Настройка PostgreSQL
export DOCS_PG_DSN="postgresql://user:password@localhost:5432/documents_cem"

# Инициализация схемы БД
psql -d documents_cem -f schema.sql
```

## 📈 Метрики качества

### Текущие показатели:
- **Точность классификации**: ~85% (при пороге 80%)
- **Время обработки**: 30-60 сек/документ
- **Размер модели**: 1.3GB (SBERT)
- **Поддерживаемые форматы**: DOCX, PDF, TXT

### Целевые показатели:
- **Точность классификации**: 95%+
- **Время обработки**: <10 сек/документ
- **Поддержка форматов**: +DOC, RTF, ODT
- **Многоязычность**: EN, DE, FR

## 🐛 Известные проблемы

1. **Медленная обработка больших PDF** - оптимизировать pdf_parser.py
2. **Высокое потребление памяти** - добавить batch processing
3. **Отсутствие версионирования шаблонов** - реализовать git-like версионирование
4. **Нет резервного копирования БД** - настроить автоматические бэкапы
5. **Отсутствие конфигурационных файлов политик** - правила жестко закодированы в audit_standart.py

## 📚 Ключевые файлы для изучения

- **Основное приложение**: `app_nw.py`
- **Классификация**: `classification.py`
- **Индексация шаблонов**: `index_templates.py`
- **Схема БД**: `schema.sql`
- **Модуль анализа стандартности**: `modules/audit_standart.py`
- **Работа с БД**: `modules/db.py`
- **Эмбеддинги**: `modules/embeddings.py`

---

**Версия документации**: 1.0  
**Последнее обновление**: $(date)  
**Автор**: AI Assistant  
**Контакты**: [ваш email]
