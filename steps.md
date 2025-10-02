# 📋 Пошаговый план развития системы классификации документов

## 🎯 Текущее состояние системы

### ✅ Что уже реализовано:
- Streamlit веб-приложение с многопользовательским режимом
- Классификация документов по SBERT эмбеддингам
- PostgreSQL база данных с vector расширением
- Анализ стандартности договоров по жестко закодированным правилам
- Парсинг DOCX/PDF документов
- Кеширование обработанных файлов
- Индексация шаблонов в базе данных

### ❌ Текущие проблемы:
- Отсутствие результатов после обработки документов
- Жестко закодированные правила анализа
- Медленная обработка больших файлов
- Отсутствие продвинутой аналитики

---

## 🚀 Этап 1: Исправление критических ошибок (1-2 дня)

### 🔧 Шаг 1.1: Диагностика и исправление проблем с БД

**Что делаем:**
```bash
# 1. Проверяем подключение к PostgreSQL
psql -d documents_cem -c "SELECT version();"

# 2. Проверяем существование расширения vector
psql -d documents_cem -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# 3. Если расширение отсутствует - устанавливаем
psql -d documents_cem -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Как делаем:**
1. Добавить отладочную информацию в `app_nw.py`
2. Проверить корректность передачи `session_id` в скрипт классификации
3. Добавить логирование в `classification.py`

**Что получаем:**
- Рабочая система с корректным сохранением результатов
- Отображение результатов классификации в веб-интерфейсе

### 🔧 Шаг 1.2: Оптимизация производительности

**Что делаем:**
```python
# В modules/embeddings.py добавляем батчинг
class EmbeddingBackend:
    def encode_batch(self, texts: List[str], batch_size: int = 16):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            results.extend(batch_embeddings)
        return np.array(results)
```

**Как делаем:**
1. Модифицировать `classification.py` для батчевой обработки
2. Добавить прогресс-бар в Streamlit интерфейс
3. Оптимизировать запросы к БД

**Что получаем:**
- Ускорение обработки в 2-3 раза
- Снижение потребления памяти на 40%
- Визуальная обратная связь для пользователя

---

## 🔄 Этап 2: Улучшение архитектуры (1-2 недели)

### 🏗️ Шаг 2.1: Вынос конфигурации в файлы

**Что делаем:**
```yaml
# policy/contract_rules.yaml
rules:
  contract_term:
    max_years: 3
    allowed_extensions: false
  payment_terms:
    min_days: 60
    exceptions: ["legal_requirement"]
  prepayment:
    requires_approval: true
    bank_guarantee_threshold: 1000000
  acceptance_period:
    min_working_days: 5
```

**Как делаем:**
1. Создать директорию `policy/` с конфигурационными файлами
2. Модифицировать `modules/audit_standart.py` для чтения из YAML
3. Добавить валидацию конфигурации

**Что получаем:**
- Гибкие правила анализа без изменения кода
- Возможность A/B тестирования разных правил
- Легкое обновление политик компании

### 🏗️ Шаг 2.2: Добавление системы логирования и мониторинга

**Что делаем:**
```python
# modules/monitoring.py
import structlog
from prometheus_client import Counter, Histogram, start_http_server

# Метрики
classification_requests = Counter('classification_requests_total', 
                                 'Total classification requests', 
                                 ['status', 'template_type'])
processing_time = Histogram('document_processing_seconds',
                           'Document processing time')

logger = structlog.get_logger()
```

**Как делаем:**
1. Интегрировать структурированное логирование во все модули
2. Добавить метрики Prometheus
3. Настроить экспорт метрик в Grafana

**Что получаем:**
- Детальная телеметрия работы системы
- Возможность отслеживания производительности
- Алерты при критических ошибках

### 🏗️ Шаг 2.3: Улучшение пользовательского интерфейса

**Что делаем:**
```python
# В app_nw.py добавляем новые возможности
def show_advanced_analytics():
    """Расширенная аналитика документов"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Обработано сегодня", get_daily_count())
    with col2:
        st.metric("Средняя точность", f"{get_avg_accuracy():.1f}%")
    with col3:
        st.metric("Время обработки", f"{get_avg_processing_time():.1f}s")
    
    # Графики трендов
    show_accuracy_trends()
    show_template_usage_stats()
```

**Как делаем:**
1. Добавить дашборд с ключевыми метриками
2. Реализовать фильтрацию и поиск по результатам
3. Добавить экспорт результатов в Excel/PDF

**Что получаем:**
- Удобный интерфейс для аналитиков
- Возможность отслеживания трендов
- Экспорт данных для внешних систем

---

## 🤖 Этап 3: Интеграция продвинутых ML моделей (2-4 недели)

### 🧠 Шаг 3.1: Интеграция Reranker модели

**Что делаем:**
```python
# modules/reranker.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DocumentReranker:
    def __init__(self, model_path="./models/reranker_ru"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def rerank_candidates(self, query: str, candidates: List[Dict], top_k: int = 5):
        """Переранжирование кандидатов по релевантности"""
        scores = []
        for candidate in candidates:
            inputs = self.tokenizer(query, candidate['text'], 
                                  return_tensors="pt", 
                                  max_length=512, 
                                  truncation=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = torch.softmax(outputs.logits, dim=-1)[0][1].item()
                scores.append(score)
        
        # Сортируем по убыванию релевантности
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in ranked_indices[:top_k]]
```

**Как делаем:**
1. Интегрировать reranker в пайплайн классификации
2. Добавить A/B тестирование с/без reranker
3. Настроить гиперпараметры для оптимальной производительности

**Что получаем:**
- Повышение точности классификации на 10-15%
- Более релевантные результаты поиска шаблонов
- Лучшее ранжирование похожих документов

### 🧠 Шаг 3.2: Интеграция локальной LLM (Phi-3-mini)

**Что делаем:**
```python
# modules/llm_analyzer.py
from llama_cpp import Llama

class ContractAnalyzer:
    def __init__(self, model_path="./models/llm/gguf/Phi-3-mini-4k-instruct-q4.gguf"):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )
    
    def analyze_differences(self, document: str, template: str) -> Dict:
        """Анализ отличий документа от шаблона с помощью LLM"""
        prompt = f"""
        <|system|>Ты эксперт по анализу договоров. Сравни договор с шаблоном и найди ключевые отличия.<|end|>
        
        <|user|>
        ШАБЛОН:
        {template[:2000]}
        
        ДОКУМЕНТ:
        {document[:2000]}
        
        Найди и опиши основные отличия в структуре JSON:
        {{
            "differences": [
                {{"section": "название_раздела", "description": "описание_отличия", "severity": "high|medium|low"}},
            ],
            "compliance_score": 0.85,
            "recommendations": ["рекомендация1", "рекомендация2"]
        }}
        <|end|>
        
        <|assistant|>
        """
        
        response = self.llm(prompt, max_tokens=1024, temperature=0.1)
        return self._parse_llm_response(response['choices'][0]['text'])
    
    def generate_summary(self, document: str) -> str:
        """Генерация краткого резюме документа"""
        prompt = f"""
        <|system|>Создай краткое резюме договора на русском языке.<|end|>
        
        <|user|>
        Договор:
        {document[:3000]}
        
        Создай структурированное резюме:
        - Тип договора
        - Стороны
        - Предмет
        - Сумма
        - Сроки
        - Ключевые условия
        <|end|>
        
        <|assistant|>
        """
        
        response = self.llm(prompt, max_tokens=512, temperature=0.1)
        return response['choices'][0]['text']
```

**Как делаем:**
1. Интегрировать LLM анализ в пайплайн обработки
2. Добавить кеширование результатов LLM анализа
3. Создать промпты для разных типов анализа

**Что получаем:**
- Глубокий семантический анализ документов
- Автоматическое выявление нестандартных условий
- Генерация рекомендаций по улучшению документов

### 🧠 Шаг 3.3: Система обратной связи и самообучения

**Что делаем:**
```python
# modules/feedback_system.py
class FeedbackCollector:
    def __init__(self, db: DB):
        self.db = db
    
    def record_user_feedback(self, doc_id: str, template_id: str, 
                           user_rating: int, comments: str):
        """Сохранение обратной связи пользователя"""
        with self.db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_feedback 
                    (doc_id, template_id, rating, comments, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                """, (doc_id, template_id, user_rating, comments))
    
    def analyze_feedback_trends(self) -> Dict:
        """Анализ трендов в обратной связи"""
        # Анализ паттернов в отзывах
        # Выявление проблемных шаблонов
        # Рекомендации по улучшению
        pass
    
    def suggest_threshold_adjustments(self) -> Dict:
        """Предложения по корректировке порогов"""
        # Анализ ложных срабатываний
        # Оптимизация порогов классификации
        pass
```

**Как делаем:**
1. Добавить интерфейс для сбора обратной связи в Streamlit
2. Реализовать анализ паттернов в отзывах
3. Создать систему автоматической корректировки параметров

**Что получаем:**
- Непрерывное улучшение качества классификации
- Адаптация системы под специфику организации
- Выявление проблемных областей

---

## 🌐 Этап 4: Масштабирование и интеграция (1-2 месяца)

### 🔌 Шаг 4.1: Создание REST API

**Что делаем:**
```python
# api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI(title="Document Classification API", version="1.0.0")

@app.post("/api/v1/classify")
async def classify_document(file: UploadFile = File(...)):
    """Классификация загруженного документа"""
    try:
        # Сохранение временного файла
        content = await file.read()
        
        # Асинхронная обработка
        result = await asyncio.get_event_loop().run_in_executor(
            None, process_document, content, file.filename
        )
        
        return {
            "status": "success",
            "document_id": result["doc_id"],
            "template_match": result["template"],
            "similarity_score": result["score"],
            "compliance_analysis": result["compliance"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/templates")
async def get_templates():
    """Получение списка доступных шаблонов"""
    # Возврат списка шаблонов из БД
    pass

@app.post("/api/v1/feedback")
async def submit_feedback(feedback: FeedbackModel):
    """Отправка обратной связи"""
    # Сохранение отзыва в БД
    pass
```

**Как делаем:**
1. Создать FastAPI приложение с асинхронной обработкой
2. Добавить аутентификацию и авторизацию (JWT)
3. Реализовать rate limiting и валидацию

**Что получаем:**
- API для интеграции с внешними системами
- Асинхронная обработка больших объемов документов
- Стандартизированный интерфейс взаимодействия

### 🔌 Шаг 4.2: Микросервисная архитектура

**Что делаем:**
```yaml
# docker-compose.yml
version: '3.8'
services:
  api-gateway:
    build: ./gateway
    ports:
      - "8000:8000"
    depends_on:
      - classification-service
      - template-service
  
  classification-service:
    build: ./services/classification
    environment:
      - SBERT_MODEL_PATH=/models/sbert_large_nlu_ru
    volumes:
      - ./models:/models
  
  template-service:
    build: ./services/templates
    depends_on:
      - postgres
  
  analytics-service:
    build: ./services/analytics
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      - POSTGRES_DB=documents_cem
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
  
  redis:
    image: redis:7-alpine
```

**Как делаем:**
1. Разделить монолитное приложение на микросервисы
2. Настроить межсервисное взаимодействие через gRPC/HTTP
3. Добавить service discovery и load balancing

**Что получаем:**
- Горизонтальное масштабирование компонентов
- Независимое развертывание сервисов
- Отказоустойчивость системы

### 🔌 Шаг 4.3: Интеграция с внешними системами

**Что делаем:**
```python
# integrations/email_processor.py
import imaplib
import email
from email.mime.multipart import MIMEMultipart

class EmailIntegration:
    def __init__(self, imap_server: str, username: str, password: str):
        self.imap = imaplib.IMAP4_SSL(imap_server)
        self.imap.login(username, password)
    
    def process_incoming_emails(self):
        """Обработка входящих писем с документами"""
        self.imap.select('INBOX')
        _, messages = self.imap.search(None, 'UNSEEN')
        
        for msg_id in messages[0].split():
            _, msg_data = self.imap.fetch(msg_id, '(RFC822)')
            email_body = msg_data[0][1]
            email_message = email.message_from_bytes(email_body)
            
            # Извлечение вложений
            attachments = self._extract_attachments(email_message)
            
            # Обработка документов
            for attachment in attachments:
                if self._is_document(attachment['filename']):
                    result = self._classify_document(attachment['content'])
                    self._send_response_email(email_message, result)

# integrations/sharepoint_sync.py
class SharePointIntegration:
    def sync_templates(self):
        """Синхронизация шаблонов с SharePoint"""
        # Загрузка новых шаблонов
        # Обновление существующих
        # Удаление устаревших
        pass
```

**Как делаем:**
1. Создать интеграции с популярными системами (SharePoint, Exchange, CRM)
2. Добавить webhook'и для real-time обработки
3. Реализовать очереди сообщений для надежной доставки

**Что получаем:**
- Автоматическая обработка документов из email
- Синхронизация с корпоративными хранилищами
- Интеграция в существующие бизнес-процессы

---

## 📊 Этап 5: Продвинутая аналитика и BI (2-3 месяца)

### 📈 Шаг 5.1: Система аналитики и отчетности

**Что делаем:**
```python
# analytics/dashboard.py
import plotly.graph_objects as go
import plotly.express as px

class AnalyticsDashboard:
    def create_compliance_trends(self, data: pd.DataFrame):
        """График трендов соответствия стандартам"""
        fig = px.line(data, x='date', y='compliance_rate', 
                     color='document_type',
                     title='Тренды соответствия стандартам по типам документов')
        return fig
    
    def create_template_usage_heatmap(self, data: pd.DataFrame):
        """Тепловая карта использования шаблонов"""
        pivot_data = data.pivot_table(
            values='usage_count', 
            index='template_name', 
            columns='month'
        )
        
        fig = px.imshow(pivot_data, 
                       title='Использование шаблонов по месяцам',
                       color_continuous_scale='Blues')
        return fig
    
    def generate_executive_report(self, period: str) -> Dict:
        """Генерация исполнительного отчета"""
        return {
            'total_processed': self._get_total_processed(period),
            'accuracy_improvement': self._calculate_accuracy_trend(period),
            'cost_savings': self._estimate_cost_savings(period),
            'risk_reduction': self._calculate_risk_metrics(period)
        }
```

**Как делаем:**
1. Создать интерактивные дашборды с Plotly/Dash
2. Реализовать автоматическую генерацию отчетов
3. Добавить алерты на аномалии в данных

**Что получаем:**
- Визуализация ключевых метрик бизнеса
- Автоматические отчеты для менеджмента
- Раннее обнаружение проблем

### 📈 Шаг 5.2: Предиктивная аналитика

**Что делаем:**
```python
# analytics/predictive.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

class PredictiveAnalytics:
    def predict_processing_load(self, historical_data: pd.DataFrame) -> Dict:
        """Прогнозирование нагрузки на систему"""
        features = ['day_of_week', 'month', 'season', 'holiday']
        X = historical_data[features]
        y = historical_data['document_count']
        
        model = RandomForestRegressor(n_estimators=100)
        
        # Валидация на временных рядах
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X, y, cv=tscv)
        
        model.fit(X, y)
        
        # Прогноз на следующие 30 дней
        future_dates = pd.date_range(
            start=historical_data.index.max() + pd.Timedelta(days=1),
            periods=30
        )
        
        future_features = self._generate_future_features(future_dates)
        predictions = model.predict(future_features)
        
        return {
            'predictions': predictions.tolist(),
            'dates': future_dates.tolist(),
            'confidence_score': scores.mean()
        }
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """Обнаружение аномалий в обработке документов"""
        from sklearn.ensemble import IsolationForest
        
        features = ['processing_time', 'similarity_score', 'file_size']
        X = data[features]
        
        iso_forest = IsolationForest(contamination=0.1)
        anomalies = iso_forest.fit_predict(X)
        
        anomaly_records = data[anomalies == -1]
        return anomaly_records.to_dict('records')
```

**Как делаем:**
1. Обучить модели прогнозирования на исторических данных
2. Реализовать детекцию аномалий в real-time
3. Создать систему рекомендаций для оптимизации

**Что получаем:**
- Прогнозирование нагрузки на систему
- Автоматическое обнаружение аномалий
- Рекомендации по оптимизации процессов

---

## 🔒 Этап 6: Безопасность и соответствие требованиям (1 месяц)

### 🛡️ Шаг 6.1: Аудит безопасности и GDPR соответствие

**Что делаем:**
```python
# security/audit_logger.py
class SecurityAuditLogger:
    def log_document_access(self, user_id: str, doc_id: str, action: str):
        """Логирование доступа к документам"""
        audit_entry = {
            'timestamp': datetime.utcnow(),
            'user_id': user_id,
            'document_id': doc_id,
            'action': action,
            'ip_address': request.remote_addr,
            'user_agent': request.headers.get('User-Agent')
        }
        
        # Шифрование чувствительных данных
        encrypted_entry = self._encrypt_audit_entry(audit_entry)
        self._store_audit_entry(encrypted_entry)
    
    def generate_gdpr_report(self, user_id: str) -> Dict:
        """Генерация отчета по GDPR для пользователя"""
        return {
            'personal_data': self._get_user_personal_data(user_id),
            'processing_activities': self._get_processing_activities(user_id),
            'retention_policy': self._get_retention_info(),
            'deletion_rights': self._get_deletion_options()
        }

# security/data_anonymization.py
class DataAnonymizer:
    def anonymize_document_content(self, text: str) -> str:
        """Анонимизация персональных данных в документах"""
        # Замена имен, адресов, телефонов, email
        anonymized_text = self._replace_names(text)
        anonymized_text = self._replace_addresses(anonymized_text)
        anonymized_text = self._replace_contacts(anonymized_text)
        return anonymized_text
```

**Как делаем:**
1. Реализовать шифрование данных в покое и при передаче
2. Добавить детальное аудитирование всех операций
3. Создать механизмы анонимизации персональных данных

**Что получаем:**
- Соответствие требованиям GDPR и других регуляций
- Защита персональных данных
- Полный аудит доступа к информации

### 🛡️ Шаг 6.2: Система управления доступом

**Что делаем:**
```python
# security/rbac.py
class RoleBasedAccessControl:
    def __init__(self):
        self.roles = {
            'viewer': ['read_documents', 'view_results'],
            'analyst': ['read_documents', 'view_results', 'create_reports'],
            'admin': ['*'],  # Все права
            'auditor': ['read_audit_logs', 'view_results']
        }
    
    def check_permission(self, user_role: str, action: str) -> bool:
        """Проверка прав доступа"""
        if user_role not in self.roles:
            return False
        
        user_permissions = self.roles[user_role]
        return '*' in user_permissions or action in user_permissions
    
    @decorator
    def require_permission(permission: str):
        """Декоратор для проверки прав доступа"""
        def wrapper(func):
            def inner(*args, **kwargs):
                if not self.check_permission(current_user.role, permission):
                    raise PermissionError(f"Access denied for action: {permission}")
                return func(*args, **kwargs)
            return inner
        return wrapper
```

**Как делаем:**
1. Реализовать Role-Based Access Control (RBAC)
2. Добавить многофакторную аутентификацию
3. Создать систему управления сессиями

**Что получаем:**
- Гранулярный контроль доступа к функциям
- Защита от несанкционированного доступа
- Соответствие корпоративным политикам безопасности

---

## 🎯 Результаты по этапам

### 📊 Метрики успеха

| Этап | Метрика | Текущее значение | Целевое значение |
|------|---------|------------------|------------------|
| 1 | Работоспособность системы | 60% | 100% |
| 1 | Время обработки документа | 60 сек | 30 сек |
| 2 | Точность классификации | 80% | 85% |
| 2 | Удобство использования | 3/5 | 4.5/5 |
| 3 | Точность с Reranker | 85% | 95% |
| 3 | Глубина анализа | Базовая | Продвинутая |
| 4 | Пропускная способность | 10 док/час | 100 док/час |
| 4 | Время отклика API | N/A | <2 сек |
| 5 | Покрытие аналитикой | 20% | 90% |
| 5 | Точность прогнозов | N/A | 85% |
| 6 | Соответствие GDPR | 30% | 100% |
| 6 | Безопасность данных | Базовая | Продвинутая |

### 🏆 Ожидаемые результаты

**После Этапа 1:**
- ✅ Полностью рабочая система классификации
- ✅ Стабильное сохранение и отображение результатов
- ✅ Улучшенная производительность

**После Этапа 2:**
- ✅ Гибкая конфигурация правил анализа
- ✅ Детальный мониторинг и логирование
- ✅ Улучшенный пользовательский интерфейс

**После Этапа 3:**
- ✅ Высокая точность классификации (95%+)
- ✅ Глубокий семантический анализ документов
- ✅ Система непрерывного обучения

**После Этапа 4:**
- ✅ Масштабируемая микросервисная архитектура
- ✅ API для интеграции с внешними системами
- ✅ Автоматизированные бизнес-процессы

**После Этапа 5:**
- ✅ Продвинутая аналитика и BI
- ✅ Предиктивные возможности
- ✅ Автоматические отчеты для менеджмента

**После Этапа 6:**
- ✅ Соответствие всем требованиям безопасности
- ✅ GDPR compliance
- ✅ Корпоративный уровень защиты данных

---

## 📋 Чек-лист готовности к продакшену

### ✅ Функциональность
- [ ] Классификация документов работает стабильно
- [ ] Анализ стандартности выдает корректные результаты
- [ ] Веб-интерфейс отзывчив и интуитивен
- [ ] API документирован и протестирован

### ✅ Производительность
- [ ] Время обработки документа < 10 сек
- [ ] Система выдерживает нагрузку 100+ документов/час
- [ ] Использование памяти оптимизировано
- [ ] База данных индексирована

### ✅ Надежность
- [ ] Система восстанавливается после сбоев
- [ ] Есть механизмы резервного копирования
- [ ] Логирование покрывает все критические операции
- [ ] Мониторинг настроен и работает

### ✅ Безопасность
- [ ] Данные шифруются при хранении и передаче
- [ ] Аутентификация и авторизация настроены
- [ ] Аудит доступа к данным ведется
- [ ] GDPR требования выполнены

### ✅ Поддержка
- [ ] Документация актуальна и полная
- [ ] Процедуры развертывания автоматизированы
- [ ] Команда обучена работе с системой
- [ ] План аварийного восстановления готов

---

**Общее время реализации**: 6-12 месяцев  
**Команда**: 3-5 разработчиков  
**Бюджет**: Зависит от масштаба внедрения  
**ROI**: Ожидается окупаемость в течение 12-18 месяцев
