from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import psycopg2
import os

DEFAULT_DSN = os.getenv(
    "DOCS_PG_DSN", "postgresql://denis.murataev:123@localhost:5432/documents_cem")

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT,
    created_at TIMESTAMP DEFAULT now(),
    is_active BOOLEAN DEFAULT TRUE,
    embedding vector(1024),
    title TEXT,
    title_emb vector(1024)
);

CREATE TABLE IF NOT EXISTS template_chunks (
    chunk_id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL REFERENCES templates(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    heading TEXT,
    text TEXT NOT NULL,
    embedding vector(1024)
);

-- Таблицы для входящих документов (аналогичные templates и template_chunks)
CREATE TABLE IF NOT EXISTS docs_inserted (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT,
    created_at TIMESTAMP DEFAULT now(),
    is_active BOOLEAN DEFAULT TRUE,
    embedding vector(1024),
    similar_id TEXT REFERENCES templates(id),
    similarity_score FLOAT,
    session_id TEXT DEFAULT 'default',
    user_choice_doc_id TEXT REFERENCES templates(id),
    title TEXT,
    title_emb vector(1024)
);

CREATE TABLE IF NOT EXISTS docs_inserted_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES docs_inserted(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    heading TEXT,
    text TEXT NOT NULL,
    embedding vector(1024)
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_chunks_ivfflat ON template_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
CREATE INDEX IF NOT EXISTS idx_templates_ivfflat ON templates USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
CREATE INDEX IF NOT EXISTS idx_docs_inserted_ivfflat ON docs_inserted USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
CREATE INDEX IF NOT EXISTS idx_docs_inserted_chunks_ivfflat ON docs_inserted_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
CREATE INDEX IF NOT EXISTS idx_docs_inserted_session ON docs_inserted (session_id);
"""


@dataclass
class DB:
    dsn: str = DEFAULT_DSN

    def _conn(self):
        return psycopg2.connect(self.dsn)

    def ensure_schema(self):
        """Создание схемы БД с проверкой подключения"""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    # Проверяем существование таблиц
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'docs_inserted'
                        );
                    """)
                    docs_table_exists = cur.fetchone()[0]

                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'docs_inserted_chunks'
                        );
                    """)
                    chunks_table_exists = cur.fetchone()[0]

                    # Создаем таблицы только если они не существуют
                    if not docs_table_exists or not chunks_table_exists:
                        print("Создание таблиц для входящих документов...")
                        cur.execute(SCHEMA_SQL)
                    else:
                        print("Таблицы уже существуют, проверяем структуру...")

                    # Проверяем наличие поля session_id и добавляем если нужно
                    try:
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = 'docs_inserted' 
                                AND column_name = 'session_id'
                            );
                        """)
                        session_id_exists = cur.fetchone()[0]

                        if not session_id_exists:
                            print("Добавление поля session_id...")
                            cur.execute(
                                "ALTER TABLE docs_inserted ADD COLUMN session_id TEXT DEFAULT 'default';")
                            # Обновляем существующие записи
                            cur.execute(
                                "UPDATE docs_inserted SET session_id = 'default' WHERE session_id IS NULL;")
                        else:
                            print("Поле session_id уже существует")
                            # Обновляем записи с NULL session_id
                            cur.execute(
                                "UPDATE docs_inserted SET session_id = 'default' WHERE session_id IS NULL;")

                        # Создаем индекс для session_id если его нет
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_docs_inserted_session ON docs_inserted (session_id);")

                        # Проверяем наличие поля user_choice_doc_id и добавляем если нужно
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = 'docs_inserted' 
                                AND column_name = 'user_choice_doc_id'
                            );
                        """)
                        user_choice_exists = cur.fetchone()[0]

                        if not user_choice_exists:
                            print("Добавление поля user_choice_doc_id...")
                            cur.execute(
                                "ALTER TABLE docs_inserted ADD COLUMN user_choice_doc_id TEXT REFERENCES templates(id);")
                        else:
                            print("Поле user_choice_doc_id уже существует")

                        # Проверяем наличие поля title в templates и добавляем если нужно
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = 'templates' 
                                AND column_name = 'title'
                            );
                        """)
                        title_templates_exists = cur.fetchone()[0]

                        if not title_templates_exists:
                            print("Добавление поля title в таблицу templates...")
                            cur.execute(
                                "ALTER TABLE templates ADD COLUMN title TEXT;")
                        else:
                            print("Поле title в таблице templates уже существует")

                        # Проверяем наличие поля title_emb в templates и добавляем если нужно
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = 'templates' 
                                AND column_name = 'title_emb'
                            );
                        """)
                        title_emb_templates_exists = cur.fetchone()[0]

                        if not title_emb_templates_exists:
                            print("Добавление поля title_emb в таблицу templates...")
                            cur.execute(
                                "ALTER TABLE templates ADD COLUMN title_emb vector(1024);")
                        else:
                            print(
                                "Поле title_emb в таблице templates уже существует")

                        # Проверяем наличие поля title в docs_inserted и добавляем если нужно
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = 'docs_inserted' 
                                AND column_name = 'title'
                            );
                        """)
                        title_docs_exists = cur.fetchone()[0]

                        if not title_docs_exists:
                            print("Добавление поля title в таблицу docs_inserted...")
                            cur.execute(
                                "ALTER TABLE docs_inserted ADD COLUMN title TEXT;")
                        else:
                            print(
                                "Поле title в таблице docs_inserted уже существует")

                        # Проверяем наличие поля title_emb в docs_inserted и добавляем если нужно
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = 'docs_inserted' 
                                AND column_name = 'title_emb'
                            );
                        """)
                        title_emb_docs_exists = cur.fetchone()[0]

                        if not title_emb_docs_exists:
                            print(
                                "Добавление поля title_emb в таблицу docs_inserted...")
                            cur.execute(
                                "ALTER TABLE docs_inserted ADD COLUMN title_emb vector(1024);")
                        else:
                            print(
                                "Поле title_emb в таблице docs_inserted уже существует")

                    except Exception as e:
                        print(
                            f"Предупреждение при проверке/добавлении session_id: {e}")

                conn.commit()
                print("База данных успешно инициализирована")
        except psycopg2.OperationalError as e:
            if "does not exist" in str(e):
                # Пытаемся создать базу данных
                self._create_database()
                # Повторная попытка
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(SCHEMA_SQL)
                    conn.commit()
                    print("База данных создана и инициализирована")
            else:
                raise e

    def _create_database(self):
        """Создание базы данных если она не существует"""
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        # Подключаемся к postgres для создания БД
        admin_dsn = self.dsn.replace("/documents_cem", "/postgres")
        try:
            conn = psycopg2.connect(admin_dsn)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            cur.execute("CREATE DATABASE documents_cem")
            cur.close()
            conn.close()
            print("База данных documents_cem создана")
        except psycopg2.Error as e:
            if "already exists" not in str(e):
                raise e

    def insert_template(self, name: str, version: Optional[str], file_path: str, document_embedding=None, title: str = None, title_embedding=None) -> str:
        """Вставка шаблона в БД с эмбеддингом всего документа. Возвращает id как строку."""
        import uuid
        template_id = str(uuid.uuid4())

        with self._conn() as conn:
            with conn.cursor() as cur:
                # Подготавливаем параметры для вставки
                params = [template_id, name, version]
                sql_parts = ["id", "name", "version"]

                if document_embedding is not None:
                    # Преобразуем эмбеддинг в строку для pgvector
                    vec_str = "[" + \
                        ",".join(
                            [f"{float(x):.6f}" for x in document_embedding]) + "]"
                    params.append(vec_str)
                    sql_parts.append("embedding")

                if title is not None:
                    params.append(title)
                    sql_parts.append("title")

                if title_embedding is not None:
                    # Преобразуем эмбеддинг названия в строку для pgvector
                    title_vec_str = "[" + \
                        ",".join(
                            [f"{float(x):.6f}" for x in title_embedding]) + "]"
                    params.append(title_vec_str)
                    sql_parts.append("title_emb")

                # Формируем SQL запрос динамически
                placeholders = ",".join(["%s"] * len(params))
                columns = ",".join(sql_parts)
                sql = f"INSERT INTO templates({columns}) VALUES ({placeholders})"

                cur.execute(sql, params)
            conn.commit()
        return template_id

    def insert_chunks(self, template_id: str, chunks: List[Dict[str, Any]], embeddings) -> None:
        """Вставка чанков в БД. template_id теперь строка."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                for i, ch in enumerate(chunks):
                    vec = embeddings[i]
                    # turn numpy vector into pgvector literal
                    vec_str = "[" + \
                        ",".join([f"{float(x):.6f}" for x in vec]) + "]"
                    chunk_id = f"{template_id}:{i}"
                    heading = " > ".join(ch.get("heading_path", []) or [])
                    cur.execute("""
                        INSERT INTO template_chunks(chunk_id, template_id, ord, heading, text, embedding)
                        VALUES (%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (chunk_id) DO NOTHING
                    """, (chunk_id, template_id, ch.get("chunk_index", i), heading, ch["text"], vec_str))
            conn.commit()

    def find_similar_documents(self, query_embedding, limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Поиск похожих документов по эмбеддингу всего документа.

        Args:
            query_embedding: вектор запроса размерности 1024
            limit: максимальное количество результатов
            threshold: минимальный порог косинусного сходства

        Returns:
            List[Dict]: список похожих документов с метаданными и скорами
        """
        # Преобразуем эмбеддинг в строку для pgvector
        vec_str = "[" + \
            ",".join([f"{float(x):.6f}" for x in query_embedding]) + "]"

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        id,
                        name,
                        version,
                        created_at,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM templates 
                    WHERE embedding IS NOT NULL 
                        AND (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vec_str, vec_str, threshold, vec_str, limit))

                results = []
                for row in cur.fetchall():
                    results.append({
                        "template_id": row[0],
                        "name": row[1],
                        "version": row[2],
                        "created_at": row[3],
                        "similarity": float(row[4])
                    })

                return results

    def find_similar_documents_enhanced(self, query_embedding, title_embedding=None,
                                        limit: int = 10, threshold: float = 0.7,
                                        document_weight: float = 0.7, title_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Улучшенный поиск похожих документов с учетом эмбеддинга документа И названия.

        Args:
            query_embedding: вектор запроса размерности 1024 (эмбеддинг документа)
            title_embedding: вектор названия размерности 1024 (опционально)
            limit: максимальное количество результатов
            threshold: минимальный порог комбинированного сходства
            document_weight: вес эмбеддинга документа (по умолчанию 0.7)
            title_weight: вес эмбеддинга названия (по умолчанию 0.3)

        Returns:
            List[Dict]: список похожих документов с комбинированными скорами
        """
        # Преобразуем эмбеддинг документа в строку для pgvector
        doc_vec_str = "[" + \
            ",".join([f"{float(x):.6f}" for x in query_embedding]) + "]"

        with self._conn() as conn:
            with conn.cursor() as cur:
                if title_embedding is not None:
                    # Поиск с учетом обоих эмбеддингов. Для шаблонов без
                    # title_emb комбинированный скор равен doc_similarity,
                    # чтобы не занижать результаты.
                    title_vec_str = "[" + \
                        ",".join(
                            [f"{float(x):.6f}" for x in title_embedding]) + "]"

                    weight_sum = document_weight + title_weight
                    if weight_sum <= 0:
                        # Если веса обнуляются, используем только схожесть
                        # документа, чтобы избежать деления на ноль и не
                        # отбрасывать релевантные шаблоны.
                        effective_doc_weight = 1.0
                        effective_title_weight = 0.0
                        weight_sum = 1.0
                    else:
                        effective_doc_weight = document_weight
                        effective_title_weight = title_weight

                    cur.execute("""
                        WITH candidate AS (
                            SELECT
                                id,
                                name,
                                version,
                                created_at,
                                title,
                                (1 - (embedding <=> %s::vector)) AS doc_similarity,
                                CASE
                                    WHEN title_emb IS NOT NULL THEN (1 - (title_emb <=> %s::vector))
                                    ELSE NULL
                                END AS title_similarity
                            FROM templates
                            WHERE embedding IS NOT NULL
                        )
                        SELECT
                            id,
                            name,
                            version,
                            created_at,
                            title,
                            doc_similarity,
                            COALESCE(title_similarity, 0) AS title_similarity,
                            CASE
                                WHEN title_similarity IS NOT NULL THEN
                                    (%s * doc_similarity + %s * title_similarity) /
                                    NULLIF(%s, 0)
                                ELSE doc_similarity
                            END AS combined_similarity
                        FROM candidate
                        WHERE CASE
                                WHEN title_similarity IS NOT NULL THEN
                                    (%s * doc_similarity + %s * title_similarity) /
                                    NULLIF(%s, 0)
                                ELSE doc_similarity
                            END >= %s
                        ORDER BY combined_similarity DESC
                        LIMIT %s
                    """, (
                        doc_vec_str,
                        title_vec_str,
                        effective_doc_weight,
                        effective_title_weight,
                        weight_sum,
                        effective_doc_weight,
                        effective_title_weight,
                        weight_sum,
                        threshold,
                        limit,
                    ))
                else:
                    # Поиск только по эмбеддингу документа (как раньше)
                    cur.execute("""
                        SELECT
                            id,
                            name,
                            version,
                            created_at,
                            title,
                            (1 - (embedding <=> %s::vector)) as doc_similarity,
                            0 as title_similarity,
                            (1 - (embedding <=> %s::vector)) as combined_similarity
                        FROM templates
                        WHERE embedding IS NOT NULL
                            AND (1 - (embedding <=> %s::vector)) >= %s
                        ORDER BY combined_similarity DESC
                        LIMIT %s
                    """, (doc_vec_str, doc_vec_str, doc_vec_str, threshold, limit))

                results = []
                for row in cur.fetchall():
                    results.append({
                        "template_id": row[0],
                        "name": row[1],
                        "version": row[2],
                        "created_at": row[3],
                        "title": row[4],
                        "doc_similarity": float(row[5]),
                        "title_similarity": float(row[6]),
                        "combined_similarity": float(row[7]),
                        # для обратной совместимости
                        "similarity": float(row[7])
                    })

                return results

    def insert_incoming_document(self, name: str, version: Optional[str], file_path: str,
                                 document_embedding, similar_template_id: str = None,
                                 similarity_score: float = None, session_id: str = None,
                                 title: str = None, title_embedding=None) -> str:
        """Вставка входящего документа в БД с эмбеддингом и ссылкой на похожий шаблон."""
        import uuid
        doc_id = str(uuid.uuid4())

        with self._conn() as conn:
            with conn.cursor() as cur:
                # Подготавливаем параметры для вставки
                params = [doc_id, name, version]
                sql_parts = ["id", "name", "version"]

                # Преобразуем эмбеддинг в строку для pgvector
                vec_str = "[" + \
                    ",".join(
                        [f"{float(x):.6f}" for x in document_embedding]) + "]"
                params.append(vec_str)
                sql_parts.append("embedding")

                if similar_template_id is not None:
                    params.append(similar_template_id)
                    sql_parts.append("similar_id")

                if similarity_score is not None:
                    params.append(similarity_score)
                    sql_parts.append("similarity_score")

                if session_id is not None:
                    params.append(session_id)
                    sql_parts.append("session_id")

                if title is not None:
                    params.append(title)
                    sql_parts.append("title")

                if title_embedding is not None:
                    # Преобразуем эмбеддинг названия в строку для pgvector
                    title_vec_str = "[" + \
                        ",".join(
                            [f"{float(x):.6f}" for x in title_embedding]) + "]"
                    params.append(title_vec_str)
                    sql_parts.append("title_emb")

                # Формируем SQL запрос динамически
                placeholders = ",".join(["%s"] * len(params))
                columns = ",".join(sql_parts)
                sql = f"INSERT INTO docs_inserted({columns}) VALUES ({placeholders})"

                cur.execute(sql, params)
            conn.commit()
        return doc_id

    def insert_incoming_chunks(self, doc_id: str, chunks: List[Dict[str, Any]], embeddings) -> None:
        """Вставка чанков входящего документа в БД."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                for i, ch in enumerate(chunks):
                    vec = embeddings[i]
                    # turn numpy vector into pgvector literal
                    vec_str = "[" + \
                        ",".join([f"{float(x):.6f}" for x in vec]) + "]"
                    chunk_id = f"{doc_id}:{i}"
                    heading = " > ".join(ch.get("heading_path", []) or [])
                    cur.execute("""
                        INSERT INTO docs_inserted_chunks(chunk_id, doc_id, ord, heading, text, embedding)
                        VALUES (%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (chunk_id) DO NOTHING
                    """, (chunk_id, doc_id, ch.get("chunk_index", i), heading, ch["text"], vec_str))
            conn.commit()

    def get_classification_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение результатов классификации входящих документов."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        di.id,
                        di.name as doc_name,
                        di.created_at,
                        di.similarity_score,
                        t.id as template_id,
                        t.name as template_name,
                        t.version as template_version,
                        di.session_id
                    FROM docs_inserted di
                    LEFT JOIN templates t ON di.similar_id = t.id
                    ORDER BY di.created_at DESC, di.similarity_score DESC
                    LIMIT %s
                """, (limit,))

                results = []
                for row in cur.fetchall():
                    results.append({
                        "doc_id": row[0],
                        "doc_name": row[1],
                        "created_at": row[2],
                        "similarity_score": float(row[3]) if row[3] else None,
                        "template_id": row[4],
                        "template_name": row[5],
                        "template_version": row[6],
                        "session_id": row[7]
                    })

                return results

    def get_classification_results_by_session(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение результатов классификации входящих документов для конкретной сессии."""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    # Проверяем существование таблицы
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'docs_inserted'
                        );
                    """)
                    table_exists = cur.fetchone()[0]

                    if not table_exists:
                        print(
                            "Таблица docs_inserted не существует, возвращаем пустой список")
                        return []

                    cur.execute("""
                        SELECT 
                            di.id,
                            di.name as doc_name,
                            di.created_at,
                            di.similarity_score,
                            t.id as template_id,
                            t.name as template_name,
                            t.version as template_version,
                            di.session_id,
                            di.user_choice_doc_id
                        FROM docs_inserted di
                        LEFT JOIN templates t ON di.similar_id = t.id
                        WHERE di.session_id = %s
                        ORDER BY di.created_at DESC, di.similarity_score DESC
                        LIMIT %s
                    """, (session_id, limit))

                    results = []
                    for row in cur.fetchall():
                        results.append({
                            "doc_id": row[0],
                            "doc_name": row[1],
                            "created_at": row[2],
                            "similarity_score": float(row[3]) if row[3] else None,
                            "template_id": row[4],
                            "template_name": row[5],
                            "template_version": row[6],
                            "session_id": row[7],
                            "user_choice_doc_id": row[8]
                        })

                    return results
        except Exception as e:
            print(f"Ошибка при получении результатов классификации: {e}")
            return []

    def clear_session_data(self, session_id: str) -> None:
        """Очистка всех данных для конкретной сессии."""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    # Проверяем существование таблиц
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'docs_inserted'
                        );
                    """)
                    docs_table_exists = cur.fetchone()[0]

                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'docs_inserted_chunks'
                        );
                    """)
                    chunks_table_exists = cur.fetchone()[0]

                    if not docs_table_exists:
                        print("Таблица docs_inserted не существует, нечего очищать")
                        return

                    # Удаляем чанки документов сессии (если таблица существует)
                    if chunks_table_exists:
                        cur.execute("""
                            DELETE FROM docs_inserted_chunks 
                            WHERE doc_id IN (
                                SELECT id FROM docs_inserted WHERE session_id = %s
                            )
                        """, (session_id,))

                    # Удаляем документы сессии
                    cur.execute("""
                        DELETE FROM docs_inserted WHERE session_id = %s
                    """, (session_id,))

                conn.commit()
                print(f"Данные сессии {session_id} успешно очищены")
        except Exception as e:
            print(f"Ошибка при очистке данных сессии: {e}")
            raise e

    def get_all_templates(self) -> List[Dict[str, Any]]:
        """Получение всех шаблонов для выпадающего меню."""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, name, version
                        FROM templates
                        ORDER BY name
                    """)

                    results = []
                    for row in cur.fetchall():
                        results.append({
                            "id": row[0],
                            "name": row[1],
                            "version": row[2] or ""
                        })

                    return results
        except Exception as e:
            print(f"Ошибка при получении шаблонов: {e}")
            return []

    def update_user_choice(self, doc_id: str, template_id: str) -> bool:
        """Обновление выбора пользователя для документа."""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE docs_inserted 
                        SET user_choice_doc_id = %s
                        WHERE id = %s
                    """, (template_id, doc_id))

                conn.commit()
                return True
        except Exception as e:
            print(f"Ошибка при обновлении выбора пользователя: {e}")
            return False
