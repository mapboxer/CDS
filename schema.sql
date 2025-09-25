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
