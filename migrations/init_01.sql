CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id TEXT,           -- e.g., original file name
    chunk_index INT,       -- chunk number in doc
    content TEXT,          -- text chunk
    metadata JSONB,        -- e.g., {'source': 'file.pdf', 'page': 1}
    embedding VECTOR(768)  -- Matches nomic-embed-text dimension; adjust if using other models
);

-- Index for fast similarity search (cosine distance)
CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);