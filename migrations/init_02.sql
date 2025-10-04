-- Migration to add user queries table
CREATE TABLE IF NOT EXISTS user_queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(768),  -- Same dimension as documents table
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_ip INET,
    session_id TEXT
);

-- Index for fast similarity search on query embeddings
CREATE INDEX IF NOT EXISTS user_queries_embedding_idx
ON user_queries USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 100);

-- Index for query text search
CREATE INDEX IF NOT EXISTS user_queries_text_idx ON user_queries USING gin(to_tsvector('english', query_text));

-- Index for timestamp-based queries
CREATE INDEX IF NOT EXISTS user_queries_created_at_idx ON user_queries (created_at);
