-- Migration to add summaries table for query summaries
CREATE TABLE IF NOT EXISTS summaries (
    id SERIAL PRIMARY KEY,
    query_id INT NOT NULL REFERENCES user_queries(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ensure one summary per query_id for now (can be relaxed later)
CREATE UNIQUE INDEX IF NOT EXISTS summaries_query_id_uniq ON summaries(query_id);

-- Support filtering by created_at
CREATE INDEX IF NOT EXISTS summaries_created_at_idx ON summaries (created_at);


