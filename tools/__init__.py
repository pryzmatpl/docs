from typing import Type, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import psycopg2
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import re

# Module-level configuration constants (kept out of Pydantic models)
DB_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'crewai_db',
    'user': 'postgres',
    'password': 'postgres'
}
OPENAI_CONFIG = {
    'model': 'text-embedding-ada-002',
    'api_key': os.getenv('OPENAI_API_KEY')
}

class DocextractToolInput(BaseModel):
    """Input schema for Docsearch."""
    path: str = Field(..., description="Directory path.")

class DocextractTool(BaseTool):
    name: str = "Document extract tool"
    description: str = "Extracts document embeddings from PDF files in a specific directory and stores them in a vector database."
    args_schema: Type[BaseModel] = DocextractToolInput

    def _run(self, path: str) -> str:
        try:
            # Validate OpenAI API key
            if not OPENAI_CONFIG['api_key']:
                return "Error: OPENAI_API_KEY environment variable not set."

            # Validate directory path
            directory = Path(path)
            if not directory.is_dir():
                return f"Error: '{path}' is not a valid directory."

            # Initialize text splitter and embeddings model
            # Smaller chunks reduce token spikes; overlap preserves context
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
            embeddings_model = OpenAIEmbeddings(
                model=OPENAI_CONFIG['model'],
                api_key=OPENAI_CONFIG['api_key']
            )

            # Connect to database
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()

            processed_files = []
            total_chunks = 0

            # Helper: conservative token estimate (~4 chars per token)
            def estimate_tokens_for_text(text: str) -> int:
                if not text:
                    return 0
                return max(1, len(text) // 4)

            # Iterate over PDF files in the directory
            for file_path in directory.glob("*.pdf"):
                try:
                    # Load and chunk PDF
                    loader = PyPDFLoader(str(file_path))
                    pages = loader.load()
                    chunks = text_splitter.split_documents(pages)
                    doc_id = file_path.stem

                    # Prepare data for database insertion
                    insert_data = []
                    texts = []
                    def clean_text_content(text: str) -> str:
                        # Remove control characters not supported by Postgres text/varchar
                        # Keep common whitespace: \t, \n, \r, \f, \v
                        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text or "")
                        return cleaned.strip()

                    for i, chunk in enumerate(chunks):
                        content = clean_text_content(chunk.page_content)
                        metadata = {
                            'source': str(file_path),
                            'doc_id': doc_id,
                            'chunk_index': i,
                            'page': chunk.metadata.get('page', 0)
                        }
                        texts.append(content)
                        insert_data.append((
                            doc_id,
                            i,
                            content,
                            json.dumps(metadata)
                        ))

                    # Clip overly long texts (defensive) and build dynamic batches
                    max_tokens_per_input = 4096  # conservative per-input cap
                    max_tokens_per_request = 200000  # extra headroom under provider cap

                    clipped_texts = []
                    for t in texts:
                        est = estimate_tokens_for_text(t)
                        if est > max_tokens_per_input:
                            # clip by characters based on estimate
                            max_chars = max_tokens_per_input * 4
                            clipped_texts.append(t[:max_chars])
                        else:
                            clipped_texts.append(t)

                    embeddings = []
                    current_batch = []
                    current_tokens = 0
                    for t in clipped_texts:
                        t_tokens = estimate_tokens_for_text(t)
                        # if adding this text would exceed request cap, flush batch
                        if current_batch and (current_tokens + t_tokens) > max_tokens_per_request:
                            batch_embeddings = embeddings_model.embed_documents(current_batch)
                            embeddings.extend(batch_embeddings)
                            current_batch = []
                            current_tokens = 0
                        current_batch.append(t)
                        current_tokens += t_tokens

                    # flush final batch
                    if current_batch:
                        batch_embeddings = embeddings_model.embed_documents(current_batch)
                        embeddings.extend(batch_embeddings)

                    # Add embeddings to insert data
                    for i, embedding in enumerate(embeddings):
                        insert_data[i] = insert_data[i] + (embedding,)

                    # Insert into database
                    insert_sql = """
                        INSERT INTO documents (doc_id, chunk_index, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """
                    cur.executemany(insert_sql, insert_data)
                    conn.commit()

                    processed_files.append(doc_id)
                    total_chunks += len(chunks)

                except Exception as e:
                    return f"Error processing file {file_path}: {str(e)}"

            cur.close()
            conn.close()

            # Return summary
            if not processed_files:
                return f"No PDF files found in directory: '{path}'"
            return (
                f"Successfully processed {len(processed_files)} files: {', '.join(processed_files)}\n"
                f"Total chunks stored: {total_chunks}"
            )

        except Exception as e:
            return f"Error in document extraction: {str(e)}"


class SummarizeToolInput(BaseModel):
    """Input schema for SummarizeTool."""
    query_id: int = Field(..., description="ID of the stored user query to attach the summary to")
    query_text: str = Field(..., description="Original user query text for context")
    contexts: List[str] = Field(..., description="List of top matched chunk contents to summarize")


class SummarizeTool(BaseTool):
    name: str = "Summarize tool"
    description: str = "Summarizes the provided contexts with respect to the user's query and stores the result in the summaries table."
    args_schema: Type[BaseModel] = SummarizeToolInput

    def _run(self, query_id: int, query_text: str, contexts: List[str]) -> str:
        # Validate API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "Error: OPENAI_API_KEY environment variable not set."

        # Prepare prompt
        joined_context = "\n\n".join([c.strip() for c in contexts if c and c.strip()])
        if not joined_context:
            return "Error: No contexts provided to summarize."

        system_prompt = (
            "You are a precise research assistant. Given a user query and a set of retrieved document excerpts, "
            "produce a concise, factual, and well-structured knowledge summary answering the query. "
            "Include key points, definitions, and any caveats. Avoid speculation."
        )
        user_prompt = (
            f"User Query:\n{query_text}\n\n"
            f"Retrieved Contexts (may be partial excerpts, do not hallucinate beyond them):\n{joined_context}\n\n"
            "Task: Provide a 5-10 bullet point knowledge summary tailored to the query."
        )

        # Call LLM via chat completions API
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            summary_text = response.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

        # Store in DB
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            # Ensure summaries table exists (idempotent) and commit before insert
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    id SERIAL PRIMARY KEY,
                    query_id INT NOT NULL REFERENCES user_queries(id) ON DELETE CASCADE,
                    summary TEXT NOT NULL,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS summaries_query_id_uniq ON summaries(query_id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS summaries_created_at_idx ON summaries (created_at)
                """
            )
            conn.commit()
            cur.execute(
                """
                INSERT INTO summaries (query_id, summary, model)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (query_id, summary_text, "gpt-3.5-turbo"),
            )
            summary_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            return f"Error storing summary: {str(e)}"

        return summary_text