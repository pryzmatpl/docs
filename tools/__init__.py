from typing import Type, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import psycopg2
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
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
    path: str = Field(str, description="Directory path.")

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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            embeddings_model = OpenAIEmbeddings(
                model=OPENAI_CONFIG['model'],
                api_key=OPENAI_CONFIG['api_key']
            )

            # Connect to database
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()

            processed_files = []
            total_chunks = 0

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

                    # Generate embeddings for all chunks in one call (more efficient)
                    embeddings = embeddings_model.embed_documents(texts)

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

        # Call LLM
        llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
        try:
            summary_text = llm.invoke(f"{system_prompt}\n\n{user_prompt}")
        except Exception as e:
            return f"Error generating summary: {str(e)}"

        # Store in DB
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
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