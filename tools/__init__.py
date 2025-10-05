from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import psycopg2
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os

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
                    for i, chunk in enumerate(chunks):
                        # Sanitize content to avoid NUL (0x00) characters which Postgres rejects
                        raw_content = chunk.page_content or ""
                        content = raw_content.replace("\x00", "")
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