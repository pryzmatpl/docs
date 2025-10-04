import psycopg2
import json
import re
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os

# Configs
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'postgres'),  # Use 'postgres' for Docker, 'localhost' for external
    'port': 5432,
    'database': 'crewai_db',
    'user': 'postgres',
    'password': 'postgres'
}
OPENAI_CONFIG = {
    'model': 'text-embedding-ada-002',  # 1536 dimensions - we'll update DB schema to match
    'api_key': os.getenv('OPENAI_API_KEY')
}


def clean_text_content(text: str) -> str:
    """Clean text content by removing problematic characters that can't be stored in PostgreSQL."""
    # Remove NUL characters and other control characters except common whitespace
    # Keep: \t (tab), \n (newline), \r (carriage return), \f (form feed), \v (vertical tab)
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Also remove any remaining problematic characters
    cleaned = cleaned.replace('\x00', '').replace('\x01', '').replace('\x02', '').replace('\x03', '')
    return cleaned.strip()


def load_pdf(pdf_path: str) -> List[Dict]:
    """Load and chunk a PDF file, returning a list of chunk dictionaries."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        doc_chunks = []
        doc_id = Path(pdf_path).stem
        for i, chunk in enumerate(chunks):
            # Clean content by removing problematic characters
            cleaned_content = clean_text_content(chunk.page_content)
            doc_chunks.append({
                'doc_id': doc_id,
                'chunk_index': i,
                'content': cleaned_content,
                'metadata': {
                    'source': str(pdf_path),
                    'page': chunk.metadata.get('page', 0),
                    'doc_id': doc_id,
                    'chunk_index': i
                }
            })
        return doc_chunks
    except Exception as e:
        raise Exception(f"Error loading PDF {pdf_path}: {str(e)}")


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI."""
    if not OPENAI_CONFIG['api_key']:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    embeddings_model = OpenAIEmbeddings(
        model=OPENAI_CONFIG['model'],
        api_key=OPENAI_CONFIG['api_key']
    )
    return embeddings_model.embed_documents(texts)


def insert_chunks_to_db(chunks: List[Dict], conn) -> int:
    """Insert chunks and their embeddings into the database."""
    try:
        texts = [chunk['content'] for chunk in chunks]
        embeddings = get_embeddings(texts)
        insert_data = []
        for chunk, embedding in zip(chunks, embeddings):
            insert_data.append((
                chunk['doc_id'],
                chunk['chunk_index'],
                chunk['content'],
                json.dumps(chunk['metadata']),
                embedding
            ))

        cur = conn.cursor()
        insert_sql = """
            INSERT INTO documents (doc_id, chunk_index, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """
        cur.executemany(insert_sql, insert_data)
        conn.commit()
        cur.close()
        return len(chunks)
    except Exception as e:
        # Rollback the transaction on error
        conn.rollback()
        raise e


def process_directory(directory_path: str) -> str:
    """Process all PDF files in the directory and store their embeddings."""
    try:
        directory = Path(directory_path)
        if not directory.is_dir():
            return f"Error: '{directory_path}' is not a valid directory."

        conn = psycopg2.connect(**DB_CONFIG)
        processed_files = []
        total_chunks = 0

        for file_path in directory.glob("*.pdf"):
            try:
                chunks = load_pdf(str(file_path))
                chunks_inserted = insert_chunks_to_db(chunks, conn)
                processed_files.append(file_path.stem)
                total_chunks += chunks_inserted
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue

        conn.close()

        if not processed_files:
            return f"No PDF files found in directory: '{directory_path}'"
        return (
            f"Successfully processed {len(processed_files)} files: {', '.join(processed_files)}\n"
            f"Total chunks stored: {total_chunks}"
        )

    except Exception as e:
        return f"Error in document processing: {str(e)}"


if __name__ == "__main__":
    directory_path = "/app/docs"
    result = process_directory(directory_path)
    print(result)