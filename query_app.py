from flask import Flask, render_template, request, jsonify, redirect, url_for
import psycopg2
import json
import os
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Tuple
import uuid
from crewai import Agent, Task, Crew, Process
from tools import DocextractTool
from tools import SummarizeTool
from langchain_openai import OpenAI

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'postgres',  # Use service name from docker-compose
    'port': 5432,
    'database': 'crewai_db',
    'user': 'postgres',
    'password': 'postgres'
}

# OpenAI configuration
OPENAI_CONFIG = {
    'model': 'text-embedding-ada-002',
    'api_key': os.getenv('OPENAI_API_KEY')
}

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(**DB_CONFIG)

def get_query_embedding(query_text: str) -> List[float]:
    """Generate embedding for a query text."""
    if not OPENAI_CONFIG['api_key']:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    embeddings_model = OpenAIEmbeddings(
        model=OPENAI_CONFIG['model'],
        api_key=OPENAI_CONFIG['api_key']
    )
    return embeddings_model.embed_query(query_text)

def store_user_query(query_text: str, query_embedding: List[float], user_ip: str = None, session_id: str = None) -> int:
    """Store user query in the database and return query ID."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Convert the embedding list to a PostgreSQL vector format
        embedding_vector = '[' + ','.join(map(str, query_embedding)) + ']'
        
        cur.execute("""
            INSERT INTO user_queries (query_text, query_embedding, user_ip, session_id)
            VALUES (%s, %s::vector, %s, %s)
            RETURNING id
        """, (query_text, embedding_vector, user_ip, session_id))
        
        query_id = cur.fetchone()[0]
        conn.commit()
        return query_id
    finally:
        cur.close()
        conn.close()

def search_similar_documents(query_embedding: List[float], limit: int = 5) -> List[Dict]:
    """Search for similar documents using vector similarity."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Convert the embedding list to a PostgreSQL vector format
        embedding_vector = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Use cosine similarity to find closest embeddings
        cur.execute("""
            SELECT doc_id, chunk_index, content, metadata, 
                   1 - (embedding <=> %s::vector) as similarity_score
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding_vector, embedding_vector, limit))
        
        results = []
        for row in cur.fetchall():
            # Handle metadata - it might already be a dict from JSONB or a string
            metadata = row[3]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            elif metadata is None:
                metadata = {}
            
            results.append({
                'doc_id': row[0],
                'chunk_index': row[1],
                'content': row[2],
                'metadata': metadata,
                'similarity_score': float(row[4])
            })
        
        return results
    finally:
        cur.close()
        conn.close()

def trigger_ingestion() -> str:
    """Trigger the document ingestion process via CrewAI agent and DocextractTool."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY environment variable not set."

        llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
        agent = Agent(
            role="Document Processor",
            goal="Extract and store document embeddings for semantic search",
            backstory="You are an expert in processing documents and preparing them for semantic search.",
            tools=[DocextractTool()],
            llm=llm,
            verbose=True
        )
        task = Task(
            description="Extract embeddings from PDF files in the '/app/docs' directory and store them in the vector database.",
            expected_output="A summary of processed files and the number of chunks stored.",
            agent=agent
        )
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"Error during ingestion: {str(e)}"

@app.route('/')
def index():
    """Main page with query interface."""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handle semantic search queries."""
    try:
        query_text = request.form.get('query', '').strip()
        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400
        
        # Generate embedding for the query
        query_embedding = get_query_embedding(query_text)
        
        # Store the query
        user_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
        session_id = request.cookies.get('session_id', str(uuid.uuid4()))
        query_id = store_user_query(query_text, query_embedding, user_ip, session_id)
        
        # Search for similar documents
        results = search_similar_documents(query_embedding, limit=5)

        # Kickoff SummarizerAgent via SummarizeTool and wait for completion
        # Prepare contexts from results
        contexts = [r.get('content', '') for r in results]

        # Create agent (for orchestration/traceability) and run the tool synchronously
        api_key = os.getenv("OPENAI_API_KEY")
        llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key) if api_key else None
        summarizer_agent = Agent(
            role="SummarizerAgent",
            goal="Summarize retrieved contexts into a concise knowledge summary for the user query.",
            backstory="You turn retrieved passages into accurate, actionable summaries.",
            tools=[SummarizeTool()],
            llm=llm,
            verbose=False
        )
        task = Task(
            description=(
                "Use SummarizeTool with provided query_id, query_text and contexts to produce and store a summary."
            ),
            expected_output="A stored summary row and the summary text returned.",
            agent=summarizer_agent
        )

        crew =  Crew(
            agents=[summarizer_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True)

        crew_output = crew.kickoff()
        if crew_output.json_dict:
            print(f"JSON Output: {json.dumps(crew_output.json_dict, indent=2)}")
        if crew_output.pydantic:
            print(f"Pydantic Output: {crew_output.pydantic}")

        summary_text = crew_output.summary_text

        return jsonify({
            'query_id': query_id,
            'query_text': query_text,
            'results': results,
            'total_results': len(results),
            'summary': summary_text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ingest', methods=['POST'])
def ingest():
    """Trigger document ingestion."""
    try:
        result = trigger_ingestion()
        return jsonify({'message': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    """Show query history."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT id, query_text, created_at, user_ip
            FROM user_queries
            ORDER BY created_at DESC
            LIMIT 50
        """)
        
        queries = []
        for row in cur.fetchall():
            queries.append({
                'id': row[0],
                'query_text': row[1],
                'created_at': row[2],
                'user_ip': row[3]
            })
        
        return render_template('history.html', queries=queries)
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50505, debug=True)
