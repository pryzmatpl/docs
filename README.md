# Semantic Knowledge Base Query Application

A Flask-based web application for querying document embeddings using semantic search. The application provides a modern UI built with Jinja templates and Bootstrap for intuitive document search.

## Features

- 🔍 **Semantic Search**: Query documents using natural language
- 📊 **Similarity Scoring**: Results ranked by semantic similarity
- 📝 **Query History**: Track all user queries with timestamps
- 🔄 **Live Ingestion**: Trigger document processing from the UI
- 🎨 **Modern UI**: Responsive design with Bootstrap and Font Awesome
- 🐳 **Docker Ready**: Complete containerized setup

## Architecture

- **Backend**: Flask application with PostgreSQL + pgvector
- **Frontend**: Jinja2 templates with Bootstrap 5
- **Embeddings**: OpenAI text-embedding-ada-002
- **Database**: PostgreSQL with vector similarity search
- **Chunking**: RecursiveCharacterTextSplitter with 20% overlap

## Quick Start

### Prerequisites

1. Docker and Docker Compose
2. OpenAI API key

### Setup

1. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Add PDF documents** to the `./docs` directory

3. **Start the services**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   - Query Interface: http://localhost:50505
   - Query History: http://localhost:50505/history

### Testing

Run the test script to verify functionality:
```bash
python test_query_app.py
```

## API Endpoints

### POST /query
Submit a semantic search query.

**Request**:
```json
{
  "query": "What is machine learning?"
}
```

**Response**:
```json
{
  "query_id": 123,
  "query_text": "What is machine learning?",
  "results": [
    {
      "doc_id": "document_name",
      "chunk_index": 0,
      "content": "Machine learning is...",
      "metadata": {
        "source": "/path/to/document.pdf",
        "page": 1,
        "doc_id": "document_name",
        "chunk_index": 0
      },
      "similarity_score": 0.85
    }
  ],
  "total_results": 5
}
```

### POST /ingest
Trigger document ingestion process.

**Response**:
```json
{
  "message": "Successfully processed 2 files: doc1, doc2\nTotal chunks stored: 45"
}
```

### GET /history
Retrieve query history.

**Response**: HTML page with query history

## Database Schema

### documents table
- `id`: Primary key
- `doc_id`: Document identifier
- `chunk_index`: Chunk position in document
- `content`: Text content
- `metadata`: JSON metadata
- `embedding`: Vector embedding (768 dimensions)

### user_queries table
- `id`: Primary key
- `query_text`: User query text
- `query_embedding`: Query vector embedding
- `created_at`: Timestamp
- `user_ip`: User IP address
- `session_id`: Session identifier

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for embedding generation

### Database Configuration
- Host: `postgres` (Docker service name)
- Port: `5432`
- Database: `crewai_db`
- User: `postgres`
- Password: `postgres`

### Chunking Configuration
- Chunk Size: 1000 characters
- Chunk Overlap: 200 characters (20% overlap)
- Splitter: RecursiveCharacterTextSplitter

## Development

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start PostgreSQL:
   ```bash
   docker-compose up postgres -d
   ```

3. Run the Flask app:
   ```bash
   python query_app.py
   ```

### File Structure
```
tender/
├── query_app.py              # Main Flask application
├── templates/
│   ├── index.html           # Main query interface
│   └── history.html         # Query history page
├── migrations/
│   ├── init_01.sql          # Documents table
│   └── init_02.sql          # User queries table
├── docker-compose.yml       # Docker services
├── Dockerfile.query         # Query app Dockerfile
├── requirements.txt         # Python dependencies
└── test_query_app.py        # Test script
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   - Ensure the environment variable is set
   - Check docker-compose.yml includes the environment variable

2. **"No relevant documents found"**
   - Run ingestion first using the "Trigger Ingestion" button
   - Ensure PDF files are in the `./docs` directory

3. **Database connection errors**
   - Verify PostgreSQL container is running: `docker-compose ps`
   - Check database logs: `docker-compose logs postgres`

4. **Port conflicts**
   - Ensure port 50505 is available
   - Modify port mapping in docker-compose.yml if needed

### Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs query-app
docker-compose logs postgres
```

## Performance Notes

- Vector similarity search uses cosine distance
- Results are limited to top 5 matches by default
- Embeddings are cached in the database
- Query history is limited to 50 recent queries

## Security Considerations

- User IP addresses are logged for analytics
- Session IDs are generated for tracking
- No authentication is implemented (add as needed)
- API endpoints are not rate-limited (consider adding)
