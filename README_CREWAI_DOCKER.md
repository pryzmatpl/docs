# CrewAI Document Extraction with Docker Compose

This setup allows you to run the CrewAI document extraction process within Docker Compose, making it easy to manage dependencies and ensure consistent execution across different environments.

## 🚀 Quick Start

### 1. Set up your environment

Create a `.env` file in the project root:
```bash
echo "OPENAI_API_KEY=your-actual-openai-api-key-here" > .env
```

### 2. Add PDF documents

Place your PDF files in the `./docs` directory:
```bash
# Example: Copy PDF files to docs directory
cp /path/to/your/documents/*.pdf ./docs/
```

### 3. Run the CrewAI extraction

```bash
# Start PostgreSQL database
docker-compose up postgres -d

# Run CrewAI document extraction
docker-compose up crewai-extraction
```

### 4. Access the query interface (optional)

```bash
# Start the full application with query interface
docker-compose up -d
# Then visit: http://localhost:50505
```

## 📋 Available Commands

### Run only the extraction:
```bash
docker-compose up crewai-extraction
```

### Run extraction and view logs:
```bash
docker-compose up crewai-extraction --build
```

### Run the full application (extraction + query interface):
```bash
docker-compose up -d
```

### Stop all services:
```bash
docker-compose down
```

## 🔧 Configuration

### Environment Variables

The `.env` file should contain:
```bash
OPENAI_API_KEY=your-actual-openai-api-key-here
```

### Docker Services

- **postgres**: PostgreSQL database with pgvector extension
- **crewai-extraction**: Runs document extraction once (restart: "no")
- **query-app**: Web interface for querying documents

### Volumes

- `./docs` → `/app/docs`: Your PDF documents
- `./.env` → `/app/.env`: Environment configuration

## 📊 What Happens During Extraction

1. **Prerequisites Check**: Validates API key and PDF files
2. **Agent Creation**: Creates Document Processor agent
3. **Task Execution**: Processes PDFs and generates embeddings
4. **Database Storage**: Stores chunks and embeddings in PostgreSQL
5. **Results Display**: Shows summary of processed files

## 🎯 Expected Output

```
📚 CrewAI Document Extraction Launcher
========================================
🔍 Checking prerequisites...
✅ Found 3 PDF file(s) to process
✅ OpenAI API key configured
✅ All prerequisites met!

🚀 Starting CrewAI Document Extraction...
==================================================
🤖 Creating Document Processor Agent...
📋 Creating extraction task...
👥 Assembling CrewAI team...

⚡ Executing CrewAI extraction process...
--------------------------------------------------
[Detailed CrewAI execution logs]

==================================================
🎉 CREWAI EXTRACTION COMPLETED!
==================================================

📊 EXTRACTION RESULTS:
------------------------------
Successfully processed 3 files: doc1, doc2, doc3
Total chunks stored: 45

✅ Document extraction process finished successfully!
```

## 🔍 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not properly set"**
   - Ensure your `.env` file contains the actual API key
   - Check that the `.env` file is in the project root

2. **"No PDF files found"**
   - Add PDF files to the `./docs` directory
   - Ensure files have `.pdf` extension

3. **Database connection errors**
   - Make sure PostgreSQL is running: `docker-compose up postgres -d`
   - Check database logs: `docker-compose logs postgres`

4. **Build errors**
   - Rebuild the image: `docker-compose up crewai-extraction --build`
   - Check Docker is running and has sufficient resources

### Viewing Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs crewai-extraction
docker-compose logs postgres
```

## 🎯 Next Steps

After successful extraction:

1. **Query your documents**: Start the query app and visit http://localhost:50505
2. **Check the database**: Verify embeddings were stored correctly
3. **Add more documents**: Place new PDFs in `./docs` and run extraction again

The extraction service runs once and exits (restart: "no"), so you can run it multiple times to process new documents.
