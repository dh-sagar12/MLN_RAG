# Multi-Knowledge Base RAG Chatbot

A full-featured Retrieval-Augmented Generation (RAG) system that supports multiple knowledge bases, document uploads, and semantic search across all knowledge bases using PostgreSQL with pgvector.

## Features

- **Multi-Knowledge Base System**: Create, manage, and organize multiple knowledge bases
- **Document Upload**: Support for PDF, TXT, DOCX, PPTX, Markdown, and HTML files
- **Automatic Processing**: Background task processing for document chunking and embedding
- **Semantic Search**: Query across all knowledge bases using vector similarity search
- **Modern UI**: Clean, responsive interface built with Starlette and Tailwind CSS
- **Streaming Support**: Real-time streaming responses for chat queries

## Tech Stack

- **Python 3.11+**
- **Starlette**: Lightweight ASGI framework for web server
- **LlamaIndex**: Document processing, chunking, and embedding orchestration
- **OpenAI**: Embeddings (text-embedding-3-small) and LLM (GPT-3.5-turbo)
- **PostgreSQL + pgvector**: Vector database for storing embeddings
- **SQLAlchemy**: ORM for database operations (sync with async wrapper where needed)
- **Tailwind CSS**: Modern, utility-first CSS framework

## Prerequisites

1. **PostgreSQL 12+** with **pgvector** extension
2. **Python 3.11+**
3. **OpenAI API Key** (for embeddings and LLM)

## Setup Instructions

### 1. Install PostgreSQL and pgvector

#### Ubuntu/Debian:
```bash
# Install PostgreSQL
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# Install pgvector
sudo apt-get install postgresql-14-pgvector  # Adjust version as needed
```

#### macOS (using Homebrew):
```bash
brew install postgresql
brew install pgvector
```

#### Docker (Recommended):
```bash
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_chatbot \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. Create Database and Enable Extension

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE rag_chatbot;

# Connect to the database
\c rag_chatbot

# Enable pgvector extension
CREATE EXTENSION vector;

# Exit
\q
```

### 3. Clone and Install Dependencies

```bash
# Navigate to project directory
cd rag_chatbot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_chatbot

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-3.5-turbo

# File Storage (optional)
STORAGE_PATH=storage/uploads

# Server (optional)
HOST=0.0.0.0
PORT=8000
```

### 5. Initialize Database

The database tables and indexes will be automatically created when you start the server for the first time.

### 6. Run the Server

```bash
uvicorn app.main:app --reload
```

The application will be available at `http://localhost:8000`

## Usage Guide

### 1. Create a Knowledge Base

1. Navigate to the home page (`http://localhost:8000`)
2. Fill in the "Create New Knowledge Base" form
3. Enter a name and optional description
4. Click "Create Knowledge Base"

### 2. Upload Documents

1. Click on a knowledge base to view its detail page
2. Use the "Upload Document" form
3. Select a file (PDF, TXT, DOCX, PPTX, MD, or HTML)
4. Click "Upload & Process"

The system will automatically:
- Extract text from the file
- Split it into chunks
- Generate embeddings
- Store them in the vector database

### 3. Process Files (Optional)

If you need to reprocess all files in a knowledge base:
1. Go to the knowledge base detail page
2. Click "Process All Files" button
3. This will regenerate embeddings for all uploaded files

### 4. Query the RAG System

1. Navigate to the Chat page (`http://localhost:8000/chat`)
2. Enter your question in the text area
3. Adjust the "Top K" value (number of chunks to retrieve, default: 5)
4. Click "Ask Question"

The system will:
- Embed your query
- Search across all knowledge bases
- Retrieve the most relevant chunks
- Generate an answer using the LLM
- Display the answer along with sources and matched knowledge bases

## API Endpoints

### Knowledge Bases

- `GET /` - List all knowledge bases
- `POST /kb/create` - Create a new knowledge base
- `GET /kb/{kb_id}` - View knowledge base details
- `POST /kb/{kb_id}/delete` - Delete a knowledge base

### File Management

- `POST /kb/{kb_id}/upload` - Upload a file to a knowledge base
- `POST /kb/{kb_id}/process` - Process all files in a knowledge base

### Chat/Query

- `GET /chat` - Chat interface page
- `POST /api/query` - Query the RAG system
  ```json
  {
    "query": "Your question here",
    "top_k": 5
  }
  ```
- `POST /api/query/stream` - Stream query response

## Project Structure

```
rag_chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # Application entry point
│   ├── config.py            # Configuration settings
│   ├── database.py          # Database connection and initialization
│   ├── models/              # SQLAlchemy models
│   │   ├── knowledge_base.py
│   │   ├── uploaded_file.py
│   │   └── embedding.py
│   ├── services/            # Business logic
│   │   ├── file_service.py  # File handling and text extraction
│   │   ├── ingest_service.py # Document processing and embedding
│   │   └── rag_service.py   # RAG querying
│   ├── routes/              # API routes
│   │   ├── kb_routes.py     # Knowledge base routes
│   │   ├── file_routes.py   # File upload routes
│   │   └── chat_routes.py   # Chat/query routes
│   ├── templates/           # Jinja2 templates
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── kb_detail.html
│   │   └── chat.html
│   └── static/              # Static files
│       └── styles.css
├── requirements.txt
└── README.md
```

## Database Schema

### knowledge_bases
- `id` (UUID, Primary Key)
- `name` (String)
- `description` (Text)
- `created_at` (DateTime)
- `updated_at` (DateTime)

### uploaded_files
- `id` (UUID, Primary Key)
- `kb_id` (UUID, Foreign Key)
- `file_name` (String)
- `file_type` (String)
- `file_size` (Integer)
- `file_path` (String)
- `created_at` (DateTime)

### embeddings
- `id` (UUID, Primary Key)
- `kb_id` (UUID, Foreign Key)
- `chunk_text` (Text)
- `embedding` (Vector(1536))
- `metadata` (JSON)
- `created_at` (DateTime)

**Index**: HNSW index on `embedding` column for fast similarity search.

## Troubleshooting

### Database Connection Issues

- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Check database credentials in `.env` file
- Ensure pgvector extension is installed: `psql -c "CREATE EXTENSION IF NOT EXISTS vector;"`

### OpenAI API Issues

- Verify your API key is set in `.env`
- Check your OpenAI account has sufficient credits
- Ensure you have access to the embedding model

### File Upload Issues

- Check file permissions on `storage/uploads` directory
- Verify file type is supported
- Check file size limits

### Embedding Generation Issues

- Ensure OpenAI API key is configured
- Check network connectivity
- Verify model names in configuration

## Development

### Running in Development Mode

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Code Style

The project follows:
- Type hints for all functions
- Async/await patterns
- Clean architecture with separation of concerns
- OOP principles

## License

This project is provided as-is for demonstration purposes.

## Contributing

Feel free to submit issues and enhancement requests!

