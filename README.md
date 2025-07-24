# docker-rag-test

## Overview

A proof-of-concept Retrieval-Augmented Generation (RAG) system that demonstrates:
- Document ingestion from multiple formats (PDF, HTML, TXT, MD, DOCX)
- Text chunking and embedding generation
- Local vector storage using ChromaDB
- Query API with FastAPI for document search and chat functionality

## Features

- **Multi-format Document Support**: Process PDF, HTML, TXT, Markdown, and DOCX files
- **Flexible Embeddings**: Support for both Sentence Transformers (local) and OpenAI embeddings
- **Local Vector Database**: ChromaDB for efficient similarity search
- **RESTful API**: FastAPI-based endpoints for document ingestion, search, and chat
- **Docker Support**: Fully containerized for easy deployment
- **Test Coverage**: Comprehensive test suite for core functionality

## For Teams Using This Template

1. **Clone this repository**
2. **Replace the documents** in `data/documents/` with your team's documents (PDF, DOCX, TXT, MD, HTML)
3. **Configure environment** by copying `.env.example` to `.env` and adding your OpenAI API key (optional)
4. **Run with Docker**: `docker-compose up --build`
5. **Start querying** your documents at `http://localhost:8000`

Your documents will be automatically processed and ready for search and chat!

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for local development)
- OpenAI API key (optional, for chat functionality)

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/docker-rag-test.git
cd docker-rag-test
```

2. Create a `.env` file from the example:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key if you want chat functionality
```

3. Add your documents to the `data/documents/` directory

4. Build and run with Docker Compose:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`. **Documents in `data/documents/` will be automatically ingested on startup.**

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
uvicorn src.api.main:app --reload
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/
```

### Ingest Additional Documents

Documents in `data/documents/` are automatically ingested on startup. To manually ingest additional documents:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "/app/data/documents"}'
```

### Upload Files
```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt"
```

### Search Documents
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "k": 5
  }'
```

### Chat with Documents
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "k": 3,
    "use_context": true
  }'
```

### Get Document Count
```bash
curl http://localhost:8000/documents/count
```

### Clear All Documents
```bash
curl -X DELETE http://localhost:8000/documents
```

## Project Structure

```
docker-rag-test/
├── src/
│   ├── ingestion/         # Document loading and text splitting
│   ├── embedding/         # Embedding generation (OpenAI/Sentence Transformers)
│   ├── storage/           # Vector database interface (ChromaDB)
│   └── api/              # FastAPI application and endpoints
├── tests/                # Test suite
├── data/
│   └── documents/        # Place your documents here for auto-ingestion
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Docker Compose configuration
└── .env.example         # Environment variables template
```

## Configuration

Environment variables can be set in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required for chat functionality)
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `EMBEDDER_TYPE`: "sentence-transformer" or "openai" (default: "sentence-transformer")
- `CHROMA_PERSIST_DIRECTORY`: Directory for ChromaDB persistence
- `AUTO_INGEST_ON_STARTUP`: Enable/disable auto-ingestion on startup (default: true)
- `AUTO_INGEST_DIRECTORY`: Directory to auto-ingest documents from (default: /app/data/documents)

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Development

The project follows these principles:
1. **Test-Driven Development**: Tests are written for core functionality
2. **Modular Design**: Clear separation between ingestion, embedding, storage, and API layers
3. **Docker-First**: Fully containerized for consistent environments
4. **Type Safety**: Uses Pydantic for data validation
5. **Async Support**: FastAPI with async endpoints for better performance

## Future Enhancements

While this is a proof-of-concept with local storage, the architecture supports easy migration to:
- Cloud vector databases (AWS S3 Vector Engine, Pinecone, Qdrant)
- Serverless deployment (AWS Lambda)
- Container orchestration (AWS ECS/Fargate)
- Managed API Gateway integration

## Original Goals

This project was built following these principles:
1. Use most supported and compatible tech stack
2. Test driven development
3. Explicit folder structure separating resources from code
4. Docker-first approach for local development
5. Design for easy cloud migration
6. Follow best practices
7. Version control with regular commits