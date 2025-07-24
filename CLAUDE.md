# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready Retrieval-Augmented Generation (RAG) system designed to:
- Ingest heterogeneous documents (PDF, HTML, TXT, MD, DOCX)
- Process documents through chunking and embedding
- Store embeddings in vector databases (local initially, with easy migration to AWS)
- Provide a query API and/or chat UI backed by an LLM

## Development Methodology

This project follows these core principles from README.md:
1. **Test-Driven Development**: Write tests first, test before committing
2. **Explicit Structure**: Separate resources (data) from code
3. **Docker-First**: Provide Dockerfiles/docker-compose for local development
4. **AWS-Ready**: Design for easy migration to Lambda/ECS deployment
5. **Version Control**: Commit changes regularly, use branches for new features

## Project Structure (Planned)

Since this is a new project, the following structure is recommended based on the project goals:
```
docker-rag-test/
├── src/                    # Application source code
│   ├── ingestion/         # Document processing pipeline
│   ├── embedding/         # Embedding generation
│   ├── storage/           # Vector database interfaces
│   └── api/              # Query API and chat interface
├── tests/                 # Test files (TDD approach)
├── data/                  # Sample documents and resources
├── docker/                # Docker configurations
├── requirements.txt       # Python dependencies
└── docker-compose.yml     # Local development setup
```

## Development Commands

As the project is in initial stages, these commands will need to be implemented:
- **Install dependencies**: `pip install -r requirements.txt` (once created)
- **Run tests**: `pytest` (after test framework setup)
- **Run linting**: `flake8` or `ruff` (to be configured)
- **Run type checking**: `mypy` (to be configured)
- **Build Docker image**: `docker build -t docker-rag-test .` (once Dockerfile exists)
- **Run locally**: `docker-compose up` (once docker-compose.yml exists)

## Technology Stack

Based on project goals and Python ecosystem:
- **Language**: Python
- **Document Processing**: Libraries for PDF, DOCX, HTML parsing
- **Embeddings**: OpenAI, Sentence Transformers, or similar
- **Vector Database**: Start with local (ChromaDB/Qdrant), migrate to AWS S3 Vector Engine
- **API Framework**: FastAPI or Flask
- **Testing**: pytest
- **Containerization**: Docker & Docker Compose

## AWS Deployment Considerations

The architecture should support:
- Lambda deployment for serverless operation
- ECS deployment for containerized services
- API Gateway integration
- S3 Vector Engine or managed vector database services

## Current Status

The project is newly initialized with:
- Git repository configured
- Python .gitignore in place
- Clear project vision in README.md
- No implementation code yet

Next steps should focus on setting up the project structure and basic testing framework before implementing features.