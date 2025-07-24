# docker-rag-test

A production-ready Retrieval-Augmented Generation (RAG) systems that:
- Ingest heterogeneous documents (PDF, HTML, TXT, MD, DOCX)
- Chunk and embed them
- Store embeddings in a vector database (e.g. Amazon S3 Vector Engine, pgvector, Pinecone, Qdrant)
- Expose a query API and/or chat UI backed by an LLM

# Goal:
- Build a replicable project that takes a bunch of documents, embeds them, and stores them in a vector database (initially local; easily swapped to Amazon S3 Vector Engine or another managed vector store).
- Connect it to an LLM so users can query the knowledge base in natural language.
- Package everything into a Docker setup for local use, but keep it simple to move to AWS Lambda or run as an API.

# Methodology:
1. Use most supported and compatible tech stack.
2. Test driven development. Write tests first and test before commiting.
3. Use an explicit folder structure separating resources (data) from code.
4. Provide Dockerfiles/docker-compose examples to run locally.
5. Provide Guidelines for migrating the vector store and the app to AWS (Lambda or ECS) and making it available via API Gateway.
6. Use best practices.
7. Commit any changes. Use branch while creating new unique features.

