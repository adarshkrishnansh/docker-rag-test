from typing import List, Optional, Dict, Any
import os
from openai import OpenAI
from src.storage.vector_store import ChromaVectorStore, SearchResult
from src.embedding.embedder import EmbedderFactory
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_splitter import TextSplitter
from src.api.config import settings
import logging

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        # Initialize embedder
        self.embedder = EmbedderFactory.create_embedder(
            embedder_type=settings.embedder_type,
            model_name=settings.embedding_model if settings.embedder_type == "sentence-transformer" else None,
            api_key=settings.openai_api_key if settings.embedder_type == "openai" else None
        )
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore(
            collection_name=settings.chroma_collection_name,
            persist_directory=settings.chroma_persist_directory
        )
        
        # Initialize OpenAI client for chat
        self.openai_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        
        # Initialize document loader and splitter
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    def ingest_documents(self, directory_path: str) -> Dict[str, Any]:
        errors = []
        chunks_created = 0
        
        try:
            # Load documents
            documents = self.document_loader.load_documents(directory_path)
            logger.info(f"Loaded {len(documents)} documents")
            
            # Process each document
            for doc in documents:
                try:
                    # Split into chunks
                    chunks = self.text_splitter.split_text(doc.content, doc.metadata)
                    
                    # Prepare data for vector store
                    texts = [chunk.content for chunk in chunks]
                    metadatas = [
                        {
                            **chunk.metadata,
                            'source': doc.source,
                            'file_name': doc.metadata.get('file_name', ''),
                            'chunk_start': chunk.start_index,
                            'chunk_end': chunk.end_index
                        }
                        for chunk in chunks
                    ]
                    
                    # Generate embeddings if using custom embedder
                    if settings.embedder_type == "openai":
                        embeddings = self.embedder.embed_texts(texts)
                        self.vector_store.add_embeddings(
                            embeddings=embeddings,
                            documents=texts,
                            metadatas=metadatas
                        )
                    else:
                        # ChromaDB will use its own embedding function
                        self.vector_store.add_documents(
                            documents=texts,
                            metadatas=metadatas
                        )
                    
                    chunks_created += len(chunks)
                    
                except Exception as e:
                    error_msg = f"Error processing {doc.source}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            return {
                "documents_processed": len(documents),
                "chunks_created": chunks_created,
                "errors": errors
            }
            
        except Exception as e:
            error_msg = f"Error during ingestion: {str(e)}"
            logger.error(error_msg)
            return {
                "documents_processed": 0,
                "chunks_created": 0,
                "errors": [error_msg]
            }
    
    def search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        if settings.embedder_type == "openai":
            # Use custom embedder
            query_embedding = self.embedder.embed_query(query)
            return self.vector_store.search_by_embedding(query_embedding, k, filter)
        else:
            # Use ChromaDB's built-in embedding
            return self.vector_store.search(query, k, filter)
    
    def chat(self, message: str, k: int = 5, use_context: bool = True, temperature: float = 0.7) -> Dict[str, Any]:
        sources = []
        context = ""
        
        if use_context:
            # Search for relevant documents
            search_results = self.search(message, k)
            
            # Build context from search results
            context_parts = []
            for result in search_results:
                context_parts.append(f"Source: {result.metadata.get('source', 'Unknown')}\n{result.content}")
                sources.append(result.metadata.get('source', 'Unknown'))
            
            context = "\n\n---\n\n".join(context_parts)
        
        # Generate response
        if self.openai_client:
            system_prompt = "You are a helpful assistant that answers questions based on the provided context."
            
            if use_context and context:
                user_prompt = f"Context:\n{context}\n\nQuestion: {message}\n\nAnswer based on the context provided. If the answer cannot be found in the context, say so."
            else:
                user_prompt = message
            
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            
            return {
                "message": message,
                "response": response.choices[0].message.content,
                "sources": list(set(sources)),  # Remove duplicates
                "context_used": use_context and bool(context)
            }
        else:
            # Fallback response when OpenAI is not configured
            return {
                "message": message,
                "response": "OpenAI API key not configured. Please set OPENAI_API_KEY to enable chat functionality.",
                "sources": sources,
                "context_used": False
            }
    
    def get_document_count(self) -> int:
        return self.vector_store.count()