from typing import List, Dict, Any, Optional
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid
from dataclasses import dataclass


@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    id: str


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Set up embedding function
        if embedding_function is None:
            # Use default sentence transformer
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            self.embedding_function = embedding_function
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except ValueError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if not documents:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Add documents to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if not embeddings:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]
        
        # Add embeddings directly
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter
        )
        
        # Convert to SearchResult objects
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                search_results.append(SearchResult(
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=1 - results['distances'][0][i],  # Convert distance to similarity
                    id=results['ids'][0][i]
                ))
        
        return search_results
    
    def search_by_embedding(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        # Perform search with embedding
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=filter
        )
        
        # Convert to SearchResult objects
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                search_results.append(SearchResult(
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=1 - results['distances'][0][i],
                    id=results['ids'][0][i]
                ))
        
        return search_results
    
    def delete(self, ids: List[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)
    
    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        update_dict = {"ids": ids}
        
        if documents is not None:
            update_dict["documents"] = documents
        if metadatas is not None:
            update_dict["metadatas"] = metadatas
        if embeddings is not None:
            update_dict["embeddings"] = embeddings
        
        self.collection.update(**update_dict)
    
    def get(self, ids: List[str]) -> Dict[str, Any]:
        return self.collection.get(ids=ids)
    
    def count(self) -> int:
        return self.collection.count()
    
    def reset(self) -> None:
        # Delete and recreate the collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )