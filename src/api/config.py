from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"
    log_level: str = "INFO"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # ChromaDB Configuration
    chroma_persist_directory: str = "/app/chroma_db"
    chroma_collection_name: str = "documents"
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding Configuration
    embedder_type: str = "sentence-transformer"  # "sentence-transformer" or "openai"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Query Configuration
    search_k: int = 5
    
    # Auto-ingestion Configuration
    auto_ingest_on_startup: bool = True
    auto_ingest_directory: str = "/app/data/documents"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()