import pytest
import os
from unittest.mock import patch
from src.api.config import Settings


def test_default_config_values():
    """Test default configuration values"""
    settings = Settings()
    
    # API Configuration
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    
    # OpenAI Configuration
    assert settings.openai_api_key is None
    assert settings.openai_model == "gpt-3.5-turbo"
    
    # ChromaDB Configuration
    assert settings.chroma_persist_directory == "/app/chroma_db"
    assert settings.chroma_collection_name == "documents"
    
    # Document Processing
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    
    # Embedding Configuration
    assert settings.embedder_type == "sentence-transformer"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    
    # Query Configuration
    assert settings.search_k == 5
    
    # Auto-ingestion Configuration
    assert settings.auto_ingest_on_startup is True
    assert settings.auto_ingest_directory == "/app/data/documents"


def test_config_from_environment_variables():
    """Test configuration loading from environment variables"""
    env_vars = {
        "API_HOST": "127.0.0.1",
        "API_PORT": "9000",
        "ENVIRONMENT": "production",
        "LOG_LEVEL": "DEBUG",
        "OPENAI_API_KEY": "test-key-123",
        "OPENAI_MODEL": "gpt-4",
        "CHROMA_PERSIST_DIRECTORY": "/custom/chroma",
        "CHROMA_COLLECTION_NAME": "custom_docs",
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "100",
        "EMBEDDER_TYPE": "openai",
        "EMBEDDING_MODEL": "text-embedding-ada-002",
        "SEARCH_K": "10",
        "AUTO_INGEST_ON_STARTUP": "false",
        "AUTO_INGEST_DIRECTORY": "/custom/docs"
    }
    
    with patch.dict(os.environ, env_vars):
        settings = Settings()
        
        # Verify all values are loaded from environment
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 9000
        assert settings.environment == "production"
        assert settings.log_level == "DEBUG"
        assert settings.openai_api_key == "test-key-123"
        assert settings.openai_model == "gpt-4"
        assert settings.chroma_persist_directory == "/custom/chroma"
        assert settings.chroma_collection_name == "custom_docs"
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 100
        assert settings.embedder_type == "openai"
        assert settings.embedding_model == "text-embedding-ada-002"
        assert settings.search_k == 10
        assert settings.auto_ingest_on_startup is False
        assert settings.auto_ingest_directory == "/custom/docs"


def test_config_case_insensitive():
    """Test that configuration is case insensitive"""
    env_vars = {
        "api_host": "lowercase.host",
        "API_PORT": "8080",
        "ChUnK_SiZe": "2000"
    }
    
    with patch.dict(os.environ, env_vars):
        settings = Settings()
        
        assert settings.api_host == "lowercase.host"
        assert settings.api_port == 8080
        assert settings.chunk_size == 2000


def test_config_type_conversion():
    """Test that configuration values are properly type-converted"""
    env_vars = {
        "API_PORT": "8080",  # String should convert to int
        "CHUNK_SIZE": "1500",  # String should convert to int
        "AUTO_INGEST_ON_STARTUP": "true",  # String should convert to bool
    }
    
    with patch.dict(os.environ, env_vars):
        settings = Settings()
        
        assert isinstance(settings.api_port, int)
        assert settings.api_port == 8080
        assert isinstance(settings.chunk_size, int)
        assert settings.chunk_size == 1500
        assert isinstance(settings.auto_ingest_on_startup, bool)
        assert settings.auto_ingest_on_startup is True


def test_config_boolean_values():
    """Test various boolean value formats"""
    # Test True values
    for true_val in ["true", "True", "TRUE", "1", "yes", "on"]:
        with patch.dict(os.environ, {"AUTO_INGEST_ON_STARTUP": true_val}):
            settings = Settings()
            assert settings.auto_ingest_on_startup is True
    
    # Test False values
    for false_val in ["false", "False", "FALSE", "0", "no", "off"]:
        with patch.dict(os.environ, {"AUTO_INGEST_ON_STARTUP": false_val}):
            settings = Settings()
            assert settings.auto_ingest_on_startup is False


def test_config_with_dotenv_file(tmp_path):
    """Test configuration loading from .env file"""
    # Create a temporary .env file
    test_env_file = tmp_path / ".env"
    env_content = """
# Test .env file
API_HOST=dotenv.host
API_PORT=7000
OPENAI_API_KEY=dotenv-key
CHUNK_SIZE=800
AUTO_INGEST_ON_STARTUP=false
"""
    test_env_file.write_text(env_content)
    
    # Test with custom env_file path
    from pydantic import ConfigDict
    
    class TestSettings(Settings):
        model_config = ConfigDict(env_file=str(test_env_file), case_sensitive=False)
    
    settings = TestSettings()
    
    assert settings.api_host == "dotenv.host"
    assert settings.api_port == 7000
    assert settings.openai_api_key == "dotenv-key"
    assert settings.chunk_size == 800
    assert settings.auto_ingest_on_startup is False


def test_config_precedence():
    """Test that environment variables take precedence over .env file"""
    # This test assumes that environment variables should override .env values
    env_vars = {
        "API_HOST": "env.override.host",
        "API_PORT": "9999"
    }
    
    with patch.dict(os.environ, env_vars):
        settings = Settings()
        
        # Environment variables should take precedence
        assert settings.api_host == "env.override.host"
        assert settings.api_port == 9999


def test_config_validation():
    """Test configuration validation"""
    # Test valid embedder types
    for embedder_type in ["sentence-transformer", "openai"]:
        with patch.dict(os.environ, {"EMBEDDER_TYPE": embedder_type}):
            settings = Settings()
            assert settings.embedder_type == embedder_type
    
    # Test valid chunk sizes
    with patch.dict(os.environ, {"CHUNK_SIZE": "1", "CHUNK_OVERLAP": "0"}):
        settings = Settings()
        assert settings.chunk_size == 1
        assert settings.chunk_overlap == 0


def test_config_optional_fields():
    """Test that optional fields work correctly"""
    # Test with no OpenAI key
    settings = Settings()
    assert settings.openai_api_key is None
    
    # Test with OpenAI key
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
        settings = Settings()
        assert settings.openai_api_key == "sk-test123"


def test_settings_singleton_behavior():
    """Test that settings behave consistently"""
    from src.api.config import settings
    
    # The settings object should be the same instance
    assert isinstance(settings, Settings)
    
    # Test that we can access all expected attributes
    required_attrs = [
        'api_host', 'api_port', 'environment', 'log_level',
        'openai_api_key', 'openai_model',
        'chroma_persist_directory', 'chroma_collection_name',
        'chunk_size', 'chunk_overlap',
        'embedder_type', 'embedding_model',
        'search_k', 'auto_ingest_on_startup', 'auto_ingest_directory'
    ]
    
    for attr in required_attrs:
        assert hasattr(settings, attr), f"Settings missing attribute: {attr}"