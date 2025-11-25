"""Configuration settings for the RAG chatbot application."""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    database_url: str = os.getenv("DATABASE_URL")
    
    # OpenAI
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL")
    openai_llm_model: str = os.getenv("OPENAI_LLM_MODEL")
    temperature: float = float(os.getenv("TEMPERATURE", 0))
    
    # File Storage
    storage_path: str = os.getenv("STORAGE_PATH", "storage/uploads")
    
    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Vector Store
    vector_dimension: int = os.getenv("VECTOR_DIMENSION")
    top_k: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

