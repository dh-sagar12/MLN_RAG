"""Configuration settings for the RAG chatbot application."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ----------------------------------------------------------------------
    # Database
    # ----------------------------------------------------------------------
    db_name: str
    db_user: str
    db_host: str
    db_password: str
    db_port: int

    # ----------------------------------------------------------------------
    # OpenAI
    # ----------------------------------------------------------------------
    openai_api_key: Optional[str]
    openai_embedding_model: str
    openai_llm_model: str
    temperature: float = 0.0

    # ----------------------------------------------------------------------
    # File Storage
    # ----------------------------------------------------------------------
    storage_path: str = "storage/uploads"

    # ----------------------------------------------------------------------
    # Server
    # ----------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # ----------------------------------------------------------------------
    # Vector Store
    # ----------------------------------------------------------------------
    vector_dimension: int = 1536
    top_k: int = 5

    # ----------------------------------------------------------------------
    # Config
    # ----------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----------------------------------------------------------------------
    # Computed property
    # ----------------------------------------------------------------------
    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


settings = Settings()
