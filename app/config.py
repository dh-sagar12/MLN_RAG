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
    # Lambda Reranker
    # ----------------------------------------------------------------------
    lambda_reranker_function_name: Optional[str] = None
    lambda_reranker_region: str = "ap-south-1"
    lambda_reranker_timeout: int = 60
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None

    # ----------------------------------------------------------------------
    # Performance Tracking
    # ----------------------------------------------------------------------
    performance_tracking_enabled: bool = True
    performance_log_dir: str = "logs/performance"
    performance_log_filename: str = "performance.json"
    performance_max_file_size_mb: int = 50
    performance_flush_interval: int = 1

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
