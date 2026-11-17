"""
Centralized configuration management for DubaiProp HITL.
Loads and validates environment variables from the .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory (dubai-realestate-hitl-agent/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Metadata
    APP_NAME: str = "DubaiProp HITL"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"

    # LLM Configuration (Groq)
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.1

    # Local Vector Database & Embedding Configuration
    QDRANT_PATH: str = str(BASE_DIR / "qdrant_storage")
    QDRANT_COLLECTION: str = "dubai_real_estate_kb"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # Knowledge Base Source Data
    KB_DATA_PATH: str = str(BASE_DIR / "data" / "dubai_real_estate_kb.json")

    # Persistent Graph Checkpointer (SQLite)
    SQLITE_DB_PATH: str = str(BASE_DIR / "dubaiprop_state.db")

    # Agentic Guardrails & Loop Limits
    MAX_RAG_RETRIES: int = 2
    HIGH_VALUE_THRESHOLD_AED: int = 1_000_000  # Inquiries >= 1M AED trigger HITL approval

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton settings instance
settings = Settings()