"""
Configuration management for the agentic AI system.

This module provides centralized configuration using pydantic-settings,
loading values from environment variables and .env files.

Design Principles:
- All configuration is type-safe and validated
- Environment variables take precedence over .env files
- Sensitive values (API keys) are never logged
- Configuration is immutable after initialization
- Sensible defaults for development

Usage:
    from src.common.config import settings

    # Access configuration
    api_key = settings.anthropic_api_key
    model = settings.default_model
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Settings are loaded in order of precedence:
    1. Environment variables
    2. .env file
    3. Default values

    Attributes:
        # LLM Provider Settings
        anthropic_api_key: Anthropic API key (required)
        openai_api_key: OpenAI API key (optional)
        default_model: Default LLM model to use
        fallback_model: Fallback model if primary fails

        # Application Settings
        environment: Deployment environment (dev/staging/prod)
        log_level: Logging level
        enable_cost_tracking: Enable LLM cost tracking
        max_retries: Maximum retries for failed operations
        timeout_seconds: Default timeout for operations

        # Database Settings
        database_url: Database connection URL (optional)
        database_pool_size: Connection pool size

        # Vector Store Settings
        vector_store_type: Type of vector store (chromadb/pinecone)
        chromadb_path: Path for ChromaDB persistence
        pinecone_api_key: Pinecone API key
        pinecone_environment: Pinecone environment

        # Observability Settings
        langsmith_api_key: LangSmith API key for tracing
        langsmith_project: LangSmith project name
        enable_langsmith: Enable LangSmith tracing
        phoenix_api_key: Phoenix API key
        enable_phoenix: Enable Phoenix tracing

        # Security Settings
        enable_prompt_sanitization: Enable prompt injection detection
        enable_sql_validation: Enable SQL injection prevention
        allowed_tables: Whitelist of allowed database tables
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables
    )

    # ========================================================================
    # LLM Provider Settings
    # ========================================================================

    anthropic_api_key: str = Field(
        ...,  # Required
        description="Anthropic API key"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (optional fallback)"
    )
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default LLM model identifier"
    )
    fallback_model: Optional[str] = Field(
        default="gpt-4o-mini",
        description="Fallback model if primary fails"
    )
    max_tokens_per_call: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum tokens per LLM call"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0=deterministic, 2=creative)"
    )

    # ========================================================================
    # Application Settings
    # ========================================================================

    environment: str = Field(
        default="development",
        description="Deployment environment"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG/INFO/WARNING/ERROR)"
    )
    enable_cost_tracking: bool = Field(
        default=True,
        description="Enable LLM cost tracking"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for failed operations"
    )
    timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Default operation timeout"
    )

    # ========================================================================
    # Database Settings
    # ========================================================================

    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL"
    )
    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Database connection pool size"
    )

    # ========================================================================
    # Vector Store Settings
    # ========================================================================

    vector_store_type: str = Field(
        default="chromadb",
        description="Vector store type (chromadb/pinecone)"
    )
    chromadb_path: str = Field(
        default="./data/chromadb",
        description="ChromaDB persistence path"
    )
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key"
    )
    pinecone_environment: Optional[str] = Field(
        default=None,
        description="Pinecone environment"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for vector store"
    )

    # ========================================================================
    # Observability Settings
    # ========================================================================

    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API key"
    )
    langsmith_project: str = Field(
        default="agentic-ai-dev",
        description="LangSmith project name"
    )
    enable_langsmith: bool = Field(
        default=False,
        description="Enable LangSmith tracing"
    )
    phoenix_api_key: Optional[str] = Field(
        default=None,
        description="Phoenix API key"
    )
    enable_phoenix: bool = Field(
        default=False,
        description="Enable Phoenix tracing"
    )

    # ========================================================================
    # Security Settings
    # ========================================================================

    enable_prompt_sanitization: bool = Field(
        default=True,
        description="Enable prompt injection detection"
    )
    enable_sql_validation: bool = Field(
        default=True,
        description="Enable SQL injection prevention"
    )
    allowed_tables: List[str] = Field(
        default_factory=list,
        description="Whitelist of allowed database tables"
    )
    max_query_results: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of query results"
    )

    # ========================================================================
    # Validators
    # ========================================================================

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ConfigurationError(
                f"Invalid log_level: {v}. Must be one of {valid_levels}"
            )
        return v_upper

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is valid."""
        valid_envs = ["development", "staging", "production"]
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ConfigurationError(
                f"Invalid environment: {v}. Must be one of {valid_envs}"
            )
        return v_lower

    @field_validator('vector_store_type')
    @classmethod
    def validate_vector_store(cls, v: str) -> str:
        """Ensure vector store type is valid."""
        valid_types = ["chromadb", "pinecone", "faiss"]
        v_lower = v.lower()
        if v_lower not in valid_types:
            raise ConfigurationError(
                f"Invalid vector_store_type: {v}. Must be one of {valid_types}"
            )
        return v_lower

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def get_llm_config(self) -> dict:
        """
        Get LLM configuration dictionary.

        Returns:
            Dictionary with LLM settings for provider initialization
        """
        return {
            "model": self.default_model,
            "max_tokens": self.max_tokens_per_call,
            "temperature": self.temperature,
        }

    def get_observability_config(self) -> dict:
        """
        Get observability configuration.

        Returns:
            Dictionary with observability settings
        """
        return {
            "langsmith": {
                "enabled": self.enable_langsmith,
                "api_key": self.langsmith_api_key,
                "project": self.langsmith_project,
            },
            "phoenix": {
                "enabled": self.enable_phoenix,
                "api_key": self.phoenix_api_key,
            },
            "cost_tracking": self.enable_cost_tracking,
        }

    def validate_required_keys(self) -> None:
        """
        Validate that all required API keys are present.

        Raises:
            ConfigurationError: If required keys are missing
        """
        if not self.anthropic_api_key or self.anthropic_api_key == "your-key-here":
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is required. "
                "Please set it in your .env file or environment variables."
            )

        # Validate observability keys if enabled
        if self.enable_langsmith and not self.langsmith_api_key:
            raise ConfigurationError(
                "LANGSMITH_API_KEY is required when enable_langsmith=True"
            )

        if self.vector_store_type == "pinecone":
            if not self.pinecone_api_key or not self.pinecone_environment:
                raise ConfigurationError(
                    "PINECONE_API_KEY and PINECONE_ENVIRONMENT are required "
                    "when vector_store_type=pinecone"
                )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Call this function to access application settings.

    Returns:
        Settings instance

    Example:
        >>> from src.common.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.default_model)
    """
    settings = Settings()
    settings.validate_required_keys()
    return settings


# Singleton instance for convenient access
settings = get_settings()
