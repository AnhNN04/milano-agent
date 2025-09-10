import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file at project startup
load_dotenv()


class AppSettings(BaseSettings):
    name: str = "Stock Assistant"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    environment: str = "development"

    openai_api_key: str
    tavily_api_key: str
    gemini_api_key: str

    model_config = SettingsConfigDict(env_prefix="APP_", case_sensitive=False)


class LLMSettings(BaseSettings):
    # General
    default_provider: str = "gemini"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 60
    max_retries: int = 3

    # OpenAI
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 2000
    openai_top_p: float = 1.0
    openai_frequency_penalty: float = 0.0
    openai_presence_penalty: float = 0.0
    openai_timeout: int = 60
    openai_max_retries: int = 3

    # Gemini
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.1
    gemini_max_tokens: int = 2000
    gemini_top_p: float = 1.0
    gemini_top_k: int = 40
    gemini_timeout: int = 60
    gemini_max_retries: int = 1

    # LangChain
    verbose: bool = False
    streaming: bool = False
    enable_callbacks: bool = True

    # Safety
    enable_content_filter: bool = True

    model_config = SettingsConfigDict(env_prefix="LLM_", case_sensitive=False)


class QdrantSettings(BaseSettings):
    host: str = "qdrant-service.ai-agent.local"
    port: int = 6333
    url: str = ""
    api_key: str = ""
    collection_name: str = "milano-agent-qdrant"
    vector_size: int = 1024
    distance: str = "Cosine"
    max_results: int = 5

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_", case_sensitive=False
    )


class S3Settings(BaseSettings):
    bucket_name: str
    documents_prefix: str = "rag-docs/"
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str

    model_config = SettingsConfigDict(env_prefix="S3_", case_sensitive=False)


class TavilySettings(BaseSettings):
    max_results: int = 5
    search_depth: str = "advanced"
    include_domains: list[str] = [
        "cafef.vn",
        "vneconomy.vn",
        "vietstock.vn",
        "investing.com",
    ]
    timeout_seconds: int = 5

    model_config = SettingsConfigDict(
        env_prefix="TAVILY_", case_sensitive=False
    )


class VnStock(BaseSettings):
    timeout_seconds: int = 5
    interval: str = (
        "1D"  # Khoảng thời gian lấy dữ liệu (vd: '1d', '1wk', '1mo')
    )

    model_config = SettingsConfigDict(
        env_prefix="VNSTOCK_", case_sensitive=False
    )


class EmbeddingsSettings(BaseSettings):
    model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    cohere_model_id: str = "cohere.embed-multilingual-v3"
    cohere_input_type: str = "search_document"
    cohere_embedding_type: str = "float"
    cohere_max_batch_size: int = 96

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDINGS_", case_sensitive=False
    )


class RedisSettings(BaseSettings):
    """Redis configuration settings"""

    host: str = "redis-service.ai-agent.local"
    port: int = 6379
    username: str = "default"  # Redis 6+ requires username
    password: Optional[str] = None
    db: int = 0
    pool_size: int = 10
    connection_timeout: int = 30
    socket_timeout: int = 30
    retry_on_timeout: bool = True
    health_check_interval: int = 30

    # Session specific settings
    default_ttl: int = 3600  # 1 hour
    cache_ttl: int = 300  # 5 minutes for analysis cache

    model_config = SettingsConfigDict(
        env_prefix="REDIS_", case_sensitive=False
    )

    @property
    def connection_url(self) -> str:
        """Generate Redis connection URL"""
        if self.url:
            return self.url

        auth_part = f":{self.password}@" if self.password else ""
        return f"redis://{auth_part}{self.host}:{self.port}/{self.db}"


class Settings:
    def __init__(self):
        self.app = AppSettings()
        self.llm = LLMSettings()
        self.qdrant = QdrantSettings()
        self.s3 = S3Settings()
        self.embeddings = EmbeddingsSettings()
        self.redis = RedisSettings()
        self.tavily = TavilySettings()
        self.vnstock = VnStock()


# Singleton settings instance
settings = Settings()
