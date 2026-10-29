from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Secrets are intentionally placeholders in .env.example.
    groq_api_key: str = os.getenv("GROQ_API_KEY", "DUMMY_GROQ_API_KEY")
    chat_model: str = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")
    postgres_dsn: str = os.getenv("POSTGRES_DSN", "postgresql://DUMMY:DUMMY@localhost:5432/DUMMY")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "DUMMY")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "DUMMY")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "DUMMY_QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "repomind_chunks")
    github_token: str = os.getenv("GITHUB_TOKEN", "DUMMY_GITHUB_TOKEN")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    max_results: int = int(os.getenv("MAX_RESULTS", "8"))
    max_graph_hops: int = int(os.getenv("MAX_GRAPH_HOPS", "3"))
    max_retrieval_rounds: int = int(os.getenv("MAX_RETRIEVAL_ROUNDS", "3"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "12"))
    allow_public_repositories: bool = _bool("ALLOW_PUBLIC_REPOSITORIES", True)
    # Production deployments should populate an explicit allowlist in an auth layer.
    environment: str = os.getenv("APP_ENV", "development")
    authorized_repositories: tuple[str, ...] = tuple(
        item.strip() for item in os.getenv("AUTHORIZED_REPOSITORIES", "").split(",") if item.strip()
    )
    api_key: str = os.getenv("REPOMIND_API_KEY", "")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
