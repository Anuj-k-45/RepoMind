from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))

def _csv_set(name: str) -> tuple[str, ...]:
    """Read a comma- or semicolon-separated list of canonical repository names."""
    raw = os.getenv(name, "")
    values = (part.strip().strip("/").lower() for part in raw.replace(";", ",").split(","))
    return tuple(dict.fromkeys(value for value in values if value))

@dataclass(frozen=True)
class Settings:
    postgres_dsn: str = os.getenv("POSTGRES_DSN", "postgresql://repomind:repomind_dev_password@localhost:5432/repomind")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "repomind_dev_password")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "repomind_chunks")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_batch_size: int = _int("EMBEDDING_BATCH_SIZE", 32)
    chunk_max_tokens: int = _int("CHUNK_MAX_TOKENS", 450)
    chunk_overlap_lines: int = _int("CHUNK_OVERLAP_LINES", 8)
    max_file_bytes: int = _int("MAX_FILE_BYTES", 1_500_000)
    max_files: int = _int("MAX_FILES", 50_000)
    max_embedding_batch_size: int = _int("MAX_EMBEDDING_BATCH_SIZE", 64)
    qdrant_distance: str = os.getenv("QDRANT_DISTANCE", "COSINE")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    allowed_repositories: tuple[str, ...] = _csv_set("REPOMIND_ALLOWED_REPOSITORIES")
    canjv_command: str = os.getenv("CANJV_COMMAND", "canjv")
    analyzer_timeout_seconds: int = _int("ANALYZER_TIMEOUT_SECONDS", 900)
    analyzer_graph_depth: int = _int("ANALYZER_GRAPH_DEPTH", 6)
    analyzer_max_graph_nodes: int = _int("ANALYZER_MAX_GRAPH_NODES", 200000)

settings = Settings()
