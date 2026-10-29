from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Symbol:
    id: str
    qualified_name: str
    name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    language: str = "java"
    is_test: bool = False
    calls: set[str] = field(default_factory=set)
    imports: set[str] = field(default_factory=set)
    signature: str | None = None
    source_code: str | None = None
    modifiers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_node_id: str | None = None

@dataclass
class CodeChunk:
    id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    symbol_id: str | None = None
    symbol_name: str | None = None
    chunk_index: int = 0
    chunk_type: str = "source"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    kind: str
    provenance: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class RepositoryFile:
    path: str
    category: str
    language: str
    size_bytes: int
    sha256: str
    line_count: int

@dataclass
class IngestionDocument:
    repository_id: int
    repository_full_name: str
    commit_sha: str
    index_generation: str
    symbols: list[Symbol]
    chunks: list[CodeChunk]
    files_seen: int
    files_skipped: int
    warnings: list[str] = field(default_factory=list)
