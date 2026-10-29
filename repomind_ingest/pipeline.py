from __future__ import annotations

import hashlib
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from .analyzer import CodeAnalyzerJava
from .config import settings
from .models import CodeChunk, RepositoryFile, Symbol
from .parser import chunk_file, fallback_line_chunks, stable_id
from .repository import checkout_repository, discover_files, category_for_path, language_for_path
from .stores import PostgresStore, Neo4jStore, QdrantStore

log = logging.getLogger("repomind.ingestion")


class IngestionPipeline:
    """Java-first, immutable-generation ingestion coordinated across all retrieval stores."""
    def __init__(self):
        self.pg = PostgresStore()
        self.neo = Neo4jStore()
        self.qdrant = QdrantStore()
        self.analyzer = CodeAnalyzerJava(self.neo.driver)

    def close(self):
        self.neo.close()

    def run(self, source: str, *, repository_full_name: str | None = None,
            ref: str | None = None, cleanup_old: bool = False) -> dict[str, Any]:
        # Resolve and authorize the repository before cloning or touching any index store.
        repo_name = (repository_full_name or self._infer_repo_name(source) or "").strip().strip("/")
        if repo_name.lower().endswith(".git"):
            repo_name = repo_name[:-4]
        repo_parts = [part for part in repo_name.split("/") if part]
        if len(repo_parts) != 2:
            raise ValueError(
                "Provide a canonical owner/repository name (use --repository owner/repository for local paths)."
            )
        repo_name = "/".join(repo_parts)
        allowed = settings.allowed_repositories
        if allowed and repo_name.lower() not in allowed:
            allowed_display = ", ".join(allowed)
            raise ValueError(
                f"Repository '{repo_name}' is not in REPOMIND_ALLOWED_REPOSITORIES. "
                f"Add it to the comma-separated allowlist in .env first. Allowed: {allowed_display}"
            )

        checkout, temp_dir = checkout_repository(source, ref)
        repo_id: int | None = None
        generation: str | None = None
        published = False
        try:
            commit = self._git(["rev-parse", "HEAD"], checkout)

            files, skipped, warnings = discover_files(checkout, max_files=settings.max_files,
                                                       max_file_bytes=settings.max_file_bytes)
            if not files:
                raise RuntimeError("No eligible Java source or repository artifact files were found.")
            java_files = [p for p in files if p.suffix.lower() == ".java"]
            if not java_files:
                raise RuntimeError("This RepoMind ingestion version requires at least one .java source file.")

            generation = f"{commit[:12]}-{uuid.uuid4().hex[:12]}"
            self.pg.ensure_schema()
            self.neo.ensure_schema()
            self.qdrant.ensure_collection()
            repo_id = self.pg.start_generation(repo_name, commit, generation)

            # Stage 1: codeanalyzer-java performs Java-aware structural analysis and emits raw facts to Neo4j.
            java_rel_paths = [p.relative_to(Path(checkout)).as_posix() for p in java_files]
            symbols, edges, analyzer_stats = self.analyzer.analyze(
                checkout, repo_name, commit, generation, java_rel_paths)
            symbols_by_file: dict[str, list[Symbol]] = {}
            for symbol in symbols:
                if symbol.file_path:
                    symbols_by_file.setdefault(symbol.file_path, []).append(symbol)

            # Stage 2: content manifest + retrieval chunks. Only Java source and approved text artifacts
            # are indexed; non-Java source languages are deliberately excluded by repository discovery.
            file_rows: list[RepositoryFile] = []
            chunks: list[CodeChunk] = []
            for path in files:
                rel = path.relative_to(Path(checkout)).as_posix()
                try:
                    raw = path.read_bytes()
                    text = raw.decode("utf-8", errors="replace")
                except OSError as exc:
                    warnings.append(f"Could not read {rel}: {type(exc).__name__}")
                    continue
                lines = text.splitlines()
                category = category_for_path(rel)
                file_rows.append(RepositoryFile(
                    path=rel, category=category, language=language_for_path(rel),
                    size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), line_count=len(lines)))

                if path.suffix.lower() == ".java":
                    # Prefer method/constructor-sized chunks. Class-level chunks are used only when
                    # the analyzer found no callable in that file; this avoids embedding the same class
                    # body repeatedly while still covering imports, fields, and module-level context.
                    file_symbols = symbols_by_file.get(rel, [])
                    callables = [s for s in file_symbols if s.kind in {"method", "constructor"}]
                    chunk_symbols = callables or [s for s in file_symbols if s.kind in {
                        "class", "interface", "enum", "record", "type"}]
                    file_chunks = chunk_file(rel, text, chunk_symbols,
                        max_lines=70, overlap=settings.chunk_overlap_lines, max_tokens=settings.chunk_max_tokens)
                    for chunk in file_chunks:
                        chunk.chunk_type = "java_symbol" if chunk.symbol_id else "java_context"
                        chunk.metadata = {"file_category": "java_source", "commit_sha": commit}
                else:
                    file_chunks = fallback_line_chunks(rel, text, max_lines=70,
                        overlap=settings.chunk_overlap_lines, max_tokens=settings.chunk_max_tokens)
                    for chunk in file_chunks:
                        chunk.chunk_type = "repository_artifact"
                        chunk.metadata = {"file_category": category, "commit_sha": commit}
                for chunk in file_chunks:
                    chunk.id = stable_id(generation, chunk.id)
                    chunk.metadata.setdefault("source_path", rel)
                chunks.extend(file_chunks)

            if not file_rows:
                raise RuntimeError("No repository files could be read; refusing to publish an empty index.")
            if not chunks:
                raise RuntimeError("No retrieval chunks were produced; refusing to publish an empty index.")

            # Store each generation separately. The PostgreSQL repository pointer remains unchanged until
            # all three stores pass validation, so readers keep using the previous complete generation.
            self.pg.write_files(repo_id, commit, generation, file_rows)
            self.pg.write_rows(repo_id, commit, generation, symbols, chunks)
            graph_stats = self.neo.write_generation(repo_id, commit, generation, symbols, edges)
            vector_count = self.qdrant.upsert_chunks(repo_id, repo_name, commit, generation, chunks)

            pg_counts = self.pg.counts(repo_id, generation)
            neo_counts = self.neo.validate_generation(repo_id, generation)
            qdrant_count = self.qdrant.count_generation(repo_id, commit, generation)
            expected_edges = len(edges) + graph_stats.get("test_edges", 0)
            problems = []
            if pg_counts["symbols"] != len(symbols):
                problems.append(f"PostgreSQL symbols {pg_counts['symbols']} != {len(symbols)}")
            if pg_counts["chunks"] != len(chunks):
                problems.append(f"PostgreSQL chunks {pg_counts['chunks']} != {len(chunks)}")
            if pg_counts["files"] != len(file_rows):
                problems.append(f"PostgreSQL files {pg_counts['files']} != {len(file_rows)}")
            if neo_counts["symbols"] != len(symbols):
                problems.append(f"Neo4j symbols {neo_counts['symbols']} != {len(symbols)}")
            if neo_counts["edges"] != expected_edges:
                problems.append(f"Neo4j edges {neo_counts['edges']} != expected {expected_edges}")
            if qdrant_count != len(chunks):
                problems.append(f"Qdrant points {qdrant_count} != {len(chunks)}")
            if vector_count != len(chunks):
                problems.append(f"Qdrant upsert count {vector_count} != {len(chunks)}")
            if problems:
                raise RuntimeError("Cross-store validation failed: " + "; ".join(problems))

            details = {
                "files_seen": len(file_rows), "files_skipped": skipped,
                "java_files": len(java_files), "parse_errors": 0,
                "symbols": len(symbols), "chunks": len(chunks), "qdrant_points": vector_count,
                "postgres": pg_counts, "neo4j": neo_counts, "graph": graph_stats,
                "analyzer": analyzer_stats, "warnings": warnings[:100],
                "embedding_model": settings.embedding_model,
                "embedding_dimension": self.qdrant.dimension,
                "chunk_contract": "File/Language/Symbol prefix + source text; see QdrantStore.embedding_text",
            }
            self.pg.publish_generation(repo_id, commit, generation, details)
            published = True

            if cleanup_old:
                # Only use when no in-flight retrieval request depends on previous generations.
                self.pg.cleanup_old_generations(repo_id, generation)
                self.neo.cleanup_old_generations(repo_id, generation)
                self.qdrant.cleanup_old_generations(repo_id, generation)
            return {"repository": repo_name, "repository_id": repo_id, "commit_sha": commit,
                    "index_generation": generation, **details}
        except Exception as exc:
            if repo_id is not None and generation is not None and not published:
                # Best-effort rollback of only the failed staging generation; never delete the active one.
                for store_name, cleanup in (
                    ("qdrant", lambda: self.qdrant.cleanup_generation(repo_id, generation)),
                    ("neo4j", lambda: self.neo.cleanup_generation(repo_id, generation)),
                    ("postgres", lambda: self.pg.cleanup_generation(repo_id, generation)),
                ):
                    try:
                        cleanup()
                    except Exception:
                        log.exception("Failed to clean staging generation from %s", store_name)
                try:
                    self.pg.mark_failed(generation, type(exc).__name__, str(exc))
                except Exception:
                    log.exception("Failed to mark ingestion generation as failed")
            raise
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _git(args: list[str], cwd: str) -> str:
        import subprocess
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                              text=True, timeout=30).stdout.strip()

    @staticmethod
    def _infer_repo_name(source: str) -> str | None:
        from urllib.parse import urlparse
        if Path(source).expanduser().exists():
            return None
        if source.startswith("git@"):
            path = source.split(":", 1)[1]
        else:
            path = urlparse(source).path.lstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        parts = [p for p in path.split("/") if p]
        return "/".join(parts[-2:]) if len(parts) >= 2 else None
