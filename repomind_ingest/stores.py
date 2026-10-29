from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from .config import settings
from .models import CodeChunk, GraphEdge, RepositoryFile, Symbol

log = logging.getLogger(__name__)


class PostgresStore:
    """Metadata, lexical retrieval, file manifest, and the active-generation pointer."""
    def __init__(self):
        self.dsn = settings.postgres_dsn

    def ensure_schema(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS repositories (
          id BIGSERIAL PRIMARY KEY, full_name TEXT NOT NULL UNIQUE,
          indexed_commit_sha TEXT, index_generation TEXT, indexed_at TIMESTAMPTZ
        );
        ALTER TABLE repositories ADD COLUMN IF NOT EXISTS indexed_commit_sha TEXT;
        ALTER TABLE repositories ADD COLUMN IF NOT EXISTS index_generation TEXT;
        ALTER TABLE repositories ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMPTZ;
        ALTER TABLE repositories ALTER COLUMN indexed_commit_sha DROP NOT NULL;
        ALTER TABLE repositories ALTER COLUMN index_generation DROP NOT NULL;
        ALTER TABLE repositories ALTER COLUMN indexed_at DROP NOT NULL;

        CREATE TABLE IF NOT EXISTS ingestion_runs (
          index_generation TEXT PRIMARY KEY,
          repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
          commit_sha TEXT NOT NULL,
          status TEXT NOT NULL CHECK (status IN ('building','published','failed')),
          started_at TIMESTAMPTZ NOT NULL DEFAULT now(), completed_at TIMESTAMPTZ,
          details JSONB NOT NULL DEFAULT '{}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS symbols (
          id TEXT PRIMARY KEY, repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
          qualified_name TEXT NOT NULL, file_path TEXT NOT NULL, kind TEXT NOT NULL,
          start_line INT, end_line INT, commit_sha TEXT NOT NULL, index_generation TEXT NOT NULL
        );
        ALTER TABLE symbols ADD COLUMN IF NOT EXISTS name TEXT;
        ALTER TABLE symbols ADD COLUMN IF NOT EXISTS signature TEXT;
        ALTER TABLE symbols ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'java';
        ALTER TABLE symbols ADD COLUMN IF NOT EXISTS is_test BOOLEAN NOT NULL DEFAULT FALSE;
        ALTER TABLE symbols ADD COLUMN IF NOT EXISTS source_code TEXT;
        ALTER TABLE symbols ADD COLUMN IF NOT EXISTS modifiers JSONB NOT NULL DEFAULT '[]'::jsonb;
        ALTER TABLE symbols ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
        ALTER TABLE symbols ADD COLUMN IF NOT EXISTS index_generation TEXT;
        CREATE INDEX IF NOT EXISTS idx_symbols_repo_generation ON symbols(repository_id,index_generation);
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(repository_id,lower(qualified_name));
        CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(repository_id,file_path,start_line);

        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY, repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
          file_path TEXT NOT NULL, start_line INT NOT NULL, end_line INT NOT NULL,
          content TEXT NOT NULL, commit_sha TEXT NOT NULL, index_generation TEXT NOT NULL,
          search_vector TSVECTOR
        );
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS index_generation TEXT;
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector TSVECTOR;
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS symbol_id TEXT;
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS symbol_name TEXT;
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'java';
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_type TEXT NOT NULL DEFAULT 'source';
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_index INT NOT NULL DEFAULT 0;
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
        CREATE OR REPLACE FUNCTION repomind_chunks_search_vector_update() RETURNS trigger AS $$
        BEGIN
          NEW.search_vector := to_tsvector('simple', coalesce(NEW.file_path, '') || ' ' ||
              coalesce(NEW.symbol_name, '') || ' ' || coalesce(NEW.content, ''));
          RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS repomind_chunks_search_vector_trigger ON chunks;
        CREATE TRIGGER repomind_chunks_search_vector_trigger
          BEFORE INSERT OR UPDATE OF content, file_path, symbol_name ON chunks
          FOR EACH ROW EXECUTE FUNCTION repomind_chunks_search_vector_update();
        UPDATE chunks SET content=content WHERE search_vector IS NULL;
        CREATE INDEX IF NOT EXISTS idx_chunks_repo_generation ON chunks(repository_id,commit_sha,index_generation);
        CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON chunks USING GIN(search_vector);
        CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(repository_id,file_path,start_line);

        CREATE TABLE IF NOT EXISTS repository_files (
          repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
          index_generation TEXT NOT NULL, commit_sha TEXT NOT NULL, path TEXT NOT NULL,
          category TEXT NOT NULL, language TEXT NOT NULL, size_bytes BIGINT NOT NULL,
          sha256 TEXT NOT NULL, line_count INT NOT NULL,
          PRIMARY KEY(repository_id,index_generation,path)
        );
        CREATE INDEX IF NOT EXISTS idx_repository_files_generation ON repository_files(repository_id,index_generation);
        """
        with psycopg.connect(self.dsn) as conn:
            conn.execute(sql)
            conn.commit()

    def start_generation(self, full_name: str, commit: str, generation: str) -> int:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                """INSERT INTO repositories(full_name) VALUES(%s)
                   ON CONFLICT(full_name) DO UPDATE SET full_name=EXCLUDED.full_name
                   RETURNING id""", (full_name,)
            ).fetchone()
            repo_id = int(row["id"])
            conn.execute(
                """INSERT INTO ingestion_runs(index_generation,repository_id,commit_sha,status)
                   VALUES(%s,%s,%s,'building')""", (generation, repo_id, commit)
            )
            conn.commit()
            return repo_id

    def write_files(
        self,
        repository_id: int,
        commit: str,
        generation: str,
        files: list[RepositoryFile],
    ) -> None:
        if not files:
            return

        rows = [
            (
                repository_id,
                generation,
                commit,
                f.path,
                f.category,
                f.language,
                f.size_bytes,
                f.sha256,
                f.line_count,
            )
            for f in files
        ]

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO repository_files (
                        repository_id,
                        index_generation,
                        commit_sha,
                        path,
                        category,
                        language,
                        size_bytes,
                        sha256,
                        line_count
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (
                        repository_id,
                        index_generation,
                        path
                    )
                    DO UPDATE SET
                        commit_sha = EXCLUDED.commit_sha,
                        category = EXCLUDED.category,
                        language = EXCLUDED.language,
                        size_bytes = EXCLUDED.size_bytes,
                        sha256 = EXCLUDED.sha256,
                        line_count = EXCLUDED.line_count
                    """,
                    rows,
                )

            conn.commit()


    def write_rows(self, repository_id: int, commit: str, generation: str,
                symbols: list[Symbol], chunks: list[CodeChunk]) -> None:
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                if symbols:
                    cur.executemany("""
                        INSERT INTO symbols(id,repository_id,qualified_name,file_path,kind,start_line,end_line,
                        commit_sha,index_generation,name,signature,language,is_test,source_code,modifiers,metadata)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                        ON CONFLICT(id) DO UPDATE SET repository_id=EXCLUDED.repository_id,
                        qualified_name=EXCLUDED.qualified_name,file_path=EXCLUDED.file_path,kind=EXCLUDED.kind,
                        start_line=EXCLUDED.start_line,end_line=EXCLUDED.end_line,commit_sha=EXCLUDED.commit_sha,
                        index_generation=EXCLUDED.index_generation,name=EXCLUDED.name,signature=EXCLUDED.signature,
                        language=EXCLUDED.language,is_test=EXCLUDED.is_test,source_code=EXCLUDED.source_code,
                        modifiers=EXCLUDED.modifiers,metadata=EXCLUDED.metadata
                    """, [(s.id, repository_id, s.qualified_name, s.file_path, s.kind, s.start_line, s.end_line,
                        commit, generation, s.name, s.signature, s.language, s.is_test, s.source_code,
                        json.dumps(s.modifiers), json.dumps(s.metadata, default=str)) for s in symbols])
                if chunks:
                    cur.executemany("""
                        INSERT INTO chunks(id,repository_id,file_path,start_line,end_line,content,commit_sha,index_generation,
                        symbol_id,symbol_name,language,chunk_type,chunk_index,metadata)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        ON CONFLICT(id) DO UPDATE SET repository_id=EXCLUDED.repository_id,file_path=EXCLUDED.file_path,
                        start_line=EXCLUDED.start_line,end_line=EXCLUDED.end_line,content=EXCLUDED.content,
                        commit_sha=EXCLUDED.commit_sha,index_generation=EXCLUDED.index_generation,
                        symbol_id=EXCLUDED.symbol_id,symbol_name=EXCLUDED.symbol_name,language=EXCLUDED.language,
                        chunk_type=EXCLUDED.chunk_type,chunk_index=EXCLUDED.chunk_index,metadata=EXCLUDED.metadata
                    """, [(c.id, repository_id, c.file_path, c.start_line, c.end_line, c.content, commit, generation,
                        c.symbol_id, c.symbol_name, c.language, c.chunk_type, c.chunk_index,
                        json.dumps(c.metadata, default=str)) for c in chunks])
            conn.commit()

    def counts(self, repository_id: int, generation: str) -> dict[str, int]:
        with psycopg.connect(self.dsn) as conn:
            symbols = conn.execute("SELECT count(*) FROM symbols WHERE repository_id=%s AND index_generation=%s",
                                (repository_id, generation)).fetchone()[0]
            chunks = conn.execute("SELECT count(*) FROM chunks WHERE repository_id=%s AND index_generation=%s",
                                (repository_id, generation)).fetchone()[0]
            files = conn.execute("SELECT count(*) FROM repository_files WHERE repository_id=%s AND index_generation=%s",
                                (repository_id, generation)).fetchone()[0]
        return {"symbols": int(symbols), "chunks": int(chunks), "files": int(files)}

    def publish_generation(self, repository_id: int, commit: str, generation: str,
                        details: dict[str, Any]) -> None:
        with psycopg.connect(self.dsn) as conn:
            conn.execute("""UPDATE repositories SET indexed_commit_sha=%s,index_generation=%s,indexed_at=now()
                            WHERE id=%s""", (commit, generation, repository_id))
            conn.execute("""UPDATE ingestion_runs SET status='published',completed_at=now(),details=%s::jsonb
                            WHERE index_generation=%s""", (json.dumps(details, default=str), generation))
            conn.commit()

    def mark_failed(self, generation: str, error_type: str, message: str = "") -> None:
        details = {"error_type": error_type, "message": message[:2000]}
        with psycopg.connect(self.dsn) as conn:
            conn.execute("""UPDATE ingestion_runs SET status='failed',completed_at=now(),details=%s::jsonb
                            WHERE index_generation=%s""", (json.dumps(details), generation))
            conn.commit()

    def cleanup_generation(self, repository_id: int, generation: str) -> None:
        with psycopg.connect(self.dsn) as conn:
            conn.execute("DELETE FROM symbols WHERE repository_id=%s AND index_generation=%s", (repository_id, generation))
            conn.execute("DELETE FROM chunks WHERE repository_id=%s AND index_generation=%s", (repository_id, generation))
            conn.execute("DELETE FROM repository_files WHERE repository_id=%s AND index_generation=%s", (repository_id, generation))
            conn.commit()

    def cleanup_old_generations(self, repository_id: int, active_generation: str) -> None:
            with psycopg.connect(self.dsn) as conn:
                conn.execute("DELETE FROM symbols WHERE repository_id=%s AND index_generation<>%s", (repository_id, active_generation))
                conn.execute("DELETE FROM chunks WHERE repository_id=%s AND index_generation<>%s", (repository_id, active_generation))
                conn.execute("DELETE FROM repository_files WHERE repository_id=%s AND index_generation<>%s", (repository_id, active_generation))
                conn.commit()



class Neo4jStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password), max_connection_pool_size=20, connection_timeout=10)

    def close(self):
        self.driver.close()

    def ensure_schema(self):
        queries = [
            "CREATE CONSTRAINT symbol_id IF NOT EXISTS FOR (n:Symbol) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT test_id IF NOT EXISTS FOR (n:Test) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX symbol_repo_generation IF NOT EXISTS FOR (n:Symbol) ON (n.repository_id, n.index_generation)",
            "CREATE INDEX test_repo_generation IF NOT EXISTS FOR (n:Test) ON (n.repository_id, n.index_generation)",
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query).consume()

    def write_generation(self, repository_id: int, commit: str, generation: str,
                         symbols: list[Symbol], edges: list[GraphEdge]) -> dict[str, int]:
        node_rows = [{
            "id": s.id, "repository_id": repository_id, "index_generation": generation,
            "commit_sha": commit, "qualified_name": s.qualified_name, "name": s.name,
            "kind": s.kind, "file_path": s.file_path, "start_line": s.start_line,
            "end_line": s.end_line, "language": s.language, "is_test": s.is_test,
            "signature": s.signature or "", "source_code": s.source_code or "",
            "modifiers": s.modifiers, "metadata_json": json.dumps(s.metadata, default=str),
        } for s in symbols]
        tests = [{"id": "test-" + s.id, "symbol_id": s.id, "repository_id": repository_id,
                  "index_generation": generation, "commit_sha": commit, "name": s.qualified_name,
                  "file_path": s.file_path, "start_line": s.start_line, "end_line": s.end_line}
                 for s in symbols if s.is_test]
        symbol_by_id = {s.id: s for s in symbols}
        call_test_edges = []
        for edge in edges:
            caller = symbol_by_id.get(edge.source_id)
            callee = symbol_by_id.get(edge.target_id)
            if edge.kind == "CALLS" and caller and callee and caller.is_test and not callee.is_test:
                call_test_edges.append({"test_id": "test-" + caller.id, "symbol_id": callee.id,
                                        "provenance": ["derived-from-direct-call"]})

        with self.driver.session() as session:
            for offset in range(0, len(node_rows), 500):
                session.run("UNWIND $rows AS row MERGE (s:Symbol {id:row.id}) SET s += row",
                            rows=node_rows[offset:offset + 500]).consume()
            for offset in range(0, len(tests), 500):
                session.run("UNWIND $rows AS row MERGE (t:Test {id:row.id}) SET t += row",
                            rows=tests[offset:offset + 500]).consume()
            # Only known normalized edge types are allowed into Cypher text.
            edge_queries = {
                "CALLS": "MATCH (a:Symbol {id:row.from_id,repository_id:$rid,index_generation:$gen}), (b:Symbol {id:row.to_id,repository_id:$rid,index_generation:$gen}) MERGE (a)-[r:CALLS {index_generation:$gen}]->(b) SET r.commit_sha=$commit,r.provenance=row.provenance,r.metadata_json=row.metadata_json",
                "CONTAINS": "MATCH (a:Symbol {id:row.from_id,repository_id:$rid,index_generation:$gen}), (b:Symbol {id:row.to_id,repository_id:$rid,index_generation:$gen}) MERGE (a)-[r:CONTAINS {index_generation:$gen}]->(b) SET r.commit_sha=$commit,r.provenance=row.provenance,r.metadata_json=row.metadata_json",
                "ANNOTATED_BY": "MATCH (a:Symbol {id:row.from_id,repository_id:$rid,index_generation:$gen}), (b:Symbol {id:row.to_id,repository_id:$rid,index_generation:$gen}) MERGE (a)-[r:ANNOTATED_BY {index_generation:$gen}]->(b) SET r.commit_sha=$commit,r.provenance=row.provenance,r.metadata_json=row.metadata_json",
                "HAS_BODY_NODE": "MATCH (a:Symbol {id:row.from_id,repository_id:$rid,index_generation:$gen}), (b:Symbol {id:row.to_id,repository_id:$rid,index_generation:$gen}) MERGE (a)-[r:HAS_BODY_NODE {index_generation:$gen}]->(b) SET r.commit_sha=$commit,r.provenance=row.provenance,r.metadata_json=row.metadata_json",
                "HAS_MODULE": "MATCH (a:Symbol {id:row.from_id,repository_id:$rid,index_generation:$gen}), (b:Symbol {id:row.to_id,repository_id:$rid,index_generation:$gen}) MERGE (a)-[r:HAS_MODULE {index_generation:$gen}]->(b) SET r.commit_sha=$commit,r.provenance=row.provenance,r.metadata_json=row.metadata_json",
                "DEFINES_CONFIG": "MATCH (a:Symbol {id:row.from_id,repository_id:$rid,index_generation:$gen}), (b:Symbol {id:row.to_id,repository_id:$rid,index_generation:$gen}) MERGE (a)-[r:DEFINES_CONFIG {index_generation:$gen}]->(b) SET r.commit_sha=$commit,r.provenance=row.provenance,r.metadata_json=row.metadata_json",
            }
            for kind, query in edge_queries.items():
                rows = [{"from_id": e.source_id, "to_id": e.target_id,
                         "provenance": e.provenance, "metadata_json": json.dumps(e.metadata, default=str)}
                        for e in edges if e.kind == kind]
                for offset in range(0, len(rows), 500):
                    session.run("UNWIND $rows AS row " + query, rows=rows[offset:offset + 500],
                                rid=repository_id, gen=generation, commit=commit).consume()
            for offset in range(0, len(call_test_edges), 500):
                session.run("""
                    UNWIND $rows AS row
                    MATCH (t:Test {id:row.test_id,repository_id:$rid,index_generation:$gen})
                    MATCH (s:Symbol {id:row.symbol_id,repository_id:$rid,index_generation:$gen})
                    MERGE (t)-[r:TESTS {index_generation:$gen}]->(s)
                    SET r.commit_sha=$commit,r.provenance=row.provenance
                """, rows=call_test_edges[offset:offset + 500], rid=repository_id,
                    gen=generation, commit=commit).consume()
        return {"symbols": len(node_rows), "tests": len(tests),
                "calls": sum(e.kind == "CALLS" for e in edges),
                "contains": sum(e.kind == "CONTAINS" for e in edges),
                "annotations": sum(e.kind == "ANNOTATED_BY" for e in edges),
                "other_edges": sum(e.kind not in {"CALLS", "CONTAINS", "ANNOTATED_BY"} for e in edges),
                "test_edges": len(call_test_edges)}

    def validate_generation(self, repository_id: int, generation: str) -> dict[str, int]:
        with self.driver.session() as session:
            symbol_row = session.run("MATCH (s:Symbol {repository_id:$rid,index_generation:$gen}) RETURN count(s) AS n",
                                     rid=repository_id, gen=generation).single()
            test_row = session.run("MATCH (t:Test {repository_id:$rid,index_generation:$gen}) RETURN count(t) AS n",
                                   rid=repository_id, gen=generation).single()
            edge_row = session.run("MATCH (a {repository_id:$rid,index_generation:$gen})-[r {index_generation:$gen}]->(b {repository_id:$rid,index_generation:$gen}) RETURN count(r) AS n",
                                   rid=repository_id, gen=generation).single()
        return {"symbols": int(symbol_row["n"] if symbol_row else 0),
                "tests": int(test_row["n"] if test_row else 0),
                "edges": int(edge_row["n"] if edge_row else 0)}

    def cleanup_generation(self, repository_id: int, generation: str) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n {repository_id:$rid,index_generation:$gen}) DETACH DELETE n",
                        rid=repository_id, gen=generation).consume()

    def cleanup_old_generations(self, repository_id: int, active_generation: str) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n {repository_id:$rid}) WHERE n.index_generation <> $gen DETACH DELETE n",
                        rid=repository_id, gen=active_generation).consume()


class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)
        self.model = SentenceTransformer(settings.embedding_model)
        dimension_getter = getattr(self.model, "get_embedding_dimension", None)
        self.dimension = dimension_getter() if callable(dimension_getter) else self.model.get_sentence_embedding_dimension()
        if not self.dimension:
            raise RuntimeError("Embedding model did not report a vector dimension.")

    def ensure_collection(self):
        if self.client.collection_exists(settings.qdrant_collection):
            info = self.client.get_collection(settings.qdrant_collection)
            vectors = info.config.params.vectors
            existing = vectors.size if hasattr(vectors, "size") else None
            if existing != self.dimension:
                raise ValueError(f"Qdrant collection dimension is {existing}, embedding model dimension is {self.dimension}. "
                                 "Create a new collection and re-index; never silently reuse incompatible vectors.")
        else:
            self.client.create_collection(collection_name=settings.qdrant_collection,
                vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE))
        # Idempotently ensure payload indexes on existing collections too.
        for field, schema in [("repository_id", models.PayloadSchemaType.INTEGER),
                              ("commit_sha", models.PayloadSchemaType.KEYWORD),
                              ("index_generation", models.PayloadSchemaType.KEYWORD),
                              ("file_path", models.PayloadSchemaType.KEYWORD),
                              ("symbol_id", models.PayloadSchemaType.KEYWORD),
                              ("language", models.PayloadSchemaType.KEYWORD),
                              ("chunk_type", models.PayloadSchemaType.KEYWORD)]:
            try:
                self.client.create_payload_index(collection_name=settings.qdrant_collection,
                    field_name=field, field_schema=schema, wait=True)
            except Exception as exc:
                # Some server/client versions report an existing index as an error. Verify its existence
                # through collection payload schema rather than failing a re-ingestion for an idempotent step.
                info = self.client.get_collection(settings.qdrant_collection)
                schema_now = getattr(info, "payload_schema", {}) or {}
                if field not in schema_now:
                    raise RuntimeError(f"Could not ensure Qdrant payload index {field!r}: {exc}") from exc

    @staticmethod
    def embedding_text(chunk: CodeChunk, repository: str = "") -> str:
        # Keep this exact preprocessing contract shared with the retrieval query builder.
        # Keep the established document-prefix contract used by the current retrieval design.
        lines = [f"File: {chunk.file_path}", f"Language: {chunk.language}",
                 f"Symbol: {chunk.symbol_name}" if chunk.symbol_name else ""]
        return "\n".join(line for line in lines if line) + "\n\n" + chunk.content

    def upsert_chunks(self, repository_id: int, repository: str, commit: str, generation: str,
                      chunks: list[CodeChunk]) -> int:
        if not chunks:
            return 0
        self.ensure_collection()
        total = 0
        batch_size = max(1, min(settings.embedding_batch_size, settings.max_embedding_batch_size))
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [self.embedding_text(c, repository) for c in batch]
            vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                        show_progress_bar=False).tolist()
            points = []
            for chunk, vector in zip(batch, vectors):
                points.append(models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{generation}:{chunk.id}")), vector=vector,
                    payload={"chunk_id": chunk.id, "repository_id": repository_id,
                        "repository": repository, "commit_sha": commit, "index_generation": generation,
                        "file_path": chunk.file_path, "start_line": chunk.start_line, "end_line": chunk.end_line,
                        "content": chunk.content, "language": chunk.language, "symbol_id": chunk.symbol_id,
                        "symbol_name": chunk.symbol_name, "chunk_type": chunk.chunk_type,
                        "chunk_index": chunk.chunk_index, "metadata": chunk.metadata},
                ))
            self.client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
            total += len(points)
        return total

    def count_generation(self, repository_id: int, commit: str, generation: str) -> int:
        count = self.client.count(collection_name=settings.qdrant_collection, count_filter=models.Filter(must=[
            models.FieldCondition(key="repository_id", match=models.MatchValue(value=repository_id)),
            models.FieldCondition(key="commit_sha", match=models.MatchValue(value=commit)),
            models.FieldCondition(key="index_generation", match=models.MatchValue(value=generation)),
        ]), exact=True)
        return int(count.count)

    def cleanup_generation(self, repository_id: int, generation: str) -> None:
        self.client.delete(settings.qdrant_collection, points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="repository_id", match=models.MatchValue(value=repository_id)),
            models.FieldCondition(key="index_generation", match=models.MatchValue(value=generation)),
        ])), wait=True)

    def cleanup_old_generations(self, repository_id: int, active_generation: str) -> None:
        # Scroll and delete stale IDs explicitly; this works across qdrant-client MatchExcept versions.
        offset = None
        stale_ids: list[Any] = []
        while True:
            points, offset = self.client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="repository_id", match=models.MatchValue(value=repository_id))]),
                limit=512, offset=offset, with_payload=["index_generation"], with_vectors=False)
            stale_ids.extend(point.id for point in points
                             if (point.payload or {}).get("index_generation") != active_generation)
            if len(stale_ids) >= 2048:
                self.client.delete(settings.qdrant_collection, points_selector=stale_ids, wait=True)
                stale_ids = []
            if offset is None:
                break
        if stale_ids:
            self.client.delete(settings.qdrant_collection, points_selector=stale_ids, wait=True)
