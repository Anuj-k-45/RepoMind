from __future__ import annotations

import logging
import re
from typing import Any

import psycopg
from psycopg.rows import dict_row
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings
import requests

from .config import settings
from .models import Evidence
from .security import validate_line_range

log = logging.getLogger(__name__)


class RepositoryStore:
    """PostgreSQL adapter. Use a read-only DB role for the online retrieval service."""

    def health(self) -> bool:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=5) as conn:
            conn.execute("SELECT 1").fetchone()
        return True

    def diagnostics(self, repository: str) -> dict[str, Any]:
        with psycopg.connect(settings.postgres_dsn, row_factory=dict_row, connect_timeout=5) as conn:
            repo = conn.execute(
                "SELECT id, full_name, indexed_commit_sha, index_generation, indexed_at "
                "FROM repositories WHERE lower(full_name)=lower(%s) LIMIT 1", (repository,)
            ).fetchone()
            if not repo:
                raise LookupError(f"Repository {repository!r} is not indexed.")
            rid, commit, generation = repo["id"], repo["indexed_commit_sha"], repo["index_generation"]
            counts = conn.execute("""
                SELECT
                  (SELECT count(*) FROM repository_files WHERE repository_id=%s AND commit_sha=%s AND index_generation=%s) AS files,
                  (SELECT count(*) FROM symbols WHERE repository_id=%s AND commit_sha=%s AND index_generation=%s) AS symbols,
                  (SELECT count(*) FROM chunks WHERE repository_id=%s AND commit_sha=%s AND index_generation=%s) AS chunks,
                  (SELECT count(*) FROM chunks WHERE repository_id=%s AND commit_sha=%s AND index_generation=%s AND search_vector IS NOT NULL) AS searchable_chunks
            """, (rid, commit, generation, rid, commit, generation, rid, commit, generation,
                  rid, commit, generation)).fetchone()
            return {"repository": dict(repo), "counts": dict(counts)}

    def resolve(self, repository: str, target_symbol: str | None) -> dict[str, Any]:
        with psycopg.connect(settings.postgres_dsn, row_factory=dict_row, connect_timeout=5) as conn:
            repo = conn.execute(
                """
                SELECT id, full_name, indexed_commit_sha, index_generation
                FROM repositories WHERE lower(full_name)=lower(%s) LIMIT 1
                """,
                (repository,),
            ).fetchone()
            if not repo:
                raise LookupError(f"Repository {repository!r} is not indexed.")
            symbol = None
            if target_symbol:
                symbol = conn.execute(
                    """
                    SELECT id, qualified_name, name, file_path, kind, start_line, end_line,
                           commit_sha, index_generation
                    FROM symbols
                    WHERE repository_id=%s AND commit_sha=%s AND index_generation=%s
                      AND COALESCE(file_path, '') <> ''
                      AND (lower(qualified_name)=lower(%s)
                           OR lower(name)=lower(%s)
                           OR lower(qualified_name) LIKE lower(%s))
                    ORDER BY CASE WHEN lower(qualified_name)=lower(%s) THEN 0
                                  WHEN lower(name)=lower(%s) THEN 1 ELSE 2 END,
                             length(qualified_name), start_line NULLS LAST
                    LIMIT 5
                    """,
                    (repo["id"], repo["indexed_commit_sha"], repo["index_generation"], target_symbol,
                     target_symbol, f"%{target_symbol}%", target_symbol, target_symbol),
                ).fetchall()
                # Do not silently pick an ambiguous substring match.
                if len(symbol) == 1:
                    symbol = symbol[0]
                elif len(symbol) > 1:
                    exact = [s for s in symbol if s["qualified_name"].casefold() == target_symbol.casefold()]
                    if not exact:
                        exact = [s for s in symbol if (s.get("name") or "").casefold() == target_symbol.casefold()]
                    symbol = exact[0] if len(exact) == 1 else None
            return {"repository": repo, "symbol": symbol}

    def metadata_search(self, repository_id: int, commit: str, generation: str,
                        target_symbol: str | None, limit: int) -> list[dict[str, Any]]:
        if not target_symbol:
            return []
        with psycopg.connect(settings.postgres_dsn, row_factory=dict_row, connect_timeout=5) as conn:
            rows = conn.execute(
                """
                SELECT id, qualified_name, name, file_path, kind, start_line, end_line,
                       commit_sha, index_generation
                FROM symbols
                WHERE repository_id=%s AND commit_sha=%s AND index_generation=%s
                  AND COALESCE(file_path, '') <> ''
                  AND (lower(qualified_name) LIKE lower(%s) OR lower(name) LIKE lower(%s))
                ORDER BY CASE WHEN lower(qualified_name)=lower(%s) THEN 0
                              WHEN lower(name)=lower(%s) THEN 1 ELSE 2 END
                LIMIT %s
                """,
                (repository_id, commit, generation, f"%{target_symbol}%", f"%{target_symbol}%",
                 target_symbol, target_symbol, limit),
            ).fetchall()
        return [self._metadata_evidence(row) for row in rows]

    @staticmethod
    def _metadata_evidence(row: dict[str, Any]) -> dict[str, Any]:
        return Evidence(
            evidence_id=f"pg-symbol:{row['id']}", source_type="postgres_metadata",
            claim=(f"Symbol {row['qualified_name']} ({row['kind']}) is defined in "
                   f"{row['file_path']} lines {row['start_line']}-{row['end_line']}."),
            repository_full_name="", commit_sha=row["commit_sha"],
            index_generation=row["index_generation"], file_path=row["file_path"],
            start_line=row["start_line"], end_line=row["end_line"],
            details={"symbol_id": row["id"], "qualified_name": row["qualified_name"]},
        ).model_dump()

    def lexical_search(self, repository_id: int, commit: str, generation: str,
                       queries: list[str], limit: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with psycopg.connect(settings.postgres_dsn, row_factory=dict_row, connect_timeout=5) as conn:
            for query in queries[:5]:
                rows = conn.execute(
                    """
                    SELECT id, file_path, start_line, end_line, content, commit_sha,
                           index_generation,
                           ts_rank(search_vector, websearch_to_tsquery('simple', %s)) AS rank
                    FROM chunks
                    WHERE repository_id=%s AND commit_sha=%s AND index_generation=%s
                      AND search_vector @@ websearch_to_tsquery('simple', %s)
                    ORDER BY rank DESC LIMIT %s
                    """,
                    (query, repository_id, commit, generation, query, limit),
                ).fetchall()
                # Natural-language phrases are AND-heavy under websearch_to_tsquery. If
                # the focused phrase has no exact hits, retry with an OR of safe tokens
                # so one missing word does not erase useful code evidence.
                tokens = list(dict.fromkeys(re.findall(r"[A-Za-z0-9_$]{3,}", query)))
                if not rows and len(tokens) > 1:
                    fallback_query = " OR ".join(tokens[:8])
                    rows = conn.execute(
                        """
                        SELECT id, file_path, start_line, end_line, content, commit_sha,
                               index_generation,
                               ts_rank(search_vector, websearch_to_tsquery('simple', %s)) AS rank
                        FROM chunks
                        WHERE repository_id=%s AND commit_sha=%s AND index_generation=%s
                          AND search_vector @@ websearch_to_tsquery('simple', %s)
                        ORDER BY rank DESC LIMIT %s
                        """,
                        (fallback_query, repository_id, commit, generation, fallback_query, limit),
                    ).fetchall()
                    if rows:
                        log.debug("Lexical fallback broadened query=%r to=%r hits=%d",
                                  query, fallback_query, len(rows))
                for row in rows:
                    results.append(Evidence(
                        evidence_id=f"pg-chunk:{row['id']}", source_type="postgres_lexical",
                        claim=row["content"][:1800], repository_full_name="",
                        commit_sha=row["commit_sha"], index_generation=row["index_generation"],
                        file_path=row["file_path"], start_line=row["start_line"],
                        end_line=row["end_line"], rank_score=float(row["rank"]),
                        details={"chunk_id": row["id"]},
                    ).model_dump())
        # Aggregate repeated hits across query variants and diversify by file path.
        merged: dict[str, dict[str, Any]] = {}
        for item in results:
            key = item["evidence_id"]
            if key not in merged:
                merged[key] = item
                merged[key]["details"] = dict(item.get("details") or {}, query_matches=1)
            else:
                merged[key]["rank_score"] = max(float(merged[key].get("rank_score") or 0),
                                                  float(item.get("rank_score") or 0))
                merged[key]["details"]["query_matches"] = int(merged[key]["details"].get("query_matches", 1)) + 1
        ranked = sorted(merged.values(), key=lambda x: (
            float(x.get("rank_score") or 0) + 0.02 * min(3, int((x.get("details") or {}).get("query_matches", 1)) - 1),
            x.get("file_path") or "", -(x.get("start_line") or 0)
        ), reverse=True)
        selected: list[dict[str, Any]] = []
        file_counts: dict[str, int] = {}
        for item in ranked:
            path = str(item.get("file_path") or "")
            if file_counts.get(path, 0) >= 2:
                continue
            selected.append(item)
            file_counts[path] = file_counts.get(path, 0) + 1
            if len(selected) >= limit:
                break
        return selected


class GraphStore:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_pool_size=20, connection_timeout=5,
        )

    def close(self) -> None:
        self.driver.close()

    def health(self) -> bool:
        self.driver.verify_connectivity()
        return True

    def generation_counts(self, repository_id: int, generation: str) -> dict[str, int]:
        with self.driver.session() as session:
            symbols = session.run(
                "MATCH (n:Symbol {repository_id:$rid,index_generation:$gen}) RETURN count(n) AS n",
                rid=repository_id, gen=generation).single()["n"]
            tests = session.run(
                "MATCH (n:Test {repository_id:$rid,index_generation:$gen}) RETURN count(n) AS n",
                rid=repository_id, gen=generation).single()["n"]
            edges = session.run(
                "MATCH (a {repository_id:$rid,index_generation:$gen})-[r {index_generation:$gen}]->"
                "(b {repository_id:$rid,index_generation:$gen}) RETURN count(r) AS n",
                rid=repository_id, gen=generation).single()["n"]
        return {"symbols": int(symbols), "tests": int(tests), "edges": int(edges)}

    def discovery_search(self, repository_id: int, generation: str, commit: str,
                         terms: list[str], limit: int) -> list[dict[str, Any]]:
        """Find symbols by identifier/path keywords and return their immediate graph neighbors.

        Unlike impact_search, this does not require a pre-resolved target symbol, which
        is essential for broad questions such as "how is Dispatch connected?".
        """
        clean_terms = list(dict.fromkeys(
            str(term).casefold().strip() for term in terms
            if str(term).strip() and len(str(term).strip()) >= 3
        ))[:8]
        if not clean_terms:
            return []
        query = """
        MATCH (target:Symbol {repository_id:$repo_id, index_generation:$generation, commit_sha:$commit})
        WHERE any(term IN $terms WHERE
            toLower(coalesce(target.name, '')) CONTAINS term OR
            toLower(coalesce(target.qualified_name, '')) CONTAINS term OR
            toLower(coalesce(target.file_path, '')) CONTAINS term)
        WITH target,
             reduce(score = 0, term IN $terms |
                score + CASE WHEN toLower(coalesce(target.name, '')) CONTAINS term THEN 3
                             WHEN toLower(coalesce(target.qualified_name, '')) CONTAINS term THEN 2
                             WHEN toLower(coalesce(target.file_path, '')) CONTAINS term THEN 1
                             ELSE 0 END) AS match_score
        ORDER BY match_score DESC, target.file_path, target.qualified_name
        LIMIT $limit
        OPTIONAL MATCH (target)-[r]-(neighbor:Symbol {
            repository_id:$repo_id, index_generation:$generation, commit_sha:$commit})
        WITH target, collect(DISTINCT CASE WHEN neighbor IS NULL THEN NULL ELSE {
            relationship:type(r), name:neighbor.qualified_name, id:neighbor.id,
            path:neighbor.file_path, kind:neighbor.kind
        } END)[0..$limit] AS neighbors
        RETURN target.id AS id, target.qualified_name AS name, target.name AS simple_name,
               target.file_path AS path, target.kind AS kind, neighbors
        ORDER BY path, name
        """
        with self.driver.session() as session:
            records = list(session.run(query, repo_id=repository_id, generation=generation,
                                       commit=commit, terms=clean_terms, limit=max(1, min(limit, 20))))
        results: list[dict[str, Any]] = []
        for record in records:
            neighbors = [item for item in (record.get("neighbors") or []) if item and item.get("id")]
            relation_summary = "; ".join(
                f"{item.get('relationship')} -> {item.get('name')} ({item.get('path')})"
                for item in neighbors[:8]
            )
            claim = (f"Repository graph contains symbol {record.get('name')} ({record.get('kind')}) "
                     f"in {record.get('path')}. Related symbols: {relation_summary or 'no immediate Symbol neighbors returned'}.")
            results.append(Evidence(
                evidence_id=f"neo4j:discovery:{record.get('id')}", source_type="neo4j_graph",
                claim=claim, repository_full_name="", commit_sha=commit,
                index_generation=generation, file_path=record.get("path"),
                details={"target_id": record.get("id"), "target_name": record.get("name"),
                         "neighbors": neighbors, "discovery_terms": clean_terms,
                         "relationship_types": sorted({x.get("relationship") for x in neighbors if x.get("relationship")})},
            ).model_dump())
        return results

    def impact_search(self, repository_id: int, generation: str, commit: str,
                      target_symbol_id: str, hops: int, limit: int) -> list[dict[str, Any]]:
        # Only validated, bounded integer interpolation is used for Cypher path length.
        hops = max(1, min(int(hops), settings.max_graph_hops, 4))
        query = f"""
        MATCH (target:Symbol {{id: $target_id, repository_id: $repo_id,
                              index_generation: $generation, commit_sha: $commit}})
        OPTIONAL MATCH p=(dependent:Symbol {{repository_id: $repo_id,
                                            index_generation: $generation,
                                            commit_sha: $commit}})
                       -[:CALLS*1..{hops}]->(target)
        WITH target, collect(DISTINCT p)[0..$limit] AS paths
        WITH target, [target] + reduce(acc=[], p IN paths | acc + nodes(p)) AS path_nodes
        UNWIND path_nodes AS affected
        WITH target, collect(DISTINCT affected)[0..$limit] AS affected_nodes
        UNWIND affected_nodes AS affected
        OPTIONAL MATCH (test:Test {{repository_id: $repo_id,
                                   index_generation: $generation, commit_sha: $commit}})
                      -[:TESTS]->(affected)
        RETURN target.id AS target_id, target.qualified_name AS target_name,
               collect(DISTINCT {{id: affected.id, name: affected.qualified_name,
                                  path: affected.file_path, kind: affected.kind}})
                   [0..$limit] AS affected_symbols,
               collect(DISTINCT {{id: test.id, name: test.name, path: test.file_path}})
                   [0..$limit] AS related_tests
        """
        with self.driver.session() as session:
            # Read the active commit from the indexed generation's graph nodes; the caller
            # supplies the exact active generation and the query remains generation-scoped.
            record = session.run(query, target_id=target_symbol_id, repo_id=repository_id,
                                 generation=generation, commit=commit, limit=limit).single()
        if not record:
            return []
        affected = [x for x in (record["affected_symbols"] or []) if x and x.get("id")]
        tests = [x for x in (record["related_tests"] or []) if x and x.get("id")]
        return [Evidence(
            evidence_id=f"neo4j:{target_symbol_id}:impact", source_type="neo4j_graph",
            claim=(f"Graph traversal around {record['target_name']} found "
                   f"{len(affected)} related symbols and {len(tests)} candidate tests. "
                   "This indicates inspection candidates, not test outcomes."),
            repository_full_name="", index_generation=generation,
            details={"target_id": target_symbol_id, "affected_symbols": affected,
                     "related_tests": tests, "max_hops": hops,
                     "relationship_types": ["CALLS", "TESTS"]},
        ).model_dump()]


class VectorStore:
    def __init__(self) -> None:
        # Local embeddings avoid requiring a paid embedding API. Ingestion MUST use
        # this exact model and identical chunk/query preprocessing.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key,
                                   timeout=8)

    def health(self) -> dict[str, Any]:
        info = self.client.get_collection(settings.qdrant_collection)
        vectors = info.config.params.vectors
        size = vectors.size if hasattr(vectors, "size") else None
        raw_status = getattr(info.status, "value", info.status)
        return {"status": str(raw_status).split(".")[-1].casefold(), "vector_size": size}

    def count_generation(self, repository_id: int, commit: str, generation: str) -> int:
        count = self.client.count(collection_name=settings.qdrant_collection, count_filter=Filter(must=[
            FieldCondition(key="repository_id", match=MatchValue(value=repository_id)),
            FieldCondition(key="commit_sha", match=MatchValue(value=commit)),
            FieldCondition(key="index_generation", match=MatchValue(value=generation)),
        ]), exact=True)
        return int(count.count)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def search(self, repository_id: int, commit: str, generation: str,
               queries: list[str], limit: int) -> list[dict[str, Any]]:
        # Search multiple focused queries, then fuse their results. The previous
        # implementation truncated the de-duplicated list to `limit`, which usually
        # meant the first query consumed the entire result budget and later queries
        # contributed nothing. This version rewards cross-query agreement and file diversity.
        merged: dict[str, dict[str, Any]] = {}
        for query in queries[:3]:
            vector = self.embeddings.embed_query(query)
            response = self.client.query_points(
                collection_name=settings.qdrant_collection, query=vector,
                query_filter=Filter(must=[
                    FieldCondition(key="repository_id", match=MatchValue(value=repository_id)),
                    FieldCondition(key="commit_sha", match=MatchValue(value=commit)),
                    FieldCondition(key="index_generation", match=MatchValue(value=generation)),
                ]),
                limit=max(limit, min(12, limit * 2)), with_payload=True,
            )
            for point in response.points:
                payload = point.payload or {}
                chunk_id = str(payload.get("chunk_id", point.id))
                key = f"qdrant:{chunk_id}"
                score = float(point.score or 0.0)
                if key not in merged:
                    merged[key] = Evidence(
                        evidence_id=key, source_type="qdrant_vector",
                        claim=str(payload.get("content", ""))[:1800], repository_full_name="",
                        commit_sha=payload.get("commit_sha"),
                        index_generation=payload.get("index_generation"),
                        file_path=payload.get("file_path"), start_line=payload.get("start_line"),
                        end_line=payload.get("end_line"), rank_score=score,
                        details={"chunk_id": chunk_id, "vector_score": score, "query_matches": 1},
                    ).model_dump()
                else:
                    item = merged[key]
                    item["rank_score"] = max(float(item.get("rank_score") or 0), score)
                    item["details"]["vector_score"] = max(float(item["details"].get("vector_score") or 0), score)
                    item["details"]["query_matches"] = int(item["details"].get("query_matches", 1)) + 1
        ranked = sorted(merged.values(), key=lambda x: (
            float(x.get("rank_score") or 0) + 0.025 * min(3, int((x.get("details") or {}).get("query_matches", 1)) - 1),
            x.get("file_path") or "", -(x.get("start_line") or 0)
        ), reverse=True)
        selected: list[dict[str, Any]] = []
        file_counts: dict[str, int] = {}
        for item in ranked:
            path = str(item.get("file_path") or "")
            if file_counts.get(path, 0) >= 2:
                continue
            selected.append(item)
            file_counts[path] = file_counts.get(path, 0) + 1
            if len(selected) >= limit:
                break
        log.info("[TRACE VECTOR FUSION] query_count=%d unique_candidates=%d selected=%d files=%d",
                 min(len(queries), 3), len(merged), len(selected), len(file_counts))
        return selected


class GitHubSource:
    def fetch(self, repository: str, commit: str, file_path: str,
              start_line: int | None, end_line: int | None) -> dict[str, Any] | None:
        if not commit or not file_path or ".." in file_path.split("/"):
            return None
        owner, repo = repository.split("/", 1)
        # The GitHub raw endpoint is fixed; path is quoted to prevent URL path tricks.
        from urllib.parse import quote
        url = f"https://raw.githubusercontent.com/{quote(owner)}/{quote(repo)}/{quote(commit)}/{quote(file_path, safe='/')}"
        headers = {"Accept": "text/plain", "User-Agent": "RepoMind/0.2"}
        token = settings.github_token
        has_token = bool(token and not token.startswith("DUMMY_"))
        if not settings.allow_public_repositories and not has_token:
            return None
        if has_token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(url, headers=headers, timeout=settings.request_timeout_seconds,
                                stream=True, allow_redirects=False)
        if response.status_code != 200:
            response.close()
            return None
        # Bound source response to 2 MB; do not trust content length alone.
        chunks, total = [], 0
        try:
            for part in response.iter_content(chunk_size=65536):
                total += len(part)
                if total > 2 * 1024 * 1024:
                    raise ValueError("Source file exceeds the 2 MB retrieval limit.")
                chunks.append(part)
        finally:
            response.close()
        lines = b"".join(chunks).decode("utf-8", errors="replace").splitlines()
        start, end = validate_line_range(start_line, end_line)
        if start > len(lines):
            start, end = 1, min(len(lines), 80)
        end = min(end, len(lines))
        excerpt = "\n".join(f"{i}: {lines[i-1]}" for i in range(start, end + 1))
        return {"file_path": file_path, "commit_sha": commit, "start_line": start,
                "end_line": end, "excerpt": excerpt,
                "url": f"https://github.com/{owner}/{repo}/blob/{commit}/{file_path}#L{start}"}
