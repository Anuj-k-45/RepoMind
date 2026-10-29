from __future__ import annotations

import json
import sys
from collections import Counter

import psycopg
from psycopg.rows import dict_row
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models

from repomind_ingest.config import settings


REPOSITORY = "LableOrg/java-maven-junit-helloworld"
COMMIT = "9e489de2e27fae2935f4b6a8eed19b90ea4be9eb"
GENERATION = "9e489de2e27f-893a85094308"
REPOSITORY_ID = 1


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check_postgres() -> tuple[set[str], int | None]:
    section("1) POSTGRESQL — repository, generation, counts, samples")
    with psycopg.connect(settings.postgres_dsn, row_factory=dict_row) as conn:
        repo = conn.execute(
            """
            SELECT id, full_name, indexed_commit_sha, index_generation, indexed_at
            FROM repositories WHERE full_name = %s
            """,
            (REPOSITORY,),
        ).fetchone()

        if not repo:
            print(f"FAIL: repository {REPOSITORY!r} not found")
            return set(), None

        print("Repository row:")
        print(json.dumps(dict(repo), default=str, indent=2))
        repo_id = int(repo["id"])
        if repo_id != REPOSITORY_ID:
            print(f"NOTE: report said repository_id={REPOSITORY_ID}; database has {repo_id}")

        run = conn.execute(
            """
            SELECT index_generation, repository_id, commit_sha, status,
                   started_at, completed_at, details
            FROM ingestion_runs WHERE index_generation = %s
            """,
            (GENERATION,),
        ).fetchone()
        print("\nIngestion run:")
        print(json.dumps(dict(run) if run else None, default=str, indent=2))

        count_rows = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM repository_files
                WHERE repository_id=%s AND index_generation=%s AND commit_sha=%s) AS files,
              (SELECT count(*) FROM symbols
                WHERE repository_id=%s AND index_generation=%s AND commit_sha=%s) AS symbols,
              (SELECT count(*) FROM chunks
                WHERE repository_id=%s AND index_generation=%s AND commit_sha=%s) AS chunks,
              (SELECT count(*) FROM symbols
                WHERE repository_id=%s AND index_generation=%s AND commit_sha=%s AND is_test) AS test_symbols,
              (SELECT count(*) FROM chunks
                WHERE repository_id=%s AND index_generation=%s AND commit_sha=%s
                  AND search_vector IS NOT NULL) AS chunks_with_search_vector
            """,
            (repo_id, GENERATION, COMMIT,
             repo_id, GENERATION, COMMIT,
             repo_id, GENERATION, COMMIT,
             repo_id, GENERATION, COMMIT,
             repo_id, GENERATION, COMMIT),
        ).fetchone()
        print("\nGeneration-scoped counts:")
        print(json.dumps(dict(count_rows), indent=2))

        samples = conn.execute(
            """
            SELECT id, qualified_name, kind, file_path, start_line, end_line, is_test
            FROM symbols
            WHERE repository_id=%s AND index_generation=%s AND commit_sha=%s
            ORDER BY file_path, start_line NULLS LAST
            LIMIT 8
            """,
            (repo_id, GENERATION, COMMIT),
        ).fetchall()
        print("\nSample symbols:")
        print(json.dumps([dict(r) for r in samples], indent=2))

        chunk_rows = conn.execute(
            """
            SELECT id, file_path, start_line, end_line, symbol_name, language,
                   chunk_type, length(content) AS content_chars
            FROM chunks
            WHERE repository_id=%s AND index_generation=%s AND commit_sha=%s
            ORDER BY file_path, start_line
            """,
            (repo_id, GENERATION, COMMIT),
        ).fetchall()
        chunk_ids = {str(row["id"]) for row in chunk_rows}
        print(f"\nPostgres chunk IDs collected for cross-store check: {len(chunk_ids)}")
        print("Sample chunks:")
        print(json.dumps([dict(r) for r in chunk_rows[:5]], indent=2))

        if repo["indexed_commit_sha"] != COMMIT:
            print("WARNING: repository active commit does not match this ingestion commit.")
        if repo["index_generation"] != GENERATION:
            print("WARNING: repository active generation does not match this generation.")
        if run and run["status"] != "published":
            print(f"WARNING: ingestion run status is {run['status']!r}, not 'published'.")

        return chunk_ids, repo_id


def check_neo4j(repo_id: int) -> None:
    section("2) NEO4J — scoped node/edge counts and relationship types")
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
        connection_timeout=10,
    )
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            counts = session.run(
                """
                MATCH (n)
                WHERE n.repository_id=$rid AND n.index_generation=$gen
                WITH
                  sum(CASE WHEN n:Symbol THEN 1 ELSE 0 END) AS symbols,
                  sum(CASE WHEN n:Test THEN 1 ELSE 0 END) AS tests,
                  count(n) AS all_nodes
                OPTIONAL MATCH (a)-[r]->(b)
                WHERE a.repository_id=$rid AND a.index_generation=$gen
                  AND b.repository_id=$rid AND b.index_generation=$gen
                  AND r.index_generation=$gen
                RETURN symbols, tests, all_nodes, count(r) AS edges
                """,
                rid=repo_id, gen=GENERATION,
            ).single()
            print("Generation-scoped graph counts:")
            print(json.dumps(dict(counts) if counts else {}, indent=2))

            rel_types = session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE a.repository_id=$rid AND a.index_generation=$gen
                  AND b.repository_id=$rid AND b.index_generation=$gen
                  AND r.index_generation=$gen
                RETURN type(r) AS relationship_type, count(r) AS count
                ORDER BY relationship_type
                """,
                rid=repo_id, gen=GENERATION,
            ).data()
            print("\nRelationship counts by type:")
            print(json.dumps(rel_types, indent=2))

            samples = session.run(
                """
                MATCH (s:Symbol {repository_id:$rid,index_generation:$gen})
                RETURN s.id AS id, s.qualified_name AS qualified_name,
                       s.kind AS kind, s.file_path AS file_path,
                       s.start_line AS start_line, s.is_test AS is_test
                ORDER BY s.file_path, s.start_line
                LIMIT 8
                """,
                rid=repo_id, gen=GENERATION,
            ).data()
            print("\nSample symbol nodes:")
            print(json.dumps(samples, indent=2))

            duplicates = session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE a.repository_id=$rid AND a.index_generation=$gen
                  AND b.repository_id=$rid AND b.index_generation=$gen
                  AND r.index_generation=$gen
                WITH type(r) AS rel_type, a.id AS from_id, b.id AS to_id, count(r) AS copies
                WHERE copies > 1
                RETURN rel_type, from_id, to_id, copies
                ORDER BY copies DESC
                LIMIT 20
                """,
                rid=repo_id, gen=GENERATION,
            ).data()
            print("\nDuplicate same-type edges between the same symbol IDs (should normally be empty):")
            print(json.dumps(duplicates, indent=2))
    finally:
        driver.close()


def check_qdrant(repo_id: int, postgres_chunk_ids: set[str]) -> None:
    section("3) QDRANT — collection, vector dimension, payload scope, cross-store IDs")
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30,
    )
    collection = settings.qdrant_collection
    info = client.get_collection(collection)
    print(f"Collection: {collection}")
    print(f"Collection status: {info.status}")
    print(f"Total points in collection: {info.points_count}")
    print(f"Vector config: {info.config.params.vectors}")

    filt = models.Filter(must=[
        models.FieldCondition(key="repository_id", match=models.MatchValue(value=repo_id)),
        models.FieldCondition(key="commit_sha", match=models.MatchValue(value=COMMIT)),
        models.FieldCondition(key="index_generation", match=models.MatchValue(value=GENERATION)),
    ])
    count = client.count(collection_name=collection, count_filter=filt, exact=True).count
    print(f"\nPoints matching this repository + commit + generation: {count}")

    point_ids: set[str] = set()
    sample_points = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            if payload.get("chunk_id") is not None:
                point_ids.add(str(payload["chunk_id"]))
            if len(sample_points) < 5:
                sample_points.append({
                    "point_id": str(point.id),
                    "chunk_id": payload.get("chunk_id"),
                    "repository_id": payload.get("repository_id"),
                    "repository": payload.get("repository"),
                    "commit_sha": payload.get("commit_sha"),
                    "index_generation": payload.get("index_generation"),
                    "file_path": payload.get("file_path"),
                    "start_line": payload.get("start_line"),
                    "end_line": payload.get("end_line"),
                    "language": payload.get("language"),
                    "symbol_name": payload.get("symbol_name"),
                    "content_chars": len(payload.get("content") or ""),
                })
        if offset is None:
            break

    print("\nSample vector payloads:")
    print(json.dumps(sample_points, indent=2))
    missing = sorted(postgres_chunk_ids - point_ids)
    extra = sorted(point_ids - postgres_chunk_ids)
    print("\nCross-store chunk ID comparison:")
    print(json.dumps({
        "postgres_chunk_ids": len(postgres_chunk_ids),
        "qdrant_chunk_ids": len(point_ids),
        "missing_in_qdrant_count": len(missing),
        "extra_in_qdrant_count": len(extra),
        "missing_in_qdrant_sample": missing[:10],
        "extra_in_qdrant_sample": extra[:10],
        "ids_match_exactly": postgres_chunk_ids == point_ids,
    }, indent=2))


def main() -> int:
    print("RepoMind read-only database verification")
    print(f"Repository: {REPOSITORY}")
    print(f"Commit: {COMMIT}")
    print(f"Generation: {GENERATION}")
    print("No database writes or deletes will be performed.")

    try:
        postgres_chunk_ids, repo_id = check_postgres()
        if repo_id is None:
            return 1
        check_neo4j(repo_id)
        check_qdrant(repo_id, postgres_chunk_ids)
    except Exception as exc:
        print(f"\nVERIFICATION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Check that your .env/config and all three database services are available.", file=sys.stderr)
        return 2

    section("VERIFICATION FINISHED")
    print("Compare the actual values above with the expected ingestion report.")
    print("Expected: Postgres 53 symbols / 48 chunks / 8 files; Qdrant 48 points;")
    print("Neo4j 53 Symbol nodes / 34 Test nodes / 95 relationships including derived TESTS edges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
