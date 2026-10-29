from __future__ import annotations

import argparse
import json
import sys

from .config import settings
from .graph import RepoMind
from .security import normalize_repository


def _authorized(repository: str) -> list[str]:
    allowlist = list(settings.authorized_repositories)
    if not allowlist:
        raise RuntimeError("Set AUTHORIZED_REPOSITORIES in .env before running retrieval.")
    normalized = normalize_repository(repository)
    if normalized.casefold() not in {normalize_repository(x).casefold() for x in allowlist}:
        raise PermissionError(f"{normalized!r} is not in AUTHORIZED_REPOSITORIES.")
    return allowlist


def main() -> int:
    parser = argparse.ArgumentParser(prog="repomind", description="RepoMind retrieval CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Ask a question about an indexed repository")
    ask.add_argument("--repository", required=True, help="owner/repository")
    ask.add_argument("--question", required=True, help="Question to answer")
    doctor = sub.add_parser("doctor", help="Check active-generation counts across the three stores")
    doctor.add_argument("--repository", required=True, help="owner/repository")
    args = parser.parse_args()

    try:
        allowlist = _authorized(args.repository)
        engine = RepoMind()
        try:
            if args.command == "ask":
                result = engine.answer_question(args.question, user_id="local-cli",
                                               allowed_repositories=allowlist,
                                               repository=args.repository)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return 0

            pg = engine.pg.diagnostics(args.repository)
            repo = pg["repository"]
            rid, commit, generation = repo["id"], repo["indexed_commit_sha"], repo["index_generation"]
            graph = engine.neo4j.generation_counts(rid, generation)
            vectors = engine.qdrant.count_generation(rid, commit, generation)
            qdrant_info = engine.qdrant.health()
            report = {
                "repository": repo["full_name"], "commit_sha": commit,
                "index_generation": generation,
                "postgres": pg["counts"], "neo4j": graph,
                "qdrant": {"points_for_active_generation": vectors, **qdrant_info},
                "consistency": {
                    "postgres_chunks_match_qdrant_points": pg["counts"]["chunks"] == vectors,
                    "postgres_symbols_match_neo4j_symbols": pg["counts"]["symbols"] == graph["symbols"],
                    "postgres_search_vectors_complete": pg["counts"]["chunks"] == pg["counts"]["searchable_chunks"],
                },
            }
            print(json.dumps(report, indent=2, default=str))
            okay = all(report["consistency"].values())
            return 0 if okay else 2
        finally:
            engine.close()
    except Exception as exc:
        print(f"RepoMind error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
