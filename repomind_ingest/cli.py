from __future__ import annotations
import argparse
import json
import logging
import sys
from .pipeline import IngestionPipeline

def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a Git repository into RepoMind stores.")
    parser.add_argument("source", help="Local repository path or approved HTTPS/SSH Git URL")
    parser.add_argument("--repository", help="Canonical owner/repository name (required for local paths)")
    parser.add_argument("--ref", help="Git branch, tag, or commit to ingest")
    parser.add_argument("--cleanup-old", action="store_true",
                        help="Delete previous generations after publication; unsafe with in-flight readers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    pipeline = IngestionPipeline()
    try:
        result = pipeline.run(args.source, repository_full_name=args.repository,
                              ref=args.ref, cleanup_old=args.cleanup_old)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        logging.exception("Ingestion failed: %s", type(exc).__name__)
        print(f"Ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        pipeline.close()

if __name__ == "__main__":
    raise SystemExit(main())
