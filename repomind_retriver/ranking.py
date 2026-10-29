from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    result_sets: list[list[dict[str, Any]]],
    *,
    k: int = 60,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """RRF is robust when source scores (BM25, cosine, graph) are not comparable."""
    fused: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = defaultdict(float)
    source_ranks: dict[str, list[dict[str, int]]] = defaultdict(list)

    for results in result_sets:
        for rank, item in enumerate(results, start=1):
            details = item.get("details") or {}
            # PostgreSQL lexical and Qdrant vector hits for the same chunk have
            # different source-specific evidence IDs. Fuse them using the shared
            # chunk ID while retaining all source IDs as provenance.
            chunk_id = details.get("chunk_id") or item.get("chunk_id")
            key = f"chunk:{chunk_id}" if chunk_id is not None else str(item.get("evidence_id") or item.get("id"))
            if key in {"None", ""}:
                continue
            scores[key] += 1.0 / (k + rank)
            source_ranks[key].append({"rank": rank})
            if key not in fused:
                fused[key] = dict(item)
                fused[key]["details"] = dict(details)
                fused[key]["details"]["source_evidence_ids"] = [item.get("evidence_id")]
                fused[key]["details"]["retrieval_sources"] = [item.get("source_type")]
            else:
                # Preserve source provenance while keeping the first non-empty claim.
                current_details = dict(fused[key].get("details") or {})
                current_details.update(details)
                current_details.setdefault("source_evidence_ids", []).append(item.get("evidence_id"))
                current_details.setdefault("retrieval_sources", []).append(item.get("source_type"))
                fused[key]["details"] = current_details
                if not fused[key].get("claim") and item.get("claim"):
                    fused[key]["claim"] = item["claim"]

    output = []
    for key, item in fused.items():
        item["rank_score"] = scores[key]
        item.setdefault("details", {})["source_ranks"] = source_ranks[key]
        item["details"]["retrieval_sources"] = list(dict.fromkeys(item["details"].get("retrieval_sources", [])))
        item["details"]["source_evidence_ids"] = list(dict.fromkeys(item["details"].get("source_evidence_ids", [])))
        output.append(item)
    return sorted(output, key=lambda x: x["rank_score"], reverse=True)[:limit]
