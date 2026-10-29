from __future__ import annotations

import re

from .models import RetrievalPlan


def fallback_plan(question: str, repository: str, *, max_results: int = 8) -> RetrievalPlan:
    """Small deterministic planner used when Groq is not configured or unavailable."""
    q = question.strip()
    lowered = q.casefold()
    if any(word in lowered for word in ("test", "regression", "coverage")):
        intent = "test_discovery"
    elif any(word in lowered for word in ("depend", "impact", "caller", "calls", "call graph", "affected")):
        intent = "dependency_analysis"
    elif any(word in lowered for word in ("architecture", "module", "overall design", "structure")):
        intent = "architecture_question"
    elif any(word in lowered for word in ("what is", "explain", "how does", "how do")):
        intent = "code_explanation"
    else:
        intent = "general_repository_question"

    # Prefer explicit code formatting or clear symbol phrases over arbitrary capitalized words.
    match = re.search(r"`([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)`", q)
    if not match:
        match = re.search(r"\b(?:class|method|symbol|function)\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)", q, re.I)
    if not match:
        match = re.search(
            r"\b(?i:calls?|callers? of|around|impact of|tests? for|dependenc(?:y|ies) of)\s+([A-Z][A-Za-z0-9_$]*)",
            q,
        )
    target = match.group(1) if match else None
    return RetrievalPlan(
        intent=intent,
        repository_full_name=repository,
        target_symbol=target,
        graph_hops=2,
        use_graph=True,
        use_vector=True,
        use_lexical=True,
        use_metadata=True,
        search_queries=[q],
        max_results=max(1, min(8, max_results)),
    )
