import pytest

from repomind_retriver.security import authorize_repository, AuthorizationError, validate_line_range
from repomind_retriver.ranking import reciprocal_rank_fusion


def test_repository_allowlist_is_case_insensitive():
    assert authorize_repository("Owner/Repo", ["owner/repo"]) == "Owner/Repo"


def test_repository_authorization_fails_closed():
    with pytest.raises(AuthorizationError):
        authorize_repository("owner/private", ["owner/public"])
    with pytest.raises(AuthorizationError):
        authorize_repository("owner/public", None)


def test_repository_name_validation():
    with pytest.raises(ValueError):
        authorize_repository("../etc/passwd", ["../etc/passwd"])


def test_line_range_is_bounded():
    start, end = validate_line_range(10, 10000)
    assert start == 10
    assert end <= 169


def test_rrf_promotes_results_present_in_multiple_lists():
    a = {"evidence_id": "A", "source_type": "lexical", "claim": "A"}
    b = {"evidence_id": "B", "source_type": "vector", "claim": "B"}
    fused = reciprocal_rank_fusion([[a, b], [a]], limit=5)
    assert fused[0]["evidence_id"] == "A"
    assert fused[0]["rank_score"] > fused[1]["rank_score"]


def test_rrf_fuses_same_chunk_across_sources():
    lexical = {"evidence_id": "pg-chunk:42", "source_type": "postgres_lexical", "claim": "text", "details": {"chunk_id": "42"}}
    vector = {"evidence_id": "qdrant:42", "source_type": "qdrant_vector", "claim": "text", "details": {"chunk_id": "42"}}
    fused = reciprocal_rank_fusion([[lexical], [vector]], limit=5)
    assert len(fused) == 1
    assert set(fused[0]["details"]["retrieval_sources"]) == {"postgres_lexical", "qdrant_vector"}


def test_fallback_plan_extracts_explicit_target_for_graph_queries():
    from repomind_retriver.planning import fallback_plan

    plan = fallback_plan("Which tests call `HelloWorld`?", "owner/repo")
    assert plan.intent == "test_discovery"
    assert plan.target_symbol == "HelloWorld"
    assert plan.repository_full_name == "owner/repo"
    assert plan.use_graph and plan.use_vector and plan.use_lexical


def test_fallback_plan_does_not_guess_a_target_from_general_prose():
    from repomind_retriver.planning import fallback_plan

    plan = fallback_plan("Explain the overall architecture", "owner/repo")
    assert plan.intent == "architecture_question"
    assert plan.target_symbol is None
