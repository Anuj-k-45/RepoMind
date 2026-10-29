from __future__ import annotations

from typing import Any, Literal, TypedDict, Annotated
from operator import add
from pydantic import BaseModel, Field, ConfigDict


Intent = Literal[
    "symbol_lookup", "dependency_analysis", "test_discovery",
    "architecture_question", "code_explanation", "general_repository_question",
]


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Intent
    repository_full_name: str = Field(min_length=3, max_length=200)
    target_symbol: str | None = Field(default=None, max_length=300)
    graph_hops: int = Field(default=2, ge=1, le=4)
    use_graph: bool = True
    use_vector: bool = True
    use_lexical: bool = True
    use_metadata: bool = True
    search_queries: list[str] = Field(default_factory=list, max_length=5)
    max_results: int = Field(default=8, ge=1, le=20)


class SufficiencyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sufficient: bool
    missing_evidence: list[str] = Field(default_factory=list, max_length=8)
    reason: str = Field(max_length=1200)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    citation_ids: list[str] = Field(default_factory=list, max_length=50)
    caveats: list[str] = Field(default_factory=list, max_length=10)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")
    evidence_id: str
    source_type: str
    claim: str
    repository_full_name: str
    commit_sha: str | None = None
    index_generation: str | None = None
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    rank_score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RepoMindState(TypedDict, total=False):
    question: str
    user_id: str
    allowed_repositories: list[str]
    requested_repository: str | None
    default_repository: str
    plan: dict[str, Any]
    repository_full_name: str
    repository_id: int
    context_error: str
    indexed_commit_sha: str
    index_generation: str
    target_symbol: str | None
    target_symbol_id: str | None
    target_file_path: str | None
    retrieval_queries: list[str]
    retrieval_round: int
    metadata_hits: list[dict[str, Any]]
    graph_hits: list[dict[str, Any]]
    vector_hits: list[dict[str, Any]]
    lexical_hits: list[dict[str, Any]]
    github_hits: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    evidence_sufficient: bool
    missing_evidence: list[str]
    sufficiency_reason: str
    answer: str
    citation_ids: list[str]
    final_response: str
    errors: Annotated[list[str], add]
    investigation_steps: list[str]
    prior_evidence_ids: list[str]
    evidence_history: Annotated[list[dict[str, Any]], add]
    source_history: Annotated[list[dict[str, Any]], add]
    novel_evidence_count: int
    no_progress_rounds: int
