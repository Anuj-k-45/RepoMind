from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from .config import settings
from .graph import RepoMind
from .security import normalize_repository

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("repomind.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    repository: str | None = Field(default=None, min_length=3, max_length=201)


class AskResponse(BaseModel):
    answer: str
    repository: str | None = None
    commit_sha: str | None = None
    index_generation: str | None = None
    evidence_count: int = 0
    citation_ids: list[str] = Field(default_factory=list)
    evidence_sufficient: bool = False
    missing_evidence: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


def _authorize_request(request: Request, x_api_key: str | None) -> None:
    configured = (settings.api_key or "").strip()
    if configured:
        if not x_api_key or not hmac.compare_digest(x_api_key, configured):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
        return
    if settings.environment.casefold() != "development":
        raise HTTPException(status_code=503, detail="REPOMIND_API_KEY must be configured outside development.")
    if not _is_loopback(request):
        raise HTTPException(status_code=403, detail="Unauthenticated API access is restricted to localhost in development.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep the embedding model and database clients warm for the process lifetime.
    app.state.repomind = RepoMind()
    try:
        yield
    finally:
        app.state.repomind.close()


app = FastAPI(
    title="RepoMind Retrieval API",
    version="0.3.0",
    description="Hybrid repository retrieval over PostgreSQL, Neo4j, and Qdrant.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "repomind-retrieval"}


@app.get("/ready")
def readiness(request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize_request(request, x_api_key)
    engine: RepoMind = request.app.state.repomind
    checks: dict[str, Any] = {}
    for name, fn in (
        ("postgres", engine.pg.health),
        ("neo4j", engine.neo4j.health),
        ("qdrant", engine.qdrant.health),
    ):
        try:
            checks[name] = fn()
        except Exception as exc:
            log.warning("Readiness check failed for %s (%s)", name, type(exc).__name__)
            checks[name] = {"ok": False, "error": type(exc).__name__}
    okay = all(value is True or (isinstance(value, dict) and value.get("status") in {"green", "ok", "healthy"})
               for value in checks.values())
    if not okay:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}


@app.get("/repositories")
def repositories(request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize_request(request, x_api_key)
    allowlist = list(settings.authorized_repositories)
    if not allowlist:
        raise HTTPException(status_code=503, detail="Set AUTHORIZED_REPOSITORIES in the server environment.")
    engine: RepoMind = request.app.state.repomind
    results = []
    for repository in allowlist:
        try:
            info = engine.pg.diagnostics(normalize_repository(repository))
            results.append({"repository": info["repository"]["full_name"],
                            "commit_sha": info["repository"]["indexed_commit_sha"],
                            "index_generation": info["repository"]["index_generation"],
                            "counts": info["counts"]})
        except Exception as exc:
            results.append({"repository": repository, "error": type(exc).__name__})
    return {"repositories": results}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request,
        x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize_request(request, x_api_key)
    allowlist = list(settings.authorized_repositories)
    if not allowlist:
        raise HTTPException(status_code=503, detail="Set AUTHORIZED_REPOSITORIES in the server environment.")
    # The allowlist comes from server configuration, never from the request or model.
    try:
        repository = normalize_repository(body.repository) if body.repository else normalize_repository(allowlist[0])
        if repository.casefold() not in {normalize_repository(r).casefold() for r in allowlist}:
            raise HTTPException(status_code=403, detail="Repository is not in the server-side allowlist.")
        engine: RepoMind = request.app.state.repomind
        return engine.answer_question(
            body.question, user_id=("api-key-user" if settings.api_key else "local-user"),
            allowed_repositories=allowlist, repository=repository,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Retrieval request failed (%s)", type(exc).__name__)
        # Do not expose SQL, provider errors, or stack traces to API clients.
        raise HTTPException(status_code=500, detail="Retrieval failed. Check server logs and /ready.") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("repomind.api:app", host=settings.api_host, port=settings.api_port)
