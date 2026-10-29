from __future__ import annotations

import re
from typing import Iterable

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")


class AuthorizationError(PermissionError):
    pass


def normalize_repository(value: str) -> str:
    value = value.strip()
    if not _REPO_RE.fullmatch(value):
        raise ValueError("Repository must use the owner/repository format.")
    owner, repo = value.split("/", 1)
    return f"{owner}/{repo}"


def authorize_repository(repository: str, allowed_repositories: Iterable[str] | None) -> str:
    """Fail closed when an explicit allowlist is supplied; never trust the planner as auth."""
    normalized = normalize_repository(repository)
    if allowed_repositories is None:
        raise AuthorizationError("No repository authorization context was provided.")
    allowed = {normalize_repository(x).casefold() for x in allowed_repositories}
    if normalized.casefold() not in allowed:
        raise AuthorizationError(f"Access to repository {normalized!r} is not authorized.")
    return normalized


def validate_line_range(start_line: int | None, end_line: int | None) -> tuple[int, int]:
    start = max(1, int(start_line or 1))
    end = max(start, int(end_line or (start + 79)))
    # Prevent unexpectedly huge source excerpts.
    end = min(end, start + 159)
    return start, end
