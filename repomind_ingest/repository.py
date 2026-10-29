from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse
import logging

log = logging.getLogger(__name__)

# Avoid indexing generated, binary, dependency, or secret-bearing paths.
EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build",
    "target", ".venv", "venv", "__pycache__", ".next", ".turbo",
    "coverage", ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
}
EXCLUDED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip",
    ".gz", ".jar", ".class", ".pyc", ".so", ".dll", ".exe", ".bin",
    ".lock", ".min.js", ".min.css", ".map", ".woff", ".woff2", ".ttf",
    ".pem", ".key", ".p12", ".pfx", ".sqlite", ".db",
}
EXCLUDED_NAMES = {
    ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519",
    "credentials.json", "secrets.yaml", "secrets.yml",
}
# Java is the only structural source language in the current RepoMind scope.
# Non-Java files are indexed only when they are useful repository artifacts.
JAVA_EXTENSIONS = {".java"}
ARTIFACT_EXTENSIONS = {
    ".md", ".mdx", ".rst", ".txt", ".xml", ".yaml", ".yml", ".toml",
    ".json", ".properties", ".gradle", ".kts", ".sql", ".sh", ".bash",
}
SUPPORTED_EXTENSIONS = JAVA_EXTENSIONS | ARTIFACT_EXTENSIONS

TEXT_FILENAMES = {"Dockerfile", "Makefile", "justfile", "Jenkinsfile", "Procfile"}

def _git(args: list[str], cwd: str | None = None) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
        timeout=120, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return completed.stdout.strip()

def validate_repo_url(url: str) -> None:
    parsed = urlparse(url)
    allowed_hosts = {"github.com", "gitlab.com", "bitbucket.org"}
    if url.startswith("git@") and ":" in url:
        host = url[4:].split(":", 1)[0].lower()
        if host not in allowed_hosts:
            raise ValueError("Remote host is not on the configured Git host allowlist.")
        return
    if parsed.scheme not in {"https", "ssh"}:
        raise ValueError("Remote repositories must use HTTPS or SSH Git URLs.")
    if parsed.hostname not in allowed_hosts:
        raise ValueError("Remote host is not on the configured Git host allowlist.")

def checkout_repository(source: str, ref: str | None = None) -> tuple[str, str | None]:
    """Return (checkout_path, temp_directory_to_delete_or_None)."""
    source_path = Path(source).expanduser().resolve()
    if source_path.exists() and source_path.is_dir():
        # Always index a clean, immutable Git revision, never the caller's mutable worktree.
        target_commit = _git(["rev-parse", ref or "HEAD"], cwd=str(source_path))
        temp_dir = tempfile.mkdtemp(prefix="repomind-local-repo-")
        try:
            _git(["clone", "--no-hardlinks", "--no-checkout", str(source_path), temp_dir])
            _git(["checkout", "--detach", target_commit], cwd=temp_dir)
            return temp_dir, temp_dir
        except Exception:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    validate_repo_url(source)
    temp_dir = tempfile.mkdtemp(prefix="repomind-ingest-")
    try:
        # Shallow clone reduces network transfer. Fetch exact ref if requested.
        _git(["clone", "--no-tags", "--filter=blob:none", "--depth=1", source, temp_dir])
        if ref:
            try:
                _git(["fetch", "--depth=1", "origin", ref], cwd=temp_dir)
                _git(["checkout", "--detach", "FETCH_HEAD"], cwd=temp_dir)
            except subprocess.CalledProcessError:
                _git(["fetch", "--no-tags", "origin", ref], cwd=temp_dir)
                _git(["checkout", "--detach", ref], cwd=temp_dir)
        commit = _git(["rev-parse", "HEAD"], cwd=temp_dir)
        return temp_dir, temp_dir
    except Exception:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

def discover_files(root: str, *, max_files: int, max_file_bytes: int) -> tuple[list[Path], int, list[str]]:
    root_path = Path(root).resolve()
    files: list[Path] = []
    skipped = 0
    warnings: list[str] = []
    for current, dirs, names in os.walk(root_path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS and not (Path(current) / d).is_symlink())
        for name in sorted(names):
            path = Path(current) / name
            rel = path.relative_to(root_path).as_posix()
            if name in EXCLUDED_NAMES or name.startswith(".env.") or path.suffix.lower() in EXCLUDED_SUFFIXES:
                skipped += 1
                continue
            if name not in TEXT_FILENAMES and path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped += 1
                continue
            try:
                if path.is_symlink() or path.stat().st_size > max_file_bytes:
                    skipped += 1
                    continue
                # NUL-byte heuristic for binary files.
                with path.open("rb") as f:
                    sample = f.read(8192)
                if b"\x00" in sample:
                    skipped += 1
                    continue
            except OSError:
                skipped += 1
                continue
            files.append(path)
            if len(files) >= max_files:
                warnings.append(f"File limit reached ({max_files}); remaining files were not indexed.")
                return files, skipped, warnings
    return files, skipped, warnings

def language_for_path(path: str) -> str:
    """Return the retrieval language for Java source and supported text artifacts only."""
    ext = Path(path).suffix.lower()
    mapping = {
        ".java": "java", ".md": "markdown", ".mdx": "markdown", ".rst": "rst",
        ".txt": "text", ".xml": "xml", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".json": "json", ".properties": "properties",
        ".gradle": "groovy", ".kts": "kotlin", ".sql": "sql",
        ".sh": "bash", ".bash": "bash",
    }
    name = Path(path).name
    if name in {"Dockerfile", "Makefile", "Jenkinsfile", "Procfile"}:
        return "text"
    return mapping.get(ext, "text")


def category_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".java":
        return "java_source"
    name = Path(path).name
    if name in {"Dockerfile", "Makefile", "Jenkinsfile", "Procfile"}:
        return "build_or_runtime_config"
    if suffix in {".xml", ".gradle", ".kts", ".properties", ".yaml", ".yml", ".toml", ".json", ".sql", ".sh", ".bash"}:
        return "configuration"
    if suffix in {".md", ".mdx", ".rst", ".txt"}:
        return "documentation"
    return "repository_artifact"
