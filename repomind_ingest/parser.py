"""Source chunking helpers. Java structure comes from codeanalyzer-java, not text parsing."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import CodeChunk, Symbol
from .repository import language_for_path


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8", errors="replace")).hexdigest()[:32]


def _estimated_tokens(line: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", line, flags=re.UNICODE))) if line.strip() else 0


def _split_line_range(lines: list[str], start: int, end: int, *, max_tokens: int,
                      max_lines: int, overlap: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor <= end:
        stop = cursor - 1
        token_count = 0
        while stop + 1 <= end and (stop + 1 - cursor) < max_lines:
            candidate = _estimated_tokens(lines[stop])
            if stop >= cursor and token_count + candidate > max_tokens:
                break
            token_count += candidate
            stop += 1
            if token_count >= max_tokens:
                break
        if stop < cursor:
            stop = cursor
        ranges.append((cursor, stop))
        if stop >= end:
            break
        cursor = max(cursor + 1, stop - overlap + 1)
    return ranges


def fallback_line_chunks(rel_path: str, text: str, max_lines: int = 70,
                         overlap: int = 8, max_tokens: int = 450) -> list[CodeChunk]:
    lines = text.splitlines()
    chunks: list[CodeChunk] = []
    if not lines:
        return chunks
    for idx, (start, end) in enumerate(_split_line_range(
        lines, 1, len(lines), max_tokens=max_tokens, max_lines=max_lines, overlap=overlap
    )):
        body = "\n".join(lines[start - 1:end])
        if body.strip():
            chunks.append(CodeChunk(stable_id(rel_path, str(start), str(end), body), rel_path,
                start, end, body, language_for_path(rel_path), chunk_index=idx))
    return chunks


def chunk_file(rel_path: str, text: str, symbols: list[Symbol], max_lines: int = 70,
               overlap: int = 8, max_tokens: int = 450) -> list[CodeChunk]:
    """Chunk by supplied analyzer source ranges, then cover all unindexed lines.

    Callers should pass non-overlapping/leaf symbol ranges where possible. This function
    deliberately does not parse code or infer relationships; source structure is external.
    """
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[CodeChunk] = []
    covered: list[tuple[int, int]] = []
    for sym in sorted(symbols, key=lambda s: (s.start_line, s.end_line)):
        if not sym.file_path or sym.file_path != rel_path or sym.kind == "annotation":
            continue
        start, end = max(1, sym.start_line), min(len(lines), sym.end_line)
        if end < start:
            continue
        for part, (chunk_start, chunk_end) in enumerate(_split_line_range(
            lines, start, end, max_tokens=max_tokens, max_lines=max_lines, overlap=overlap
        )):
            body = "\n".join(lines[chunk_start - 1:chunk_end])
            if body.strip():
                chunk_id = stable_id(rel_path, sym.id, str(chunk_start), str(chunk_end), body)
                chunks.append(CodeChunk(chunk_id, rel_path, chunk_start, chunk_end, body,
                    sym.language, sym.id, sym.qualified_name, part, "java_symbol"))
            covered.append((chunk_start, chunk_end))

    covered.sort()
    uncovered: list[tuple[int, int]] = []
    cursor = 1
    for start, end in covered:
        if start > cursor:
            uncovered.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= len(lines):
        uncovered.append((cursor, len(lines)))
    for gap_start, gap_end in uncovered:
        for chunk_start, chunk_end in _split_line_range(
            lines, gap_start, gap_end, max_tokens=max_tokens, max_lines=max_lines, overlap=overlap
        ):
            body = "\n".join(lines[chunk_start - 1:chunk_end])
            if body.strip():
                chunks.append(CodeChunk(stable_id(rel_path, str(chunk_start), str(chunk_end), body),
                    rel_path, chunk_start, chunk_end, body, language_for_path(rel_path),
                    chunk_index=len(chunks), chunk_type="java_context"))
    if not chunks:
        chunks = fallback_line_chunks(rel_path, text, max_lines=max_lines,
                                      overlap=overlap, max_tokens=max_tokens)
    return sorted(chunks, key=lambda c: (c.start_line, c.end_line, c.id))
