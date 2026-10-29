"""Adapter for codeanalyzer-java (canjv) and its Neo4j graph output.

The analyzer's raw graph is treated as an external schema. This module normalizes
only facts supported by node labels, properties, and observed relationship types;
it does not infer missing compiler facts from names.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver
else:
    Driver = Any

from .config import settings
from .models import GraphEdge, Symbol

log = logging.getLogger(__name__)

SYMBOL_LABELS = {"JSymbol", "JField", "JRecordComponent"}
ANNOTATION_LABELS = {"JAnnotation"}


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8", errors="replace")).hexdigest()[:32]


def _first(props: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = props.get(key)
        if value is not None and value != "":
            return value
    return default


def _relative_path(raw_path: Any, repo_root: Path, known_paths: set[str]) -> str:
    if not raw_path:
        return ""
    value = str(raw_path).replace("\\", "/")
    # Analyzer versions may emit absolute paths, root-relative paths, or URI-like values.
    try:
        candidate = Path(value).resolve()
        root = repo_root.resolve()
        if candidate.is_relative_to(root):
            rel = candidate.relative_to(root).as_posix()
            if rel in known_paths:
                return rel
    except (OSError, ValueError):
        pass
    value = value.removeprefix("file://")
    value = value.lstrip("./")
    if value in known_paths:
        return value
    matches = [p for p in known_paths if value.endswith("/" + p) or value == p]
    if len(matches) == 1:
        return matches[0]
    # Keep only a safe relative-looking suffix; never expose local absolute paths in retrieval.
    parts = [p for p in value.split("/") if p and p not in {".", ".."}]
    for i in range(len(parts)):
        suffix = "/".join(parts[i:])
        if suffix in known_paths:
            return suffix
    return ""


class CodeAnalyzerJava:
    def __init__(self, driver: Driver):
        self.driver = driver

    def analyze(self, repo_root: str, repo_name: str, commit: str, generation: str,
                java_paths: list[str]) -> tuple[list[Symbol], list[GraphEdge], dict[str, Any]]:
        if not java_paths:
            return [], [], {"analyzer": "codeanalyzer-java", "app_name": None, "raw_nodes": 0, "raw_edges": 0}
        app_name = self._app_name(repo_name, generation)
        self._run_cli(repo_root, app_name)
        nodes, relationships = self._read_app_graph(app_name)
        known_paths = set(java_paths)
        path_by_stem: dict[str, list[str]] = {}
        path_by_declared_type: dict[str, list[str]] = {}
        package_by_path: dict[str, str] = {}
        for path in java_paths:
            path_by_stem.setdefault(Path(path).stem.casefold(), []).append(path)
            try:
                source_text = (Path(repo_root) / path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                source_text = ""
            package_match = re.search(r"(?m)^\s*package\s+([\w.]+)\s*;", source_text)
            package_by_path[path] = package_match.group(1) if package_match else ""
            for type_match in re.finditer(r"\b(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)", source_text):
                path_by_declared_type.setdefault(type_match.group(1).casefold(), []).append(path)
        resolved_paths: dict[str, str] = {}
        for raw_id, labels, props in nodes:
            direct = _relative_path(_first(props, "file_path", "filePath", "path", "source_path", "source_file", "sourceFile", "file", "filename", "uri"), Path(repo_root), known_paths)
            if direct:
                resolved_paths[raw_id] = direct
        # Some analyzer graph versions keep source-file ownership on an Artifact/owner edge
        # rather than duplicating the path on every symbol. Resolve type nodes by Java filename,
        # then inherit a source path over observed ownership relationships.
        for raw_id, labels, props in nodes:
            if raw_id in resolved_paths or "JType" not in set(labels):
                continue
            type_name = str(_first(props, "name", "simple_name", "simpleName", default=""))
            type_name = type_name.rsplit(".", 1)[-1]
            candidates = path_by_stem.get(type_name.casefold(), []) or path_by_declared_type.get(type_name.casefold(), [])
            if len(candidates) > 1:
                qname = str(_first(props, "qualified_name", "qualifiedName", "fully_qualified_name", "fullyQualifiedName", "qname", default=""))
                package = qname.rsplit(".", 1)[0] if "." in qname else ""
                package_matches = [candidate for candidate in candidates if package_by_path.get(candidate, "") == package]
                if len(package_matches) == 1:
                    candidates = package_matches
            if len(candidates) == 1:
                resolved_paths[raw_id] = candidates[0]
        owner_by_child: dict[str, set[str]] = {}
        for source_raw, rel_type, target_raw, _ in relationships:
            if rel_type in {"J_HAS_METHOD", "J_HAS_FIELD", "J_HAS_BODY_NODE"}:
                owner_by_child.setdefault(target_raw, set()).add(source_raw)
        for _ in range(3):
            changed = False
            for raw_id, labels, props in nodes:
                if raw_id in resolved_paths:
                    continue
                candidates = {resolved_paths[parent] for parent in owner_by_child.get(raw_id, set()) if parent in resolved_paths}
                if not candidates:
                    qname = str(_first(props, "qualified_name", "qualifiedName", "fully_qualified_name", "fullyQualifiedName", "qname", default=""))
                    before_params = qname.split("(", 1)[0]
                    owner_name = before_params.rsplit(".", 1)[0].rsplit(".", 1)[-1] if "." in before_params else before_params
                    candidates = set(path_by_stem.get(owner_name.casefold(), []))
                if len(candidates) == 1:
                    resolved_paths[raw_id] = next(iter(candidates))
                    changed = True
            if not changed:
                break
        raw_to_symbol: dict[str, str] = {}
        symbols: list[Symbol] = []
        symbol_ids_seen: set[str] = set()
        for raw_id, labels, props in nodes:
            label_set = set(labels)
            if not (label_set & SYMBOL_LABELS or label_set & ANNOTATION_LABELS):
                continue
            if "JExternal" in label_set:
                continue
            path = resolved_paths.get(raw_id) or _relative_path(_first(props, "file_path", "filePath", "path", "source_path", "source_file", "sourceFile", "file", "filename", "uri"), Path(repo_root), known_paths)
            name = str(_first(props, "name", "simple_name", "simpleName", "signature", default="<unknown>")).strip()
            signature = _first(props, "signature", "declaration")
            kind = self._kind(labels, props)
            start = self._safe_int(_first(props, "start_line", "startLine", "line", default=1), 1)
            end = self._safe_int(_first(props, "end_line", "endLine", default=start), start)
            if end < start:
                end = start
            # Java source symbols need a mapped source path for pinned citations. Annotation nodes
            # are the exception: analyzer versions often expose only their names, but their edges matter.
            is_annotation = "JAnnotation" in label_set
            if (not path and not is_annotation) or name == "<unknown>":
                # Analyzer annotation nodes often only carry {name}; retain them as graph concepts
                # so J_ANNOTATED_BY edges survive normalization. They are never used as source chunks.
                if not is_annotation or name == "<unknown>":
                    continue
                path = ""
            qualified = str(_first(props, "qualified_name", "qualifiedName", "fully_qualified_name", "fullyQualifiedName", "qname", default=name))
            identity_parts = [repo_name, commit, generation, path, kind, qualified, str(signature or ""), str(start)]
            if is_annotation:
                identity_parts.append(raw_id)  # source-less annotation concepts may repeat by name
            sid = stable_id(*identity_parts)
            raw_to_symbol[raw_id] = sid
            if sid in symbol_ids_seen:
                continue
            symbol_ids_seen.add(sid)
            modifiers = props.get("modifiers", [])
            if not isinstance(modifiers, list):
                modifiers = [str(modifiers)] if modifiers else []
            source_code = _first(props, "code", "source_code", "declaration")
            metadata = {k: v for k, v in props.items() if k not in {
                "code", "source_code", "declaration", "name", "qualified_name", "signature",
                "file_path", "filePath", "path", "source_path", "source_file", "sourceFile", "file", "filename", "uri", "start_line", "end_line",
                "modifiers", "id", "kind", "type", "language"
            }}
            is_test = ("test" in path.lower() or "tests" in path.lower() or
                       name.lower().endswith("test") or name.lower().startswith("test"))
            symbols.append(Symbol(
                id=sid, qualified_name=qualified, name=name, kind=kind, file_path=path,
                start_line=start, end_line=end, language="java", is_test=is_test,
                signature=str(signature) if signature is not None else None,
                source_code=str(source_code) if source_code is not None else None,
                modifiers=[str(m) for m in modifiers], metadata=metadata, raw_node_id=raw_id,
            ))

        edges: list[GraphEdge] = []
        dedupe: set[tuple[str, str, str]] = set()
        for source_raw, rel_type, target_raw, rel_props in relationships:
            source_id = raw_to_symbol.get(source_raw)
            target_id = raw_to_symbol.get(target_raw)
            if not source_id or not target_id or source_id == target_id:
                continue
            kind = self._edge_kind(rel_type)
            if kind is None:
                continue
            key = (source_id, target_id, kind)
            if key in dedupe:
                continue
            dedupe.add(key)
            prov = rel_props.get("prov", rel_props.get("provenance", []))
            if isinstance(prov, str):
                prov = [prov]
            edges.append(GraphEdge(source_id, target_id, kind,
                                   [str(p) for p in (prov or [])],
                                   {k: v for k, v in rel_props.items() if k not in {"prov", "provenance"}}))

        # Build a deterministic count summary, keeping raw and normalized counts separate.
        stats = {
            "analyzer": "codeanalyzer-java", "app_name": app_name,
            "raw_nodes": len(nodes), "raw_edges": len(relationships),
            "normalized_symbols": len(symbols), "normalized_edges": len(edges),
            "labels_seen": sorted({label for _, labels, _ in nodes for label in labels}),
            "relationship_types_seen": sorted({rel for _, rel, _, _ in relationships}),
        }
        if not any(symbol.file_path and symbol.kind != "annotation" for symbol in symbols):
            raise RuntimeError(
                "codeanalyzer-java completed but no source-grounded Java symbols were normalized. "
                "Inspect analyzer graph properties (especially source path fields) before publishing."
            )
        return symbols, edges, stats

    @staticmethod
    def _app_name(repo_name: str, generation: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", repo_name).strip("-")[:40] or "repository"
        return f"repomind-{slug}-{generation[:12]}"

    @staticmethod
    def _run_cli(repo_root: str, app_name: str) -> None:
        command = os.getenv("CANJV_COMMAND", settings.canjv_command)
        args = [command, "-i", repo_root, "--emit", "neo4j", "--app-name", app_name,
                "--neo4j-uri", settings.neo4j_uri, "--neo4j-user", settings.neo4j_username]
        env = os.environ.copy()
        env["NEO4J_URI"] = settings.neo4j_uri
        env["NEO4J_USERNAME"] = settings.neo4j_username
        env["NEO4J_USER"] = settings.neo4j_username
        env["NEO4J_PASSWORD"] = settings.neo4j_password
        timeout = settings.analyzer_timeout_seconds
        log.info("Running codeanalyzer-java for %s", app_name)
        try:
            completed = subprocess.run(args, cwd=repo_root, env=env, capture_output=True, text=True,
                                       timeout=timeout, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not find analyzer executable '{command}'. Install codeanalyzer-java and ensure canjv is on PATH, "
                "or set CANJV_COMMAND to its executable path."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"codeanalyzer-java exceeded timeout of {timeout} seconds") from exc
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip()[-6000:]
            raise RuntimeError(f"codeanalyzer-java failed (exit={completed.returncode}):\n{tail}")
        log.debug("codeanalyzer-java output: %s", (completed.stdout or "").strip()[-2000:])

    def _read_app_graph(self, app_name: str) -> tuple[list[tuple[str, list[str], dict[str, Any]]], list[tuple[str, str, str, dict[str, Any]]]]:
        """Read the app's connected subgraph using bounded BFS to avoid path explosion."""
        with self.driver.session() as session:
            app = session.run("MATCH (a:JApplication {name:$name}) RETURN elementId(a) AS id", name=app_name).single()
            if not app:
                raise RuntimeError(f"Analyzer returned success but Neo4j has no JApplication named {app_name!r}")
            seen = {app["id"]}
            frontier = [app["id"]]
            for _depth in range(max(1, settings.analyzer_graph_depth)):
                if not frontier or len(seen) >= settings.analyzer_max_graph_nodes:
                    break
                rows = session.run("""
                    UNWIND $ids AS id
                    MATCH (n) WHERE elementId(n)=id
                    MATCH (n)-[r]-(m)
                    RETURN DISTINCT elementId(m) AS neighbor, labels(m) AS labels,
                           properties(m) AS props
                """, ids=frontier)
                next_frontier = []
                for row in rows:
                    neighbor = row["neighbor"]
                    labels = set(row["labels"] or [])
                    props = row["props"] or {}
                    if not neighbor or neighbor in seen:
                        continue
                    # Shared annotation/external nodes can be linked to more than one application.
                    # Include them as terminal evidence, but never traverse through them to another app.
                    if "JApplication" in labels and props.get("name") != app_name:
                        continue
                    seen.add(neighbor)
                    if not (labels & {"JExternal", "JAnnotation"}):
                        next_frontier.append(neighbor)
                    if len(seen) >= settings.analyzer_max_graph_nodes:
                        break
                frontier = next_frontier
            if len(seen) >= settings.analyzer_max_graph_nodes:
                log.warning("Analyzer graph reached node cap %d for %s", settings.analyzer_max_graph_nodes, app_name)
            ids = list(seen)
            nodes: list[tuple[str, list[str], dict[str, Any]]] = []
            for offset in range(0, len(ids), 1000):
                batch = ids[offset:offset + 1000]
                for row in session.run("""
                    UNWIND $ids AS id
                    MATCH (n) WHERE elementId(n)=id
                    RETURN elementId(n) AS raw_id, labels(n) AS labels, properties(n) AS props
                """, ids=batch):
                    nodes.append((row["raw_id"], list(row["labels"]), dict(row["props"])))
            relationships: list[tuple[str, str, str, dict[str, Any]]] = []
            for offset in range(0, len(ids), 500):
                batch = ids[offset:offset + 500]
                for row in session.run("""
                    MATCH (a)-[r]->(b)
                    WHERE elementId(a) IN $ids
                    RETURN elementId(a) AS source_id, type(r) AS rel_type,
                           elementId(b) AS target_id, properties(r) AS props
                """, ids=batch):
                    relationships.append((row["source_id"], row["rel_type"], row["target_id"], dict(row["props"])))
        return nodes, relationships

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _kind(labels: list[str], props: dict[str, Any]) -> str:
        label_set = set(labels)
        if "JAnnotation" in label_set:
            return "annotation"
        if "JField" in label_set:
            return "field"
        if "JRecordComponent" in label_set:
            return "record_component"
        if "JCallable" in label_set:
            raw_kind = str(props.get("kind", "method")).lower()
            if "constructor" in raw_kind or "<init>" in str(props.get("signature", "")):
                return "constructor"
            return "method"
        if "JType" in label_set:
            raw_kind = str(props.get("kind", "type")).lower()
            for candidate in ("interface", "enum", "record", "annotation", "class"):
                if candidate in raw_kind or candidate in label_set:
                    return candidate
            return "type"
        return str(props.get("kind", "symbol")).lower()

    @staticmethod
    def _edge_kind(rel_type: str) -> str | None:
        mapping = {
            "J_CALLS": "CALLS",
            "J_HAS_METHOD": "CONTAINS",
            "J_HAS_FIELD": "CONTAINS",
            "J_ANNOTATED_BY": "ANNOTATED_BY",
            "J_HAS_BODY_NODE": "HAS_BODY_NODE",
            "J_HAS_MODULE": "HAS_MODULE",
            "DEFINES_CONFIG": "DEFINES_CONFIG",
        }
        return mapping.get(rel_type)
