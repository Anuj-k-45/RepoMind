from repomind_ingest.analyzer import CodeAnalyzerJava, _relative_path
from pathlib import Path


def test_analyzer_relationship_normalization():
    assert CodeAnalyzerJava._edge_kind("J_CALLS") == "CALLS"
    assert CodeAnalyzerJava._edge_kind("J_HAS_METHOD") == "CONTAINS"
    assert CodeAnalyzerJava._edge_kind("UNKNOWN_REL") is None


def test_analyzer_kind_normalization():
    assert CodeAnalyzerJava._kind(["JSymbol", "JCallable"], {"kind": "constructor", "signature": "Foo.<init>()"}) == "constructor"
    assert CodeAnalyzerJava._kind(["JSymbol", "JType"], {"kind": "record"}) == "record"
    assert CodeAnalyzerJava._kind(["JAnnotation"], {"name": "GetMapping"}) == "annotation"


def test_analyzer_path_normalization_uses_repository_relative_path():
    root = Path("/tmp/example-repo")
    known = {"src/main/java/demo/App.java"}
    assert _relative_path("/tmp/example-repo/src/main/java/demo/App.java", root, known) == "src/main/java/demo/App.java"
    assert _relative_path("C:\\work\\example-repo\\src\\main\\java\\demo\\App.java", root, known) == "src/main/java/demo/App.java"


def test_normalizer_preserves_source_symbols_and_call_provenance(tmp_path):
    from repomind_ingest.analyzer import CodeAnalyzerJava

    source = tmp_path / "src" / "main" / "java" / "demo" / "Service.java"
    source.parent.mkdir(parents=True)
    source.write_text("package demo;\nclass Service {\n  void run() {\n    new Service();\n  }\n}\n", encoding="utf-8")
    nodes = [
        ("app", ["JApplication", "JCanNode"], {"name": "app"}),
        ("type", ["JSymbol", "JType", "JCanNode"], {"name": "Service", "qualified_name": "demo.Service", "kind": "class", "start_line": 2, "end_line": 6}),
        ("method", ["JSymbol", "JCallable", "JCanNode"], {"name": "run", "qualified_name": "demo.Service.run()", "signature": "run()", "kind": "method", "start_line": 3, "end_line": 5, "code": "void run() { new Service(); }"}),
        ("ctor", ["JSymbol", "JCallable", "JCanNode"], {"name": "Service.<init>", "qualified_name": "demo.Service.<init>()", "signature": "Service()", "kind": "constructor", "start_line": 2, "end_line": 2}),
        ("annotation", ["JAnnotation"], {"name": "Override"}),
    ]
    relationships = [
        ("type", "J_HAS_METHOD", "method", {}),
        ("type", "J_HAS_METHOD", "ctor", {}),
        ("method", "J_ANNOTATED_BY", "annotation", {}),
        ("method", "J_CALLS", "ctor", {"prov": ["rta"]}),
    ]
    adapter = CodeAnalyzerJava(driver=None)
    adapter._run_cli = lambda repo_root, app_name: None
    adapter._read_app_graph = lambda app_name: (nodes, relationships)
    symbols, edges, stats = adapter.analyze(str(tmp_path), "owner/repo", "abc123", "gen123", ["src/main/java/demo/Service.java"])

    by_name = {s.name: s for s in symbols}
    assert by_name["Service"].file_path == "src/main/java/demo/Service.java"
    assert by_name["run"].file_path == "src/main/java/demo/Service.java"
    assert any(s.kind == "annotation" and s.file_path == "" for s in symbols)
    assert any(e.kind == "CALLS" and e.provenance == ["rta"] for e in edges)
    assert any(e.kind == "ANNOTATED_BY" for e in edges)
    assert stats["normalized_symbols"] == len(symbols)
