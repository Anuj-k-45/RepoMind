from repomind_ingest.parser import chunk_file, fallback_line_chunks
from repomind_ingest.models import Symbol
from repomind_ingest.repository import language_for_path, category_for_path, discover_files


def test_java_symbol_chunk_and_context_coverage():
    text = "package demo;\n\nclass Service {\n  void run() {\n    helper();\n  }\n}\n"
    symbol = Symbol("s1", "demo.Service.run()", "run", "method", "src/Service.java", 4, 6)
    chunks = chunk_file("src/Service.java", text, [symbol], max_lines=30, overlap=2)
    assert chunks
    assert any(c.symbol_id == "s1" and "helper();" in c.content for c in chunks)
    assert min(c.start_line for c in chunks) == 1
    assert max(c.end_line for c in chunks) == len(text.splitlines())


def test_artifact_chunks_preserve_ranges_and_content():
    text = "\n".join(f"line_{i}" for i in range(1, 121))
    chunks = fallback_line_chunks("README.md", text, max_lines=30, overlap=5)
    assert chunks
    assert min(c.start_line for c in chunks) == 1
    assert max(c.end_line for c in chunks) == 120
    assert any("line_120" in c.content for c in chunks)


def test_java_only_structural_scope_and_artifact_classification():
    assert language_for_path("src/Main.java") == "java"
    assert category_for_path("src/Main.java") == "java_source"
    assert category_for_path("pom.xml") == "configuration"
    assert category_for_path("README.md") == "documentation"
