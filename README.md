# RepoMind Retrieval Application

RepoMind is the query-time application for the existing RepoMind ingestion/indexing pipeline. It reads the active index generation from PostgreSQL and retrieves repository evidence from PostgreSQL lexical search, Qdrant vector search, and Neo4j code relationships. It fuses evidence, optionally fetches source excerpts from the exact indexed Git commit, evaluates evidence sufficiency, and returns a cited answer.

This package is deliberately **read-only against the three index stores**. It does not reindex, publish, clean up, or delete database data.

## 1. Retrieval flow

```text
Question + server-side repository allowlist
                 |
                 v
      Query planning (Groq or local fallback)
                 |
                 v
 Resolve repository's active commit + generation in PostgreSQL
                 |
       +---------+----------+----------+
       |                    |          |
       v                    v          v
 PostgreSQL metadata   PostgreSQL FTS  Qdrant vector search
       |                    |          |
       +--------------------+----------+
                            |
                            v
               Neo4j graph traversal (when a target symbol resolves)
                            |
                            v
       Reciprocal Rank Fusion + exact generation validation
                            |
                            v
    Optional GitHub source excerpts pinned to indexed commit SHA
                            |
                            v
       Evidence sufficiency check (LLM or deterministic fallback)
                            |
                            v
             Cited answer + sources + caveats
```

### Generation safety

Every search is scoped to the active `repositories.indexed_commit_sha` and `repositories.index_generation`:

- PostgreSQL searches filter on `repository_id`, `commit_sha`, and `index_generation`.
- Qdrant searches filter on `repository_id`, `commit_sha`, and `index_generation` payload fields.
- Neo4j graph traversals filter nodes by `repository_id`, `commit_sha`, and `index_generation`.
- Evidence is checked again before fusion. Evidence from another generation is discarded.

The retrieval code matches the current ingestion contract: PostgreSQL has `repositories`, `repository_files`, `symbols`, and `chunks`; Neo4j has `Symbol` and `Test` nodes with `CALLS`, `CONTAINS`, `ANNOTATED_BY`, and derived `TESTS` relationships; Qdrant payloads contain chunk identifiers and line/file metadata. The graph query uses `CALLS` and `TESTS` rather than assuming an uncreated `DEPENDS_ON` relationship. Symbol resolution excludes rows with empty `file_path` so analyzer-created external references do not masquerade as source definitions.

**Do not run the old generic schema SQL against your populated ingestion database.** The retrieval adapter consumes the schema already created by the ingestion package; it does not need a schema migration.

## 2. Requirements

- Python 3.11 or 3.12 recommended (especially for PyTorch / sentence-transformers compatibility).
- Existing PostgreSQL, Neo4j, and Qdrant services used by the ingestion application.
- The Qdrant collection populated by ingestion (`repomind_chunks` by default).
- Optional Groq API key for stronger query planning, evidence assessment, and answer synthesis. Without it, RepoMind uses a deterministic planner and evidence-summary fallback; the application still runs retrieval, but answer quality is more extractive.
- GitHub access is optional. When reachable, source excerpts are pinned to the exact indexed commit, not the moving branch.

The embedding model must match ingestion. The current index uses `BAAI/bge-small-en-v1.5` and 384-dimensional vectors. Do not change the embedding model for query-time retrieval unless the collection is re-indexed with the same model and preprocessing.

## 3. Windows PowerShell setup

From the extracted project directory:

```powershell
.\setup.ps1
```

Then edit `.env` (created from `.env.example`) with the same service endpoints and credentials used by your ingestion application. Use a PostgreSQL read-only user for retrieval where possible.

Minimum important settings:

```dotenv
POSTGRES_DSN=postgresql://USER:PASSWORD@localhost:5433/DATABASE
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=repomind_chunks
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
AUTHORIZED_REPOSITORIES=LableOrg/java-maven-junit-helloworld
```

Use your actual PostgreSQL port/database. The `5433` above is only an example for setups where PostgreSQL is mapped to that port. `AUTHORIZED_REPOSITORIES` is a server-side comma-separated allowlist. A repository supplied in a question/request is never itself authorization.

For better answers, set `GROQ_API_KEY` to a real key. Keep `.env` private and do not commit it. In non-development deployments, configure `REPOMIND_API_KEY` and a proper authenticated gateway; do not expose the development server directly to the internet.

## 4. Check the three stores before querying

```powershell
.\doctor.ps1 -Repository "LableOrg/java-maven-junit-helloworld"
```

This reports the active commit/generation and compares:

- PostgreSQL symbol count against Neo4j `Symbol` node count.
- PostgreSQL chunk count against Qdrant point count for the active generation.
- PostgreSQL chunks with search vectors against total PostgreSQL chunks.

For the currently verified sample generation, the expected baseline is 53 symbols, 48 chunks, 8 files, 53 Neo4j symbols, 34 Neo4j tests, 95 graph edges, and 48 Qdrant points. If you re-index, use the new active generation's counts instead of hard-coding these values.

## 5. Ask a retrieval question

```powershell
.\ask.ps1 -Repository "LableOrg/java-maven-junit-helloworld" -Question "Explain how the HelloWorld test is connected to the application code."
```

Or directly:

```powershell
.\.venv\Scripts\python.exe -m repomind.cli ask `
  --repository "LableOrg/java-maven-junit-helloworld" `
  --question "Which tests call the main application method?"
```

The CLI prints JSON with the answer, repository, commit SHA, index generation, evidence count, citation IDs, evidence sufficiency, missing evidence, and non-sensitive error summaries.

## 6. Start the HTTP API

```powershell
.\start_api.ps1
```

The development server binds to `127.0.0.1:8000`.

- `GET http://127.0.0.1:8000/health` — process liveness.
- `GET http://127.0.0.1:8000/ready` — checks PostgreSQL, Neo4j, and Qdrant connectivity.
- `GET http://127.0.0.1:8000/repositories` — active generation and counts for server-allowlisted repositories.
- `POST http://127.0.0.1:8000/ask` — retrieve evidence and answer a question.

Example PowerShell request:

```powershell
$body = @{
  repository = "LableOrg/java-maven-junit-helloworld"
  question = "Explain the test coverage around HelloWorld."
} | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/ask" `
  -ContentType "application/json" `
  -Body $body
```

If `REPOMIND_API_KEY` is configured, include `-Headers @{ "X-API-Key" = "YOUR_KEY" }` in `Invoke-RestMethod`. Without a key, unauthenticated requests are allowed only from loopback in `APP_ENV=development`. Set an explicit server-side repository allowlist in all cases.

OpenAPI docs are available at `http://127.0.0.1:8000/docs` while the server is running.

## 7. Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Unit tests cover repository allowlist enforcement, line-range bounds, and reciprocal-rank fusion. `doctor` is the read-only integration check against the real stores. A full end-to-end answer test additionally needs reachable services and, for generated answers, a valid Groq key.

## 8. Important limitations

- This retrieval service does not run tests or prove a candidate test passes/fails. A `Test` node is a discovery/impact hint, not test execution evidence.
- The graph currently models direct calls and derived test-to-symbol edges. It does not invent dependency edges that ingestion has not produced.
- RRF is rank fusion, not a calibrated relevance probability. Evaluate top-k results on representative questions before production.
- The API's allowlist is not a substitute for your organization’s full identity/tenant authorization system. Use a trusted gateway/authentication layer in multi-user deployments.
- The first local embedding-model load can download model files and take time.
- The requirements pin the Qdrant Python client to the 1.13 series to align with the currently verified Qdrant server 1.13.2 and avoid the large client/server version gap observed during verification. If your server version differs, align the client pin with that server and run the integration checks before deployment.
