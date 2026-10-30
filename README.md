# RepoMind — Repository Intelligence & Codebase Reasoning

<p align="center"><img src="assets/banner.png" alt="RepoMind — Agentic Java Repository Intelligence" width="100%"></p>

<p align="center">
  <strong>Turn a repository into a searchable, connected, source-grounded knowledge system.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/LLM-Groq-f55036?style=for-the-badge" alt="Groq" />
  <img src="https://img.shields.io/badge/Orchestration-LangGraph-6B46C1?style=for-the-badge" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Vector%20Search-Qdrant-DC244C?style=for-the-badge" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Knowledge%20Graph-Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white" alt="Neo4j" />
  <img src="https://img.shields.io/badge/Lexical%20Index-PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

> **RepoMind is a repository-level intelligence system, not just a code chatbot.** It combines Java-aware structural analysis, a symbol-and-relationship graph, semantic vector search, PostgreSQL lexical retrieval, repository metadata, pinned GitHub source excerpts, and a bounded agentic LangGraph workflow that selects retrieval channels, checks evidence sufficiency, follows newly discovered identifiers, and explains findings with source references.

> [!IMPORTANT]
> **Current scope:** Java is the only structurally analyzed source language. Markdown and selected text/configuration/build artifacts can also be indexed to provide repository context. The project is designed to make evidence traceable to a specific repository, commit, index generation, file, and line range. Retrieval quality still depends on the indexed revision, source availability, and evidence returned by the stores; RepoMind should explicitly acknowledge missing evidence rather than be treated as infallible.

---

## Contents

- [Why RepoMind?](#why-repomind)
- [What it can do](#what-it-can-do)
- [Visual architecture guide](#visual-architecture-guide)
- [System architecture](#system-architecture)
- [The ingestion and indexing pipeline](#the-ingestion-and-indexing-pipeline)
- [The retrieval and reasoning workflow](#the-retrieval-and-reasoning-workflow)
  - [Why this is agentic rather than conventional RAG](#why-this-is-agentic-rather-than-conventional-rag)
  - [How the LangGraph makes decisions](#how-the-langgraph-makes-decisions)
  - [Hybrid retrieval: four complementary channels](#hybrid-retrieval-four-complementary-channels)
  - [Fallbacks and graceful degradation](#fallbacks-and-graceful-degradation)
  - [A worked investigation: tracing a service integration](#a-worked-investigation-tracing-a-service-integration)
- [The three-store data architecture](#the-three-store-data-architecture)
- [Evidence, source grounding, and version safety](#evidence-source-grounding-and-version-safety)
- [Streamlit application](#streamlit-application)
- [Repository layout](#repository-layout)
- [Technology stack](#technology-stack)
- [Requirements and prerequisites](#requirements-and-prerequisites)
- [Installation and local setup](#installation-and-local-setup)
- [Configure the environment](#configure-the-environment)
- [Run RepoMind](#run-repomind)
- [CLI reference](#cli-reference)
- [Tests and diagnostics](#tests-and-diagnostics)
- [Troubleshooting](#troubleshooting)
- [Security and operational notes](#security-and-operational-notes)
- [Current limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why RepoMind?

Understanding an unfamiliar codebase is rarely a single-search problem. A question such as *“How does Dispatch connect to the rest of the application?”* may require finding a listener, identifying the event it consumes, locating the event contract, finding the publisher in another service, checking queue bindings, and following the result into another projection or command handler.

A single vector search can find conceptually similar text but may miss exact identifiers. A lexical search can find a class or routing key but may not explain how it relates to callers and consumers. A graph can expose structural relationships but may not retrieve the implementation details needed to explain their behavior. RepoMind combines these complementary evidence sources and then uses an LLM to synthesize a readable answer from the retrieved material.

### The design principles

- **Code-aware, not code-only:** use Java structural analysis for symbols and relationships, while indexing useful documentation and configuration artifacts for surrounding context.
- **Hybrid retrieval:** combine exact text search, semantic similarity, graph traversal, and metadata lookup instead of relying on a single retrieval strategy.
- **Evidence before explanation:** collect source-backed evidence and inspect its sufficiency before asking the model to produce the final response.
- **Version-pinned investigation:** associate indexed facts with a commit and index generation, and fetch source excerpts from the indexed commit rather than silently switching to the latest branch state.
- **Multi-repository isolation:** select and authorize repositories explicitly, and scope retrieval to the selected repository's active index generation.
- **Inspectable execution:** expose the investigation trace and retrieved evidence through the Streamlit UI so an answer is easier to audit and debug.
- **Incremental publication:** build a new index generation and validate the stores before moving the repository's active-generation pointer to it.

## What it can do

| Capability | What it means in practice |
|---|---|
| Java structural ingestion | Run `codeanalyzer-java` over Java files and normalize its output into symbols and graph relationships. |
| Repository artifact indexing | Index supported documentation, build files, configuration, and SQL/shell artifacts for contextual retrieval. |
| Symbol and file lookup | Find symbol names, qualified names, kinds, file paths, and source ranges through PostgreSQL metadata. |
| Relationship exploration | Use Neo4j to investigate calls and other normalized structural relationships. |
| Semantic retrieval | Retrieve conceptually relevant chunks through embeddings stored in Qdrant. |
| Lexical retrieval | Search exact identifiers and text using PostgreSQL full-text search. |
| Evidence fusion | Combine and rank candidates from different retrieval channels. |
| Source excerpt retrieval | Retrieve relevant source excerpts from GitHub at the indexed commit when source access is available. |
| Adaptive investigation | Assess evidence, identify gaps, and run follow-up retrieval when the current evidence is insufficient and the retry budget allows it. |
| Source-grounded answers | Ask the LLM to explain the retrieved evidence and return source references where supported by the available evidence. |
| Execution trace | Inspect LangGraph node activity, evidence output, and recent investigations in Streamlit. |
| Index consistency checks | Compare active-generation counts across PostgreSQL, Neo4j, and Qdrant. |

**Examples of questions to ask**

- “Explain the end-to-end flow when a new order is created.”
- “Where is this class instantiated, and which components call it?”
- “How does the Dispatch service connect to Orders and Restaurants?”
- “Which event contract links this publisher to its consumers?”
- “What is the difference between the write-side handler and the read-model projection?”
- “Which tests or modules may be affected by changing this method?”
- “Where is this configuration value used?”

These are example prompts, not guarantees that every question can be answered from every repository. The repository must be indexed, the relevant source must be available, and the current retrieval evidence must support the requested conclusion.

---

## Visual architecture guide

Every major subsystem has a dedicated diagram. Each one sits next to the section it explains, and they are collected here as a quick map.

| Diagram | What it shows | Where it is used |
|---|---|---|
| [`Overall_architecture.png`](assets/Overall_architecture.png) | Ingestion and query planes, the three-store versioned index, the bounded retry loop | [System architecture](#system-architecture) |
| [`deep_dive.png`](assets/deep_dive.png) | Layer-by-layer view: inputs, ingestion, processing, indexing, retrieval, outputs, serving | [System architecture](#system-architecture) |
| [`ingestion.png`](assets/ingestion.png) | Checkout, analysis, chunking, embedding, store writes, validation, publication | [Ingestion pipeline](#the-ingestion-and-indexing-pipeline) |
| [`retrival.png`](assets/retrival.png) | LangGraph planning, parallel retrieval, evidence fusion, assessment, refine loop | [Retrieval workflow](#the-retrieval-and-reasoning-workflow) |
| [`db_view.png`](assets/db_view.png) | PostgreSQL, Neo4j and Qdrant sharing one repository / commit / generation contract | [Data architecture](#the-three-store-data-architecture) |
| [`evidence.png`](assets/evidence.png) | How multiple evidence channels become one cited, grounded answer | [Evidence and version safety](#evidence-source-grounding-and-version-safety) |

---

## System architecture

RepoMind has two main execution paths that share the same indexed repository data:

1. **Offline ingestion:** analyze a repository revision, build searchable representations, validate the outputs, and publish an active index generation.
2. **Online retrieval:** investigate a question against the active generation, fetch source excerpts pinned to the indexed commit, evaluate evidence, and generate a grounded explanation.

<p align="center"><img src="assets/Overall_architecture.png" alt="RepoMind end-to-end architecture: ingestion and query planes over a versioned three-store index" width="100%"></p>

The current repository contains two Python packages:

- `repomind_ingest` — repository checkout, file discovery, CodeAnalyzer-Java integration, chunking, embedding, and writes to the index stores.
- `repomind_retriver` — query planning, graph/vector/lexical/metadata retrieval, evidence fusion, source fetching, evidence assessment, and answer generation.

`app.py` is the Streamlit interface. The retrieval package also contains a CLI and a FastAPI application entry point; the standard interactive workflow described here uses Streamlit directly and does not require a separate API server unless you choose to run it.

### Architecture at a glance

| Layer | Main component | Responsibility |
|---|---|---|
| Repository source | Git repository or local Git working copy | Provides the revision to index. |
| Ingestion orchestration | `repomind_ingest.pipeline.IngestionPipeline` | Coordinates checkout, analysis, chunking, writes, validation, and publication. |
| Structural analysis | `codeanalyzer-java` / `canjv` | Extracts Java-aware program structure from Java files. |
| Chunking and embeddings | Ingestion parser + Sentence Transformers | Creates retrieval-sized chunks and vector representations. |
| Relational/lexical store | PostgreSQL | Stores repository metadata, symbols, chunks, file manifest, and lexical search vectors. |
| Relationship store | Neo4j | Stores normalized symbols and structural edges for relationship-oriented retrieval. |
| Vector store | Qdrant | Stores embedded chunks with repository/version metadata for semantic retrieval. |
| Workflow orchestration | LangGraph | Coordinates planning, parallel retrieval, evidence fusion, source fetch, assessment, retry, and answer generation. |
| LLM provider | Groq through LangChain | Performs query/investigation planning and evidence assessment/answer synthesis where configured. |
| Source grounding | GitHub API / repository source access | Fetches excerpts from the indexed commit for direct verification. |
| User interface | Streamlit | Lets users choose a repository, ask questions, and inspect trace/evidence/history. |

### Layer-by-layer view

The same system, broken down by layer: inputs, ingestion, processing, indexing, retrieval, outputs, and the serving and deployment infrastructure underneath.

<p align="center"><img src="assets/deep_dive.png" alt="RepoMind layer-by-layer architecture" width="100%"></p>

### Why three retrieval stores?

Each store answers a different class of question:

- **PostgreSQL** is the reliable home for repository/version metadata, symbol records, file manifests, chunks, and exact-text search.
- **Neo4j** represents relationships between program entities, making it useful for questions about callers, callees, dependencies, and structural connections.
- **Qdrant** retrieves conceptually similar code and artifact chunks when the question uses different wording from the source.

The stores are complementary. RepoMind combines their results rather than treating any single one as the complete truth.

---

## The ingestion and indexing pipeline

Ingestion is not just “embed every file.” The pipeline builds several coordinated representations of a repository revision and validates them before making the new generation active.

<p align="center"><img src="assets/ingestion.png" alt="RepoMind ingestion and index publication pipeline" width="100%"></p>

### 1. Resolve and pin a repository revision

The ingestion CLI accepts a local Git repository path or an approved HTTPS/SSH Git URL, with an optional branch, tag, or commit reference. For a local path, pass the canonical repository identity using `--repository owner/repository`. The pipeline resolves a commit and uses a detached checkout so that the index represents a specific revision rather than a changing working tree.

### 2. Discover eligible files

The discovery layer applies file and directory exclusions to avoid generated outputs, dependency directories, binary assets, credentials, and oversized files. The current supported set includes:

- **Java source:** `.java` — structurally analyzed and chunked using symbol ranges.
- **Documentation:** Markdown, MDX, reStructuredText, and text files.
- **Configuration/build/runtime artifacts:** XML, YAML, TOML, JSON, properties, Gradle/Kotlin build files, SQL, shell scripts, and selected conventional files such as `Dockerfile`, `Makefile`, and `Jenkinsfile`.

This does **not** mean every listed language is structurally parsed. Java is the structural source language in the current implementation; supported non-Java artifacts provide additional searchable context.

The pipeline also enforces configurable limits for maximum file count and maximum file size. A repository with no eligible Java source files is rejected by the current ingestion implementation.

### 3. Analyze Java structure with CodeAnalyzer-Java

`codeanalyzer-java` is invoked through the `canjv` command. Its raw graph output is normalized into RepoMind's internal symbol and edge representations. The ingestion code resolves analyzer entities to Java paths using available source-path metadata and, where necessary, Java type/file ownership information.

The normalized symbol model carries fields such as:

- stable symbol identifier;
- qualified and simple name;
- symbol kind;
- source file path and line range;
- language and test marker;
- signature/source information where available;
- modifiers and metadata;
- observed call/import relationships and analyzer provenance.

The exact relationship coverage depends on what the analyzer emits for the repository and which relationships the adapter can normalize. A missing relationship should not be interpreted as proof that the relationship does not exist.

### 4. Build retrieval chunks

Java chunks are created preferentially around analyzer-provided method and constructor ranges. When there are no suitable callables in a file, class/interface/enum/record-level ranges may be used. Remaining uncovered source ranges are also chunked so imports, fields, and surrounding context are not automatically lost.

Supported non-Java artifacts use bounded line-based chunking. Chunk size and overlap are configurable. Chunk records preserve useful metadata such as file path, language, symbol association, chunk type, and commit.

### 5. Create embeddings

The configured embedding model is `BAAI/bge-small-en-v1.5`, which produces 384-dimensional vectors. **The ingestion and retrieval embedding model and preprocessing contract must match.** Changing the model without rebuilding the vector index can make vector search incompatible or lower quality.

### 6. Write to the three stores

The pipeline writes coordinated generation-scoped records:

- **PostgreSQL:** repository state, ingestion runs, file manifest, symbols, chunks, and lexical search vectors.
- **Neo4j:** normalized symbols and structural relationships for graph retrieval.
- **Qdrant:** embedded chunks with payload filters for repository, commit, and index generation.

### 7. Validate before publishing

Before the new generation becomes active, ingestion compares expected and stored counts, including symbols, chunks, files, graph edges, and Qdrant points. If cross-store validation fails, the generation is not published as the active one. This prevents readers from switching to a partially written index as if it were complete.

The repository record acts as the active-generation pointer. Older generations may be retained to avoid breaking in-flight readers. The optional `--cleanup-old` flag deletes older generations after publication and is explicitly marked unsafe when active readers may still depend on them.

---

## The retrieval and reasoning workflow

A user question enters a LangGraph state graph. The workflow plans an investigation, retrieves evidence from multiple sources, merges it, fetches pinned source excerpts, evaluates whether the evidence can answer the question, and may refine the investigation before generating the final response.

<p align="center"><img src="assets/retrival.png" alt="RepoMind LangGraph retrieval and reasoning workflow" width="100%"></p>

### Workflow stages

| Stage | Purpose |
|---|---|
| `plan_query` | Classifies the question and selects a suitable retrieval strategy and search terms. |
| `resolve_context` | Resolves the requested repository and its active commit/index-generation context, subject to authorization. |
| `plan_investigation` | Creates an investigation plan for the type of question, including relevant evidence to look for. |
| `metadata_search` | Looks up symbol/file metadata in PostgreSQL. |
| `neo4j_graph_search` | Searches structural relationships in Neo4j. |
| `qdrant_vector_search` | Performs semantic retrieval against the vector index. |
| `postgres_lexical_search` | Finds exact terms and identifiers through PostgreSQL lexical search. |
| `merge_evidence` | Deduplicates, ranks, and combines evidence from retrieval channels. |
| `fetch_github_sources` | Attempts to retrieve relevant source excerpts from the indexed commit. |
| `assess_evidence` | Evaluates whether the collected evidence supports the requested answer and identifies gaps. |
| `refine_retrieval` | Revises terms/targets based on missing evidence when more retrieval rounds are permitted. |
| `generate_answer` | Produces a readable response from the gathered evidence and available source references. |

The four retrieval channels are executed as parallel branches in the graph. This is important: graph search, semantic search, lexical search, and metadata lookup can return different evidence for the same question.

### Adaptive retrieval, not unlimited looping

When the evidence assessment finds gaps, RepoMind can refine the query and search again. The maximum number of retrieval rounds is configurable through `MAX_RETRIEVAL_ROUNDS`. A retry budget limits cost and runtime; it does not guarantee that every investigation will find enough evidence.

The evidence assessment is particularly important for architecture and integration questions. Finding a listener is not enough to prove who publishes its event. Finding a queue binding is not enough to prove the publisher uses a matching exchange and routing key. Strong answers should distinguish confirmed links from inferred or unverified ones.

### Answer generation

The LLM receives a bounded selection of retrieved evidence rather than an unfiltered dump of the entire repository. The final response is intended to explain the behavior, connect components, and cite source material when the retrieved evidence supports it. If the necessary implementation or cross-service evidence cannot be found, the system should communicate that limitation rather than invent a connection.

### Why this is agentic rather than conventional RAG

A conventional RAG pipeline is often described as a mostly fixed sequence: embed the question, retrieve the nearest chunks, put those chunks into a prompt, and generate an answer. That pattern is useful for document question answering, but it can be too shallow for repository-level questions where the answer depends on code relationships, event wiring, exact identifiers, or several services.

RepoMind uses **bounded agentic retrieval**: the model and deterministic policies help decide how to investigate, the graph coordinates specialist retrieval nodes, an evidence assessment checks whether the investigation has enough support, and a conditional edge can send the workflow back for another targeted search. The system does not simply accept the first batch of nearest-neighbor chunks as the answer.

| Conventional single-pass RAG | RepoMind's investigation loop |
|---|---|
| Usually runs one fixed retrieval recipe | The planner selects useful retrieval channels, with deterministic policy floors for common query types |
| Often depends mainly on vector similarity | Combines metadata, lexical, semantic, and graph retrieval |
| Retrieves once and answers immediately | Assesses evidence, records gaps, and can run bounded follow-up rounds |
| Treats a relevant chunk as a likely answer | Looks for implementation evidence and corroborating relationships; integration claims need concrete links |
| Usually relies on indexed text alone | Can fetch source excerpts from the exact indexed Git commit |
| May fail when the LLM planner is unavailable | Has deterministic planning and assessment fallbacks, plus an evidence-summary answer path |
| Can mix stale snippets if version scope is weak | Filters evidence by repository, indexed commit, and index generation |

The distinction is not that RepoMind is an unconstrained autonomous agent. Its autonomy is deliberately bounded: it operates within a defined set of retrieval nodes, repository authorization rules, source-fetch limits, evidence budgets, and a maximum retry count. It does not execute arbitrary repository code or invent new tools at runtime.

### How the LangGraph makes decisions

LangGraph supplies the **stateful control flow**. Each node reads the current investigation state and returns updates such as the plan, candidate evidence, source excerpts, missing-evidence list, or next-round queries. The graph—not a single giant prompt—coordinates the sequence and the retry decision.

#### 1. `plan_query`: choose a retrieval strategy

When the Groq model is available, the planner produces a structured `RetrievalPlan`: intent, target symbol where identifiable, retrieval-channel flags, search queries, result budget, and graph depth. The model is asked to plan retrieval, not answer the user's question or make up symbol IDs and paths.

A deterministic policy then applies safety floors so the planner cannot under-retrieve common cases: relationship questions enable graph, lexical, metadata, and vector retrieval; symbol/file questions ensure metadata and lexical search; conceptual questions ensure semantic and lexical search; exact-source requests locate the file before using a commit-pinned source path. Query terms are also normalized and focused. Broad service-integration questions receive short discovery probes such as `dispatch`, `dispatch listener event`, `dispatch publisher queue`, and `dispatch controller client`, rather than relying only on a long natural-language sentence.

#### 2. `resolve_context`: establish the allowed version

Before retrieval, RepoMind resolves the selected repository to its repository ID, active indexed commit, and index generation. Repository authorization is server-controlled; the LLM cannot override the allowlist. Retrieval adapters and a final evidence guard use that context to avoid combining evidence from different repositories or index generations.

#### 3. `plan_investigation`: turn a broad question into probes

For broad service-integration questions, the workflow creates a bounded investigation plan: locate the target service's entry points and listeners, find likely publishers/callers in other services, identify event contracts and queue/exchange/routing-key configuration, and distinguish verified links from documentation-only or inferred relationships. This plan guides retrieval; it is not itself evidence that those connections exist.

#### 4. Run retrieval specialists in parallel

The graph fans out to four retrieval nodes. Each node is a specialist with a different evidence source. Planner flags can disable a channel; a disabled node returns no hits without calling its adapter. The graph joins the results at `merge_evidence`.

#### 5. `merge_evidence`: fuse rather than concatenate

The merge stage combines candidate lists using reciprocal-rank fusion (RRF), deduplicates overlapping evidence, preserves retrieval provenance, and prioritizes useful implementation evidence. In service-discovery mode, production Java implementation and integration files are favored over README material, and source selection is diversified so one file does not crowd out every other participant. Results are then guarded against repository/commit/generation mismatches.

#### 6. `fetch_github_sources`: verify source at the indexed commit

For ordinary investigations, the source-fetch stage enriches retrieved candidates with excerpts from the exact indexed revision. For service-discovery questions, it can also inspect the GitHub tree at that commit to find bounded sets of target-service and cross-service Java candidates, including listener, consumer, publisher, event, queue, configuration, controller, and client files. A filename match is only a candidate—not proof of an integration. Source fetching has time and size limits and does not silently switch to the latest branch.

For explicit full-file requests, RepoMind follows a stricter path: it returns the complete file fetched at the pinned commit, or clearly says that a complete matching file could not be retrieved. It does not reconstruct a supposedly exact file from partial chunks.

#### 7. `assess_evidence`: decide whether to answer or investigate more

The evidence assessor evaluates the collected evidence against the question and produces a sufficiency decision plus missing evidence. For integration questions, the deterministic guard checks more than the number of files: it looks for target-service implementation, cross-service implementation, relevant integration participants, and a concrete shared event/routing/queue/exchange identifier or API relationship that verifies the link. An answer with ten unrelated Java files is not necessarily a verified integration explanation.

#### 8. Conditional routing: stop or refine

After assessment, a conditional edge chooses the next step:

- **Sufficient evidence:** move to `generate_answer`.
- **Insufficient evidence and retry budget available:** move to `refine_retrieval`, then search again.
- **Retry budget exhausted:** generate a qualified answer that exposes unresolved gaps.
- **No progress on a follow-up round:** stop looping and generate a qualified answer rather than repeatedly issuing the same searches.

The `MAX_RETRIEVAL_ROUNDS` setting bounds the investigation. This makes the workflow adaptive without allowing an unbounded agent loop.

### Hybrid retrieval: four complementary channels

Hybrid retrieval means more than running vector search and keyword search side by side. RepoMind combines four distinct evidence channels and then fuses their ranked candidates.

| Channel / node | Store | Strongest use cases | Typical limitation it compensates for |
|---|---|---|---|
| **Metadata — `metadata_search`** | PostgreSQL | Exact symbol, qualified name, file path, language/kind, source range | Natural-language similarity may not identify the exact class or file |
| **Graph — `neo4j_graph_search`** | Neo4j | Callers/callees, impact, dependencies, relationships, identifier-led discovery | Text chunks may describe entities without exposing how they connect |
| **Semantic — `qdrant_vector_search`** | Qdrant | Conceptual explanations, architecture questions, alternate wording | Exact string matching misses paraphrases and concepts expressed differently |
| **Lexical — `postgres_lexical_search`** | PostgreSQL full-text index | Class names, method names, event types, literals, file names, routing keys | Embedding similarity may rank conceptually related but identifier-incorrect code |

#### Graph retrieval is not just “searching a graph”

When a target symbol resolves, the graph adapter can run a relationship/impact search from that symbol ID. If a target cannot be resolved—or the question is a broad service investigation—the workflow can perform graph discovery using identifiers extracted from the question, search queries, and previously fetched source. It deduplicates the graph results before evidence fusion. This allows a broad query to begin with a service term and later become anchored to concrete classes or event names discovered in the repository.

The graph remains bounded by the relationships CodeAnalyzer-Java emitted and the ingestion adapter normalized. It cannot prove runtime behavior that was not represented in the indexed structural facts.

#### Keyword and query refinement

The planner preserves exact identifiers and adds short, focused probes. For example, a question about a Dispatch service may start with `dispatch`, then use probes such as `dispatch listener event` or `dispatch publisher queue`. If the first pass discovers `OrderCreatedV1`, `RabbitTemplate`, or a queue name, those identifiers can anchor later retrieval.

This is especially important because lexical search over a long natural-language query may be too restrictive or emphasize generic words. Focused terms improve recall, while metadata and graph retrieval help recover the structural context around the matches.

### Fallbacks and graceful degradation

RepoMind is designed to preserve useful evidence when an individual model call or retrieval adapter fails. Fallbacks do not make missing evidence magically appear; they allow the investigation to continue with the channels that still work and to disclose the limitation.

| Failure or edge case | Recovery behavior | What the user should expect |
|---|---|---|
| Groq planner unavailable or structured planning fails | Use deterministic `fallback_plan`, then apply deterministic channel-selection policy | Retrieval can continue with a simpler plan; nuanced intent detection may be weaker |
| A retrieval adapter fails | Log the error and return an empty result list for that channel | Other retrieval channels can still contribute evidence |
| A retrieval channel is intentionally not selected | The node skips the adapter call | Less work is performed, but the plan may not retrieve evidence from that channel |
| Repository is not indexed or context resolution fails | Context is marked unavailable; store retrieval cannot proceed normally | The answer points to the missing index/context rather than fabricating repository facts |
| Candidate evidence belongs to a different commit or generation | Reject it during stamping/fusion | Stale or cross-generation evidence is not allowed into the final evidence set |
| GitHub tree/source fetch fails, times out, or exceeds size limits | Continue with indexed evidence where possible; record source-fetch limitations | Source verification may be incomplete and should be stated |
| LLM evidence assessment fails | Use deterministic sufficiency checks, including stricter integration coverage checks | The assessment is less nuanced, but key evidence requirements still apply |
| Streaming answer generation fails | Attempt structured answer generation | A second LLM path may still produce the answer |
| Structured answer generation also fails or no LLM is configured | Return a compact evidence-grounded summary with source references where available | The response is a retrieval summary, not a synthesized explanation |
| Evidence remains insufficient | Include an evidence limitation and missing-evidence notes | The answer should expose uncertainty instead of treating guesses as facts |
| User requests an exact full file but complete source is unavailable | Refuse to reconstruct from partial chunks | The response explains that the complete file could not be retrieved |

### A worked investigation: tracing a service integration

Consider the question: **“How is the Dispatch service connected to the whole application?”** This is not a single-symbol lookup. The relationship may span listeners, event contracts, queue configuration, publishers in other services, and downstream consumers.

1. **Classify the question.** The planner recognizes a broad architecture/service-integration investigation and enables broad candidate discovery rather than assuming one exact target symbol.
2. **Start with focused terms.** The initial probes include the service name and likely integration roles such as listeners, events, publishers, queues, controllers, and clients.
3. **Search in parallel.** PostgreSQL metadata and lexical search find names and exact terms; Qdrant finds semantically related chunks; Neo4j searches relationships and discovered identifiers.
4. **Fuse and inspect candidates.** RRF merges ranked evidence. Implementation Java and integration configuration are prioritized over documentation, while duplicate evidence is collapsed.
5. **Fetch pinned source.** Candidate Java files are fetched from the indexed commit. For broad integration questions, the repository tree can surface likely cross-service participants that were not among the first retrieved chunks.
6. **Assess the missing links.** The assessor asks whether the evidence contains both sides of a connection—for example, a consumer/listener and a publisher, tied by a shared event contract or compatible exchange/routing configuration.
7. **Refine using discovered identifiers.** If the publisher is missing, the next round targets publisher APIs such as `RabbitTemplate.convertAndSend`, event names, routing keys, or exchange configuration. If a listener or binding is missing, follow-up probes target `@RabbitListener`, queue declarations, and bindings. Identifiers found in fetched source are fed into the next retrieval pass.
8. **Stop responsibly.** If the required links are established, the LLM synthesizes the architecture with source references. If the bounded rounds end before the links are verified, the response should explicitly say what remains unverified.

This is the core difference from “retrieve five chunks and ask the LLM”: the first result set is treated as a starting hypothesis for investigation, not automatically as sufficient proof.

---

## The three-store data architecture

<p align="center"><img src="assets/db_view.png" alt="RepoMind three-store data architecture sharing one version contract" width="100%"></p>

### PostgreSQL — metadata and lexical retrieval

The relational store contains the repository index state and the searchable records needed for metadata and text search. The schema includes:

- `repositories` — canonical repository identity and active commit/generation pointer;
- `ingestion_runs` — generation status (`building`, `published`, or `failed`) and run details;
- `symbols` — normalized symbol metadata and source locations;
- `chunks` — source/artifact chunks, their line ranges, content, and associated metadata;
- `repository_files` — file manifest, category/language, size, hash, and line count.

A PostgreSQL `tsvector` and GIN index support lexical search over chunk paths, symbol names, and content. The schema is maintained in `sql/001_schema.sql` and is also ensured by the ingestion store implementation.

### Neo4j — code relationship graph

Neo4j stores normalized structural entities and relationships. It is used when the question depends on relationships rather than text similarity alone—for example, tracing calls or understanding how symbols connect. Repository and generation metadata are used to scope graph retrieval to the relevant index.

Graph contents reflect the structural facts extracted and normalized by the analyzer. They should not be assumed to represent every possible runtime relationship, framework convention, or dynamic call.

### Qdrant — semantic vector retrieval

Qdrant stores embeddings for chunks along with payload metadata. At query time, semantic search finds code or documentation that is conceptually related to the question even if it does not use the same words. Payload filters scope results to the selected repository and indexed version.

### Shared version contract

The most important cross-store invariant is the shared repository/version context:

- repository identity (`repository_id` and canonical repository name);
- indexed commit SHA;
- index generation identifier.

This metadata allows the system to avoid accidentally combining chunks from one repository or generation with graph facts from another. Cross-store writes are coordinated by the ingestion pipeline and validated before publication; PostgreSQL, Neo4j, and Qdrant do not participate in a single distributed database transaction.

---

## Evidence, source grounding, and version safety

RepoMind aims to answer *why* a component behaves a certain way, not merely return a list of similar files.

<p align="center"><img src="assets/evidence.png" alt="RepoMind evidence channels fused into a grounded, cited answer" width="100%"></p>

### Evidence sources

- **Structural evidence:** normalized symbols and relationships.
- **Lexical evidence:** exact identifier or phrase matches.
- **Semantic evidence:** conceptually relevant chunks.
- **Metadata evidence:** symbol kind, qualified name, file path, and source range.
- **Source evidence:** excerpts retrieved from the repository at the indexed commit when available.

### What source-grounded means

A strong explanation connects claims to specific files and relevant source excerpts. For example, a claim that a service consumes an event should ideally be supported by the listener and its queue/binding configuration. A claim that another service publishes that event should be supported independently by publisher code and the matching event/routing contract.

The presence of a class name in a graph or a semantically similar chunk is a lead for investigation, not sufficient proof of an integration by itself.

### Commit pinning

The indexed commit is stored as part of the repository generation. Source retrieval is intended to use that commit so that a later branch change does not silently alter the code being explained. Source retrieval may fail because of access restrictions, rate limits, deleted repositories, or unavailable files; in those cases, the answer quality depends on the evidence already indexed.

### Citations and limitations

Citation quality depends on the provenance carried through retrieval and source fetching. Always open the referenced file and check the surrounding code for critical engineering decisions. RepoMind is an investigative assistant, not a substitute for reviewing code, running tests, or validating production behavior.

---

## Streamlit application

The default user experience is the Streamlit app in `app.py`. It calls the retrieval graph directly and is designed to make the investigation inspectable rather than showing only a final text response.

<!-- VISUAL ASSET SLOT — STREAMLIT UI
Capture a real screenshot from the running application and save it as assets/streamlit-ui.png. Show the repository selector/sidebar,
chat conversation, and bottom composer if possible. Redact tokens, private repository names, personal file paths, and sensitive query content.
Do not generate a fictional screenshot; this slot is for the actual UI.
-->

<!-- <p align="center"><img src="assets/streamlit-ui.png" alt="RepoMind Streamlit interface" width="100%"></p> -->

### Main tabs

- **Ask / Conversation:** choose a repository and ask questions in a chat-style interface. The app shows prior turns in the current session and the response produced by the retrieval workflow.
- **Trace:** inspect the LangGraph execution timeline and node/event details. A node returning no hits can still have completed successfully; zero results are not automatically a runtime error.
- **Evidence:** inspect retrieved evidence and channel output. The interface is intended to be read-only with respect to the index.
- **History:** review recent investigations held by the application session.

The Streamlit UI does not replace ingestion. Index a repository first, then select it for retrieval.

---

## Repository layout

```text
RepoMind/
├── app.py                         # Streamlit chat, trace, evidence, and history UI
├── requirements.txt               # Combined Python dependencies
├── .env.example                   # Safe configuration template; no real secrets
├── docker/
│   └── docker-compose.yml         # Local PostgreSQL, Neo4j, and Qdrant infrastructure
├── sql/
│   └── 001_schema.sql             # PostgreSQL retrieval/index schema
├── repomind_ingest/               # Ingestion package
│   ├── analyzer.py                # CodeAnalyzer-Java adapter and graph normalization
│   ├── cli.py                     # Ingestion command-line entry point
│   ├── config.py                  # Ingestion settings loaded from environment
│   ├── models.py                  # Symbol, chunk, file, and edge data models
│   ├── parser.py                  # Stable IDs and chunking helpers
│   ├── pipeline.py                # End-to-end ingestion and generation publication
│   ├── repository.py              # Git checkout and eligible-file discovery
│   ├── stores.py                  # PostgreSQL, Neo4j, and Qdrant ingestion adapters
│   └── verify_repomind_stores.py  # Store consistency verification utility
├── repomind_retriver/             # Retrieval package (name as present in this repository)
│   ├── adapters.py                # PostgreSQL, Neo4j, Qdrant, and source access adapters
│   ├── api.py                     # Optional FastAPI entry point
│   ├── cli.py                     # Retrieval ask/doctor commands
│   ├── config.py                  # Retrieval settings
│   ├── graph.py                   # LangGraph workflow and answer orchestration
│   ├── models.py                  # Workflow state and evidence models
│   ├── planning.py                # Query/investigation planning helpers
│   ├── ranking.py                 # Candidate ranking/fusion helpers
│   └── security.py                # Repository identity/authorization helpers
└── tests/
    ├── test_analyzer_adapter.py   # Analyzer normalization tests
    ├── test_chunking.py           # Chunking tests
    └── test_security_ranking.py   # Security and ranking tests
```

> The inner retrieval package is currently spelled **`repomind_retriver`** in the repository. Commands and imports below intentionally use that exact spelling.

---

## Technology stack

| Technology | Role in RepoMind |
|---|---|
| Python 3.11+ | Ingestion, retrieval, orchestration, CLI, and UI integration. |
| CodeAnalyzer-Java (`canjv`) | Java structural analysis. The executable is installed separately and must be available on `PATH`. |
| LangGraph | Stateful multi-step retrieval and reasoning workflow. |
| LangChain Core / LangChain Groq | LLM integration and orchestration utilities. |
| Sentence Transformers | Local embedding generation using the configured embedding model. |
| `BAAI/bge-small-en-v1.5` | Current 384-dimensional embedding model. |
| PostgreSQL + `psycopg` | Relational metadata, chunks, file manifest, lexical search, and generation state. |
| Neo4j | Structural symbol graph and relationship queries. |
| Qdrant | Vector storage and semantic nearest-neighbor retrieval. |
| Git / GitHub API | Immutable source revision checkout and pinned source excerpt retrieval. |
| Streamlit | Interactive investigation UI. |
| FastAPI + Uvicorn | Optional HTTP API included in the codebase. |
| Docker Compose | Local database infrastructure for development. |
| pytest | Automated tests. |

---

## Requirements and prerequisites

Before starting, ensure you have:

1. **Python 3.11 or 3.12** available. The setup scripts and package code target Python 3.11+.
2. **Git** installed and available on `PATH`.
3. **CodeAnalyzer-Java CLI** installed separately, with the `canjv` executable available on `PATH`.
4. **Docker Desktop / Docker Engine with Compose** for the provided local infrastructure, or equivalent running PostgreSQL, Neo4j, and Qdrant instances.
5. A **Groq API key** for LLM-powered planning/assessment/answer generation.
6. **GitHub access** to the repositories you want to inspect. A GitHub token is useful for private repositories and authenticated API access.
7. Enough disk space and memory for the embedding model and repository data. The first embedding-model use may download model files.

The provided Compose file uses PostgreSQL 16, Neo4j 5.26 Community, and Qdrant 1.13.2. The host ports are bound to localhost by default. PostgreSQL is exposed on host port `5432` in the supplied Compose file; if your local PostgreSQL instance uses a different port, update `POSTGRES_DSN` accordingly.

---

## Installation and local setup

### 1. Clone the project

```powershell
git clone <YOUR-REPOMIND-REPOSITORY-URL>
cd RepoMind
```

Replace the placeholder with your repository URL. If you already have the project locally, open a terminal at the directory containing `app.py`, `requirements.txt`, and `.env.example`.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

### 3. Configure the environment

Copy the example file and edit it:

```powershell
Copy-Item .env.example .env
```

Or on macOS/Linux:

```bash
cp .env.example .env
```

Fill in your own credentials and confirm the repository allowlists. See [Configure the environment](#configure-the-environment).

### 4. Start the local databases

From the repository root:

```powershell
docker compose -f docker/docker-compose.yml up -d
```

Check container status:

```powershell
docker compose -f docker/docker-compose.yml ps
```

The Compose file initializes PostgreSQL with `sql/001_schema.sql` when the database volume is first created. For an existing database volume, Docker's initialization scripts are not automatically rerun; apply required schema changes using your normal migration process or the ingestion store's schema initialization.

Default local endpoints from the supplied Compose file:

| Service | Local endpoint | Notes |
|---|---|---|
| PostgreSQL | `localhost:5432` | Compose defaults to database/user `repomind`; password comes from `POSTGRES_PASSWORD` or the development fallback in Compose. |
| Neo4j Browser | `http://localhost:7474` | Bolt endpoint `neo4j://localhost:7687`. |
| Qdrant HTTP | `http://localhost:6333` | gRPC port `6334`. |

If you already have these services running outside Docker, use their real endpoints and credentials in `.env` instead of starting duplicate containers.

### 5. Verify CodeAnalyzer-Java

```powershell
canjv --help
```

If the command is not found, install/configure CodeAnalyzer-Java separately and ensure its executable directory is on `PATH`. The Python dependency declaration does not guarantee that a standalone CLI executable has been installed correctly.

### 6. Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Resolve test failures before treating the environment as ready.

---

## Configure the environment

The repository's `.env.example` is the configuration template. The table below explains the main variables; check the current package configuration code before adding new variables, because an environment variable only has an effect when the application reads it.

### Core services and secrets

| Variable | Purpose | Example / guidance |
|---|---|---|
| `APP_ENV` | Environment label for retrieval/API behavior. | `development` |
| `LOG_LEVEL` | Ingestion logging verbosity. | `INFO` or `DEBUG` |
| `GROQ_API_KEY` | Groq credential for the LLM. | Keep the real key only in `.env`. |
| `CHAT_MODEL` | Model name used by retrieval. | Must be available to your Groq account/provider integration. |
| `POSTGRES_DSN` | PostgreSQL connection string. | `postgresql://user:password@localhost:5432/database` |
| `POSTGRES_PASSWORD` | Password used by the supplied Compose service. | Keep consistent with the Compose database configuration. |
| `NEO4J_URI` | Neo4j Bolt connection URI. | `neo4j://127.0.0.1:7687` |
| `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j credentials. | Use the credentials configured for your instance. |
| `QDRANT_URL` | Qdrant endpoint. | `http://localhost:6333` |
| `QDRANT_API_KEY` | Optional Qdrant API key. | Leave empty for local unauthenticated Qdrant. |
| `QDRANT_COLLECTION` | Vector collection name. | `repomind_chunks` |
| `GITHUB_TOKEN` | Token for GitHub source/API access. | Required for private repository access; also helps with API rate limits. |
| `ALLOW_PUBLIC_REPOSITORIES` | Retrieval setting for public repository handling. | `true` or `false`; repository authorization still matters. |

### Repository authorization

| Variable | Used for |
|---|---|
| `REPOMIND_ALLOWED_REPOSITORIES` | Ingestion-side allowlist. |
| `AUTHORIZED_REPOSITORIES` | Retrieval CLI/API repository allowlist. |
| `DEFAULT_REPOSITORY` | Intended default repository selection for the UI. |
| `REPOMIND_UI_USER_ID` | Local Streamlit user identifier used by the application. |

Use canonical `owner/repository` names separated by commas, for example:

```dotenv
REPOMIND_ALLOWED_REPOSITORIES=owner/repository-one,owner/repository-two
AUTHORIZED_REPOSITORIES=owner/repository-one,owner/repository-two
DEFAULT_REPOSITORY=owner/repository-one
REPOMIND_UI_USER_ID=local-streamlit-user
```

Keep ingestion and retrieval authorization aligned when the same repositories should be ingestible and queryable. The retrieval CLI checks `AUTHORIZED_REPOSITORIES`; ingestion checks `REPOMIND_ALLOWED_REPOSITORIES` when that allowlist is non-empty. These settings are separate and should not be assumed to be interchangeable.

### Embeddings and ingestion limits

| Variable | Purpose | Current default / note |
|---|---|---|
| `EMBEDDING_MODEL` | Embedding model for indexing and retrieval. | `BAAI/bge-small-en-v1.5`; keep consistent across both paths. |
| `EMBEDDING_BATCH_SIZE` | Initial embedding batch size for ingestion. | `32` |
| `MAX_EMBEDDING_BATCH_SIZE` | Maximum supported embedding batch size. | `64` |
| `CHUNK_MAX_TOKENS` | Approximate chunk token budget. | `450` |
| `CHUNK_OVERLAP_LINES` | Overlap for chunking uncovered ranges/artifacts. | `8` |
| `MAX_FILE_BYTES` | Maximum eligible file size. | `1500000` bytes |
| `MAX_FILES` | Maximum eligible files to discover. | `50000` |
| `CANJV_COMMAND` | CodeAnalyzer-Java executable. | `canjv` |
| `ANALYZER_TIMEOUT_SECONDS` | Analyzer process timeout. | `900` |
| `ANALYZER_GRAPH_DEPTH` | Analyzer graph depth setting. | `6` |
| `ANALYZER_MAX_GRAPH_NODES` | Analyzer graph node cap. | `200000` |

### Retrieval controls

| Variable | Purpose | Current default / note |
|---|---|---|
| `MAX_RESULTS` | Retrieval candidate/result limit. | `8` |
| `MAX_GRAPH_HOPS` | Graph traversal bound. | `3` |
| `MAX_RETRIEVAL_ROUNDS` | Maximum adaptive retrieval rounds. | `3` in the example; runtime cost grows with additional rounds. |
| `REQUEST_TIMEOUT_SECONDS` | Request/source access timeout used by retrieval configuration. | `12` |

The example file may also include optional tuning variables for LLM token budgets or parallel source fetching. Do not assume these settings are active simply because they appear in `.env.example`; confirm the current code reads each variable before relying on it operationally.

### Optional API settings

`repomind_retriver/api.py` contains an optional FastAPI entry point. If you run it separately, configure `REPOMIND_API_KEY`, `API_HOST`, and `API_PORT`. The default host is `127.0.0.1`, which keeps local development bound to localhost. The main Streamlit workflow imports the retrieval graph directly and does not need the API server unless you specifically use that integration.

---

## Run RepoMind

### Start the Streamlit UI

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux:**

```bash
.venv/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Select an authorized repository, ask a question, and use Trace/Evidence to inspect the investigation.

### Ingest a GitHub repository

The ingestion CLI lives in `repomind_ingest`, separate from the retrieval package.

```powershell
.\.venv\Scripts\python.exe -m repomind_ingest.cli https://github.com/OWNER/REPOSITORY.git --repository OWNER/REPOSITORY --ref main
```

Replace `OWNER/REPOSITORY` with the actual repository and choose a branch, tag, or commit in `--ref`. The repository must satisfy the ingestion allowlist if `REPOMIND_ALLOWED_REPOSITORIES` is configured.

For a local Git repository:

```powershell
.\.venv\Scripts\python.exe -m repomind_ingest.cli "C:\path\to\local\git-repository" --repository OWNER/REPOSITORY --ref main
```

For verbose logs:

```powershell
.\.venv\Scripts\python.exe -m repomind_ingest.cli https://github.com/OWNER/REPOSITORY.git --repository OWNER/REPOSITORY --ref main --verbose
```

The command prints a JSON summary with repository identity, commit SHA, generation, file/symbol/chunk counts, vector count, analyzer/store statistics, warnings, and embedding metadata.

> [!WARNING]
> Avoid `--cleanup-old` unless you understand the impact on in-flight retrieval requests. It removes previous generations after the new generation is published.

### Ask a question from the retrieval CLI

The CLI package name is `repomind_retriver` (including the current spelling):

```powershell
.\.venv\Scripts\python.exe -m repomind_retriver.cli ask --repository OWNER/REPOSITORY --question "Explain the request flow through this application"
```

This command requires the repository to be present in `AUTHORIZED_REPOSITORIES`. It prints the answer payload as JSON.

### Run the store consistency doctor

```powershell
.\.venv\Scripts\python.exe -m repomind_retriver.cli doctor --repository OWNER/REPOSITORY
```

The doctor compares active-generation counts and reports consistency checks such as:

- PostgreSQL chunk count matches Qdrant point count;
- PostgreSQL symbol count matches Neo4j symbol count;
- PostgreSQL searchable chunk count matches the chunk count.

A non-zero exit code indicates that a check or operation failed and should be investigated.

### Optional FastAPI server

If you need the HTTP API independently of Streamlit:

```powershell
.\.venv\Scripts\python.exe -m repomind_retriver.api
```

The API host and port are configurable. Check `/health` and `/ready` in `repomind_retriver/api.py` for the current health/readiness behavior. Do not expose the development server publicly without reviewing authentication, network binding, and deployment configuration.

---

## CLI reference

### Ingestion CLI

```text
python -m repomind_ingest.cli SOURCE [--repository OWNER/REPOSITORY] [--ref REF] [--cleanup-old] [--verbose]
```

| Argument | Description |
|---|---|
| `SOURCE` | Local Git repository path or approved HTTPS/SSH Git URL. |
| `--repository` | Canonical `owner/repository` identity; required for local paths. |
| `--ref` | Branch, tag, or commit to index. Defaults to the source's default/current revision. |
| `--cleanup-old` | Delete old generations after publication; unsafe with in-flight readers. |
| `--verbose` | Enable debug-level ingestion logs. |

### Retrieval CLI

```text
python -m repomind_retriver.cli ask --repository OWNER/REPOSITORY --question "..."
python -m repomind_retriver.cli doctor --repository OWNER/REPOSITORY
```

| Command | Description |
|---|---|
| `ask` | Run the retrieval workflow for a question against an authorized indexed repository. |
| `doctor` | Compare active-generation counts across PostgreSQL, Neo4j, and Qdrant. |

---

## Tests and diagnostics

### Run the test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current test modules cover:

- analyzer adapter normalization;
- chunking behavior;
- security and repository handling;
- ranking behavior.

### Verify stores after ingestion

Run the retrieval doctor:

```powershell
.\.venv\Scripts\python.exe -m repomind_retriver.cli doctor --repository OWNER/REPOSITORY
```

The ingestion result also includes counts and validation details. Compare the active generation, commit SHA, PostgreSQL symbols/chunks/files, Neo4j symbols/edges, and Qdrant point count when investigating a mismatch.

### Read the UI trace

If an answer is incomplete, inspect the Trace and Evidence tabs before assuming the LLM is the cause. The issue may be:

- no candidate returned by a retrieval channel;
- a useful symbol exists but its relationship was not extracted;
- the query did not include the identifier needed to find a publisher/consumer;
- source fetching could not access the pinned file;
- the evidence budget or retrieval-round limit was reached;
- the indexed commit is older than the code you expected to query.

---

## Troubleshooting

### `canjv` is not found

Install CodeAnalyzer-Java separately and add its executable to `PATH`. Verify from the same shell used to run ingestion:

```powershell
Get-Command canjv
canjv --help
```

### The repository is rejected by the allowlist

Check `REPOMIND_ALLOWED_REPOSITORIES` in `.env`. Use canonical `owner/repository` names separated by commas. For retrieval, separately check `AUTHORIZED_REPOSITORIES`. Restart the Streamlit process after changing environment variables so the new settings are loaded.

### PostgreSQL connection fails

Check the actual host port, database name, username, password, and `POSTGRES_DSN`. The included Compose file uses port `5432` and database/user `repomind` by default; an existing local PostgreSQL setup may use different values. Check service status with:

```powershell
docker compose -f docker/docker-compose.yml ps
```

### Neo4j authentication or connection fails

Confirm `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`. If the Docker volume was initialized with a previous password, changing the Compose environment variable may not change the credentials stored in that existing database.

### Qdrant collection or vector dimensions mismatch

Ensure `QDRANT_URL` and `QDRANT_COLLECTION` match the ingestion setup and that the configured embedding model matches the model used for indexing. The current configured model produces 384-dimensional embeddings. Do not change the model and expect existing vectors to become compatible automatically.

### Groq returns rate-limit or oversized-request errors

Check the selected model, current provider limits, request size, and number of evidence chunks passed into the LLM. Reduce the evidence context or use a model/account with sufficient limits if necessary. More retrieval rounds can improve coverage but may increase the number of LLM calls; do not increase the retry budget blindly.

### The answer does not mention a relevant file

Use the Evidence tab to check whether the file was indexed and retrieved. Confirm that the file type is supported, it was not excluded by path/size rules, the correct commit was ingested, and the repository is selected. For Java structural questions, check analyzer output and graph relationships as well as vector and lexical candidates.

### Source excerpts cannot be fetched

Check GitHub access, token permissions, repository visibility, rate limits, and whether the file exists at the indexed commit. Public repository access does not guarantee that every API request will succeed.

### Store counts do not match

Run the doctor command and inspect the active commit/generation in each store. Do not manually delete records from only one store. Prefer a controlled re-ingestion and cross-store validation. Use old-generation cleanup only when safe for active readers.

### Streamlit imports fail

Ensure the terminal is at the repository root, the virtual environment is active/used, and dependencies are installed from the current `requirements.txt`. The project contains two packages, so keep the names straight:

- ingestion: `repomind_ingest`;
- retrieval: `repomind_retriver`.

---

## Security and operational notes

- **Never commit `.env`.** Commit `.env.example` with placeholders only. If a credential is exposed, revoke/rotate it rather than simply deleting it from the latest commit.
- Keep API and database ports bound to localhost during local development unless you have intentionally configured secure remote access.
- Use least-privilege credentials for database access in deployments. The local development Compose defaults are not production secrets.
- Keep ingestion and retrieval repository allowlists explicit. An ingestion allowlist does not replace retrieval authorization.
- Treat GitHub tokens as secrets and grant only the permissions needed to access the repositories being indexed.
- Do not run `--cleanup-old` while readers may still be using the previous generation.
- Avoid manually editing or deleting only one store's generation records; the stores are coordinated by repository, commit, and generation identifiers.
- Review generated answers against their cited files before using them for security-sensitive or production-impacting decisions.
- The provided Compose stack is a local development setup, not a hardened production deployment configuration.

---

## Current limitations

The following boundaries are important to understand:

1. **Java-first structural analysis:** only Java source files are structurally analyzed. Other supported files are indexed as textual artifacts.
2. **Analyzer-dependent graph coverage:** the graph is limited to relationships emitted and normalized from CodeAnalyzer-Java output. Dynamic dispatch, reflection, framework wiring, and runtime behavior may not be fully represented.
3. **Cross-store consistency is validated, not transactionally distributed:** the ingestion pipeline coordinates writes and publishes only after validation, but the three databases do not share a distributed transaction.
4. **Source fetching depends on GitHub access:** inaccessible or unavailable source files reduce the ability to validate retrieval results against exact source excerpts.
5. **LLM output is evidence-bounded, not formally verified:** evidence assessment and citations improve traceability but do not guarantee correctness.
6. **Retrieval is budgeted:** maximum results, graph hops, timeouts, and retrieval rounds trade completeness against latency and provider cost.
7. **Local Streamlit session history:** the UI's recent investigation history is application-session state, not a durable multi-user audit database.
8. **The optional API requires separate operational review:** authentication, exposure, and deployment hardening should be evaluated before running it beyond localhost.

---

## Roadmap

This roadmap separates existing capabilities from future engineering opportunities. Items below are suggestions for future work, not claims that they are already implemented.

| Area | Current state | Potential next step |
|---|---|---|
| Java structural ingestion | Implemented through CodeAnalyzer-Java adapter | Broaden analyzer regression fixtures and graph coverage validation. |
| Repository artifact indexing | Implemented for selected text/config/build formats | Improve file-type policies and language-specific retrieval metadata. |
| Multi-store indexing | PostgreSQL, Neo4j, and Qdrant writes with generation validation | Add stronger reconciliation and operational recovery tooling. |
| Hybrid retrieval | Metadata, graph, vector, and lexical channels | Benchmark retrieval quality on a curated set of codebase questions. |
| Adaptive investigation | Bounded evidence assessment and refinement workflow | Improve claim-level verification and missing-evidence diagnostics. |
| Source grounding | Pinned GitHub source retrieval when available | Add caching, request observability, and robust rate-limit handling. |
| UI | Streamlit Ask, Trace, Evidence, and History tabs | Add exportable investigation reports and improved saved-session support. |
| Evaluation | Unit tests for core adapters, chunking, security, and ranking | Add an end-to-end evaluation dataset with expected evidence and citations. |
| Security | Repository allowlists and optional API-key configuration | Add deployment-grade secret management, identity integration, and audit controls. |
| Operations | Docker Compose for local stores | Add documented backup/restore, monitoring, and deployment profiles. |
| Language coverage | Java structural analysis | Explore additional language analyzers behind explicit, testable adapters. |

---

## Contributing

Contributions are welcome, especially around retrieval quality, analyzer normalization, store consistency, test coverage, and documentation.

A useful contribution workflow is:

1. Create a focused branch.
2. Add or update tests for behavior changes.
3. Run `python -m pytest -q`.
4. Keep secrets, local databases, `.venv`, caches, and generated artifacts out of commits.
5. Document any new environment variables and whether they are actually consumed by the application.
6. For changes to ingestion schemas or metadata contracts, verify that ingestion and retrieval remain compatible across PostgreSQL, Neo4j, and Qdrant.

For retrieval changes, include at least one example question, the expected evidence sources, and how a reviewer can verify the result.

## License

No license file was included in the inspected project archive. Until a license is added by the repository owner, reuse and redistribution permissions should not be assumed. Add a `LICENSE` file before publishing the project for broad reuse.

---

<p align="center">
  <strong>RepoMind</strong><br>
  <em>Explore code. Connect evidence. Understand the system.</em>
</p>