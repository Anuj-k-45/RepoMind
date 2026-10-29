from __future__ import annotations

import os
import time
import traceback
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Load the project's existing environment configuration before importing RepoMind.
load_dotenv()

try:
    from repomind_retriver.graph import RepoMind
    try:
        from repomind_retriver.graph import _STREAM_ANSWER_TOKENS
    except Exception:
        _STREAM_ANSWER_TOKENS = None
except Exception as exc:
    RepoMind = None  # handled in the UI with a useful diagnostic
    _STREAM_ANSWER_TOKENS = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


st.set_page_config(
    page_title="RepoMind · Repository Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
      --rm-bg: #101216;
      --rm-panel: #17191f;
      --rm-panel-2: #1d2027;
      --rm-border: rgba(255,255,255,.09);
      --rm-text: #eceef2;
      --rm-muted: #9a9faa;
      --rm-accent: #d9e2f2;
      --rm-green: #83c9a4;
      --rm-amber: #e5c58b;
      --rm-red: #e49b9b;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--rm-bg); color: var(--rm-text); }
    [data-testid="stHeader"] { background: rgba(16,18,22,.92); }
    [data-testid="stSidebar"] { background: #14161b; border-right: 1px solid var(--rm-border); }
    [data-testid="stSidebar"] > div { padding-top: 1rem; }
    .block-container { padding-top: 1.2rem; padding-bottom: 8.5rem; max-width: 1120px; }
    /* Keep the chat composer pinned to the viewport, even though it is rendered
       inside the Ask tab. The extra bottom padding keeps the final chat turn
       from being obscured by the fixed composer. */
    [data-testid="stChatInput"] {
      position: fixed !important;
      bottom: 1rem !important;
      left: calc(50% + 168px) !important;
      transform: translateX(-50%) !important;
      width: min(calc(100vw - 368px), 1120px) !important;
      max-width: 1120px !important;
      z-index: 9999 !important;
      box-shadow: 0 8px 32px rgba(0,0,0,.38);
      backdrop-filter: blur(16px);
    }
    /* On narrow layouts the sidebar collapses, so center the composer in the viewport. */
    @media (max-width: 760px) {
      [data-testid="stChatInput"] {
        left: 50% !important;
        width: calc(100vw - 1.5rem) !important;
        bottom: .5rem !important;
      }
      .block-container { padding-bottom: 8rem; }
    }
    h1, h2, h3 { letter-spacing: -.025em; }
    h1 { font-weight: 650 !important; }
    .rm-brand { display:flex; align-items:center; gap:11px; margin-bottom: 1.4rem; }
    .rm-logo { width:38px;height:38px;display:flex;align-items:center;justify-content:center;
      border:1px solid rgba(217,226,242,.18); border-radius:12px;
      background:#20232b; font-size:20px; }
    .rm-brand-title { font-size:1rem;font-weight:700;letter-spacing:-.02em; }
    .rm-brand-sub { font-size:.74rem;color:var(--rm-muted);margin-top:2px; }
    .rm-eyebrow { text-transform:uppercase;letter-spacing:.12em;font-size:.68rem;font-weight:700;color:#aab4c5; }
    .rm-hero { padding: .4rem 0 1.1rem; margin-bottom: .8rem; border-bottom:1px solid var(--rm-border); }
    .rm-hero h1 { margin:.25rem 0 .25rem 0;font-size:1.65rem; }
    .rm-hero p { color:var(--rm-muted);max-width:760px;margin:0;line-height:1.55;font-size:.92rem; }
    .rm-chip { display:inline-block;padding:4px 8px;border:1px solid var(--rm-border);border-radius:7px;
      color:#b9c0cd;font-size:.68rem;margin:10px 5px 0 0;background:rgba(255,255,255,.025); }
    .rm-card { border:1px solid var(--rm-border);border-radius:12px;background:rgba(255,255,255,.025);padding:14px 15px; }
    .rm-muted { color:var(--rm-muted);font-size:.82rem; }
    .rm-mono { font-family:'JetBrains Mono',monospace;font-size:.76rem;overflow-wrap:anywhere; }
    .rm-section { font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--rm-muted);font-weight:700;margin: .2rem 0 .8rem; }
    .rm-step { display:flex;gap:11px;padding:9px 0;border-bottom:1px solid rgba(255,255,255,.06); }
    .rm-step:last-child { border-bottom:0; }
    .rm-dot { width:8px;height:8px;border-radius:50%;margin-top:6px;flex:0 0 8px;background:#707784; }
    .rm-dot.running { background:var(--rm-amber); }
    .rm-dot.done { background:var(--rm-green); }
    .rm-dot.error { background:var(--rm-red); }
    .rm-step-title { font-weight:600;font-size:.86rem; }
    .rm-step-detail { color:var(--rm-muted);font-size:.76rem;margin-top:3px;line-height:1.45; }
    .stButton > button { border-radius:9px; font-weight:600; min-height:2.4rem; }
    .stButton > button[kind="primary"] { background:#e4e8ef;color:#16181d;border:0; }
    .stTextArea textarea, .stTextInput input { background:#17191f;border-color:var(--rm-border);border-radius:10px; }
    div[data-testid="stMetric"] { background:rgba(255,255,255,.025);border:1px solid var(--rm-border);padding:12px 14px;border-radius:11px; }
    div[data-testid="stMetricLabel"] { color:var(--rm-muted); }
    div[data-testid="stExpander"] { border:1px solid var(--rm-border);border-radius:10px;background:rgba(255,255,255,.018); }
    .stTabs [data-baseweb="tab-list"] { gap:5px; border-bottom:1px solid var(--rm-border); }
    .stTabs [data-baseweb="tab"] { border-radius:7px 7px 0 0;padding:8px 12px; }
    .stTabs [aria-selected="true"] { background:rgba(255,255,255,.06); }
    [data-testid="stChatMessage"] { padding: .8rem 1rem; border-radius: 14px; margin-bottom: .65rem; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: rgba(255,255,255,.035); }
    [data-testid="stChatInput"] { border:1px solid rgba(255,255,255,.16); border-radius:16px; background:#1a1d23; }
    [data-testid="stChatInput"] textarea { background:transparent !important; }
    [data-testid="stChatInputSubmitButton"] { color:#16181d !important; }
    .rm-footer { text-align:center;color:#747b88;font-size:.7rem;margin-top:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

NODE_LABELS = {
    "plan_query": ("Query planner", "Classifying intent and selecting retrieval strategy"),
    "resolve_context": ("Repository context", "Resolving repository, commit, and active index generation"),
    "metadata_search": ("Symbol metadata · PostgreSQL", "Looking up symbol and file metadata"),
    "neo4j_graph_search": ("Relationship graph · Neo4j", "Tracing callers, callees, dependencies, or tests"),
    "qdrant_vector_search": ("Semantic search · Qdrant", "Searching embeddings for conceptually relevant chunks"),
    "postgres_lexical_search": ("Lexical search · PostgreSQL", "Finding exact identifiers and text matches"),
    "merge_evidence": ("Evidence fusion", "Combining and ranking retrieval results"),
    "fetch_github_sources": ("Pinned source fetch · GitHub", "Fetching source excerpts from the indexed commit"),
    "assess_evidence": ("Evidence assessment", "Checking whether the collected evidence can answer the question"),
    "refine_retrieval": ("Retrieval refinement", "Expanding the query based on missing evidence"),
    "generate_answer": ("Answer generation", "Writing a grounded answer and validating citations"),
}
NODE_ORDER = list(NODE_LABELS)


def _allowed_repositories() -> list[str]:
    """Local single-user allowlist. Do not expose this UI publicly without real auth."""
    raw = os.getenv("REPOMIND_ALLOWED_REPOSITORIES", "").strip()
    if raw:
        return list(dict.fromkeys(x.strip() for x in raw.split(",") if x.strip()))
    # DEFAULT_REPOSITORY is a server-controlled single-repository allowlist fallback.
    default = os.getenv("DEFAULT_REPOSITORY", "").strip()
    return [default] if default else []


def _initial_state(question: str, repository: str, allowed: list[str]) -> dict[str, Any]:
    return {
        "question": question.strip(),
        "user_id": os.getenv("REPOMIND_UI_USER_ID", "local-streamlit-user"),
        "allowed_repositories": allowed,
        "requested_repository": repository,
        "default_repository": repository or (allowed[0] if allowed else ""),
        "errors": [],
    }


def _safe_text(value: Any, limit: int = 260) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _merge_state(state: dict[str, Any], patch: Any) -> None:
    if not isinstance(patch, dict):
        return
    for key, value in patch.items():
        if key == "errors" and isinstance(value, list):
            existing = state.get(key, [])
            state[key] = list(existing) + value
        else:
            state[key] = value


def _node_detail(node: str, patch: dict[str, Any], state: dict[str, Any]) -> str:
    if node == "plan_query":
        plan = patch.get("plan") or state.get("plan") or {}
        if plan:
            channels = [label for key, label in [
                ("use_metadata", "metadata"), ("use_graph", "graph"),
                ("use_vector", "vector"), ("use_lexical", "lexical")
            ] if plan.get(key, True)]
            return f"Intent: {plan.get('intent', 'unknown')} · channels: {', '.join(channels) or 'none'}"
    if node == "resolve_context":
        sha = patch.get("indexed_commit_sha") or state.get("indexed_commit_sha")
        generation = patch.get("index_generation") or state.get("index_generation")
        if sha:
            return f"Commit {str(sha)[:12]} · generation {str(generation or 'unknown')[:24]}"
        return "Context resolution completed; inspect errors if no active generation was found."
    counts = {
        "metadata_search": ("metadata_hits", "metadata hits"),
        "neo4j_graph_search": ("graph_hits", "graph hits"),
        "qdrant_vector_search": ("vector_hits", "vector hits"),
        "postgres_lexical_search": ("lexical_hits", "lexical hits"),
        "fetch_github_sources": ("github_hits", "source excerpts"),
        "merge_evidence": ("evidence", "fused evidence items"),
    }
    if node in counts:
        key, label = counts[node]
        items = patch.get(key, state.get(key, [])) or []
        return f"{len(items)} {label}"
    if node == "assess_evidence":
        enough = patch.get("evidence_sufficient", state.get("evidence_sufficient"))
        reason = patch.get("sufficiency_reason", state.get("sufficiency_reason", ""))
        return ("Sufficient" if enough else "Insufficient / uncertain") + (f" · {_safe_text(reason, 150)}" if reason else "")
    if node == "refine_retrieval":
        return f"Retrieval round {state.get('retrieval_round', 0)} · query expanded"
    if node == "generate_answer":
        answer = patch.get("final_response") or patch.get("answer") or state.get("final_response") or state.get("answer")
        return f"Answer assembled · {len(str(answer or '')):,} characters"
    return "Node completed"


def _stream_graph(rm: Any, question: str, repository: str, allowed: list[str]):
    """Stream real LangGraph updates. Tokens stream only if the installed graph exposes that API."""
    initial = _initial_state(question, repository, allowed)
    aggregate: dict[str, Any] = dict(initial)
    events: list[dict[str, Any]] = []
    start = time.perf_counter()

    # Stream the compiled graph directly so we can retain state patches for the
    # evidence explorer while also receiving custom answer-token events. The
    # ContextVar is only available in the streaming-enabled graph module.
    stream_token = None
    if _STREAM_ANSWER_TOKENS is not None:
        try:
            stream_token = _STREAM_ANSWER_TOKENS.set(True)
        except Exception:
            stream_token = None

    try:
        # Multi-mode streams yield (mode, payload). "updates" are genuine graph-node
        # results; "custom" carries explicit events if the graph emits any.
        for packet in rm.app.stream(initial, config={"recursion_limit": 40}, stream_mode=["updates", "custom"]):
            mode, payload = packet if isinstance(packet, tuple) and len(packet) == 2 else ("updates", packet)
            if mode == "custom":
                if isinstance(payload, dict):
                    event_type = str(payload.get("type", "event"))
                    if event_type in ("answer_token", "token", "llm_token"):
                        yield {"type": "answer_token", "text": str(payload.get("text") or payload.get("token") or "")}
                    elif event_type == "answer_complete":
                        yield {"type": "answer_complete", "data": payload}
                    else:
                        yield {"type": event_type, "data": payload}
                continue
            if not isinstance(payload, dict):
                continue
            for node, patch in payload.items():
                if not isinstance(patch, dict):
                    continue
                _merge_state(aggregate, patch)
                event = {
                    "type": "node_end",
                    "node": node,
                    "label": NODE_LABELS.get(node, (node, ""))[0],
                    "detail": _node_detail(node, patch, aggregate),
                    "elapsed": time.perf_counter() - start,
                    "patch": patch,
                }
                events.append(event)
                yield event
        if aggregate.get("context_error") == "not_found":
            raise LookupError("Repository is not indexed.")
        if aggregate.get("context_error") == "unavailable":
            raise RuntimeError("Could not resolve repository context. Check PostgreSQL connectivity and server logs.")
        if not aggregate.get("indexed_commit_sha") or not aggregate.get("index_generation"):
            raise LookupError("Repository has no published active index generation.")
        result = {
            "answer": aggregate.get("final_response", aggregate.get("answer", "")),
            "repository": aggregate.get("repository_full_name"),
            "commit_sha": aggregate.get("indexed_commit_sha"),
            "index_generation": aggregate.get("index_generation"),
            "evidence_count": len(aggregate.get("evidence", [])) + len(aggregate.get("github_hits", [])),
            "citation_ids": aggregate.get("citation_ids", []),
            "evidence_sufficient": aggregate.get("evidence_sufficient", False),
            "missing_evidence": aggregate.get("missing_evidence", []),
            "errors": aggregate.get("errors", []),
            "state": aggregate,
            "elapsed": time.perf_counter() - start,
        }
        yield {"type": "complete", "result": result}
    except Exception as exc:
        yield {"type": "error", "message": str(exc), "exception_type": type(exc).__name__}
        return
    finally:
        if stream_token is not None and _STREAM_ANSWER_TOKENS is not None:
            try:
                _STREAM_ANSWER_TOKENS.reset(stream_token)
            except Exception:
                pass


def _render_timeline(events: list[dict[str, Any]], running_node: str | None = None):
    seen: dict[str, dict[str, Any]] = {}
    for event in events:
        node = event.get("node")
        if node:
            seen[node] = event
    pieces = []
    for node in NODE_ORDER:
        if node in seen:
            e = seen[node]
            status = "error" if e.get("type") == "node_error" else "done"
            detail = e.get("detail", "Completed")
            elapsed = e.get("elapsed")
            if elapsed is not None:
                detail += f" · {elapsed:.1f}s elapsed"
        elif running_node == node:
            status, detail = "running", "Working…"
        else:
            # A branch can be intentionally skipped by the node's own guard.
            status, detail = "pending", "Waiting for graph execution"
        pieces.append(
            f'<div class="rm-step"><span class="rm-dot {status}"></span>'
            f'<div><div class="rm-step-title">{NODE_LABELS[node][0]}</div>'
            f'<div class="rm-step-detail">{detail}</div></div></div>'
        )
    st.markdown('<div class="rm-card">' + "".join(pieces) + "</div>", unsafe_allow_html=True)


def _render_evidence(state: dict[str, Any], result: dict[str, Any] | None = None):
    groups = [
        ("Merged evidence", state.get("evidence", [])),
        ("GitHub source excerpts", state.get("github_hits", [])),
        ("Metadata hits", state.get("metadata_hits", [])),
        ("Neo4j graph hits", state.get("graph_hits", [])),
        ("Qdrant vector hits", state.get("vector_hits", [])),
        ("PostgreSQL lexical hits", state.get("lexical_hits", [])),
    ]
    for title, items in groups:
        with st.expander(f"{title} · {len(items or [])}", expanded=(title == "Merged evidence")):
            if not items:
                st.caption("No items returned in this group.")
                continue
            for i, ev in enumerate(items):
                if not isinstance(ev, dict):
                    st.write(ev)
                    continue
                path = ev.get("file_path") or ev.get("source_type") or "Source"
                lines = ""
                if ev.get("start_line"):
                    lines = f":{ev.get('start_line')}"
                    if ev.get("end_line") and ev.get("end_line") != ev.get("start_line"):
                        lines += f"-{ev.get('end_line')}"
                st.markdown(f"**{i+1}. `{path}{lines}`**")
                meta = [str(ev.get("source_type") or ""), str(ev.get("evidence_id") or "")]
                if ev.get("score") is not None:
                    meta.append(f"score {ev['score']}")
                if ev.get("commit_sha"):
                    meta.append(f"commit {str(ev['commit_sha'])[:12]}")
                st.caption(" · ".join(x for x in meta if x))
                claim = ev.get("claim") or ev.get("excerpt") or ""
                if claim:
                    st.code(str(claim)[:7000], language="java" if str(path).endswith(".java") else "text")
                url = (ev.get("details") or {}).get("url") or ev.get("url")
                if url:
                    st.markdown(f"[Open pinned source ↗]({url})")
                st.divider()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "rm_messages": [],
    "rm_runs": [],
    "rm_last_result": None,
    "rm_last_state": {},
    "rm_events": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar
with st.sidebar:
    st.markdown(
        '<div class="rm-brand"><div class="rm-logo">⌘</div><div>'
        '<div class="rm-brand-title">RepoMind</div>'
        '<div class="rm-brand-sub">Repository intelligence workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rm-section">Workspace</div>', unsafe_allow_html=True)
    allowed = _allowed_repositories()
    default_repo = os.getenv("DEFAULT_REPOSITORY", "").strip()
    repo_options = allowed[:]
    if not repo_options:
        repo_value = ""
        st.error("No server-side repository allowlist is configured.")
        st.caption("Set REPOMIND_ALLOWED_REPOSITORIES or DEFAULT_REPOSITORY in .env before using the UI.")
    else:
        preferred = default_repo if default_repo in repo_options else repo_options[0]
        repo_value = st.selectbox("Indexed repository", repo_options, index=repo_options.index(preferred))

    st.markdown('<div class="rm-section" style="margin-top:1.3rem">Runtime</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rm-card"><div class="rm-muted">Chat model</div><div class="rm-mono">{os.getenv("CHAT_MODEL", "configured in settings")}</div>'
                f'<div class="rm-muted" style="margin-top:10px">Execution</div><div>LangGraph · streaming updates</div>'
                f'<div class="rm-muted" style="margin-top:10px">Index mutation</div><div style="color:#5de0a0">Disabled in this UI</div></div>',
                unsafe_allow_html=True)
    st.caption("Local developer UI. Do not expose it publicly without authentication and a server-managed repository allowlist.")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.rm_messages = []
        st.session_state.rm_last_result = None
        st.session_state.rm_last_state = {}
        st.session_state.rm_events = []
        st.rerun()
    if st.button("Clear run history", use_container_width=True):
        st.session_state.rm_runs = []
        st.rerun()
    with st.expander("Configuration help"):
        st.code(
            "REPOMIND_ALLOWED_REPOSITORIES=LableOrg/java-maven-junit-helloworld,Anuj-k-45/QuickBite-AI\n"
            "AUTHORIZED_REPOSITORIES=LableOrg/java-maven-junit-helloworld,Anuj-k-45/QuickBite-AI\n"
            "DEFAULT_REPOSITORY=Anuj-k-45/QuickBite-AI\n"
            "REPOMIND_UI_USER_ID=local-streamlit-user",
            language="dotenv",
        )
        st.caption("Set these in your existing .env file. Secrets remain server-side; never paste API keys into chat.")

# Header
st.markdown(
    """
    <div class="rm-hero">
      <div class="rm-eyebrow">Repository intelligence</div>
      <h1>RepoMind</h1>
      <p>Ask questions about implementation, architecture, dependencies, and tests. Answers are grounded in source evidence from the indexed commit.</p>
      <span class="rm-chip">Hybrid retrieval</span>
      <span class="rm-chip">Graph-aware analysis</span>
      <span class="rm-chip">Commit-pinned sources</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if RepoMind is None:
    st.error("RepoMind could not be imported. Start Streamlit from the project root so the `repomind` package is importable.")
    st.code(f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}")
    st.info("Expected import: `from repomind.graph import RepoMind`. If your graph module has a different filename, update that import near the top of `streamlit_app.py`.")
    st.stop()

if not repo_value.strip() or not allowed or repo_value not in allowed:
    st.warning("Configure a server-controlled repository allowlist before asking a question. The UI will not authorize repositories typed into the browser.")
    st.stop()

# Lazy cached engine: no database writes or indexing calls are made here.
@st.cache_resource(show_spinner=False)
def get_repomind():
    return RepoMind()


try:
    rm = get_repomind()
except Exception as exc:
    st.error("RepoMind could not initialize. Check `.env` and that PostgreSQL, Neo4j, and Qdrant are reachable.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

last = st.session_state.get("rm_last_result") or {}

main_tab, trace_tab, evidence_tab, history_tab = st.tabs(
    ["✦ Ask RepoMind", "⌁ Execution trace", "▤ Evidence explorer", "◷ History"]
)

with main_tab:
    # Chat history is shown before the composer and kept in chronological order.
    # New turns are appended to rm_messages, so each new question/answer appears
    # underneath the previous exchange rather than being prepended to the top.
    if st.session_state.rm_messages:
        st.markdown("### Conversation")
        for item in st.session_state.rm_messages[-8:]:
            with st.chat_message("user"):
                st.markdown(item["question"])
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(item["answer"])
                st.caption(
                    f"{item.get('repository')} · {item.get('evidence_count', 0)} evidence items"
                    + (f" · commit {str(item.get('commit_sha'))[:12]}" if item.get("commit_sha") else "")
                )
                if item.get("events"):
                    with st.expander(f"⌁ Execution flow · {len(item['events'])} steps", expanded=False):
                        _render_timeline(item["events"])
        st.markdown("---")

    st.caption(f"Repository: `{repo_value}` · Active indexed generation")
    question = st.chat_input("Message RepoMind…", key="rm_chat_input")

    if question:
        if not allowed:
            st.error("No repository allowlist is configured. Set `REPOMIND_ALLOWED_REPOSITORIES` in `.env`.")
        else:
            with st.chat_message("user"):
                st.markdown(question.strip())
            with st.chat_message("assistant", avatar="🧠"):
                answer_slot = st.empty()
                trace_slot = st.empty()
                status_slot = st.status("Investigating repository…", expanded=False)
                token_text = ""
                start = time.perf_counter()
                run_events: list[dict[str, Any]] = []
                final_result = None
                final_state: dict[str, Any] = {}
                try:
                    with status_slot:
                        for event in _stream_graph(rm, question, repo_value, allowed):
                            kind = event.get("type")
                            if kind == "node_end":
                                run_events.append(event)
                                node = event.get("node", "")
                                patch = event.get("patch", {})
                                if isinstance(patch, dict):
                                    _merge_state(final_state, patch)
                                status_slot.update(
                                    label=f"Investigating… {NODE_LABELS.get(node, (node, ''))[0]}",
                                    state="running",
                                    expanded=False,
                                )
                                # Keep the detailed event stream attached to this assistant
                                # turn, collapsed by default rather than printing log lines.
                                trace_slot.empty()
                                with trace_slot.expander(
                                    f"⌁ Execution flow · {len(run_events)} steps",
                                    expanded=False,
                                ):
                                    _render_timeline(run_events)
                                if node == "generate_answer":
                                    token_text = str(patch.get("final_response") or patch.get("answer") or "")
                                    answer_slot.markdown(token_text)
                            elif kind in ("answer_token", "token"):
                                token_text += str(event.get("text") or "")
                                answer_slot.markdown(token_text + " ▌")
                            elif kind == "answer_complete":
                                data = event.get("data", {})
                                token_text = str(data.get("answer") or data.get("text") or token_text)
                                answer_slot.markdown(token_text)
                            elif kind == "complete":
                                final_result = event.get("result")
                                if isinstance(final_result, dict):
                                    final_state = final_result.get("state", final_state)
                                    answer_slot.markdown(final_result.get("answer", token_text))
                            elif kind == "error":
                                raise RuntimeError(event.get("message", "RepoMind execution failed"))
                            elif kind == "graph_error":
                                data = event.get("data", {})
                                raise RuntimeError(data.get("message") or data.get("error") or "Graph execution failed")
                            elif kind in ("node_start", "graph_update", "event"):
                                run_events.append(event)
                    if final_result is None:
                        # The graph stream adapter can expose only events; use the latest known
                        # answer and state if it didn't provide a normalized complete payload.
                        answer = str(final_state.get("final_response") or final_state.get("answer") or token_text)
                        final_result = {
                            "answer": answer,
                            "repository": final_state.get("repository_full_name", repo_value),
                            "commit_sha": final_state.get("indexed_commit_sha"),
                            "index_generation": final_state.get("index_generation"),
                            "evidence_count": len(final_state.get("evidence", [])) + len(final_state.get("github_hits", [])),
                            "citation_ids": final_state.get("citation_ids", []),
                            "evidence_sufficient": final_state.get("evidence_sufficient", False),
                            "missing_evidence": final_state.get("missing_evidence", []),
                            "errors": final_state.get("errors", []),
                            "state": final_state,
                            "elapsed": time.perf_counter() - start,
                        }
                    final_result["elapsed"] = final_result.get("elapsed", time.perf_counter() - start)
                    st.session_state.rm_last_result = final_result
                    st.session_state.rm_last_state = final_result.get("state", final_state)
                    st.session_state.rm_events = run_events
                    trace_slot.empty()
                    with trace_slot.expander(
                        f"⌁ Execution flow · {len(run_events)} steps",
                        expanded=False,
                    ):
                        _render_timeline(run_events)
                    st.session_state.rm_messages.append({
                        "question": question.strip(),
                        "answer": final_result.get("answer", ""),
                        "repository": repo_value,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "commit_sha": final_result.get("commit_sha"),
                        "evidence_count": final_result.get("evidence_count", 0),
                        "events": run_events,
                    })
                    st.session_state.rm_runs.insert(0, {
                        "question": question.strip(),
                        "repository": repo_value,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed": final_result.get("elapsed"),
                        "evidence_count": final_result.get("evidence_count", 0),
                        "sufficient": final_result.get("evidence_sufficient", False),
                    })
                    status_slot.update(label="Investigation complete", state="complete", expanded=False)
                    st.rerun()
                except Exception as exc:
                    status_slot.update(label="Investigation failed", state="error", expanded=True)
                    st.error(f"{type(exc).__name__}: {exc}")
                    msg = str(exc).lower()
                    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
                        st.warning("The LLM provider rate-limited this request. Wait briefly and retry; avoid submitting several runs at once.")
                    elif "not indexed" in msg or "active index generation" in msg:
                        st.info("Confirm that this exact repository is in the allowlist and has a published active index generation.")
                    elif "connection" in msg or "timed out" in msg:
                        st.info("Check the local services and the connection values in `.env`.")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())


with trace_tab:
    if last:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Evidence items", last.get("evidence_count", 0))
        metric_cols[1].metric("Validated citations", len(last.get("citation_ids", [])))
        metric_cols[2].metric("Evidence", "Sufficient" if last.get("evidence_sufficient") else "Needs review")
        elapsed = last.get("elapsed")
        metric_cols[3].metric("Duration", f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "Complete")
    st.markdown("### Graph execution")
    st.caption("This timeline is built from actual LangGraph node updates. A node that returns no hits is still a completed node, not necessarily an error.")
    events = st.session_state.rm_events
    if events:
        _render_timeline(events)
        st.markdown("#### Event log")
        for i, event in enumerate(events, 1):
            with st.expander(f"{i:02d} · {event.get('label') or event.get('node') or event.get('type', 'event')}"):
                st.json({k: v for k, v in event.items() if k not in ("patch",)})
                if event.get("patch"):
                    st.json(event["patch"])
    else:
        st.info("Run a question to see real planner, retrieval, source-fetch, assessment, and answer-generation events.")
    if last:
        st.markdown("#### Run outcome")
        st.json({
            "repository": last.get("repository"),
            "commit_sha": last.get("commit_sha"),
            "index_generation": last.get("index_generation"),
            "evidence_count": last.get("evidence_count"),
            "evidence_sufficient": last.get("evidence_sufficient"),
            "missing_evidence": last.get("missing_evidence"),
            "errors": last.get("errors"),
        })

with evidence_tab:
    st.markdown("### Evidence explorer")
    st.caption("Inspect source-backed evidence and retrieval channel output. The UI does not write to the index.")
    state = st.session_state.rm_last_state or {}
    if state:
        _render_evidence(state, last)
    else:
        st.info("Evidence appears here after your first completed investigation.")

with history_tab:
    st.markdown("### Recent investigations")
    runs = st.session_state.rm_runs
    if not runs:
        st.info("No completed runs in this session yet.")
    else:
        for i, run in enumerate(runs[:30], 1):
            with st.expander(f"{i:02d} · {run['question'][:95]}"):
                st.write(f"**Repository:** `{run['repository']}`")
                st.write(f"**When:** {run['timestamp']}")
                st.write(f"**Evidence:** {run['evidence_count']} · **Sufficient:** {run['sufficient']}")
                if run.get("elapsed") is not None:
                    st.write(f"**Duration:** {run['elapsed']:.1f}s")

st.markdown(
    '<div class="rm-footer">RepoMind · Evidence first. Commit pinned. Java-first repository intelligence.</div>',
    unsafe_allow_html=True,
)
