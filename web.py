"""
RAG Web App — Weaviate + Groq + FastEmbed
Beautiful light-mode chat interface served locally via Flask.
"""

import json
import os
import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string


from dotenv import load_dotenv
from flask import send_from_directory
from flask_cors import CORS
load_dotenv()

# Prevent Windows EINVAL crashes from HF/tqdm progress bars on non-TTY stderr.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Import SQLite CRUD functions
from history_db import init_history_db, add_interaction, get_history
from code_history_db import (
    add_code_entry,
    delete_code_entry,
    get_code_entries,
    get_code_entry,
    update_code_entry,
)
from student_management.query_codegen import (
    build_crud_prompt,
    extract_python_code,
    infer_filename,
    is_crud_query,
    save_query_file,
)
app = Flask(__name__)
CORS(app)

@app.route("/history", methods=["GET"])
def history_endpoint():
    """Return stored chat interactions from SQLite."""
    # Ensure DB is initialized (redundant if already called elsewhere)
    init_history_db()
    history = get_history()
    # Return as JSON list of dicts
    return jsonify(history)



_PROJECT_DIR = Path(__file__).resolve().parent


@app.route("/")
def home_page():
    return send_from_directory(_PROJECT_DIR, "index.html")

@app.route("/about")
def about_page():
    return send_from_directory(_PROJECT_DIR, "about.html")


@app.route("/compare")
def compare_page():
    return send_from_directory(_PROJECT_DIR, "index.html")


def _run_rag_delegate(query: str, *, model: str) -> str:
    from delegate_to_rag_agent import delegate_to_rag_agent
    return delegate_to_rag_agent(query, model=model)


def _run_data_delegate(
    query: str,
    *,
    model: str,
    api_key: str,
    context_data: str = "",
) -> str:
    from delegate_to_data_agent import delegate_to_data_agent
    return delegate_to_data_agent(
        query,
        context_data=context_data,
        model=model,
        api_key=api_key,
    )


def _run_agent_delegate(query: str, *, model: str, api_key: str) -> str:
    from context_pipeline.config import PipelineConfig
    from context_pipeline.pipeline import run_agent_pipeline

    try:
        docs_dir = Path(os.environ.get("RAG_DOCS_DIR", "docs")).resolve()
    except (OSError, ValueError):
        docs_dir = Path("docs").resolve()
    retriever = os.environ.get("RAG_RETRIEVER", "vector").strip().lower()
    if retriever not in ("vector", "bm25"):
        retriever = "bm25"
    cfg = PipelineConfig.from_env()
    result = run_agent_pipeline(
        query=query,
        docs_dir=docs_dir,
        retriever_mode=retriever,
        model=model,
        api_key=api_key,
        chat_messages=[{"role": "user", "content": query}],
        cfg=cfg,
    )
    return _sanitize_delegate_answer(
        result.answer or "",
        query=query,
        model=model,
        api_key=api_key,
    )


def _detect_delegate_agent(query: str) -> str:
    """Lightweight routing for /delegate when agent=auto."""
    import re

    if re.search(
        r"\d+\s*[\*x×+\-/]\s*\d+|\bcalculate\b|\bcompute\b|\bmultiply\b|\bsum\b|\baverage\b|\bmean\b",
        query,
        re.I,
    ):
        return "data"
    return "agent"


def _sanitize_delegate_answer(answer: str, *, query: str, model: str, api_key: str) -> str:
    """Never return raw tool-call syntax to the UI — execute or unwrap it."""
    from context_pipeline.agent import sanitize_answer_for_ui

    return sanitize_answer_for_ui(answer, query=query, model=model, api_key=api_key)

_MAX_PIPELINE_TURNS = 4  # user+assistant pairs cap sent to Groq pipelines


def _trim_messages_for_pipeline(messages: list[dict]) -> list[dict]:
    """Keep recent turns only and strip heavy metadata to stay under Groq TPM limits."""
    slim: list[dict] = []
    for m in messages:
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content", "")
        if isinstance(content, str) and len(content) > 800:
            content = content[:800] + "…"
        slim.append({"role": role, "content": content, "ts": m.get("ts")})
    return slim[-_MAX_PIPELINE_TURNS:]


# ---------------------------------------------------------------------------
# Delegate endpoint – wrapper around RAG or Data agent delegation
# ---------------------------------------------------------------------------
@app.route("/delegate", methods=["POST"])
def delegate():
    """
    Accepts a JSON or form payload with a ``query`` field and returns the
    answer produced by either the RAG or Data agent.

    Supported fields:
      - query: required user question
      - context_data: optional text for the Data agent
      - agent: optional override, one of "rag", "data", "agent", or "auto"
      - model: optional Groq model name
      - api_key: optional Groq API key
    """
    try:
        agent_type = None
        payload = request.get_json(silent=True)
        raw_body = request.get_data(as_text=True).strip()
        if not payload and raw_body:
            # Support raw text payloads that embed the delegate function name.
            if raw_body.startswith("delegate_to_data_agent>"):
                raw_body = raw_body.split(">", 1)[1].strip()
                payload = json.loads(raw_body)
                agent_type = "data"
            elif raw_body.startswith("delegate_to_rag_agent>"):
                raw_body = raw_body.split(">", 1)[1].strip()
                payload = json.loads(raw_body)
                agent_type = "rag"
            else:
                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError:
                    payload = None
        if not payload:
            payload = request.form
        if not payload:
            return jsonify({"error": "Missing request payload"}), 400
        # Extract required fields
        query = payload.get("query", "").strip()
        if not query:
            return jsonify({"error": "Missing 'query' field"}), 400
        max_query_chars = 80
        if len(query) > max_query_chars:
            query = query[:max_query_chars] + "..."
            
        context_data = payload.get("context_data", "")
        # Overall character budget (~600 chars ≈ 150 tokens) for query+context – tighter limit
        max_total_chars = 600
        combined_len = len(query) + len(context_data)
        if combined_len > max_total_chars:
            allowed_context = max_total_chars - len(query)
            if allowed_context < 0:
                allowed_context = 0
            context_data = context_data[:allowed_context] + "..."
        # Additional safeguard: never let context_data alone exceed ~400 chars
        max_context_chars = 400
        if len(context_data) > max_context_chars:
            context_data = context_data[:max_context_chars] + "..."

        # Estimate token count (4 chars ≈ 1 token) and enforce a hard ceiling (~2500 tokens)
        est_tokens = (len(query) + len(context_data)) // 4
        if est_tokens > 2500:
            # Cut context_data further to respect token limit
            allowed_context = max(0, (2500 * 4) - len(query))
            context_data = context_data[:allowed_context] + "..."

        if agent_type is None:
            agent_type = (payload.get("agent") or "auto").strip().lower()
        # coding is an alias for the Python execution / math agent
        if agent_type in ("coding", "code"):
            agent_type = "data"

        model = payload.get("model") or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        api_key = payload.get("api_key") or os.environ.get("GROQ_API_KEY", "").strip()

        if "mixtral" in model.lower():
            model = "llama-3.1-8b-instant"

        if not api_key:
            return jsonify({"error": "Missing GROQ_API_KEY"}), 400

        if agent_type == "auto":
            agent_type = _detect_delegate_agent(query)

        if agent_type == "data":
            answer = _run_data_delegate(query, model=model, api_key=api_key, context_data=context_data)
        elif agent_type == "agent":
            answer = _run_agent_delegate(query, model=model, api_key=api_key)
        else:
            agent_type = "rag"
            answer = _run_rag_delegate(query, model=model)

        # Store interaction in history DB
        add_interaction(query, answer)
        return jsonify({"answer": answer, "agent": agent_type})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/query-explorer")
def query_explorer_page():
    return send_from_directory(_PROJECT_DIR / "student_management", "code_explorer.html")


@app.route("/code", methods=["GET", "POST"])
def generate_code():
    if request.method == "GET":
        return send_from_directory(_PROJECT_DIR, "code.html")
    """Generate ready‑to‑use source code for a user request.

    Expected JSON payload:
        {
            "description": "Brief natural‑language description of the desired code",
            "language": "optional, e.g. 'html', 'js', 'python' (defaults to a best‑guess)",
            "include_tests": false  # optional boolean – not used yet
        }
    The endpoint forwards the description to the Groq LLM with a system prompt that asks for **only** a fenced code block
    (no explanations). The LLM response is returned under the ``code`` key.
    """
    try:
        payload = request.get_json(silent=True) or {}
        description = payload.get("description", "").strip()
        if not description:
            return jsonify({"error": "Missing 'description' field"}), 400
        crud = is_crud_query(description)
        if crud:
            prompt = build_crud_prompt(description)
        else:
            prompt = (
                "Generate the complete source code for the following request. "
                "Return ONLY a markdown fenced code block with the appropriate language tag. "
                f"Do not include any explanation or surrounding text. Request: {description}"
            )
        model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        raw_answer, _ = answer_with_groq(
            model=model,
            api_key=api_key,
            query=prompt,
            context=""
        )
        import re
        from student_management.query_runner import run_query
        if crud:
            code = extract_python_code(raw_answer)
            filename = infer_filename(description)
            saved_path = save_query_file(code, filename)
            entry_id = add_code_entry(description, code)
            try:
                output = run_query(filename)
            except Exception as run_exc:
                output = f"Execution error: {run_exc}"
            return jsonify({
                "code": code,
                "id": entry_id,
                "saved_file": str(saved_path),
                "filename": saved_path.name,
                "crud": True,
                "output": output,
            })
        match = re.search(r"```(\w+)?\n([\s\S]*?)\n```", raw_answer)
        code = f"```{match.group(1) or ''}\n{match.group(2)}\n```" if match else raw_answer
        entry_id = add_code_entry(description, code)
        return jsonify({
            "code": code,
            "id": entry_id,
            "crud": False,
            "output": "Code generated (non-CRUD query, execution skipped)."
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/code-history", methods=["GET"])
def list_code_history():
    return jsonify(get_code_entries())


@app.route("/code-history", methods=["POST"])
def create_code_history():
    payload = request.get_json(silent=True) or {}
    description = (payload.get("description") or "").strip()
    code = (payload.get("code") or "").strip()
    if not description or not code:
        return jsonify({"error": "description and code are required"}), 400
    entry_id = add_code_entry(description, code)
    return jsonify({"id": entry_id, "description": description, "code": code}), 201


@app.route("/code-history/<int:entry_id>", methods=["GET"])
def read_code_history(entry_id):
    entry = get_code_entry(entry_id)
    if not entry:
        return jsonify({"error": "Not found"}), 404
    return jsonify(entry)


@app.route("/code-history/<int:entry_id>", methods=["PUT"])
def update_code_history(entry_id):
    payload = request.get_json(silent=True) or {}
    description = payload.get("description")
    code = payload.get("code")
    if description is None and code is None:
        return jsonify({"error": "No fields to update"}), 400
    if not update_code_entry(entry_id, description, code):
        return jsonify({"error": "Not found"}), 404
    return jsonify(get_code_entry(entry_id))


@app.route("/code-history/<int:entry_id>", methods=["DELETE"])
def delete_code_history(entry_id):
    if not delete_code_entry(entry_id):
        return jsonify({"error": "Not found"}), 404
    return jsonify({"msg": "deleted"})

# ---------------------------------------------------------------------------
# Placeholder for legacy delegate handling

# CRUD API for items
@app.route("/items", methods=["GET"]) 
def list_items():
    items = get_items()
    return jsonify(items)

@app.route("/items", methods=["POST"]) 
def create_item_endpoint():
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    description = payload.get("description")
    if not name:
        return jsonify({"error": "Missing 'name' field"}), 400
    item_id = add_item(name, description)
    return jsonify({"id": item_id, "name": name, "description": description}), 201

@app.route("/items/<int:item_id>", methods=["GET"]) 
def get_item_endpoint(item_id):
    item = get_item(item_id)
    if not item:
        return jsonify({"error": "Item not found"}), 404
    return jsonify(item)

@app.route("/items/<int:item_id>", methods=["PUT"]) 
def update_item_endpoint(item_id):
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    description = payload.get("description")
    if name is None and description is None:
        return jsonify({"error": "No fields to update"}), 400
    success = update_item(item_id, name, description)
    if not success:
        return jsonify({"error": "Item not found"}), 404
    return jsonify({"msg": "Item updated"})

@app.route("/items/<int:item_id>", methods=["DELETE"]) 
def delete_item_endpoint(item_id):
    success = delete_item(item_id)
    if not success:
        return jsonify({"error": "Item not found"}), 404
    return jsonify({"msg": "Item deleted"})

# Delegate handling moved to /delegate endpoint – nothing to do here
# ---------------------------------------------------------------------------
# Delegate handling moved to /delegate endpoint – nothing to do here

from RAG_groq import answer_with_groq, format_context, make_retriever

# ---------------------------------------------------------------------------
# HTML / CSS / JS  — single-file light-mode chat UI
# ---------------------------------------------------------------------------

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="color-scheme" content="light" />
  <title>Weaviate RAG · Powered by Groq</title>
  <meta name="description" content="Local RAG chat powered by Weaviate vector search and Groq LLM" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    /* ── Reset & Tokens ──────────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      /* 'only light' stops Chrome / OS from auto-inverting the page to dark */
      color-scheme: only light;
      --bg:         #ffffff;
      --surface:    #ffffff;
      --surface2:   #ffffff;
      --border:     #e2e8f0;
      --accent:     #3d5afe;
      --accent2:    #7c4dff;
      --accent-glow:rgba(61, 90, 254, 0.18);
      --green:      #059669;
      --green-dim:  rgba(5, 150, 105, 0.1);
      --red:        #c62828;
      --red-dim:    rgba(198, 40, 40, 0.08);
      --text:       #0f172a;
      --text-muted: #5c6b80;
      --text-dim:   #8b9aad;
      --user-bg:    linear-gradient(135deg, #dbeafe, #e0e7ff);
      --bot-bg:     linear-gradient(135deg, #fafafa, #f4f4f5);
      --radius:     14px;
      --font:       'Inter', system-ui, sans-serif;
    }

    /* ── Animations ── */
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.4); }
      70% { box-shadow: 0 0 0 6px rgba(5, 150, 105, 0); }
      100% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0); }
    }
    @keyframes pulse-red {
      0% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0.4); }
      70% { box-shadow: 0 0 0 6px rgba(198, 40, 40, 0); }
      100% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0); }
    }

    html {
      color-scheme: only light;
      background-color: #f8fafc;
    }

    html, body {
      height: 100%;
      background-color: #ffffff;
      background: #ffffff;
      color: var(--text);
      font-family: var(--font);
      font-size: 15px;
      line-height: 1.6;
    }

    /* ── Layout ──────────────────────────────────────────────────────── */
    .layout {
      display: grid;
      grid-template-columns: 260px 1fr 280px;
      grid-template-rows: 100vh;
      height: 100vh;
      overflow: hidden;
      background: #ffffff;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    .sidebar {
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      padding: 24px 18px;
      gap: 20px;
      overflow-y: auto;
    }

    .rightbar {
      background: var(--surface);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      padding: 24px 18px;
      gap: 20px;
      overflow-y: auto;
    }
    .stat-card {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 9px;
      padding: 12px;
      margin-bottom: 12px;
    }
    .stat-card h4 { font-size: 11px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; letter-spacing: 0.5px; }
    .stat-row { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
    .stat-row:last-child { margin-bottom: 0; }
    .stat-val { font-weight: 600; color: var(--text); }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--border);
    }
    .brand-icon {
      width: 36px; height: 36px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px;
      flex-shrink: 0;
      box-shadow: 0 0 16px var(--accent-glow);
    }
    .brand-name { font-size: 15px; font-weight: 700; letter-spacing: -0.3px; }
    .brand-sub  { font-size: 11px; color: var(--text-muted); }

    .section-label {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 1px;
      text-transform: uppercase;
      color: var(--text-dim);
      margin-bottom: 8px;
    }

    /* Status chip */
    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
      width: 100%;
    }
    .status-chip.ok   { background: var(--green-dim); color: var(--green);  border: 1px solid rgba(5, 150, 105, 0.22); }
    .status-chip.err  { background: var(--red-dim);   color: var(--red);    border: 1px solid rgba(198, 40, 40, 0.2); }
    .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .dot.green { background: var(--green); box-shadow: 0 0 4px rgba(5, 150, 105, 0.45); animation: pulse 2s infinite; }
    .dot.red   { background: var(--red);   box-shadow: 0 0 4px rgba(198, 40, 40, 0.35); animation: pulse-red 2s infinite; }

    /* Form controls */
    .field { display: flex; flex-direction: column; gap: 6px; }
    .field label { font-size: 12px; font-weight: 500; color: var(--text-muted); }

    input[type=text], input[type=password], input[type=number], select {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 9px;
      color: var(--text);
      font: inherit;
      font-size: 13px;
      padding: 9px 12px;
      width: 100%;
      outline: none;
      transition: border-color .2s, box-shadow .2s;
    }
    input:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }
    select option { background: var(--surface2); }

    /* Sidebar badge */
    .badge-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 12px;
      background: var(--surface2);
      border-radius: 9px;
      font-size: 12px;
      border: 1px solid var(--border);
    }
    .badge-row span { color: var(--text-muted); }
    .badge-row strong { color: var(--accent); }

    /* ── Main area ───────────────────────────────────────────────────── */
    .main {
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: #ffffff;
    }

    /* Top bar */
    .topbar {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--surface);
      flex-shrink: 0;
    }
    .topbar-title { font-size: 16px; font-weight: 600; }
    .topbar-model {
      font-size: 12px;
      padding: 4px 10px;
      background: var(--surface2);
      border-radius: 20px;
      border: 1px solid var(--border);
      color: var(--text-muted);
    }

    /* Chat window */
    .chat-window {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      scroll-behavior: smooth;
      background: #ffffff;
    }

    .chat-window::-webkit-scrollbar { width: 5px; }
    .chat-window::-webkit-scrollbar-track { background: transparent; }
    .chat-window::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

    /* Empty state */
    .empty-state {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      color: var(--text-muted);
      user-select: none;
    }
    .empty-icon {
      font-size: 48px;
      opacity: .5;
      animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
      0%,100% { transform: translateY(0); }
      50%      { transform: translateY(-8px); }
    }
    .empty-state h3 { font-size: 18px; color: var(--text); font-weight: 600; }
    .empty-state p  { font-size: 13px; text-align: center; max-width: 320px; }

    /* Message bubbles */
    .msg-row { display: flex; gap: 12px; animation: pop-in .25s ease; }
    @keyframes pop-in {
      from { opacity:0; transform:translateY(10px); }
      to   { opacity:1; transform:translateY(0); }
    }
    .msg-row.user     { flex-direction: row-reverse; }
    .msg-row.user .bubble   { background: var(--user-bg); border-bottom-right-radius: 4px; }
    .msg-row.assistant .bubble { background: var(--bot-bg); border-bottom-left-radius: 4px; }

    /* ── Agent thinking steps ── */
    .agent-steps { margin-bottom: 10px; }
    .agent-steps summary {
      cursor: pointer; font-size: 11.5px; font-weight: 600;
      color: var(--accent); user-select: none; padding: 4px 0;
    }
    .agent-step {
      display: flex; gap: 8px; align-items: flex-start;
      padding: 5px 8px; margin: 3px 0;
      border-radius: 7px; font-size: 12px;
    }
    .agent-step.search      { background: rgba(61,90,254,.07); }
    .agent-step.observation { background: rgba(5,150,105,.07); }
    .step-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }
    .step-body { flex: 1; overflow: hidden; }
    .step-label { font-weight: 600; font-size: 11px;
                  color: var(--text-muted); margin-bottom: 2px; }
    .step-preview { font-size: 11.5px; color: var(--text);
                    white-space: pre-wrap; word-break: break-word;
                    max-height: 80px; overflow: hidden; }
    .confidence-badge {
      display: inline-block; font-size: 10px; font-weight: 700;
      padding: 2px 7px; border-radius: 10px; margin-left: 6px;
      vertical-align: middle;
    }
    .conf-high   { background: rgba(5,150,105,.15); color: var(--green); }
    .conf-medium { background: rgba(234,179,8,.15);  color: #92400e; }
    .conf-low    { background: rgba(198,40,40,.10);  color: var(--red); }

    .avatar {
      width: 34px; height: 34px; border-radius: 10px;
      flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px;
    }
    .avatar.user   { background: linear-gradient(135deg, #93c5fd, #a5b4fc); border: 1px solid #818cf8; }
    .avatar.bot    { background: linear-gradient(135deg,var(--accent),var(--accent2)); box-shadow: 0 0 12px var(--accent-glow); }

    .bubble {
      max-width: 78%;
      padding: 12px 16px;
      border-radius: var(--radius);
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 14px;
      line-height: 1.65;
      border: 1px solid var(--border);
    }
    .role-tag {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: .5px;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 4px;
    }

    /* Error banner */
    .error-banner {
      margin: 0 24px;
      padding: 12px 16px;
      background: var(--red-dim);
      border: 1px solid rgba(198, 40, 40, 0.22);
      border-radius: 10px;
      color: var(--red);
      font-size: 13px;
      display: flex; align-items: center; gap: 8px;
      flex-shrink: 0;
    }

    /* Context drawer */
    .ctx-drawer {
      margin: 0 24px 4px;
      flex-shrink: 0;
    }
    .ctx-drawer details > summary {
      cursor: pointer;
      list-style: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 8px;
      transition: background .15s;
    }
    .ctx-drawer details > summary:hover { background: var(--surface2); }
    .ctx-drawer details > summary::marker { display: none; }
    .ctx-drawer pre {
      margin-top: 6px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text-muted);
      max-height: 220px;
      overflow-y: auto;
    }

    /* Input bar */
    .input-bar {
      padding: 16px 24px;
      border-top: 1px solid var(--border);
      background: var(--surface);
      flex-shrink: 0;
    }
    .input-row {
      display: flex;
      gap: 10px;
      align-items: flex-end;
    }
    .textarea-wrap { flex: 1; position: relative; }
    textarea#query {
      width: 100%;
      min-height: 52px;
      max-height: 160px;
      resize: vertical;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 12px;
      color: var(--text);
      font: inherit;
      font-size: 14px;
      padding: 14px 16px;
      outline: none;
      transition: border-color .2s, box-shadow .2s;
    }
    textarea#query:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }
    textarea#query::placeholder { color: var(--text-dim); }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      padding: 13px 20px;
      border-radius: 12px;
      border: none;
      cursor: pointer;
      font: inherit;
      font-size: 14px;
      font-weight: 600;
      transition: all .2s;
      white-space: nowrap;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      color: #fff;
      box-shadow: 0 4px 18px var(--accent-glow);
    }
    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 24px var(--accent-glow);
    }
    .btn-ghost {
      background: var(--surface2);
      color: var(--text-muted);
      border: 1px solid var(--border);
    }
    .btn-ghost:hover {
      background: #e2e8f0;
      color: var(--text);
    }

    /* Key shortcut hint */
    .shortcut-hint {
      font-size: 11px;
      color: var(--text-dim);
      margin-top: 6px;
      text-align: right;
    }

    /* ── Token status bar (Light Mode / Bottom Right) ─────────────────────────────── */
    .token-statusbar {
      position: fixed;
      bottom: 0;
      right: 0;
      z-index: 9999;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 6px 16px;
      background: var(--surface);
      border-top-left-radius: 12px;
      border: 1px solid var(--border);
      border-bottom: none;
      border-right: none;
      font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
      font-size: 11.5px;
      color: var(--text);
      user-select: none;
      box-shadow: -2px -2px 12px rgba(0,0,0,0.05);
      transition: opacity .3s;
    }
    .token-statusbar:hover { opacity: 1 !important; }

    .tsb-icon { font-size: 14px; }

    .tsb-label { color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: .5px; }

    .tsb-used  { color: var(--accent2); font-weight: 700; }
    .tsb-sep   { color: var(--text-dim); }
    .tsb-rem   { color: var(--green); font-weight: 600; }

    /* mini bar */
    .tsb-bar-wrap {
      width: 72px;
      height: 6px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 4px;
      overflow: hidden;
    }
    .tsb-bar-fill {
      height: 100%;
      border-radius: 4px;
      transition: width .5s ease, background .5s ease;
    }

    /* flash animation when a new response arrives */
    @keyframes tsb-flash {
      0%   { box-shadow: -2px -2px 12px rgba(0,0,0,0.05), 0 0 0 0 rgba(61, 90, 254, 0.4); }
      60%  { box-shadow: -2px -2px 12px rgba(0,0,0,0.05), 0 0 0 6px rgba(61, 90, 254, 0); }
      100% { box-shadow: -2px -2px 12px rgba(0,0,0,0.05), 0 0 0 0 rgba(61, 90, 254, 0); }
    }
    /* ── Chat Performance History Drawer & FAB ─────────────────────────────── */
    .history-fab {
      position: fixed;
      bottom: 50px; /* Above the token statusbar which is at bottom: 0 */
      right: 20px;
      z-index: 9998;
      width: 50px;
      height: 50px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      color: #ffffff;
      border: 1px solid rgba(255, 255, 255, 0.2);
      box-shadow: 0 4px 20px rgba(61, 90, 254, 0.35);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .history-fab:hover {
      transform: translateY(-3px) scale(1.06);
      box-shadow: 0 8px 28px rgba(61, 90, 254, 0.5);
    }
    .history-fab:active {
      transform: translateY(-1px) scale(0.98);
    }

    .history-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(15, 23, 42, 0.2);
      backdrop-filter: blur(4px);
      z-index: 10000;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }
    .history-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }

    .history-drawer {
      position: fixed;
      top: 0;
      right: -420px; /* Hide off-screen */
      width: 420px;
      max-width: 90vw;
      height: 100vh;
      background: var(--surface);
      border-left: 1px solid var(--border);
      box-shadow: -8px 0 32px rgba(15, 23, 42, 0.12);
      z-index: 10001;
      transition: right 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex;
      flex-direction: column;
    }
    .history-drawer.open {
      right: 0;
    }

    .history-drawer-header {
      padding: 20px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--surface2);
      flex-shrink: 0;
    }
    .history-drawer-header h3 {
      font-size: 16px;
      font-weight: 600;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .history-close-btn {
      background: transparent;
      border: none;
      font-size: 22px;
      cursor: pointer;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      transition: all 0.2s;
    }
    .history-close-btn:hover {
      background: var(--border);
      color: var(--text);
    }

    .history-drawer-content {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      scroll-behavior: smooth;
    }
    .history-drawer-content::-webkit-scrollbar { width: 5px; }
    .history-drawer-content::-webkit-scrollbar-track { background: transparent; }
    .history-drawer-content::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

    .history-empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      color: var(--text-muted);
      margin-top: 80px;
      text-align: center;
      padding: 0 20px;
    }
    .history-empty-icon {
      font-size: 40px;
      opacity: 0.6;
    }

    .history-card {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
    }
    .history-card:hover {
      border-color: var(--accent);
      box-shadow: 0 4px 16px rgba(61, 90, 254, 0.08);
      transform: translateY(-2px);
    }

    .history-card-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      color: var(--text-dim);
      border-bottom: 1px dashed var(--border);
      padding-bottom: 8px;
    }
    .history-card-time {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .history-card-query {
      font-size: 13.5px;
      font-weight: 600;
      color: var(--text);
      line-height: 1.5;
      cursor: pointer;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      transition: color 0.15s;
    }
    .history-card-query:hover {
      color: var(--accent);
    }
    .history-card-query.expanded {
      -webkit-line-clamp: unset;
      display: block;
    }

    .history-card-response {
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.6;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      cursor: pointer;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
      white-space: pre-wrap;
      word-break: break-word;
      transition: border-color 0.15s, color 0.15s;
    }
    .history-card-response:hover {
      border-color: var(--text-dim);
      color: var(--text);
    }
    .history-card-response.expanded {
      -webkit-line-clamp: unset;
      display: block;
    }

    .history-card-stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px 12px;
      margin-top: 4px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }
    .history-stat-box {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .history-stat-label {
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-dim);
      font-weight: 500;
    }
    .history-stat-value {
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .history-stat-value.highlight {
      color: var(--accent2);
    }
    .history-stat-value.latency {
      color: var(--green);
    }

    /* ── Responsive ──────────────────────────────────────────────────── */
    @media (max-width: 1024px) {
      .layout { grid-template-columns: 260px 1fr; }
      .rightbar { display: none; }
    }
    @media (max-width: 768px) {
      .layout { grid-template-columns: 1fr; grid-template-rows: auto 1fr auto; }
      .sidebar { border-right: none; border-bottom: 1px solid var(--border); padding: 16px; flex-direction: row; flex-wrap: wrap; gap: 12px; }
      .brand   { border-bottom: none; padding-bottom: 0; }
      .rightbar { display: none; }
    }
  </style>
</head>
<body>
<div class="layout">

  <!-- ── Sidebar ── -->
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-icon">🧠</div>
      <div>
        <div class="brand-name">Weaviate RAG</div>
        <div class="brand-sub">Powered by Groq</div>
      </div>
    </div>

    <!-- Weaviate status -->
    <div>
      <div class="section-label">Vector Store</div>
      {% if retriever == "vector" %}
        {% if weaviate_ok %}
          <div class="status-chip ok"><span class="dot green"></span>Weaviate connected</div>
        {% else %}
          <div class="status-chip err"><span class="dot red"></span>Weaviate offline — switch to BM25 or start Docker</div>
        {% endif %}
      {% else %}
        <div class="status-chip ok"><span class="dot green"></span>BM25 active — Weaviate not required</div>
      {% endif %}
    </div>

    <!-- Config form (inside the POST form below via JS submit) -->
    <form id="cfg-form" action="/" method="post" style="display:contents">
      {# chat_history is JSON — must be HTML-escaped or the first " breaks the attribute and the page. #}
      <input type="hidden" name="chat_history" id="hidden_chat_history" value="{{ chat_history|e }}" />

      <div>
        <div class="section-label">🔑 Groq API Key</div>
        <div class="field">
          <input id="api_key" name="api_key" type="password"
                 placeholder="sk-…  (or set GROQ_API_KEY)"
                 value="{{ api_key|e }}" autocomplete="off" />
        </div>
      </div>

      <div>
        <div class="section-label">🧠 Model</div>
        <div class="field">
          <input id="model" name="model" type="text"
                 value="{{ model|e }}" placeholder="llama-3.1-8b-instant" />
        </div>
      </div>

      <div>
        <div class="section-label">🔍 Retrieval Mode</div>
        <div class="field">
          <select id="retriever" name="retriever">
            <option value="vector" {% if retriever == "vector" %}selected{% endif %}>🔷 Vector (Weaviate + FastEmbed)</option>
            <option value="bm25"   {% if retriever == "bm25"   %}selected{% endif %}>🔑 BM25 (keyword)</option>
          </select>
        </div>
      </div>

      <div>
        <div class="section-label">⚙️ Pipeline Mode</div>
        <div class="field">
          <select id="rag_mode" name="rag_mode">
            <option value="simple" {% if rag_mode == "simple" %}selected{% endif %}>⚡ Simple RAG</option>
            <option value="context" {% if rag_mode == "context" %}selected{% endif %}>⚙️ Context Engineering</option>
            <option value="agent" {% if rag_mode == "agent" %}selected{% endif %}>🤖 AI Agent</option>
            <option value="autogen" {% if rag_mode == "autogen" %}selected{% endif %}>🤖 AutoGen Agent</option>
          </select>
        </div>
      </div>

      <div>
        <div class="section-label">📑 Top-K Chunks</div>
        <div class="field">
          <input id="top_k" name="top_k" type="number" min="1" max="10" value="{{ top_k|e }}" />
        </div>
      </div>

      <div class="badge-row">
        <span>Docs directory</span>
        <strong>docs/</strong>
      </div>

      <!-- Hidden query for sidebar form submissions -->
      <input type="hidden" name="query" id="hidden_query" />
      <input type="hidden" name="action" id="hidden_action" value="" />
    </form>

    <div style="margin-top:auto">
      <button class="btn btn-ghost" style="width:100%;font-size:13px;"
              onclick="setAction('clear');document.getElementById('cfg-form').submit()">
        🗑 Clear Chat
      </button>
    </div>
  </aside>

  <!-- ── Main ── -->
  <main class="main">

    <!-- Top bar -->
    <div class="topbar">
      <span class="topbar-title">💬 Chat</span>
      <span class="topbar-model">{{ model|e }}</span>
    </div>

    <!-- Chat window -->
    <div class="chat-window" id="chat-window">
      {% if messages %}
        {% for m in messages %}
          <div class="msg-row {{ m.role }}">
            {% if m.role == "user" %}
              <div class="avatar user">👤</div>
            {% else %}
              <div class="avatar bot">🤖</div>
            {% endif %}
            <div class="bubble">
              <div class="role-tag">{{ m.role|e }}
                {% if m.role == "assistant" and m.metadata %}
                  {% set meta = m.metadata | tojson | safe %}
                {% endif %}
              </div>

              {# ── Agent thinking steps ── #}
              {% if m.role == "assistant" and m.get("agent_steps") %}
                {% set steps = m.agent_steps %}
                {% set conf  = m.get("confidence", "medium") %}
                <details class="agent-steps">
                  <summary>🤖 Agent thinking · {{ steps | selectattr("kind","equalto","search") | list | length }} search(es)
                    <span class="confidence-badge conf-{{ conf }}">{{ conf }} confidence</span>
                  </summary>
                  {% for step in steps %}
                    {% if step.kind == "search" %}
                      <div class="agent-step search">
                        <span class="step-icon">🔍</span>
                        <div class="step-body">
                          <div class="step-label">Searching knowledge base</div>
                          <div class="step-preview">{{ step.label|e }}</div>
                        </div>
                      </div>
                    {% elif step.kind == "observation" %}
                      <div class="agent-step observation">
                        <span class="step-icon">📄</span>
                        <div class="step-body">
                          <div class="step-label">Retrieved context</div>
                          <div class="step-preview">{{ step.content[:300]|e }}{% if step.content|length > 300 %}…{% endif %}</div>
                        </div>
                      </div>
                    {% endif %}
                  {% endfor %}
                </details>
              {% endif %}

              {{ m.content|e }}
            </div>
          </div>
        {% endfor %}
      {% else %}
        <div class="empty-state">
          <div class="empty-icon">🔍</div>
          <h3>Ask anything about your docs</h3>
          <p>Drop <code>.txt</code> or <code>.md</code> files into the <code>docs/</code> folder, then start chatting.</p>
        </div>
      {% endif %}
    </div>

    <!-- Error banner -->
    {% if error %}
      <div class="error-banner">⚠️ {{ error|e }}</div>
    {% endif %}

    <!-- Retrieved context drawer -->
    {% if context %}
      <div class="ctx-drawer">
        <details>
          <summary>📄 Retrieved context ({{ context|length }} chars)</summary>
          <pre>{{ context|e }}</pre>
        </details>
      </div>
    {% endif %}

    <!-- Input bar -->
    <div class="input-bar">
      <div class="input-row">
        <div class="textarea-wrap">
          <textarea id="query" form="cfg-form"
                    placeholder="Ask a question about your documents…"
                    spellcheck="false"
                    required>{{ query_val|e }}</textarea>
        </div>
        <button class="btn btn-primary" id="send-btn"
                onclick="submitQuery(event)">
          ➤ Send
        </button>
      </div>
      <div class="shortcut-hint">Shift+Enter for new line · Enter to send</div>
    </div>

  </main>

  <!-- ── Right Sidebar ── -->
  <aside class="rightbar">
    <div class="brand-name" style="padding-bottom:18px; border-bottom: 1px solid var(--border);">📊 Diagnostics</div>
    
    <div class="stat-card">
      <h4>🚀 Pipeline Info</h4>
      <div class="stat-row"><span>💾 SQLite DB Messages</span> <span class="stat-val">{{ total_db_msgs }} total</span></div>
      {% if pipeline_meta %}
        {% if pipeline_meta.get("rag_mode") == "agent" %}
          <div class="stat-row" style="margin-top:4px;"><span>🔍 Searches Done</span> <span class="stat-val">{{ pipeline_meta.get("search_count", 0) }}</span></div>
          <div class="stat-row"><span>🎯 Confidence</span>
            <span class="confidence-badge conf-{{ pipeline_meta.get('confidence','medium') }}">{{ pipeline_meta.get("confidence","medium") }}</span>
          </div>
          {% if pipeline_meta.get("sources_used") %}
          <div class="stat-row" style="flex-direction:column; align-items:flex-start; gap:3px;">
            <span style="color:var(--text-muted); font-size:11px;">Queries used:</span>
            {% for src in pipeline_meta.sources_used %}
              <span style="font-size:11px; color:var(--text);">• {{ src|e }}</span>
            {% endfor %}
          </div>
          {% endif %}
        {% else %}
          <div class="stat-row" style="margin-top: 4px;"><span>⚖️ Adaptive Alpha</span> <span class="stat-val">{{ pipeline_meta.adaptive_alpha }}</span></div>
          <div class="stat-row" style="font-size: 11px; color: var(--text-muted); justify-content: flex-end; margin-bottom: 8px; margin-top: -2px;">
            <span>Semantic: {{ (pipeline_meta.adaptive_alpha * 100) | round | int }}% | Keyword: {{ ((1.0 - pipeline_meta.adaptive_alpha) * 100) | round | int }}%</span>
          </div>
          <div class="stat-row"><span>🎯 Reranker Engine</span> <span class="stat-val">{{ pipeline_meta.rerank_mode }}</span></div>
        {% endif %}
      {% endif %}
    </div>

    <div class="stat-card" style="font-size: 12.5px;">
      <h4>🧠 Active Algorithms</h4>
      <div class="stat-row"><span>⚖️ Alpha Selection</span> <span class="stat-val" style="font-size: 11px; font-weight: normal; color: var(--text-muted);">Regex Heuristics</span></div>
      <div class="stat-row"><span>⏳ Memory Decay</span> <span class="stat-val" style="font-size: 11px; font-weight: normal; color: var(--text-muted);">exp(-λ * t) Weighting</span></div>
      <div class="stat-row"><span>📐 Base Matcher</span> <span class="stat-val" style="font-size: 11px; font-weight: normal; color: var(--text-muted);">Bi-Encoder Cosine Sim</span></div>
      <div class="stat-row"><span>⚡ Fused Reranker</span> <span class="stat-val" style="font-size: 11px; font-weight: normal; color: var(--text-muted);">Hybrid Multi + Add</span></div>
      <div class="stat-row" style="font-size: 10px; color: var(--text-muted); justify-content: flex-end; margin-top: 2px;">
        <span>Score = (Base * Recency * Fresh * Import) + Overlap</span>
      </div>
    </div>

    {% if usage %}
    <div class="stat-card">
      <h4>🪙 Token Usage</h4>
      <div class="stat-row"><span>📝 Prompt</span> <span class="stat-val">{{ usage.prompt_tokens }}</span></div>
      <div class="stat-row"><span>✨ Completion</span> <span class="stat-val">{{ usage.completion_tokens }}</span></div>
      <div class="stat-row"><span>∑ Total</span> <span class="stat-val">{{ usage.total_tokens }}</span></div>
    </div>
    {% endif %}

    {% if pipeline_meta %}
    <div class="stat-card">
      <h4>⚙️ Context Pipeline Stats</h4>
      {% if pipeline_meta.retrieval_count is not none %}
      <div class="stat-row"><span>🔎 Retrieval Calls</span> <span class="stat-val">{{ pipeline_meta.retrieval_count }}</span></div>
      {% endif %}
      <div class="stat-row"><span>📥 Raw Retrieved</span> <span class="stat-val">{{ pipeline_meta.retrieved_raw or pipeline_meta.retrieved_raw_total }}</span></div>
      <div class="stat-row"><span>✂️ After Dedupe</span> <span class="stat-val">{{ pipeline_meta.after_dedupe or pipeline_meta.after_dedupe_total }}</span></div>
      <div class="stat-row"><span>🎯 After Re-rank</span> <span class="stat-val">{{ pipeline_meta.after_rerank or pipeline_meta.after_rerank_total }}</span></div>
      <div class="stat-row"><span>🗜️ Final Compressed</span> <span class="stat-val">{{ pipeline_meta.final_chunks or pipeline_meta.final_chunks_total }}</span></div>
      <div class="stat-row"><span>✅ Dedupe Keep%</span> <span class="stat-val">{{ pipeline_meta.dedupe_keep_pct }}%</span></div>
      <div class="stat-row"><span>❌ Dedupe Removed%</span> <span class="stat-val">{{ pipeline_meta.dedupe_removed_pct }}%</span></div>
      <div class="stat-row"><span>🗜️ Compression Kept%</span> <span class="stat-val">{{ pipeline_meta.compression_retained_pct }}%</span></div>
      <div class="stat-row"><span>📉 Compression Reduced%</span> <span class="stat-val">{{ pipeline_meta.compression_reduction_pct }}%</span></div>
      {% if pipeline_meta.rag_mode == 'agent' and pipeline_meta.retrieval_stats %}
      <div class="stat-row" style="flex-direction:column; align-items:flex-start; gap:4px; margin-top:10px;">
        <span style="font-size: 12px; color: var(--text-muted);">🔎 Per retrieval summary</span>
        {% for stat in pipeline_meta.retrieval_stats %}
          <div style="font-size: 11px; color: var(--text);">• {{ stat.query }} → raw {{ stat.retrieved_raw }}, deduped {{ stat.after_dedupe }}, reranked {{ stat.after_rerank }}, final {{ stat.final_chunks }}</div>
        {% endfor %}
      </div>
      {% endif %}
      
      <div class="stat-row" style="margin-top: 10px;"><span><strong>🔍 Keyword Matching</strong></span></div>
      {% if pipeline_meta.query_tokens is not none %}
      <div class="stat-row"><span>🔡 Query Tokens</span> <span class="stat-val">{{ pipeline_meta.query_tokens }}</span></div>
      {% endif %}
      {% if pipeline_meta.query_normalized_len is not none %}
      <div class="stat-row"><span>📏 Normalized Length</span> <span class="stat-val">{{ pipeline_meta.query_normalized_len }}</span></div>
      {% endif %}
    </div>
    
    {% if pipeline_meta.budget %}
    <div class="stat-card">
      <h4>🍰 Token Budget Split</h4>
      <div class="stat-row"><span>📥 Input Limit</span> <span class="stat-val">{{ pipeline_meta.budget.input_total }}</span></div>
      <div class="stat-row" style="margin-top:4px;"><span>🤖 System</span> <span class="stat-val">{{ pipeline_meta.budget.system }}</span></div>
      <div class="stat-row"><span>💬 History</span> <span class="stat-val">{{ pipeline_meta.budget.history }}</span></div>
      <div class="stat-row"><span>📄 Documents</span> <span class="stat-val">{{ pipeline_meta.budget.documents }}</span></div>
      <div class="stat-row"><span>❓ Query</span> <span class="stat-val">{{ pipeline_meta.budget.query }}</span></div>
    </div>
    {% endif %}
    {% endif %}
    
    {% if not usage and not pipeline_meta %}
      <div style="font-size: 12px; color: var(--text-muted); text-align: center; margin-top: 20px;">
        Submit a query to see pipeline statistics and token usage.
      </div>
    {% endif %}
  </aside>
</div>

<!-- ── VS Code-style Token Status Bar ── -->
{% set CTX_WINDOW = 6000 %}
{% if usage %}
  {% set used_tokens = usage.total_tokens %}
{% else %}
  {% set used_tokens = 0 %}
{% endif %}
{% set remaining = CTX_WINDOW - used_tokens %}
{% set pct = ((used_tokens / CTX_WINDOW) * 100) | round(1) %}

<div class="token-statusbar" id="tokenStatusBar"
     title="Context window: {{ CTX_WINDOW }} tokens total">

  <span class="tsb-icon">🪙</span>

  <div>
    <div class="tsb-label">tokens</div>
    <div>
      <span class="tsb-used">{{ used_tokens }}</span>
      <span class="tsb-sep"> / </span>
      <span style="color:var(--text-dim); font-size:11px;">{{ CTX_WINDOW }}</span>
    </div>
  </div>

  <div class="tsb-bar-wrap">
    <div class="tsb-bar-fill" id="tsbFill"
         style="width: {{ [pct, 100] | min }}%;
                background: {% if pct < 50 %}var(--green){% elif pct < 80 %}#f59e0b{% else %}var(--red){% endif %};"></div>
  </div>

  <div>
    <div class="tsb-label">remaining</div>
    <div class="tsb-rem">{{ [remaining, 0] | max }}</div>
  </div>

  {% if usage %}
  <span style="color:var(--text-muted); font-size:10px;">· ⬆ {{ usage.prompt_tokens }} ⬇ {{ usage.completion_tokens }}</span>
  {% endif %}
</div>

<script>
  // Scroll chat to bottom on load
  const cw = document.getElementById('chat-window');
  if (cw) cw.scrollTop = cw.scrollHeight;

  // Flash the token status bar when a fresh response just arrived
  {% if usage %}
  (function() {
    const bar = document.getElementById('tokenStatusBar');
    if (!bar) return;
    bar.classList.add('flash');
    bar.addEventListener('animationend', () => bar.classList.remove('flash'), { once: true });
  })();
  {% endif %}

  // Enter = submit, Shift+Enter = newline
  document.getElementById('query').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery(e);
    }
  });

  function setAction(val) {
    document.getElementById('hidden_action').value = val;
  }

  function submitQuery(e) {
    e.preventDefault();
    const qEl = document.getElementById('query');
    const val = qEl.value.strip ? qEl.value.strip() : qEl.value.trim();
    if (!val) return;

    document.getElementById('hidden_query').value = val;
    // Clear the box instantly for a better UX
    qEl.value = '';
    
    // Remove the textarea's 'required' to allow sidebar form submit
    qEl.removeAttribute('required');
    setAction('send');
    document.getElementById('cfg-form').submit();
  }

  // --- Updated history drawer logic ---
  function toggleHistory(open) {
    const overlay = document.getElementById('historyOverlay');
    const drawer = document.getElementById('historyDrawer');
    if (open) {
      renderHistory();
      overlay.classList.add('open');
      drawer.classList.add('open');
    } else {
      overlay.classList.remove('open');
      drawer.classList.remove('open');
    }
  }

  // Fetch and render stored history from server
  async function renderHistory() {
    const container = document.getElementById('historyDrawerContent');
    container.innerHTML = '';
    try {
      const response = await fetch('/history');
      const history = await response.json();
      if (!Array.isArray(history) || history.length === 0) {
        container.innerHTML = `
          <div class="history-empty-state">
            <div class="history-empty-icon">📜</div>
            <p>No queries recorded yet.</p>
            <p style="font-size:12px; color:var(--text-dim); margin-top: 4px;">Start a conversation to see performance metrics here.</p>
          </div>
        `;
        return;
      }
      // Transform rows into pairs of user and assistant messages
      const pairs = history.map(row => ({
        user: { content: row.user_message, ts: row.timestamp },
        assistant: { content: row.assistant_message, ts: row.timestamp }
      }));

      // Render from newest to oldest
      pairs.reverse().forEach(pair => {
        const card = document.createElement('div');
        card.className = 'history-card';

        const timeStr = formatTime(pair.user.ts || pair.assistant.ts);
        const metaHtml = `
          <div class="history-card-meta">
            <span class="history-card-time">📅 ${timeStr}</span>
          </div>
        `;
        const statsHtml = ''; // Placeholder for future stats

        const qContent = pair.user.content;
        const aContent = pair.assistant ? pair.assistant.content : '(Waiting for response...)';

        card.innerHTML = `
          ${metaHtml}
          <div class="history-card-query" onclick="this.classList.toggle('expanded')" title="Click to expand query">
            <strong>Q:</strong> ${escapeHtml(qContent)}
          </div>
          <div class="history-card-response" onclick="this.classList.toggle('expanded')" title="Click to expand response">
            <strong>A:</strong> ${escapeHtml(aContent)}
          </div>
          ${statsHtml}
        `;
        container.appendChild(card);
      });

// duplicate block removed

  }

  function escapeHtml(text) {
    if (!text) return '';
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
</script>

<!-- ── Performance History floating elements ── -->
<button class="history-fab" id="openHistoryBtn" title="View Performance History" onclick="toggleHistory(true)">
  ⏳
</button>

<div class="history-overlay" id="historyOverlay" onclick="toggleHistory(false)"></div>

<div class="history-drawer" id="historyDrawer">
  <div class="history-drawer-header">
    <h3>⏳ RAG Performance History</h3>
    <button class="history-close-btn" onclick="toggleHistory(false)">✕</button>
  </div>
  <div class="history-drawer-content" id="historyDrawerContent">
    <!-- Dynamic cards inserted here by JS -->
  </div>
</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

# Reuse the app object defined earlier so routes like /delegate remain registered.


def _check_weaviate() -> bool:
    """Check embedded Weaviate status via vector_store helper."""
    try:
        from vector_store import check_weaviate_ready
        return check_weaviate_ready()
    except Exception:
        return False


@app.route("/", methods=["GET", "POST"])
def index():
    query_val = ""
    api_key   = os.environ.get("GROQ_API_KEY", "").strip()
    retriever = os.environ.get("RAG_RETRIEVER", "bm25").strip().lower()
    if retriever not in ("vector", "bm25"):
        retriever = "bm25"
    rag_mode  = os.environ.get("RAG_MODE", "simple").strip().lower()
    if rag_mode not in ("simple", "context", "agent", "autogen"):
        rag_mode = "simple"
    model     = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    if "mixtral" in model.lower():
        model = "llama-3.1-8b-instant"
    top_k     = "4"
    import chat_memory
    messages: list[dict] = chat_memory.get_messages("default_session")
    context   = ""
    error     = ""
    chat_history_json = "[]"
    usage = None
    pipeline_meta = None
    total_db_msgs = 0

    try:
        import chat_memory
        total_db_msgs = len(chat_memory.get_messages("default_session"))
    except Exception:
        pass

    weaviate_ok = _check_weaviate() if retriever == "vector" else None

    if request.method == "POST":
        action       = (request.form.get("action") or "").strip()
        api_key      = (request.form.get("api_key") or api_key).strip()
        model        = (request.form.get("model") or model).strip()
        if "mixtral" in model.lower():
            model = "llama-3.1-8b-instant"
        top_k        = (request.form.get("top_k") or "4").strip()
        retriever    = (request.form.get("retriever") or retriever).strip().lower()
        if retriever not in ("vector", "bm25"):
            retriever = "bm25"
        rag_mode     = (request.form.get("rag_mode") or rag_mode).strip().lower()
        if rag_mode not in ("simple", "context", "agent", "autogen"):
            rag_mode = "simple"
        chat_history_json = (request.form.get("chat_history") or "[]").strip()

        weaviate_ok = _check_weaviate() if retriever == "vector" else None
        if retriever == "vector" and not weaviate_ok:
            error = "Weaviate offline — automatically using BM25 keyword retrieval."
            retriever = "bm25"

        try:
            import chat_memory
            messages = chat_memory.get_messages("default_session")
        except Exception:
            messages = []

        if action == "clear":
            import chat_memory
            chat_memory.clear_messages("default_session")
            messages = []
            context = ""
            error = ""
        else:
            query_val = (request.form.get("query") or "").strip()
            # Simple arithmetic handling: multiplication (e.g., "8 * 8" or "8 multiplied by 8")
            import re
            mul_match = re.match(r"^\s*(\d+)\s*(?:\*|multiplied\s+by)\s*(\d+)\s*$", query_val, re.I)
            if mul_match:
                a = int(mul_match.group(1))
                b = int(mul_match.group(2))
                answer = str(a * b)
                # Directly create assistant message without calling LLM pipeline
                bot_ts = time.time()
                chat_memory.save_message("default_session", "assistant", answer, bot_ts)
                messages.append({"role": "assistant", "content": answer, "ts": bot_ts})
                # Skip further processing
                usage = None
                pipeline_meta = None
                # Clear query and continue rendering
                query_val = ""
                # Jump to rendering stage by skipping the rest of the try block
                # Set a flag to indicate processed
                processed_simple = True
                # Legacy delegate handling removed
                pass

                # Continue to rendering without further processing
                import chat_memory
                user_ts = time.time()
                chat_memory.save_message("default_session", "user", query_val, user_ts)
                messages.append({"role": "user", "content": query_val, "ts": user_ts})
                
            if not query_val:
                error = "Please type a question before sending."
            else:
                import chat_memory
                user_ts = time.time()
                chat_memory.save_message("default_session", "user", query_val, user_ts)
                messages.append({"role": "user", "content": query_val, "ts": user_ts})
                
                # Start tracking latency
                start_time = time.time()
                try:
                    if locals().get('processed_simple'):
                        # Skip pipeline processing for simple arithmetic
                        pass
                    else:
                        top_k_int = max(1, int(top_k))
                    try:
                        docs_dir = Path(os.environ.get("RAG_DOCS_DIR", "docs")).resolve()
                    except (OSError, ValueError):
                        docs_dir = Path("docs").resolve()
                    final_key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
                    if not final_key:
                        error = "Groq API key missing — enter it in the sidebar or set GROQ_API_KEY in your .env file."
                        latency = time.time() - start_time
                        metadata = {
                            "latency": round(latency, 3),
                            "docs_pct": 0.0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "model": model,
                            "retriever": retriever,
                            "rag_mode": rag_mode,
                            "timestamp": time.time(),
                            "error": error
                        }
                        metadata_json = json.dumps(metadata)
                        bot_ts = time.time()
                        chat_memory.save_message("default_session", "assistant", error, bot_ts, metadata=metadata_json)
                        messages.append({"role": "assistant", "content": error, "ts": bot_ts, "metadata": metadata_json})
                    else:
                        if rag_mode == "agent":
                            from context_pipeline.config import PipelineConfig
                            from context_pipeline.pipeline import run_agent_pipeline

                            cfg = PipelineConfig.from_env()
                            pipeline_messages = _trim_messages_for_pipeline(messages)
                            result = run_agent_pipeline(
                                query=query_val,
                                docs_dir=docs_dir,
                                retriever_mode=retriever,
                                model=model,
                                api_key=final_key,
                                chat_messages=pipeline_messages,
                                cfg=cfg,
                            )
                            answer = result.answer or ""
                            context = result.context_display
                            usage = result.meta.get("usage")
                            pipeline_meta = result.meta

                        elif rag_mode == "autogen":
                            from context_pipeline.config import PipelineConfig
                            from context_pipeline.autogen_pipeline import run_autogen_pipeline

                            cfg = PipelineConfig.from_env()
                            pipeline_messages = _trim_messages_for_pipeline(messages)
                            result = run_autogen_pipeline(
                                query=query_val,
                                docs_dir=docs_dir,
                                retriever_mode=retriever,
                                model=model,
                                api_key=final_key,
                                chat_messages=pipeline_messages,
                                cfg=cfg,
                            )
                            answer = result.answer or ""
                            context = result.context_display
                            usage = result.meta.get("usage")
                            pipeline_meta = result.meta

                        elif rag_mode == "context":
                            # Advanced context-engineering path (hybrid oversample → dedupe → rerank →
                            # memory decay → compression → token budget → Groq).
                            from context_pipeline.config import PipelineConfig
                            from context_pipeline.pipeline import run_context_pipeline

                            os.environ["RAG_TOP_K"] = str(top_k_int)
                            cfg = PipelineConfig.from_env()
                            pipeline_messages = _trim_messages_for_pipeline(messages)
                            result = run_context_pipeline(
                                query=query_val,
                                docs_dir=docs_dir,
                                retriever_mode=retriever,
                                model=model,
                                api_key=final_key,
                                chat_messages=pipeline_messages,
                                cfg=cfg,
                            )
                            answer = result.answer
                            context = result.context_display
                            usage = result.meta.get("usage")
                            pipeline_meta = result.meta
                        else:
                            # Run the LangGraph state graph
                            from project.langchain.graph import _get_compiled
                            graph = _get_compiled()
                            res = graph.invoke({"query": query_val})
                            answer = res.get("answer", "")
                            compressed_chunks = res.get("compressed") or res.get("retrieved") or []
                            context = format_context(compressed_chunks)
                            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                        latency = time.time() - start_time
                        
                        # Calculate Documents Percentage Used
                        from context_pipeline.token_budget import count_tokens, compute_token_budget
                        from context_pipeline.config import PipelineConfig
                        doc_tokens = count_tokens(context or "", model)
                        if rag_mode in ("context", "agent", "autogen") and pipeline_meta and "budget" in pipeline_meta:
                            doc_budget = pipeline_meta["budget"]["documents"]
                        else:
                            try:
                                cfg = PipelineConfig.from_env()
                                budget = compute_token_budget(cfg, model)
                                doc_budget = budget.documents
                            except Exception:
                                doc_budget = 4750
                        docs_pct = round((doc_tokens / doc_budget) * 100, 1) if doc_budget > 0 else 0.0

                        if not answer:
                            answer = "(Model returned an empty response — try a different model or question.)"
                            error = "Groq returned an empty response."
                        else:
                            # Universal gate: never persist delegate_to_* syntax in chat
                            answer = _sanitize_delegate_answer(
                                answer,
                                query=query_val,
                                model=model,
                                api_key=final_key,
                            )

                        # Assemble metadata payload
                        metadata = {
                            "latency": round(latency, 3),
                            "docs_pct": docs_pct,
                            "prompt_tokens": usage.get("prompt_tokens", 0) if usage else 0,
                            "completion_tokens": usage.get("completion_tokens", 0) if usage else 0,
                            "total_tokens": usage.get("total_tokens", 0) if usage else 0,
                            "model": model,
                            "retriever": retriever,
                            "rag_mode": rag_mode,
                            "timestamp": time.time(),
                            "adaptive_alpha": pipeline_meta.get("adaptive_alpha") if (rag_mode == "context" and pipeline_meta) else None
                        }
                        metadata_json = json.dumps(metadata)
                        
                        bot_ts = time.time()
                        chat_memory.save_message("default_session", "assistant", answer, bot_ts, metadata=metadata_json)
                        msg_entry: dict = {"role": "assistant", "content": answer, "ts": bot_ts, "metadata": metadata_json}
                        if rag_mode in ("agent", "autogen") and pipeline_meta:
                            msg_entry["agent_steps"] = pipeline_meta.get("agent_steps", [])
                            msg_entry["confidence"]  = pipeline_meta.get("confidence", "medium")
                        messages.append(msg_entry)
                except Exception as exc:
                    latency = time.time() - start_time
                    error = f"Error: {exc}"
                    metadata = {
                        "latency": round(latency, 3),
                        "docs_pct": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "model": model,
                        "retriever": retriever,
                        "rag_mode": rag_mode,
                        "timestamp": time.time(),
                        "error": error
                    }
                    metadata_json = json.dumps(metadata)
                    bot_ts = time.time()
                    chat_memory.save_message("default_session", "assistant", error, bot_ts, metadata=metadata_json)
                    messages.append({"role": "assistant", "content": error, "ts": bot_ts, "metadata": metadata_json})
                
                # Always clear the input box after an attempt to send
                query_val = ""

        chat_history_json = json.dumps(messages)
        
        try:
            import chat_memory
            total_db_msgs = len(chat_memory.get_messages("default_session"))
        except Exception:
            pass

    return render_template_string(
        HTML,
        query_val=query_val,
        retriever=retriever,
        api_key=api_key,
        model=model,
        top_k=top_k,
        messages=messages,
        chat_history=chat_history_json,
        context=context,
        error=error,
        weaviate_ok=weaviate_ok,
        rag_mode=rag_mode,
        usage=usage,
        pipeline_meta=pipeline_meta,
        total_db_msgs=total_db_msgs,
    )


if __name__ == "__main__":
    import socket
    import threading

    import uvicorn

    def _warmup_heavy_models() -> None:
        """Load cross-encoder in background so the first chat request does not hang."""
        try:
            from context_pipeline.reranking import _get_cross_encoder
            _get_cross_encoder()
        except Exception:
            pass

    threading.Thread(target=_warmup_heavy_models, daemon=True).start()

    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "YOUR_IP"
    port = int(os.environ.get("PORT", "7860"))
    print("\nWeaviate RAG + FastAPI")
    print(f"  UI:             http://127.0.0.1:{port}")
    print(f"  Code Generator: http://127.0.0.1:{port}/code")
    print(f"  Query Explorer: http://127.0.0.1:{port}/query-explorer")
    print(f"  FastAPI:        http://127.0.0.1:{port}/api/students")
    print(f"  Student Docs:   http://127.0.0.1:{port}/api/students/docs")
    print(f"  Tools API:      http://127.0.0.1:{port}/api/tools")
    print(f"  Tools Docs:     http://127.0.0.1:{port}/api/tools/docs")
    print(f"  Tool Agent:     POST http://127.0.0.1:{port}/api/tools/agent")
    print(f"  Network:        http://{local_ip}:{port}\n")

    import socket as _socket
    probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    in_use = probe.connect_ex(("127.0.0.1", port)) == 0
    probe.close()
    if in_use:
        print(f"ERROR: Port {port} is already in use. Stop the other server first.\n")
        raise SystemExit(1)

    from asgi import root as asgi_app
    uvicorn.run(asgi_app, host="0.0.0.0", port=port, log_level="info")
