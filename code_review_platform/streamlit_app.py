import streamlit as st
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CodeSentinel AI — Multi-Agent Code Review",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root / Body ───────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1426 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* ── Sidebar ────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1426 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #cbd5e1 !important;
}

/* ── Header Banner ─────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e3a5f 100%);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 18px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 25px 50px rgba(0,0,0,0.5);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #818cf8, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 400;
}

/* ── Score Ring Card ────────────────────────── */
.score-card {
    background: linear-gradient(135deg, #1e1b4b, #1a2744);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
.score-ring {
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1;
}
.grade-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 100px;
    font-size: 0.9rem;
    font-weight: 700;
    margin-top: 10px;
    letter-spacing: 1px;
}

/* ── Metric Card ────────────────────────────── */
.metric-card {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 20px 24px;
    height: 100%;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(99,102,241,0.5); }
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 4px;
}

/* ── Section Headers ────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}
.section-header .icon { font-size: 1.3rem; }
.section-header .title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a5b4fc;
}

/* ── Finding Cards ──────────────────────────── */
.finding-card {
    background: rgba(17,24,39,0.7);
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: background 0.2s;
}
.finding-card:hover { background: rgba(30,41,59,0.9); }
.finding-card.high  { border-left-color: #ef4444; }
.finding-card.medium { border-left-color: #f59e0b; }
.finding-card.low   { border-left-color: #22c55e; }
.finding-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.finding-message {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 6px;
    line-height: 1.5;
}
.finding-suggestion {
    font-size: 0.78rem;
    color: #60a5fa;
    font-style: italic;
    line-height: 1.4;
}
.finding-meta {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;
}
.sev-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.sev-high   { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.sev-medium { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.sev-low    { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.line-badge {
    font-size: 0.68rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Pass / Fail Badge ──────────────────────── */
.status-pass {
    background: rgba(34,197,94,0.15);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
}
.status-fail {
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
}

/* ── Strength / Rec Items ───────────────────── */
.strength-item {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: #86efac;
    display: flex;
    gap: 8px;
    align-items: flex-start;
}
.rec-item {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.83rem;
    color: #fde68a;
    display: flex;
    gap: 8px;
    align-items: flex-start;
}

/* ── History Table ──────────────────────────── */
.history-row {
    background: rgba(17,24,39,0.7);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.2s, background 0.2s;
    cursor: pointer;
}
.history-row:hover {
    border-color: rgba(99,102,241,0.4);
    background: rgba(30,41,59,0.8);
}

/* ── Agent Source Pill ──────────────────────── */
.source-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.25);
    margin-left: 8px;
}

/* ── Streamlit overrides ────────────────────── */
.stTextArea textarea {
    background: rgba(17,24,39,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stSelectbox > div > div {
    background: rgba(17,24,39,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stTextInput > div > div > input {
    background: rgba(17,24,39,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 28px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.6) !important;
}
div[data-testid="stTabs"] button {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a5b4fc !important;
    border-bottom-color: #6366f1 !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #4f46e5, #818cf8) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] {
    background: rgba(17,24,39,0.5) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_score_color(score: int) -> str:
    if score >= 90: return "#4ade80"
    if score >= 75: return "#a3e635"
    if score >= 60: return "#fbbf24"
    if score >= 40: return "#f97316"
    return "#f87171"

def get_grade_color(grade: str) -> str:
    colors = {"A+": "#4ade80", "A": "#86efac", "B": "#a3e635",
              "C": "#fbbf24", "D": "#f97316", "F": "#f87171"}
    return colors.get(grade, "#94a3b8")

def severity_class(sev: str) -> str:
    return {"high": "high", "medium": "medium", "low": "low"}.get(sev.lower(), "low")

def render_finding(f: dict):
    sev = f.get("severity", "low").lower()
    card_cls = severity_class(sev)
    line = f.get("line_number")
    line_html = f'<span class="line-badge">Line {line}</span>' if line else ""
    st.markdown(f"""
    <div class="finding-card {card_cls}">
        <div class="finding-meta">
            <span class="sev-badge sev-{sev}">{sev.upper()}</span>
            <span class="sev-badge" style="background:rgba(99,102,241,0.1);color:#a5b4fc;border:1px solid rgba(99,102,241,0.2)">
                {f.get('type','Unknown')}
            </span>
            {line_html}
        </div>
        <div class="finding-title">{f.get('message','')}</div>
        <div class="finding-suggestion">💡 {f.get('suggestion','')}</div>
    </div>
    """, unsafe_allow_html=True)

def render_agent_section(icon: str, title: str, report: dict):
    passed = report.get("pass", True)
    status_html = '<span class="status-pass">✓ PASS</span>' if passed else '<span class="status-fail">✗ FAIL</span>'
    source = report.get("source", "")
    source_html = f'<span class="source-pill">⚡ {source}</span>' if source else ""
    st.markdown(f"""
    <div class="section-header">
        <span class="icon">{icon}</span>
        <span class="title">{title}</span>
        {status_html}
        {source_html}
    </div>
    """, unsafe_allow_html=True)
    summary = report.get("summary", "")
    if summary:
        st.caption(summary)
    findings = report.get("findings", [])
    if findings:
        for f in findings:
            render_finding(f)
    else:
        st.markdown('<p style="color:#4ade80;font-size:0.85rem;">✅ No issues found.</p>', unsafe_allow_html=True)

def call_review_api(code: str, language: str, filename: str) -> dict:
    resp = requests.post(f"{API_BASE}/review", json={"code": code, "language": language, "filename": filename}, timeout=120)
    resp.raise_for_status()
    return resp.json()

def get_all_reviews() -> list:
    try:
        resp = requests.get(f"{API_BASE}/reviews", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []

def get_review_by_id(rid: int) -> dict:
    resp = requests.get(f"{API_BASE}/reviews/{rid}", timeout=10)
    resp.raise_for_status()
    return resp.json()

def delete_review(rid: int) -> bool:
    try:
        resp = requests.delete(f"{API_BASE}/reviews/{rid}", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:2.8rem;">🛡️</div>
        <div style="font-size:1.1rem; font-weight:700; color:#a5b4fc; margin-top:6px;">CodeSentinel AI</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Multi-Agent Review Platform</div>
    </div>
    <hr style="border-color:rgba(99,102,241,0.2); margin: 10px 0 20px;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🔍 Code Review", "📚 Review History", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style="border-color:rgba(99,102,241,0.2); margin: 20px 0 16px;">
    <div style="font-size:0.72rem; color:#475569; padding: 0 4px;">
        <div style="margin-bottom:8px;"><span style="color:#6366f1;">●</span> Bug Detection Agent</div>
        <div style="margin-bottom:8px;"><span style="color:#8b5cf6;">●</span> Code Quality Agent</div>
        <div style="margin-bottom:8px;"><span style="color:#3b82f6;">●</span> Security Review Agent</div>
        <div style="margin-bottom:8px;"><span style="color:#06b6d4;">●</span> Scoring Supervisor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:absolute; bottom:20px; left:0; right:0; text-align:center; font-size:0.7rem; color:#334155;">
        Powered by Gemini API
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: CODE REVIEW
# ─────────────────────────────────────────────
if page == "🔍 Code Review":
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🛡️ CodeSentinel AI</div>
        <div class="hero-subtitle">Multi-agent pipeline: Bug Detection · Code Quality · Security Review · Intelligent Scoring</div>
    </div>
    """, unsafe_allow_html=True)

    # Template snippets – must be defined BEFORE widgets so session_state is set first
    templates = {
        "SQL Injection": 'import sqlite3\n\ndef get_user(user_id):\n    conn = sqlite3.connect("db.sqlite")\n    cursor = conn.cursor()\n    query = f"SELECT * FROM users WHERE id = {user_id}"\n    cursor.execute(query)\n    return cursor.fetchone()',
        "Hardcoded Secret": 'import requests\n\napi_key = "sk-abc123secrettoken9876"\n\ndef fetch_data():\n    headers = {"Authorization": f"Bearer {api_key}"}\n    return requests.get("https://api.example.com/data", headers=headers)',
        "Division by Zero": 'def calculate_ratio(a, b):\n    return a / b\n\nresult = calculate_ratio(10, 0)\nprint(result)',
        "Mutable Default Arg": 'def add_item(item, collection=[]):\n    collection.append(item)\n    return collection\n\nprint(add_item("a"))\nprint(add_item("b"))',
        "Clean Code": 'def calculate_average(numbers: list[float]) -> float:\n    """\n    Calculate the arithmetic mean of a list of numbers.\n\n    Args:\n        numbers: A list of numeric values.\n\n    Returns:\n        The arithmetic mean as a float.\n\n    Raises:\n        ValueError: If the list is empty.\n    """\n    if not numbers:\n        raise ValueError("Cannot calculate average of empty list.")\n    return sum(numbers) / len(numbers)\n\nif __name__ == "__main__":\n    data = [10.5, 20.3, 15.8, 9.2]\n    print(f"Average: {calculate_average(data):.2f}")',
    }

    # Code Input Panel
    col_input, col_options = st.columns([3, 1])
    with col_options:
        st.markdown("**⚙️ Options**")
        language = st.selectbox("Language", ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c"], index=0)
        filename = st.text_input("Filename", value="main.py")
        st.markdown("<br>", unsafe_allow_html=True)

        # Template selectbox – on change, store selection in a staging key (NOT code_textarea key)
        st.markdown("**📋 Load Template**")
        template = st.selectbox(
            "Templates",
            ["— select —", "SQL Injection", "Hardcoded Secret", "Division by Zero", "Mutable Default Arg", "Clean Code"],
            label_visibility="collapsed",
            key="template_select"
        )
        if template in templates:
            # Store in a staging key; textarea will read from it on next render
            st.session_state["_pending_template"] = templates[template]

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    with col_input:
        st.markdown("**📝 Paste your code below**")
        # Populate textarea from pending template if set
        default_code = st.session_state.pop("_pending_template", st.session_state.get("code_textarea", ""))
        code_input = st.text_area(
            label="code",
            label_visibility="collapsed",
            height=320,
            placeholder="# Paste your Python (or other language) code here...\n\ndef example():\n    pass",
            value=default_code,
            key="code_textarea"
        )

    # ── Run Analysis ─────────────────────────────
    if run_btn:
        if not code_input.strip():
            st.error("⚠️ Please paste some code before running the analysis.")
        else:
            with st.spinner("🤖 Agents are analyzing your code…"):
                try:
                    result = call_review_api(code_input, language, filename)
                    st.session_state["last_result"] = result
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the API server at `http://127.0.0.1:8000`. Make sure `main.py` is running.")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ API Error: {e}")
                    st.stop()

    # ── Results ──────────────────────────────────
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        scoring = result.get("scoring_report", {})
        bug_report = result.get("bug_report", {})
        quality_report = result.get("quality_report", {})
        security_report = result.get("security_report", {})

        score = scoring.get("score", result.get("overall_score", 0))
        grade = scoring.get("grade", "?")
        score_color = get_score_color(score)
        grade_color = get_grade_color(grade)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        # Top metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            sc = get_score_color(score)
            st.markdown(f"""
            <div class="metric-card" style="border-color:{sc}40; text-align:center;">
                <div class="label">Overall Score</div>
                <div class="value" style="color:{sc}">{score}</div>
                <div class="sub">out of 100</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            gc = get_grade_color(grade)
            st.markdown(f"""
            <div class="metric-card" style="border-color:{gc}40; text-align:center;">
                <div class="label">Grade</div>
                <div class="value" style="color:{gc}">{grade}</div>
                <div class="sub">code quality</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            bug_count = len(bug_report.get("findings", []))
            bc = "#f87171" if bug_count > 0 else "#4ade80"
            st.markdown(f"""
            <div class="metric-card" style="border-color:{bc}40; text-align:center;">
                <div class="label">Bugs</div>
                <div class="value" style="color:{bc}">{bug_count}</div>
                <div class="sub">detected</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            sec_count = len(security_report.get("findings", []))
            sc2 = "#f87171" if sec_count > 0 else "#4ade80"
            st.markdown(f"""
            <div class="metric-card" style="border-color:{sc2}40; text-align:center;">
                <div class="label">Security</div>
                <div class="value" style="color:{sc2}">{sec_count}</div>
                <div class="sub">vulnerabilities</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            q_score = quality_report.get("score", 10)
            qc = get_score_color(int(q_score * 10))
            st.markdown(f"""
            <div class="metric-card" style="border-color:{qc}40; text-align:center;">
                <div class="label">Quality</div>
                <div class="value" style="color:{qc}">{q_score:.1f}</div>
                <div class="sub">out of 10</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Score progress bar
        st.markdown(f"**Score: {score}/100**")
        st.progress(score / 100)

        # Summary
        summary = scoring.get("summary", result.get("overall_summary", ""))
        if summary:
            st.info(f"📋 **Executive Summary:** {summary}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Strengths & Recommendations side by side
        strengths = scoring.get("strengths", [])
        recs = scoring.get("recommendations", [])
        if strengths or recs:
            col_s, col_r = st.columns(2)
            with col_s:
                st.markdown("""
                <div class="section-header">
                    <span class="icon">✅</span>
                    <span class="title">Strengths</span>
                </div>
                """, unsafe_allow_html=True)
                if strengths:
                    for s in strengths:
                        st.markdown(f'<div class="strength-item"><span>✦</span><span>{s}</span></div>', unsafe_allow_html=True)
                else:
                    st.caption("No strengths noted.")
            with col_r:
                st.markdown("""
                <div class="section-header">
                    <span class="icon">🎯</span>
                    <span class="title">Recommendations</span>
                </div>
                """, unsafe_allow_html=True)
                if recs:
                    for r in recs:
                        st.markdown(f'<div class="rec-item"><span>→</span><span>{r}</span></div>', unsafe_allow_html=True)
                else:
                    st.caption("No recommendations.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Agent Detail Tabs
        tab1, tab2, tab3 = st.tabs(["🐛 Bug Detection", "🎨 Code Quality", "🔒 Security"])
        with tab1:
            render_agent_section("🐛", "Bug Detection Agent", bug_report)
        with tab2:
            render_agent_section("🎨", "Code Quality Agent", quality_report)
        with tab3:
            render_agent_section("🔒", "Security Review Agent", security_report)

        # Raw JSON expander
        with st.expander("🗂️ View Raw JSON Response"):
            st.json(result)

        # Save confirmation
        rid = result.get("id")
        if rid:
            st.success(f"✅ Review saved to history with ID **#{rid}**")

# ─────────────────────────────────────────────
# PAGE: REVIEW HISTORY
# ─────────────────────────────────────────────
elif page == "📚 Review History":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📚 Review History</div>
        <div class="hero-subtitle">Browse and inspect all past code review sessions stored in the database.</div>
    </div>
    """, unsafe_allow_html=True)

    reviews = get_all_reviews()
    if not reviews:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#475569;">
            <div style="font-size:3rem; margin-bottom:16px;">📭</div>
            <div style="font-size:1.1rem; font-weight:600;">No reviews yet</div>
            <div style="font-size:0.85rem; margin-top:6px;">Run a code analysis to see results here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"**{len(reviews)} review(s) on record**")
        st.markdown("<br>", unsafe_allow_html=True)

        for rev in reviews:
            score = rev.get("overall_score", 0)
            sc = get_score_color(score)
            ts = rev.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts)
                ts_fmt = dt.strftime("%b %d, %Y  %H:%M")
            except Exception:
                ts_fmt = ts

            col_main, col_score, col_btn, col_del = st.columns([5, 1, 1, 1])
            with col_main:
                st.markdown(f"""
                <div style="padding: 4px 0;">
                    <span style="font-weight:600; color:#e2e8f0; font-size:0.9rem;">#{rev['id']} — {rev['filename']}</span>
                    <br>
                    <span style="font-size:0.75rem; color:#64748b;">{ts_fmt}</span>
                    <br>
                    <span style="font-size:0.78rem; color:#94a3b8;">{rev.get('overall_summary','')[:100]}…</span>
                </div>
                """, unsafe_allow_html=True)
            with col_score:
                st.markdown(f"""
                <div style="text-align:center; padding-top:6px;">
                    <span style="font-size:1.6rem; font-weight:700; color:{sc}">{score}</span>
                    <br><span style="font-size:0.7rem; color:#64748b;">/ 100</span>
                </div>
                """, unsafe_allow_html=True)
            with col_btn:
                if st.button("👁 View", key=f"view_{rev['id']}"):
                    st.session_state["viewing_review_id"] = rev["id"]
            with col_del:
                if st.button("🗑 Del", key=f"del_{rev['id']}"):
                    if delete_review(rev["id"]):
                        st.success(f"Deleted review #{rev['id']}")
                        st.rerun()
                    else:
                        st.error("Failed to delete.")
            st.divider()

        # Detailed view
        if "viewing_review_id" in st.session_state:
            rid = st.session_state["viewing_review_id"]
            try:
                detail = get_review_by_id(rid)
                scoring = detail.get("scoring_report", {})
                score = scoring.get("score", detail.get("overall_score", 0))
                grade = scoring.get("grade", "?")

                st.markdown(f"### 🔍 Review #{rid} — `{detail.get('filename','')}`")
                sc = get_score_color(score)
                gc = get_grade_color(grade)

                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center;">
                        <div class="label">Score</div>
                        <div class="value" style="color:{sc}">{score} / 100</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center;">
                        <div class="label">Grade</div>
                        <div class="value" style="color:{gc}">{grade}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs(["🐛 Bugs", "🎨 Quality", "🔒 Security"])
                with tab1:
                    render_agent_section("🐛", "Bug Detection", detail.get("bug_report", {}))
                with tab2:
                    render_agent_section("🎨", "Code Quality", detail.get("quality_report", {}))
                with tab3:
                    render_agent_section("🔒", "Security", detail.get("security_report", {}))

                with st.expander("📄 View Submitted Code"):
                    st.code(detail.get("code", ""), language="python")

                if st.button("← Back to list"):
                    del st.session_state["viewing_review_id"]
                    st.rerun()
            except Exception as e:
                st.error(f"Could not load review #{rid}: {e}")

# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">ℹ️ About CodeSentinel AI</div>
        <div class="hero-subtitle">Architecture, agents, and how the platform works.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🏗️ Architecture

    CodeSentinel AI uses a **multi-agent orchestration pipeline** powered by Google Gemini. Each agent specialises
    in a different dimension of code quality, and results are aggregated by a scoring supervisor.
    """)

    c1, c2 = st.columns(2)
    agents = [
        ("🐛", "Bug Detection Agent", "#ef4444",
         "Identifies syntax errors, logical bugs, runtime exceptions, infinite loops, and unhandled edge cases using Gemini AI with AST fallback."),
        ("🎨", "Code Quality Agent", "#8b5cf6",
         "Evaluates naming conventions, PEP 8 adherence, cyclomatic complexity, readability, docstring completeness, and maintainability."),
        ("🔒", "Security Review Agent", "#3b82f6",
         "Detects OWASP Top 10 vulnerabilities, hardcoded secrets, insecure cryptography, SQL injection, command injection, and RCE risks."),
        ("📊", "Scoring Supervisor", "#06b6d4",
         "Aggregates all agent reports into a deterministic 0–100 score and A+–F grade with prioritised recommendations."),
    ]
    for i, (icon, name, color, desc) in enumerate(agents):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}40; margin-bottom:16px;">
                <div style="font-size:1.6rem; margin-bottom:8px;">{icon}</div>
                <div style="font-weight:700; color:{color}; margin-bottom:6px;">{name}</div>
                <div style="font-size:0.83rem; color:#94a3b8; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    ## 🔌 API Endpoints

    | Method | Endpoint | Description |
    |--------|----------|-------------|
    | `POST` | `/review` | Submit code for multi-agent analysis |
    | `GET`  | `/reviews` | List all past reviews |
    | `GET`  | `/reviews/{id}` | Fetch a specific review |
    | `DELETE` | `/reviews/{id}` | Delete a review |
    | `GET`  | `/` | Serve the built-in HTML frontend |

    ## 🚀 Running the Stack

    ```bash
    # Terminal 1 — FastAPI backend
    python main.py

    # Terminal 2 — Streamlit frontend
    streamlit run streamlit_app.py
    ```
    """)
