import streamlit as st
from text_extracr import process_folder
from component_builder import build_weighted_components
from matcher import score_student_answer
from objective_2 import flatten_text
from objective2_llm import compare_ai_with_human
from question_groupby import group_by_question
from merge_pages import merge_extractions
from upload_pics import prepare_upload_folder, clear_temp_folder, folder_has_images

DEFAULT_MODEL_FOLDER   = "model_answer"
DEFAULT_STUDENT_FOLDER = "student_answer"

st.set_page_config(page_title="MarkMaster", page_icon="📘", layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp { background: #0d0d12 !important; font-family: 'DM Mono', monospace !important; }
.stApp > header { background: transparent !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
.block-container { padding: 2rem 2.5rem 5rem !important; max-width: 1300px !important; }

/* ── Background atmosphere ── */
.stApp::before {
    content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 60% 40% at 5% 10%, rgba(240,192,96,0.05) 0%, transparent 50%),
        radial-gradient(ellipse 50% 35% at 95% 90%, rgba(120,100,240,0.05) 0%, transparent 50%);
}

/* ── Logo ── */
.logo-wrap { padding: 2.5rem 0 2rem; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 2.5rem; }
.logo-title {
    font-family: 'DM Serif Display', serif; font-size: 3.2rem; line-height: 1; letter-spacing:-0.02em;
    background: linear-gradient(120deg, #f0c060 0%, #ffe8a0 40%, #a090f0 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.logo-sub {
    font-family: 'Lora', serif; font-style: italic;
    font-size: 0.88rem; color: rgba(255,255,255,0.35); margin-top: 0.35rem;
}
.live-chip {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.2);
    padding: 0.35rem 0.85rem; border-radius: 100px;
    font-size: 0.65rem; color: #4ade80; letter-spacing: 0.12em;
}
.live-dot { width:6px; height:6px; border-radius:50%; background:#4ade80; box-shadow:0 0 8px #4ade80; animation: blink 2s ease infinite; }

/* ── Step headers ── */
.step-header {
    display: flex; align-items: center; gap: 0.9rem;
    margin: 2.2rem 0 1.2rem;
}
.step-num {
    width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
    background: rgba(240,192,96,0.12); border: 1px solid rgba(240,192,96,0.3);
    font-size: 0.7rem; color: #f0c060; display: flex; align-items: center; justify-content: center;
    font-weight: 500;
}
.step-title { font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(255,255,255,0.5); }
.step-line { flex: 1; height: 1px; background: rgba(255,255,255,0.06); }

/* ── Upload zones ── */
.upload-label {
    font-size: 0.8rem; color: rgba(255,255,255,0.6); margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.upload-icon-badge {
    width: 22px; height: 22px; border-radius: 6px;
    background: rgba(240,192,96,0.12); border: 1px solid rgba(240,192,96,0.25);
    display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem;
}
.file-count-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    margin-top: 0.6rem; padding: 0.3rem 0.75rem;
    background: rgba(74,222,128,0.07); border: 1px solid rgba(74,222,128,0.18);
    border-radius: 100px; font-size: 0.68rem; color: #4ade80;
}

/* ── Marks input row ── */
.marks-row {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.3rem 1.6rem;
    display: flex; align-items: center; gap: 1.5rem;
}
.marks-row-label { flex: 1; }
.marks-row-label .title { font-size: 0.85rem; color: rgba(255,255,255,0.8); margin-bottom: 0.2rem; }
.marks-row-label .hint  { font-size: 0.7rem; color: rgba(255,255,255,0.3); }

/* ── Progress steps ── */
.progress-wrap {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-top: 1.2rem;
}
.prog-step {
    display: flex; align-items: center; gap: 0.9rem;
    padding: 0.6rem 0.8rem; border-radius: 9px; font-size: 0.76rem;
    transition: all 0.3s;
}
.prog-step.done    { color: #4ade80; background: rgba(74,222,128,0.06); }
.prog-step.running { color: #f0c060; background: rgba(240,192,96,0.07); }
.prog-step.waiting { color: rgba(255,255,255,0.25); }
.prog-icon { font-size: 0.9rem; width: 20px; text-align: center; flex-shrink: 0; }

/* ── Score hero ── */
.score-hero {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 2.8rem 2rem; text-align: center;
    position: relative; overflow: hidden; margin-bottom: 2rem;
}
.score-hero::after {
    content: ''; position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(ellipse 55% 55% at 50% 50%, rgba(240,192,96,0.06), transparent);
}
.score-eyebrow { font-size: 0.6rem; letter-spacing: 0.3em; text-transform: uppercase; color: rgba(255,255,255,0.3); margin-bottom: 0.9rem; }
.score-big {
    font-family: 'DM Serif Display', serif; font-size: 5.5rem; line-height: 1;
    background: linear-gradient(135deg, #f0c060, #fffbe0, #a090f0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.score-denom { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: rgba(255,255,255,0.28); }
.score-pct { font-family: 'Lora', serif; font-style: italic; font-size: 0.95rem; color: rgba(255,255,255,0.38); margin-top: 0.5rem; }

/* ── Question result cards ── */
.q-result-card {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 13px; padding: 1.3rem 1.5rem; margin-bottom: 0.7rem;
    transition: border-color 0.2s, background 0.2s;
}
.q-result-card:hover { border-color: rgba(160,144,240,0.28); background: rgba(160,144,240,0.035); }
.q-label { font-family: 'DM Serif Display', serif; font-size: 1.25rem; color: #f0c060; }
.q-num   { font-family: 'DM Serif Display', serif; font-size: 1.7rem; color: #fff; }
.q-denom { font-size: 0.72rem; color: rgba(255,255,255,0.3); }
.q-bar-track { height: 3px; background: rgba(255,255,255,0.06); border-radius: 2px; margin-top: 0.9rem; }
.q-bar-fill  { height: 100%; border-radius: 2px; transition: width 1s ease; }

/* ── Grade badges ── */
.gbadge {
    display: inline-block; font-size: 0.6rem; padding: 0.2rem 0.6rem;
    border-radius: 100px; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 500;
}
.gb-excellent { background: rgba(74,222,128,0.1);  color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }
.gb-good      { background: rgba(78,205,196,0.1);  color: #4ecdc4; border: 1px solid rgba(78,205,196,0.25); }
.gb-partial   { background: rgba(251,191,36,0.1);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.gb-weak      { background: rgba(248,113,113,0.1); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }
.gb-none      { background: rgba(120,120,140,0.1); color: #888;    border: 1px solid rgba(120,120,140,0.2);  }

/* ── Breakdown table ── */
.bd-table { width: 100%; border-collapse: collapse; }
.bd-row { border-bottom: 1px solid rgba(255,255,255,0.04); }
.bd-row:last-child { border-bottom: none; }
.bd-row.is-overall { background: rgba(240,192,96,0.04); }
.bd-cell { padding: 0.85rem 0.5rem; font-size: 0.77rem; vertical-align: middle; }
.bd-topic { color: rgba(255,255,255,0.82); }
.bd-topic.overall { color: #f0c060; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; }
.bd-understanding { color: rgba(255,255,255,0.3); font-size: 0.68rem; }
.bd-marks { font-family: 'DM Serif Display', serif; font-size: 1rem; color: #fff; text-align: right; white-space: nowrap; }
.bd-marks span { color: rgba(255,255,255,0.3); font-family: 'DM Mono', monospace; font-size: 0.72rem; }

/* ── Moderation card ── */
.mod-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; overflow: hidden; margin-bottom: 1rem;
}
.mod-card-header {
    padding: 1rem 1.5rem; background: rgba(255,255,255,0.025);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex; align-items: center; justify-content: space-between;
}
.mod-q-title { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: #f0c060; }
.mod-ai-score { font-size: 0.72rem; color: rgba(255,255,255,0.35); }
.mod-card-body { padding: 1.4rem 1.5rem; }

/* ── Diff indicator ── */
.diff-row {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.5rem 0.8rem; border-radius: 8px; margin-top: 0.6rem;
    font-size: 0.72rem;
}
.diff-agree  { background: rgba(74,222,128,0.07);  color: #4ade80; border: 1px solid rgba(74,222,128,0.18);  }
.diff-higher { background: rgba(251,191,36,0.07);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.18);  }
.diff-lower  { background: rgba(248,113,113,0.07); color: #f87171; border: 1px solid rgba(248,113,113,0.18); }

/* ── Analysis result ── */
.analysis-result {
    margin-top: 1.2rem; padding: 1.2rem 1.4rem;
    background: rgba(78,205,196,0.04); border: 1px solid rgba(78,205,196,0.16);
    border-radius: 12px; font-family: 'Lora', serif; font-size: 0.9rem;
    color: rgba(255,255,255,0.72); line-height: 1.8;
}

/* ── Info strip ── */
.info-strip {
    display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center;
    padding: 0.9rem 1.4rem; margin-bottom: 1.8rem;
    background: rgba(120,100,240,0.05); border: 1px solid rgba(120,100,240,0.12);
    border-radius: 11px; font-size: 0.7rem; color: rgba(255,255,255,0.38);
}
.info-strip strong { color: #a090f0; }

/* ── Streamlit native overrides ── */
.stFileUploader > div {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(255,255,255,0.1) !important;
    border-radius: 13px !important; transition: border-color 0.25s !important;
}
.stFileUploader > div:hover { border-color: rgba(240,192,96,0.35) !important; }
.stFileUploader label { display: none !important; }
section[data-testid="stFileUploaderDropzone"] p { color: rgba(255,255,255,0.4) !important; font-size: 0.8rem !important; }

/* ── BUTTON: dark text on gold, always visible ── */
.stButton > button {
    background: linear-gradient(130deg, #f0c060 0%, #f8dc80 100%) !important;
    color: #111111 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.92rem !important; font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border: none !important; border-radius: 11px !important;
    padding: 0.85rem 2rem !important; width: 100% !important;
    box-shadow: 0 4px 22px rgba(240,192,96,0.25) !important;
    transition: all 0.22s !important;
    text-shadow: none !important; -webkit-text-fill-color: #111111 !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 28px rgba(240,192,96,0.38) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
/* Force all child text inside button to be dark */
.stButton > button *, .stButton > button span, .stButton > button p {
    color: #111111 !important; -webkit-text-fill-color: #111111 !important;
}
.stButton > button:disabled {
    background: rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.25) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.25) !important;
    box-shadow: none !important;
}

/* ── NUMBER INPUT: dark background, white text ── */
/* Target every possible Streamlit wrapper */
div[data-testid="stNumberInput"] input,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
input[type="number"],
.stNumberInput input {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
    border-radius: 10px !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.1rem !important;
    text-align: center !important;
    caret-color: #c07800 !important;
}
div[data-testid="stNumberInput"] input:focus,
input[type="number"]:focus {
    border-color: #c07800 !important;
    box-shadow: 0 0 0 2px rgba(192,120,0,0.15) !important;
    outline: none !important;
}

/* ── TEXT AREA: dark background, white text ── */
div[data-testid="stTextArea"] textarea,
div[data-baseweb="textarea"] textarea,
textarea {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
    border-radius: 10px !important;
    font-family: 'Lora', serif !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    caret-color: #6050c0 !important;
}
div[data-testid="stTextArea"] textarea:focus,
textarea:focus {
    border-color: #6050c0 !important;
    box-shadow: 0 0 0 2px rgba(96,80,192,0.12) !important;
    outline: none !important;
}
div[data-testid="stTextArea"] textarea::placeholder,
textarea::placeholder {
    color: rgba(0,0,0,0.35) !important;
    -webkit-text-fill-color: rgba(0,0,0,0.35) !important;
}
div[data-testid="stTextArea"] label,
.stTextArea label {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.72rem !important;
    font-family: 'DM Mono', monospace !important;
}

.stExpander {
    background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 13px !important; margin-bottom: 0.7rem !important;
}
.stExpander summary { color: rgba(255,255,255,0.65) !important; font-size: 0.82rem !important; }
.stExpander summary:hover { color: rgba(255,255,255,0.9) !important; }

.stSpinner > div > div { border-top-color: #f0c060 !important; }

.stAlert { border-radius: 11px !important; }
.stInfo    { background: rgba(120,100,240,0.07) !important; border-color: rgba(120,100,240,0.2) !important; }
.stSuccess { background: rgba(74,222,128,0.06)  !important; border-color: rgba(74,222,128,0.2)  !important; }
.stWarning { background: rgba(251,191,36,0.07)  !important; border-color: rgba(251,191,36,0.2)  !important; }
.stError   { background: rgba(248,113,113,0.07) !important; border-color: rgba(248,113,113,0.2) !important; }
div[data-testid="stAlert"] p { color: rgba(255,255,255,0.75) !important; font-size: 0.82rem !important; }

div[data-testid="stMarkdownContainer"] h2 { font-family: 'DM Serif Display', serif !important; font-weight: 400 !important; font-size: 1.5rem !important; color: rgba(255,255,255,0.88) !important; }
div[data-testid="stMarkdownContainer"] h3 { font-family: 'DM Serif Display', serif !important; font-weight: 400 !important; font-size: 1.1rem !important; color: rgba(255,255,255,0.55) !important; }
div[data-testid="stMarkdownContainer"] p  { font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; color: rgba(255,255,255,0.55) !important; line-height: 1.7 !important; }

hr { border-color: rgba(255,255,255,0.06) !important; margin: 2.5rem 0 !important; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
@keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def grade_from_ratio(r: float) -> str:
    if r >= 0.90: return "Excellent"
    if r >= 0.70: return "Good"
    if r >= 0.45: return "Partial"
    if r >  0:    return "Weak"
    return "Not Attempted"

GRADE_CSS = {
    "Excellent":     ("gb-excellent", "✦"),
    "Good":          ("gb-good",      "◈"),
    "Partial":       ("gb-partial",   "◇"),
    "Weak":          ("gb-weak",      "○"),
    "Not Attempted": ("gb-none",      "—"),
}

def grade_badge(grade: str) -> str:
    cls, icon = GRADE_CSS.get(grade, ("gb-none","—"))
    return f'<span class="gbadge {cls}">{icon} {grade}</span>'

def bar_color(r: float) -> str:
    if r >= 0.90: return "#4ade80"
    if r >= 0.70: return "#4ecdc4"
    if r >= 0.45: return "#fbbf24"
    if r >  0:    return "#f87171"
    return "#3a3a4a"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "processed": False, "question_scores": {}, "question_breakdowns": {},
    "model_answer": None, "student_answer": None, "analysis_results": {},
    "temp_model_folder": None, "temp_student_folder": None,
    "model_by_q": {}, "student_by_q": {}, "total_marks_used": 10.0,
    "model_pages_count": 0, "student_pages_count": 0,
    "model_q_count": 0, "student_q_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="logo-wrap">
  <div style="display:flex;align-items:flex-end;justify-content:space-between">
    <div>
      <div class="logo-title">MarkMaster</div>
      <div class="logo-sub">AI-Assisted Academic Grading System</div>
    </div>
    <div class="live-chip"><div class="live-dot"></div>System Ready</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-header">
  <div class="step-num">1</div>
  <div class="step-title">Upload Answer Sheets</div>
  <div class="step-line"></div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="upload-label">
      <span class="upload-icon-badge">📄</span>
      Model Answer — reference pages
    </div>
    """, unsafe_allow_html=True)
    model_uploads = st.file_uploader(
        "model_answer", type=["png","jpg","jpeg"],
        accept_multiple_files=True, label_visibility="collapsed",
        help="Upload all pages of the model/correct answer in order"
    )
    if model_uploads:
        st.markdown(
            f'<div class="file-count-badge">✓ {len(model_uploads)} '
            f'page{"s" if len(model_uploads)>1 else ""} ready</div>',
            unsafe_allow_html=True
        )

with col2:
    st.markdown("""
    <div class="upload-label">
      <span class="upload-icon-badge">✍️</span>
      Student Answer — pages to grade
    </div>
    """, unsafe_allow_html=True)
    student_uploads = st.file_uploader(
        "student_answer", type=["png","jpg","jpeg"],
        accept_multiple_files=True, label_visibility="collapsed",
        help="Upload all pages of the student's handwritten answer"
    )
    if student_uploads:
        st.markdown(
            f'<div class="file-count-badge">✓ {len(student_uploads)} '
            f'page{"s" if len(student_uploads)>1 else ""} ready</div>',
            unsafe_allow_html=True
        )

# Validate uploads and show clear guidance
if model_uploads and not student_uploads:
    st.info("📋 Model answer uploaded. Now upload the student's answer pages to continue.")
elif student_uploads and not model_uploads:
    st.info("📋 Student answer uploaded. Now upload the model answer pages to continue.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-header">
  <div class="step-num">2</div>
  <div class="step-title">Exam Configuration</div>
  <div class="step-line"></div>
</div>
""", unsafe_allow_html=True)

cfg1, cfg2 = st.columns([1, 3], gap="large")
with cfg1:
    st.markdown("**Total exam marks**")
    TOTAL_MARKS = st.number_input(
        "Total marks", min_value=1.0, max_value=500.0,
        step=5.0, value=10.0, label_visibility="collapsed",
        help="Total marks for the exam. These will be divided equally across all detected questions."
    )
with cfg2:
    st.markdown(
        f'<div style="padding-top:2rem;font-size:0.75rem;color:rgba(255,255,255,0.3);line-height:1.7">'
        f'Marks are distributed equally across all detected questions.<br>'
        f'E.g. 20 marks with 4 questions = 5 marks each.</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RUN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-header">
  <div class="step-num">3</div>
  <div class="step-title">Grade</div>
  <div class="step-line"></div>
</div>
""", unsafe_allow_html=True)

ready = bool(model_uploads or folder_has_images(DEFAULT_MODEL_FOLDER)) and \
        bool(student_uploads or folder_has_images(DEFAULT_STUDENT_FOLDER))

run_col, _ = st.columns([1, 2])
with run_col:
    run = st.button(
        "🚀  Start AI Grading" if ready else "⬆️  Upload files above to begin",
        use_container_width=True,
        disabled=not ready
    )

if not ready:
    st.markdown(
        '<div style="font-size:0.72rem;color:rgba(255,255,255,0.25);margin-top:0.5rem;text-align:center">'
        'Both model and student answer images are required</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
if run and ready:
    steps_placeholder = st.empty()

    def show_steps(current):
        steps = [
            ("📂", "Preparing uploaded files"),
            ("🔍", "Extracting text from images"),
            ("🔗", "Merging multi-page answers"),
            ("🧠", "Scoring with AI models"),
            ("✅", "Generating report"),
        ]
        html = '<div class="progress-wrap">'
        for i, (icon, label) in enumerate(steps):
            if i < current:
                cls = "done"; disp_icon = "✓"
            elif i == current:
                cls = "running"; disp_icon = icon
            else:
                cls = "waiting"; disp_icon = icon
            html += f'<div class="prog-step {cls}"><span class="prog-icon">{disp_icon}</span>{label}</div>'
        html += '</div>'
        steps_placeholder.markdown(html, unsafe_allow_html=True)

    show_steps(0)
    model_folder   = prepare_upload_folder(model_uploads,   "model_")   if model_uploads   else DEFAULT_MODEL_FOLDER
    student_folder = prepare_upload_folder(student_uploads, "student_") if student_uploads else DEFAULT_STUDENT_FOLDER

    show_steps(1)
    with st.spinner("Extracting text from images (this takes a moment)..."):
        model_pages    = process_folder(model_folder)
        student_pages  = process_folder(student_folder)

    show_steps(2)
    model_answer   = merge_extractions(model_pages)
    student_answer = merge_extractions(student_pages)
    model_q_count   = sum(len(m.get("questions",[])) for m in model_answer)
    student_q_count = sum(len(s.get("questions",[])) for s in student_answer)

    show_steps(3)
    with st.spinner("Scoring answers with BERTScore + Sentence Embeddings..."):
        model_components   = build_weighted_components(model_answer, TOTAL_MARKS)
        student_components = build_weighted_components(student_answer, TOTAL_MARKS)
        model_by_q   = group_by_question(model_components)
        student_by_q = group_by_question(student_components)

        unmatched = [q for q in model_by_q if q not in student_by_q]
        per_q     = TOTAL_MARKS / len(model_by_q) if model_by_q else TOTAL_MARKS

        question_scores = {}; question_breakdowns = {}; total_score = 0.0
        for qid in model_by_q:
            if qid in student_by_q:
                score, breakdown = score_student_answer(
                    model_by_q[qid], student_by_q[qid], total_marks=per_q
                )
            else:
                score = 0.0
                breakdown = [{
                    "topic": m["topic"], "marks_available": float(round(m["weight"],2)),
                    "marks_earned": 0.0, "contextual_understanding": 0.0,
                    "grade": "Not Attempted"
                } for m in model_by_q[qid] if m.get("topic","").strip()]
            question_scores[qid] = score
            question_breakdowns[qid] = breakdown
            total_score += score

    show_steps(4)

    st.session_state.update({
        "model_answer": model_answer, "student_answer": student_answer,
        "model_by_q": model_by_q, "student_by_q": student_by_q,
        "question_scores": question_scores, "question_breakdowns": question_breakdowns,
        "total_score": total_score, "total_marks_used": TOTAL_MARKS,
        "processed": True, "analysis_results": {},
        "model_pages_count": len(model_pages), "student_pages_count": len(student_pages),
        "model_q_count": model_q_count, "student_q_count": student_q_count,
    })
    if unmatched:
        st.warning(f"No student answer found for: **{', '.join(unmatched)}**. These questions scored 0.")

    clear_temp_folder(model_folder)
    clear_temp_folder(student_folder)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.processed:
    TM   = st.session_state.total_marks_used
    QS   = st.session_state.question_scores
    QB   = st.session_state.question_breakdowns
    TS   = st.session_state.total_score
    per_q = TM / len(QS) if QS else TM

    # ── STEP 4: Score summary ─────────────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
      <div class="step-num">4</div>
      <div class="step-title">Results</div>
      <div class="step-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # Info strip
    st.markdown(f"""
    <div class="info-strip">
      <span>📄 Model: <strong>{st.session_state.model_pages_count} pages → {st.session_state.model_q_count} question(s)</strong></span>
      <span>✍️ Student: <strong>{st.session_state.student_pages_count} pages → {st.session_state.student_q_count} question(s)</strong></span>
      <span>⚖️ Per question: <strong>{round(per_q, 1)} marks</strong></span>
    </div>
    """, unsafe_allow_html=True)

    # Score hero
    overall_ratio = TS / TM if TM else 0
    overall_grade = grade_from_ratio(overall_ratio)
    overall_pct   = round(overall_ratio * 100, 1)

    st.markdown(f"""
    <div class="score-hero">
      <div class="score-eyebrow">Total AI Score</div>
      <div>
        <span class="score-big">{round(TS, 2)}</span>
        <span class="score-denom"> / {TM}</span>
      </div>
      <div class="score-pct">{overall_pct}% &nbsp;·&nbsp; {overall_grade}</div>
    </div>
    """, unsafe_allow_html=True)

    # Question result cards
    st.markdown("#### Question-wise Scores")
    n = min(len(QS), 4)
    cols = st.columns(n, gap="medium")
    for i, (qid, score) in enumerate(QS.items()):
        ratio     = score / per_q if per_q else 0
        grade_str = grade_from_ratio(ratio)
        color     = bar_color(ratio)
        pct_fill  = round(ratio * 100, 1)
        attempted = qid in st.session_state.student_by_q
        flag_html = "" if attempted else '<span style="color:#fbbf24;font-size:0.65rem">⚠ Not attempted</span>'
        with cols[i % n]:
            st.markdown(f"""
            <div class="q-result-card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem">
                <span class="q-label">{qid}</span>
                {grade_badge(grade_str)}
              </div>
              <div style="display:flex;align-items:baseline;gap:0.4rem">
                <span class="q-num">{round(score,2)}</span>
                <span class="q-denom">/ {round(per_q,2)} marks &nbsp;{flag_html}</span>
              </div>
              <div class="q-bar-track">
                <div class="q-bar-fill" style="width:{pct_fill}%;background:{color}"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── STEP 5: Breakdown ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="step-header" style="margin-top:2.5rem">
      <div class="step-num">5</div>
      <div class="step-title">Topic-by-Topic Breakdown</div>
      <div class="step-line"></div>
    </div>
    <p>Click a question below to see how each topic was scored.</p>
    """, unsafe_allow_html=True)

    for qid, breakdown in QB.items():
        score     = QS.get(qid, 0)
        ratio     = score / per_q if per_q else 0
        grade_str = grade_from_ratio(ratio)

        with st.expander(f"{qid}  ·  {round(score,2)} / {round(per_q,2)} marks  ·  {grade_str}"):
            rows = ""
            for item in breakdown:
                topic    = item.get("topic","")
                avail    = item.get("marks_available", 0)
                earned   = item.get("marks_earned", 0)
                cu       = item.get("contextual_understanding", 0)
                grade_i  = item.get("grade","Not Attempted")
                is_ov    = topic == "Overall Answer"
                pct_cu   = round(cu * 100, 1)
                row_cls  = "bd-row is-overall" if is_ov else "bd-row"
                topic_cls = "bd-topic overall" if is_ov else "bd-topic"

                rows += f"""
                <tr class="{row_cls}">
                  <td class="bd-cell {topic_cls}" style="width:45%">{topic}</td>
                  <td class="bd-cell" style="width:25%">{grade_badge(grade_i)}<br>
                    <span class="bd-understanding">{pct_cu}% semantic match</span></td>
                  <td class="bd-cell bd-marks" style="width:30%;text-align:right">
                    {round(earned,2)} <span>/ {round(avail,2)}</span>
                  </td>
                </tr>"""

            st.markdown(
                f'<table class="bd-table"><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True
            )

    # ── STEP 6: Moderation ───────────────────────────────────────────────────
    st.markdown("""
    <div class="step-header" style="margin-top:2.5rem">
      <div class="step-num">6</div>
      <div class="step-title">Human Review &amp; Moderation</div>
      <div class="step-line"></div>
    </div>
    <p>
      Enter your own mark for each question and optionally add feedback on specific topics.
      The AI will compare its score with yours and explain any differences.
    </p>
    """, unsafe_allow_html=True)

    model_text   = flatten_text(st.session_state.model_answer)
    student_text = flatten_text(st.session_state.student_answer)

    for qid in QS:
        ai_score    = QS[qid]
        breakdown_q = QB[qid]

        st.markdown(f"""
        <div class="mod-card">
          <div class="mod-card-header">
            <span class="mod-q-title">{qid}</span>
            <span class="mod-ai-score">AI gave: {round(ai_score,2)} / {round(per_q,2)}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"Open moderation panel — {qid}"):
            left, right = st.columns([1, 2], gap="large")

            with left:
                st.markdown("**Your score for this question**")
                human_score = st.number_input(
                    f"Human score — {qid}",
                    min_value=0.0, max_value=float(TM),
                    step=0.5, value=float(round(ai_score, 1)),
                    key=f"hs_{qid}",
                    label_visibility="collapsed",
                    help=f"Enter the score you would give this student for {qid}"
                )

                diff = human_score - ai_score
                if abs(diff) < 0.26:
                    diff_cls  = "diff-agree"
                    diff_text = "✓ Your score agrees with the AI"
                elif diff > 0:
                    diff_cls  = "diff-higher"
                    diff_text = f"↑ You scored {round(diff,2)} marks higher than AI"
                else:
                    diff_cls  = "diff-lower"
                    diff_text = f"↓ You scored {abs(round(diff,2))} marks lower than AI"

                st.markdown(
                    f'<div class="diff-row {diff_cls}">{diff_text}</div>',
                    unsafe_allow_html=True
                )

            with right:
                st.markdown("**Topic feedback** *(optional — helps the AI understand your reasoning)*")
                human_feedback = {}
                for idx, item in enumerate(breakdown_q):
                    tn = item.get("topic","")
                    if tn and tn != "Overall Answer":
                        fb = st.text_area(
                            label=tn,
                            placeholder=f"e.g. Student explained {tn} correctly but missed details...",
                            key=f"fb_{qid}_{tn}_{idx}",
                            height=68
                        )
                        if fb.strip():
                            human_feedback[tn] = fb.strip()

            btn_c, _ = st.columns([1, 3])
            with btn_c:
                if st.button(f"⚡ Analyse discrepancy — {qid}", key=f"analyze_{qid}", use_container_width=True):
                    with st.spinner("Comparing AI and human judgment..."):
                        result = compare_ai_with_human(
                            model_text=model_text, student_text=student_text,
                            ai_score=ai_score, ai_breakdown=breakdown_q,
                            human_score=human_score,
                            human_feedback=human_feedback if human_feedback else None
                        )
                        st.session_state.analysis_results[qid] = result

            if st.session_state.analysis_results.get(qid):
                st.markdown(
                    f'<div class="analysis-result">'
                    f'<strong style="color:#4ecdc4;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase">Analysis</strong>'
                    f'<br><br>{st.session_state.analysis_results[qid]}</div>',
                    unsafe_allow_html=True
                )