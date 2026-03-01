import streamlit as st
import json
import os

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

st.set_page_config(page_title="MarkMaster", layout="wide")
st.title("📘 MarkMaster – AI Assisted Grading")


# ── Session state ─────────────────────────────────────
for key in [
    "processed", "analysis_done", "question_scores",
    "question_breakdowns", "model_answer", "student_answer",
    "analysis_results", "temp_model_folder", "temp_student_folder",
    "model_by_q", "student_by_q", "total_marks_used"
]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── Input ─────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Model Answer Images")
    model_uploads = st.file_uploader(
        "Upload model answer pages",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
with col2:
    st.subheader("Student Answer Images")
    student_uploads = st.file_uploader(
        "Upload student answer pages",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

st.header("AI Grading")
TOTAL_MARKS = st.number_input(
    "Enter total exam marks",
    min_value=1.0, step=5.0, value=10.0
)


# ── Process ───────────────────────────────────────────
if st.button("🚀 Start Processing", use_container_width=True):

    with st.spinner("Preparing files..."):
        model_folder   = prepare_upload_folder(model_uploads, "model_")   if model_uploads   else DEFAULT_MODEL_FOLDER
        student_folder = prepare_upload_folder(student_uploads, "student_") if student_uploads else DEFAULT_STUDENT_FOLDER
        st.session_state.temp_model_folder   = model_folder
        st.session_state.temp_student_folder = student_folder

    if not folder_has_images(model_folder):
        st.error("❌ Model answer images not found.")
        st.stop()
    if not folder_has_images(student_folder):
        st.error("❌ Student answer images not found.")
        st.stop()

    with st.spinner("Extracting and merging pages..."):
        model_pages    = process_folder(model_folder)
        student_pages  = process_folder(student_folder)
        model_answer   = merge_extractions(model_pages)
        student_answer = merge_extractions(student_pages)

        model_q_count   = sum(len(m.get("questions", [])) for m in model_answer)
        student_q_count = sum(len(s.get("questions", [])) for s in student_answer)

        st.info(
            f"📄 Model: {len(model_pages)} pages → {model_q_count} questions after merge  |  "
            f"Student: {len(student_pages)} pages → {student_q_count} questions after merge"
        )

    with st.spinner("Scoring..."):
        model_components   = build_weighted_components(model_answer, TOTAL_MARKS)
        student_components = build_weighted_components(student_answer, TOTAL_MARKS)

        model_by_q   = group_by_question(model_components)
        student_by_q = group_by_question(student_components)

        # ── KEY FIX: only score questions present in BOTH model and student ──
        # Questions in model but not in student = not attempted = 0
        # Don't let missing questions drag down per_question_marks calculation
        matched_qids   = [qid for qid in model_by_q if qid in student_by_q]
        unmatched_qids = [qid for qid in model_by_q if qid not in student_by_q]

        if unmatched_qids:
            st.warning(
                f"⚠️ No student answer found for: {', '.join(unmatched_qids)}. "
                f"These questions will score 0. "
                f"Please upload the student pages for these questions if available."
            )

        question_scores     = {}
        question_breakdowns = {}
        total_score         = 0.0

        # Marks per question based on total model questions (not just matched)
        # so unmatched questions still count as 0 toward total
        per_question_marks = TOTAL_MARKS / len(model_by_q) if model_by_q else TOTAL_MARKS

        for qid in model_by_q:
            if qid in student_by_q:
                score, breakdown = score_student_answer(
                    model_by_q[qid],
                    student_by_q[qid],
                    total_marks=per_question_marks
                )
            else:
                # Not attempted — score 0, show breakdown with Not Attempted
                score = 0.0
                breakdown = []
                for m in model_by_q[qid]:
                    if m.get("topic", "").strip():
                        breakdown.append({
                            "topic":                    m["topic"],
                            "marks_available":          float(round(m["weight"], 2)),
                            "marks_earned":             0.0,
                            "contextual_understanding": 0.0,
                            "coverage":                 0.0,
                            "grade":                    "Not Attempted"
                        })

            question_scores[qid]     = score
            question_breakdowns[qid] = breakdown
            total_score             += score

        st.session_state.model_answer        = model_answer
        st.session_state.student_answer      = student_answer
        st.session_state.model_by_q          = model_by_q
        st.session_state.student_by_q        = student_by_q
        st.session_state.question_scores     = question_scores
        st.session_state.question_breakdowns = question_breakdowns
        st.session_state.total_score         = total_score
        st.session_state.total_marks_used    = TOTAL_MARKS
        st.session_state.processed           = True
        st.session_state.analysis_results    = {}

        clear_temp_folder(st.session_state.temp_model_folder)
        clear_temp_folder(st.session_state.temp_student_folder)


# ── Results ───────────────────────────────────────────
if st.session_state.processed:

    total_marks_display = st.session_state.total_marks_used or TOTAL_MARKS

    st.success(
        f"✅ TOTAL AI SCORE: "
        f"{round(st.session_state.total_score, 2)} / {total_marks_display}"
    )

    st.subheader("📊 Question-wise Scores")
    per_q = total_marks_display / len(st.session_state.question_scores) \
            if st.session_state.question_scores else total_marks_display

    for qid, score in st.session_state.question_scores.items():
        attempted = qid in (st.session_state.student_by_q or {})
        label     = f"**{qid}** → {round(score, 2)} / {round(per_q, 2)} marks"
        if not attempted:
            label += " ⚠️ Not attempted"
        st.write(label)

    st.divider()
    st.subheader("📑 Question Breakdown")
    for qid in st.session_state.question_breakdowns:
        if st.button(f"Show Breakdown — {qid}", key=f"break_{qid}"):
            st.json(st.session_state.question_breakdowns[qid])

    # ── Objective 2 ───────────────────────────────────
    st.divider()
    st.header("Objective 2: Human vs AI Moderation")

    model_text   = flatten_text(st.session_state.model_answer)
    student_text = flatten_text(st.session_state.student_answer)

    for qid in st.session_state.question_scores:
        ai_score_q  = st.session_state.question_scores[qid]
        breakdown_q = st.session_state.question_breakdowns[qid]

        with st.expander(f"Moderation Panel — {qid}"):
            st.write(f"**AI Score:** {round(ai_score_q, 2)} / {round(per_q, 2)}")

            human_score = st.number_input(
                f"Human score for {qid}",
                min_value=0.0,
                max_value=float(total_marks_display),
                step=0.5,
                key=f"human_score_input_{qid}"
            )

            st.write("### Topic Feedback")
            human_feedback = {}
            for idx, item in enumerate(breakdown_q):
                if "topic" in item and item["topic"] != "Overall Answer":
                    topic_name = item["topic"]
                    fb = st.text_area(
                        label=f"Feedback for: {topic_name}",
                        key=f"fb_area_{qid}_{topic_name}_{idx}"
                    )
                    if fb.strip():
                        human_feedback[topic_name] = fb.strip()

            if st.button(f"Run Analysis — {qid}", key=f"btn_analyze_{qid}"):
                with st.spinner(f"Analyzing {qid}..."):
                    analysis = compare_ai_with_human(
                        model_text=model_text,
                        student_text=student_text,
                        ai_score=ai_score_q,
                        ai_breakdown=breakdown_q,
                        human_score=human_score,
                        human_feedback=human_feedback if human_feedback else None
                    )
                    st.session_state.analysis_results[qid] = analysis
                    st.success(f"Analysis for {qid} complete!")

            if st.session_state.analysis_results and \
               qid in st.session_state.analysis_results:
                st.markdown("---")
                st.markdown("### 📝 Analysis Result")
                st.info(st.session_state.analysis_results[qid])