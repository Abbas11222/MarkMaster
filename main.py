import json
from text_extracr import process_folder
from component_builder import build_weighted_components
from matcher import score_student_answer
from question_groupby import group_by_question
from merge_pages import merge_extractions

MODEL_FOLDER   = "model_answer"
STUDENT_FOLDER = "student_answer"
TOTAL_MARKS    = 20

# ── Extract ──────────────────────────────────────────
model_pages   = process_folder(MODEL_FOLDER)
student_pages = process_folder(STUDENT_FOLDER)

# ── Merge multi-page answers ─────────────────────────
model_answer   = merge_extractions(model_pages)
student_answer = merge_extractions(student_pages)

print(f"\nModel:   {len(model_pages)} pages → {len(model_answer)} questions after merge")
print(f"Student: {len(student_pages)} pages → {len(student_answer)} questions after merge\n")

# ── Build components ─────────────────────────────────
model_components   = build_weighted_components(model_answer, TOTAL_MARKS)
student_components = build_weighted_components(student_answer, TOTAL_MARKS)

# ── Group by question ────────────────────────────────
model_by_q   = group_by_question(model_components)
student_by_q = group_by_question(student_components)

# ── Score per question ───────────────────────────────
question_scores     = {}
question_breakdowns = {}
total_score         = 0.0

per_question_marks = TOTAL_MARKS / len(model_by_q) if model_by_q else TOTAL_MARKS

for qid in model_by_q:
    q_model   = model_by_q[qid]
    q_student = student_by_q.get(qid, [])
    q_marks   = per_question_marks

    score, breakdown = score_student_answer(
        q_model,
        q_student,
        total_marks=q_marks
    )

    question_scores[qid]     = score
    question_breakdowns[qid] = breakdown
    total_score             += score

# ── Print results ────────────────────────────────────
print("\n================ FINAL RESULT ================\n")

for qid in question_scores:
    print(f"{qid} SCORE: {round(question_scores[qid], 2)} / {round(per_question_marks, 2)}")

print(f"\nTOTAL SCORE: {round(total_score, 2)} / {TOTAL_MARKS}\n")

print("Detailed Breakdown:")
print(json.dumps(question_breakdowns, indent=2))