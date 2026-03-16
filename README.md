# MarkMaster — AI-Assisted Academic Grading System

MarkMaster is an AI-powered exam grading system that scores handwritten student answer sheets against a model answer using semantic similarity. It supports two grading modes: Classic (upload model answer images) and RAG (grade from course materials stored in a vector database).

---

## Features

- **Handwritten OCR** — extracts text from handwritten exam papers using LLM vision
- **Semantic Scoring** — combines BERTScore and Sentence Embeddings for accurate grading
- **Multi-page Support** — handles answers that span multiple pages with smart boundary detection
- **Topic-by-topic Breakdown** — shows exactly which topics earned marks and which were missed
- **RAG Mode** — teacher uploads course materials (PDF, DOCX, PPTX, images) once; system generates model answers for any question from the knowledge base
- **Human Moderation** — examiner can enter their own score and get an AI analysis comparing it with the system score
- **Dark UI** — clean editorial interface built with Streamlit

---

## Grading Modes

### Classic Mode
```
Teacher uploads model answer images + student answer images → AI score + breakdown
```

### RAG Mode
```
1. Teacher uploads course materials → knowledge base built in ChromaDB
2. Teacher types exam questions → system retrieves relevant content and generates model answer
3. Teacher uploads student answer images → scored against RAG-generated model answer
```

---

## Project Structure

```
MarkMaster/
│
├── config.py                 ← shared models & API client (loaded once)
├── streamlit_app.py          ← main UI entry point
│
├── text_extracr.py           ← image → structured JSON (OCR via LLM)
├── merge_pages.py            ← stitch multi-page answers together
│
├── component_builder.py      ← flatten topics into weighted components
├── question_groupby.py       ← group components by question ID
├── matcher.py                ← semantic scoring (BERTScore + embeddings)
│
├── objective_2.py            ← flatten_text() helper for moderation
├── objective2_llm.py         ← AI vs human score analysis
│
├── loader.py                 ← load PDF, DOCX, PPTX, TXT, images
├── chunker.py                ← split pages into overlapping chunks
├── vector_store.py           ← ChromaDB store and retrieve
├── rag_answer_builder.py     ← retrieve chunks → synthesize model answer
│
├── upload_pics.py            ← handle Streamlit image uploads
│
├── .env.example              ← copy to .env and add your API key
├── .streamlit/
│   └── config.toml           ← dark theme configuration
└── requirements.txt
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/MarkMaster.git
cd MarkMaster
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
# Copy the example file
cp .env.example .env

# Open .env and add your Groq API key
GROQ_API_KEY=your_actual_key_here
```

Get your free Groq API key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`

---

## How Scoring Works

Each student answer is scored in two ways and the higher score is used:

**Holistic Score** — compares the entire answer as a whole using sentence embeddings (70%) and BERTScore (30%).

**Topic Score** — each topic in the model answer is matched to the best matching topic in the student answer using the same blend of embeddings and BERTScore.

```
Final Score = max(Holistic Score, Topic Score)
```

This means students always get the benefit of the doubt — if they wrote everything correctly but in a different structure, the holistic score saves them.

---

## Scoring Formula

```
Semantic similarity → linear score curve
similarity ≥ 0.90  →  Excellent  (90–100% of marks)
similarity ≥ 0.70  →  Good       (70–89% of marks)
similarity ≥ 0.45  →  Partial    (45–69% of marks)
similarity ≥ 0.20  →  Weak       (20–44% of marks)
similarity <  0.20  →  0 marks
```

---

## Models Used

| Model | Purpose |
|---|---|
| `meta-llama/llama-4-scout-17b-16e-instruct` | OCR extraction and RAG synthesis (Groq API) |
| `all-MiniLM-L6-v2` | Sentence embeddings for semantic similarity |
| `distilbert-base-uncased` | BERTScore token-level matching |

---

## Requirements

- Python 3.9+
- Groq API key (free at console.groq.com)
- Internet connection for first model download (~100MB)

---

## Tech Stack

- **UI** — Streamlit
- **LLM API** — Groq (Llama 4 Scout)
- **Embeddings** — sentence-transformers
- **Semantic scoring** — bert-score
- **Vector database** — ChromaDB
- **Document loading** — PyMuPDF, python-docx, python-pptx

---

## Notes

- The `chroma_db/` folder is created automatically when you first build a knowledge base in RAG mode. Do not delete it between sessions — it stores your uploaded course materials.
- The `.env` file is listed in `.gitignore` and will never be uploaded to GitHub. Keep your API key safe.
- GPU is used automatically if available (CUDA). Falls back to CPU if not.