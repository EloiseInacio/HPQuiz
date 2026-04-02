# Wizarding World Quiz

A Harry Potter quiz web app powered by a RAG pipeline over the full book series.
Questions are generated offline from the source text and served through a playful Flask interface.

## Overview

The project has two phases:

1. **Offline** — a RAG pipeline reads the HP books, retrieves relevant passages, and prompts a local LLM to generate grounded quiz questions stored in SQLite.
2. **Online** — a Flask web app serves the questions as interactive quizzes with difficulty selection, per-question scoring, and an emoji score summary.

## Stack

| Component | Choice |
|-----------|--------|
| Embedding model | `all-MiniLM-L6-v2` (384-dim, local) |
| Vector store | ChromaDB, persistent at `./chroma_db` |
| LLM (question generation) | `Qwen/Qwen2.5-1.5B-Instruct` (local, ~3 GB) |
| Question store | SQLite (`questions.db`) |
| Web framework | Flask + Tailwind CSS |
| Python env | conda `hpquiz`, Python 3.11 |

## Repo layout

```
HPQuiz/
├── build_index.py          # PDF → ChromaDB (run once)
├── generate_questions.py   # ChromaDB → questions.db
├── app.py                  # Flask web app
├── templates/              # Jinja2 templates (base, index, question, summary)
├── run.sh                  # Launch the web app
├── hp_rag.ipynb            # RAG testing notebook
├── requirements.txt
└── RAG_DESIGN.md           # Design documentation
```

`harrypotter.pdf` and `chroma_db/` are gitignored — see setup below.

## Setup

```bash
conda create -n hpquiz python=3.11
conda activate hpquiz
pip install -r requirements.txt

# Build the vector index (requires harrypotter.pdf in the project root)
python build_index.py

# Generate questions (downloads ~3 GB model on first run)
python generate_questions.py --n 100
```

## Running the app

```bash
./run.sh
# → http://127.0.0.1:5000
```

## Question generation

`generate_questions.py` samples random chunks from ChromaDB, prompts the LLM with a random question type (factual, character, event, spell/object, location, lore), and applies a multi-layer filter pipeline before saving:

| Filter | Purpose |
|--------|---------|
| Parse validation | Reject malformed Q/A output |
| Quality filter | Reject meta-questions and non-questions |
| Copyright guard | Reject answers containing direct quotes (8-word n-gram check) |
| Grounding check | Reject answers whose tokens don't appear in the retrieved context |
| Retrieval check | Reject answers the vector index can't retrieve (likely hallucinated) |

Difficulty is estimated by counting how many of the top-100 ChromaDB hits fall within L2 distance < 1.0 for the question embedding.

## Web app

Users pick difficulty (easy / medium / hard / any) and number of questions (1–20). Questions are shown one at a time with a free-text answer field. After each submission the correct answer is revealed with a pass/fail indicator. The summary page shows the final score, an emoji-rated message, and a collapsible per-question breakdown.

## Limitations

- Answer scoring is keyword-based (token set intersection). Short or proper-noun answers score well; verbose paraphrases may not.
- Generation quality is bounded by the local 1.5B model. A swap to `gpt-4o-mini` is a two-line change (replace `pipeline` with `openai.OpenAI()`).

## Disclaimer

This project is a technical experiment. It is not affiliated with J.K. Rowling, Warner Bros., or any Wizarding World entity. The books are used for retrieval only; no copyrighted text is reproduced or served to users.
