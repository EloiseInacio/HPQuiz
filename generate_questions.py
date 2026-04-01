"""
generate_questions.py — Generate a Harry Potter Q&A question bank via RAG.

Usage:
    python generate_questions.py [--n 100] [--db questions.db]
                                 [--threshold 1.0] [--k-difficulty 100]

Requires the ChromaDB index to be built first:
    python build_index.py
"""

import argparse
import random
import re
import sqlite3

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

CHAT_MODEL        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHROMA_PATH       = "./chroma_db"
COLLECTION_NAME   = "hp_books"
DEFAULT_DB        = "questions.db"
DEFAULT_K_DIFF    = 100
DEFAULT_N         = 100
DEFAULT_THRESHOLD = 1.0  # L2 distance threshold; ~cosine_sim 0.5 for normalised vectors
EMBED_MODEL       = "all-MiniLM-L6-v2"
MAX_ANSWER_WORDS  = 20
MIN_QUOTE_WORDS   = 8   # sequences of this many consecutive words trigger the copyright guardrail

DIFFICULTY_THRESHOLDS = {"easy": 15, "medium": 5}

QUESTION_TYPES = [
    "factual",
    "about a specific character",
    "about a key event or scene",
    "about a spell, potion, or magical object",
    "about a location in the wizarding world",
    "about wizarding history or lore",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS questions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    question         TEXT    NOT NULL UNIQUE,
    answer           TEXT    NOT NULL,
    source_chunk     TEXT    NOT NULL,
    difficulty       TEXT    NOT NULL,
    similarity_count INTEGER NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

SYSTEM_PROMPT = (
    "You are a Harry Potter quiz master. "
    "You create quiz questions strictly grounded in the provided book passages. "
    "Answers must be concise: a name, a place, or a single sentence of at most 15 words. "
    "You must respond using exactly this format, with no other text:\n"
    "Q: <your question>\n"
    "A: <your answer>"
)

GENERATION_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Write exactly one {question_type} quiz question about Harry Potter "
    "based on the context above. "
    "The answer must be a name, a place, or one sentence of at most 15 words. "
    "Use this exact format with no other text:\n"
    "Q: <question>\n"
    "A: <answer>"
)


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def load_models() -> tuple:
    embedder = SentenceTransformer(EMBED_MODEL)
    gen = pipeline("text-generation", model=CHAT_MODEL, dtype="auto", device_map="auto")
    return embedder, gen


def get_chunk(collection, chunk_id: int) -> str:
    return collection.get(ids=[str(chunk_id)])["documents"][0]


def generate_raw(chunk_text: str, question_type: str, gen, embedder, collection, k_context: int = 1) -> str:
    q_emb = embedder.encode([chunk_text]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k_context, include=["documents"])
    related = results["documents"][0]
    context = chunk_text + "\n\n---\n\n" + "\n\n---\n\n".join(related)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": GENERATION_PROMPT_TEMPLATE.format(
            context=context,
            question_type=question_type,
        )},
    ]
    prompt = gen.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Prime generation with "Q:" so the model is constrained to start with the question.
    # return_full_text=False returns only the generated tokens; we prepend the primer back.
    out = gen(prompt + "Q:", max_new_tokens=192, return_full_text=False)
    return "Q:" + out[0]["generated_text"].strip()


def parse_qa(raw: str) -> tuple | None:
    pattern = re.compile(r"Q\s*:\s*(.+?)\s*\nA\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
    fallback = re.compile(r"Question\s*:\s*(.+?)\s*\nAnswer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)

    match = pattern.search(raw) or fallback.search(raw)
    if not match:
        return None

    question = match.group(1).strip()
    answer = re.sub(r"<[^>]+>", "", match.group(2)).strip()

    if len(question) < 10 or len(answer) < 5:
        return None
    if len(answer.split()) > MAX_ANSWER_WORDS:
        return None

    return question, answer


def estimate_difficulty(question: str, embedder, collection, k: int, threshold: float) -> tuple:
    q_emb = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k, include=["distances"])
    count = sum(1 for d in results["distances"][0] if d < threshold)

    if count > DIFFICULTY_THRESHOLDS["easy"]:
        return "easy", count
    if count >= DIFFICULTY_THRESHOLDS["medium"]:
        return "medium", count
    return "hard", count


_META_TERMS = frozenset({"context", "passage", "excerpt", "paraphrase", "summarize", "format"})


def contains_direct_quote(text: str, source_chunk: str, min_words: int = MIN_QUOTE_WORDS) -> bool:
    words = text.lower().split()
    chunk_lower = source_chunk.lower()
    return any(
        " ".join(words[i:i + min_words]) in chunk_lower
        for i in range(len(words) - min_words + 1)
    )


def is_valid_question(question: str) -> bool:
    q_lower = question.lower()
    if any(term in q_lower for term in _META_TERMS):
        return False
    if question.startswith('"'):
        return False
    if not question.rstrip().endswith("?"):
        return False
    return True


def save_question(conn: sqlite3.Connection, question: str, answer: str, source_chunk: str,
                  difficulty: str, similarity_count: int) -> bool:
    try:
        conn.execute(
            "INSERT INTO questions (question, answer, source_chunk, difficulty, similarity_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (question, answer, source_chunk, difficulty, similarity_count),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HP quiz Q&A pairs via RAG.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of questions to generate")
    parser.add_argument("--db", type=str, default=DEFAULT_DB, help="SQLite output path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="L2 distance threshold for difficulty estimation")
    parser.add_argument("--k-difficulty", type=int, default=DEFAULT_K_DIFF,
                        help="Number of chunks to query for difficulty estimation")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    total = collection.count()
    if total == 0:
        raise SystemExit("Index is empty. Run build_index.py first.")

    k_diff = min(args.k_difficulty, total)
    conn = init_db(args.db)
    embedder, gen = load_models()

    saved = 0
    attempts = 0
    max_attempts = args.n * 3
    print(f"=== HP Quiz — question generator ===")
    print(f"Target: {args.n} questions | Max attempts: {max_attempts} | DB: {args.db}\n")

    while saved < args.n and attempts < max_attempts:
        chunk_id = random.randrange(total)
        chunk_text = get_chunk(collection, chunk_id)

        if len(chunk_text.strip()) < 50:
            continue  # boundary artefact, do not count as attempt

        attempts += 1
        question_type = random.choice(QUESTION_TYPES)
        raw = generate_raw(chunk_text, question_type, gen, embedder, collection)
        parsed = parse_qa(raw)

        if parsed is None:
            print(f"  [{attempts}] parse failure")
            continue

        question, answer = parsed
        if not is_valid_question(question):
            print(f"  [{attempts}] quality filter")
            continue

        if contains_direct_quote(question, chunk_text) or contains_direct_quote(answer, chunk_text):
            print(f"  [{attempts}] copyright filter")
            continue

        difficulty, sim_count = estimate_difficulty(question, embedder, collection, k_diff, args.threshold)

        if save_question(conn, question, answer, chunk_text, difficulty, sim_count):
            saved += 1
            print(f"  [{attempts}] saved {saved}/{args.n} ({difficulty}) — {question[:70]}")
        else:
            print(f"  [{attempts}] duplicate, skipping")

    conn.close()
    print(f"\nDone. {saved} questions saved to '{args.db}' in {attempts} attempts.")


if __name__ == "__main__":
    main()
