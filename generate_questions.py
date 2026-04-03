"""
generate_questions.py — Generate a Harry Potter Q&A question bank via RAG.

Usage:
    python generate_questions.py [--n 100] [--output questions.db]
                                 [--threshold 1.0] [--k-difficulty 100]
                                 [--collection hp_books] [--db ./chroma_db]

Requires the ChromaDB index to be built first:
    python build_index.py
"""

import argparse
import random
import re
import sqlite3
import string

import chromadb
import fitz
from sentence_transformers import SentenceTransformer
from transformers import pipeline

CHAT_MODEL        = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CHROMA_DB = "./chroma_db"
DEFAULT_COLLECTION = "hp_books"
PDF_PATH          = "harrypotter.pdf"
DEFAULT_DB        = "questions.db"
DEFAULT_K_DIFF    = 100
DEFAULT_N         = 100
DEFAULT_THRESHOLD = 1.0  # L2 distance threshold; ~cosine_sim 0.5 for normalised vectors
EMBED_MODEL       = "all-MiniLM-L6-v2"
MAX_ANSWER_WORDS  = 20
MIN_QUOTE_WORDS   = 8   # sequences of this many consecutive words trigger the copyright guardrail

DIFFICULTY_THRESHOLDS = {"easy": 15, "medium": 5}

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "for", "from", "had", "has", "have", "he", "her",
    "him", "his", "how", "i", "in", "is", "it", "its", "of", "on",
    "or", "our", "she", "so", "that", "the", "their", "them", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "who",
    "why", "with", "you", "your",
})

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
    book             TEXT,
    chapter          TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

SYSTEM_PROMPT = (
    "You are a Harry Potter quiz master. "
    "You write quiz questions based exclusively on the passage provided by the user. "
    "Rules:\n"
    "- Both the question and the answer must be grounded in the passage. "
    "Do not use any knowledge outside it.\n"
    "- The answer must be a word, name, or short phrase that appears in the passage.\n"
    "- Answers must be at most 15 words.\n"
    "- If the passage does not contain a clear, answerable fact, output nothing.\n"
    "You must respond using exactly this format, with no other text:\n"
    "Q: <your question>\n"
    "A: <your answer>"
)

GENERATION_PROMPT_TEMPLATE = (
    "<passage>\n{context}\n</passage>\n\n"
    "Using only the passage above, write exactly one {question_type} quiz question. "
    "The answer must be a word, name, or short phrase taken directly from the passage. "
    "Do not invent any facts. "
    "If the passage does not contain a clear fact to ask about, output nothing.\n"
    "Format (no other text):\n"
    "Q: <question>\n"
    "A: <answer>"
)


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_TABLE_SQL)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(questions)")}
    for col, typedef in [("book", "TEXT"), ("chapter", "TEXT")]:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE questions ADD COLUMN {col} {typedef}")
    conn.commit()
    return conn


def load_models() -> tuple:
    embedder = SentenceTransformer(EMBED_MODEL)
    gen = pipeline("text-generation", model=CHAT_MODEL, dtype="auto", device_map="auto")
    return embedder, gen


def get_chunk(collection, chunk_id: int) -> tuple:
    result = collection.get(ids=[str(chunk_id)], include=["documents", "metadatas"])
    return result["documents"][0], result["metadatas"][0].get("page", 0)


def build_chapter_map(pdf_path: str) -> list:
    """Return list of (toc_page_1indexed, book, chapter) sorted by page."""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # [(level, title, page), ...], pages are 1-indexed
    doc.close()
    entries = []
    current_book = ""
    for level, title, page in toc:
        if level == 1:
            current_book = title
        elif level == 2 and current_book:
            entries.append((page, current_book, title))
    return entries


def get_chapter_info(fitz_page: int, chapter_map: list) -> tuple:
    """Return (book, chapter) for a 0-indexed fitz page number."""
    book, chapter = "Harry Potter", ""
    for start_page, b, c in chapter_map:
        if fitz_page + 1 >= start_page:  # fitz is 0-indexed; TOC is 1-indexed
            book, chapter = b, c
        else:
            break
    return book, chapter


def generate_raw(chunk_text: str, question_type: str, gen, embedder, collection, k_context: int = 1) -> tuple:
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
    return "Q:" + out[0]["generated_text"].strip(), context


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

_VAGUE_REFS = re.compile(
    r"\b(the|this|that)\s+"
    r"(change|changes|situation|conversation|discussion|"
    r"incident|matter|affair|task|event|occurrence|following|above)\b",
    re.IGNORECASE,
)


def answer_retrievable(answer: str, embedder, collection, threshold: float) -> bool:
    """Return True if the answer can be retrieved from the index with distance < threshold."""
    emb = embedder.encode([answer]).tolist()
    results = collection.query(query_embeddings=emb, n_results=1, include=["distances"])
    return results["distances"][0][0] < threshold


def is_vague(question: str) -> bool:
    """Return True if the question contains a context-dependent reference."""
    return bool(_VAGUE_REFS.search(question))


def is_tautological(question: str, answer: str) -> bool:
    """Return True if most answer content words already appear in the question."""
    table = str.maketrans("", "", string.punctuation)
    answer_tokens   = set(answer.lower().translate(table).split()) - STOPWORDS
    question_tokens = set(question.lower().translate(table).split()) - STOPWORDS
    if not answer_tokens:
        return False
    return len(answer_tokens & question_tokens) / len(answer_tokens) >= 0.5


def answer_in_chunk(answer: str, chunk: str) -> bool:
    """Return True if at least half of the answer's content words appear in the chunk."""
    table = str.maketrans("", "", string.punctuation)
    answer_tokens = set(answer.lower().translate(table).split()) - STOPWORDS
    chunk_tokens  = set(chunk.lower().translate(table).split())
    if not answer_tokens:
        return False
    return len(answer_tokens & chunk_tokens) / len(answer_tokens) >= 0.5


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
                  difficulty: str, similarity_count: int, book: str, chapter: str) -> bool:
    try:
        conn.execute(
            "INSERT INTO questions"
            " (question, answer, source_chunk, difficulty, similarity_count, book, chapter)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (question, answer, source_chunk, difficulty, similarity_count, book, chapter),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HP quiz Q&A pairs via RAG.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of questions to generate")
    parser.add_argument("--output", type=str, default=DEFAULT_DB, help="SQLite output path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="L2 distance threshold for difficulty estimation")
    parser.add_argument("--k-difficulty", type=int, default=DEFAULT_K_DIFF,
                        help="Number of chunks to query for difficulty estimation")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help="ChromaDB collection name (default: hp_books)")
    parser.add_argument("--db", default=DEFAULT_CHROMA_DB,
                        help="ChromaDB persist directory (default: ./chroma_db)")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.db)
    collection = client.get_collection(args.collection)

    total = collection.count()
    if total == 0:
        raise SystemExit("Index is empty. Run build_index.py first.")

    k_diff = min(args.k_difficulty, total)
    chapter_map = build_chapter_map(PDF_PATH)
    conn = init_db(args.output)
    embedder, gen = load_models()

    saved = 0
    attempts = 0
    max_attempts = args.n * 3
    print(f"=== HP Quiz — question generator ===")
    print(f"Target: {args.n} questions | Max attempts: {max_attempts} | Output: {args.output}\n")

    while saved < args.n and attempts < max_attempts:
        chunk_id = random.randrange(total)
        chunk_text, chunk_page = get_chunk(collection, chunk_id)

        if len(chunk_text.strip()) < 50:
            continue  # boundary artefact, do not count as attempt

        attempts += 1
        question_type = random.choice(QUESTION_TYPES)
        raw, full_context = generate_raw(chunk_text, question_type, gen, embedder, collection)
        parsed = parse_qa(raw)

        if parsed is None:
            print(f"  [{attempts}] parse failure")
            continue

        question, answer = parsed
        if not is_valid_question(question):
            print(f"  [{attempts}] quality filter")
            continue

        if is_vague(question):
            print(f"  [{attempts}] vague filter")
            continue

        if contains_direct_quote(question, chunk_text) or contains_direct_quote(answer, chunk_text):
            print(f"  [{attempts}] copyright filter")
            continue

        if is_tautological(question, answer):
            print(f"  [{attempts}] tautology filter")
            continue

        if not answer_in_chunk(answer, full_context):
            print(f"  [{attempts}] grounding filter (answer not in context)")
            continue

        if not answer_retrievable(answer, embedder, collection, args.threshold):
            print(f"  [{attempts}] retrieval filter (answer not found in index)")
            continue

        difficulty, sim_count = estimate_difficulty(question, embedder, collection, k_diff, args.threshold)

        book, chapter = get_chapter_info(chunk_page, chapter_map)
        if save_question(conn, question, answer, chunk_text, difficulty, sim_count, book, chapter):
            saved += 1
            print(f"  [{attempts}] saved {saved}/{args.n} ({difficulty}) — {question[:70]}")
        else:
            print(f"  [{attempts}] duplicate, skipping")

    conn.close()
    print(f"\nDone. {saved} questions saved to '{args.output}' in {attempts} attempts.")


if __name__ == "__main__":
    main()
