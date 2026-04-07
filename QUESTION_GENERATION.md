# Question Generation Pipeline

This document covers `generate_questions.py` end-to-end: how a question-answer pair is
produced from the ChromaDB index, what checks it must pass to be saved, and how each
parameter affects the output.

---

## Overview

The pipeline runs a loop until `--n` questions are saved or the attempt budget is exhausted.
Each iteration:

1. Sample a random seed chunk, embed a short prefix, retrieve k related chunks
2. Assemble the context window from the anchor and related chunks
3. Prompt the LLM to generate a Q/A pair
4. Parse the raw output into `(question, answer)`
5. Run the filter pipeline — discard on any failure
6. Estimate difficulty
7. Look up book and chapter from the PDF table of contents
8. Save to SQLite

```
random seed chunk
     │
     ▼ embed 30-word prefix
 RAG retrieval  ──► anchor + k-1 related chunks
     │
     ▼
context assembly
     │
     ▼
   LLM prompt
     │
     ▼
  raw output
     │
     ▼
   parse_qa ──► None → parse failure
     │
     ▼
 is_valid_question ──► False → quality filter
     │
 is_vague ──────────► True  → vague filter
     │
 contains_direct_quote ──► True → copyright filter
     │
 is_tautological ────► True  → tautology filter
     │
 answer_in_chunk ────► False → grounding filter
     │
 answer_retrievable ─► False → retrieval filter
     │
     ▼
 estimate_difficulty
     │
 get_chapter_info
     │
     ▼
  save to DB
```

---

## Step 1 — Semantic cluster sampling

```python
seed_id   = random.randrange(total)
seed_text, _ = get_chunk(collection, seed_id)
seed      = " ".join(seed_text.split()[:SEED_WORDS])   # first 30 words
seed_emb  = embedder.encode([seed]).tolist()
results   = collection.query(query_embeddings=seed_emb, n_results=k, ...)
anchor_text, anchor_page = docs[0], metas[0].get("page", 0)
related   = docs[1:]
```

Rather than using a random chunk directly as the generation anchor, the pipeline picks a
random chunk as a **topic seed**, extracts its first 30 words as a short query, and retrieves
the `k` nearest neighbours from ChromaDB. The closest retrieved chunk becomes the **anchor**
(used for generation and book/chapter lookup); the remaining `k-1` chunks are the **related**
set. Together they form a semantically coherent topic cluster.

`k` is controlled by `--k-context` (default 5). Anchor and related chunks are all from the
same semantic neighbourhood, so validity checks operate on topically consistent context.

Anchors shorter than 50 characters are skipped without counting as an attempt — they are
boundary artefacts near page transitions.

---

## Step 2 — Context assembly

```python
full_context = anchor_text + "\n\n---\n\n" + "\n\n---\n\n".join(related)
```

The anchor chunk is prepended to the `k-1` related chunks, separated by `---` delimiters.
At `k=5`, the context is approximately 2,000–3,000 tokens — within Qwen2.5-1.5B's 32k
context window.

`full_context` is passed to the LLM and reused downstream by the grounding filter.

---

## Step 3 — LLM generation

`generate_raw(full_context, question_type, gen)` takes the pre-assembled context and returns
a raw string. Retrieval is the caller's responsibility; this function is purely generative.

The model is `Qwen/Qwen2.5-1.5B-Instruct`, loaded via the HuggingFace `transformers`
pipeline with `dtype="auto"` and `device_map="auto"`.

Two prompts are used:

**System prompt** — sets the task and hard constraints:
- Questions and answers must be grounded in the provided passage only.
- The answer must be a word, name, or short phrase present in the passage.
- Answers must be at most 15 words.
- If no clear fact is available, output nothing.
- Strict output format: `Q: ...\nA: ...`

**User prompt** — contains the assembled context and the randomly selected question type:

```
Passage:
<context>

Using only the passage above, write exactly one <question_type> quiz question.
The answer must be a word, name, or short phrase taken directly from the passage.
Do not invent any facts.
...
Q: <question>
A: <answer>
```

The six question types, sampled uniformly at random, are:
- factual
- about a specific character
- about a key event or scene
- about a spell, potion, or magical object
- about a location in the wizarding world
- about wizarding history or lore

The prompt is formatted using `tokenizer.apply_chat_template` and primed with `"Q:"` before
generation. This constrains the model to start its output with the question, improving
format compliance. `return_full_text=False` avoids having to strip the input prompt from
the output. `max_new_tokens=192` limits generation length.

---

## Step 4 — Parsing

`parse_qa` applies two regexes in order:

1. Primary: `Q\s*:\s*(.+?)\s*\nA\s*:\s*(.+)` (case-insensitive, dotall)
2. Fallback: `Question\s*:\s*(.+?)\s*\nAnswer\s*:\s*(.+)`

HTML-like tags (`<...>`) are stripped from the answer (the model sometimes wraps answers
in placeholder tags). The pair is rejected if:

- The question is shorter than 10 characters
- The answer is shorter than 5 characters
- The answer exceeds `MAX_ANSWER_WORDS` (20 words)

---

## Step 5 — Filter pipeline

Filters run cheapest-first. A rejection at any stage discards the attempt and starts the
next iteration.

### Quality filter — `is_valid_question`

Rejects questions that:
- Contain meta-terms: `context`, `passage`, `excerpt`, `paraphrase`, `summarize`, `format`
  (the model sometimes generates questions about the passage itself rather than its content)
- Start with a quotation mark (direct-speech questions rarely work standalone)
- Do not end with `?`

### Vague filter — `is_vague`

Rejects questions containing a demonstrative determiner (`the`, `this`, `that`) followed by
a generic context-dependent noun:

```
change, situation, conversation, discussion, incident, matter,
affair, task, event, occurrence, following, above
```

These patterns indicate the question only makes sense in reference to the source passage
— for example, "What did Harry think about the situation?" is unanswerable without context.

### Copyright filter — `contains_direct_quote`

Rejects questions or answers that reproduce 8 or more consecutive words from the source
chunk verbatim (case-insensitive). This is a sliding-window n-gram check:

```python
" ".join(words[i:i + 8]) in chunk_lower
```

The threshold of 8 words is a conservative proxy for the legal definition of substantial
reproduction. Shorter sequences (names, spells, places) are permitted.

### Tautology filter — `is_tautological`

Rejects pairs where 50% or more of the answer's content words already appear in the
question. Content words are defined as tokens remaining after stopword removal and
punctuation stripping.

This catches answers that simply repeat part of the question, e.g.:
> Q: What is Mad-Eye Moody's full name? A: Mad-Eye Moody

### Grounding filter — `answer_in_chunk`

Verifies that at least 50% of the answer's content words appear somewhere in the full
context (seed chunk + all `k_context` retrieved chunks). This catches hallucinated answers
that have no basis in the retrieved text.

The check is token-level, not substring-level, so minor inflection differences (e.g.
"wizard" vs "wizards") pass correctly.

### Retrieval filter — `answer_retrievable`

Embeds the answer string with `all-MiniLM-L6-v2` and queries ChromaDB for the single
nearest chunk. If the L2 distance to that nearest chunk exceeds `--threshold` (default
1.0, equivalent to cosine similarity ≈ 0.5 for normalised vectors), the answer is
considered absent from the entire corpus and the pair is discarded.

This is a post-hoc check against the full index, complementary to the grounding filter:
the grounding filter checks the generation context; the retrieval filter checks the whole
corpus. An answer absent from both is almost certainly hallucinated.

**L2 / cosine equivalence for normalised vectors:**
```
cosine_sim = 1 - L2² / 2
L2 = 1.0  →  cosine_sim ≈ 0.50
L2 = 0.8  →  cosine_sim ≈ 0.68  (tighter, fewer passes)
L2 = 1.2  →  cosine_sim ≈ 0.28  (looser, more passes)
```

---

## Step 6 — Difficulty estimation

```python
count = sum(1 for d in results["distances"][0] if d < threshold)
```

The question is embedded and the top `k` (default 100) chunks are retrieved. The number of
chunks with L2 distance below `threshold` is the *similarity count* — a proxy for how
often the relevant information recurs across the books.

| similarity_count | difficulty |
|-----------------|------------|
| > 15            | easy       |
| 5 – 15          | medium     |
| < 5             | hard       |

The rationale: if many chunks are close to the question, the answer is widely distributed
in the text (e.g. "Who is Harry Potter?"), making it easier. Rare, specific facts produce
few matches and are classified as hard.

---

## Step 7 — Book and chapter lookup

```python
book, chapter = get_chapter_info(chunk_page, chapter_map)
```

At startup, `build_chapter_map` parses the PDF table of contents with PyMuPDF, extracting
all level-1 entries (book titles) and level-2 entries (chapter titles) as a sorted list of
`(toc_page_1indexed, book, chapter)` tuples.

`get_chapter_info` converts the 0-indexed fitz page to 1-indexed and does a linear scan
through the sorted list, returning the last entry whose start page is ≤ the chunk page.

This metadata is stored in the database and displayed to the user in the web app. It is
never included in the LLM prompt.

---

## Step 8 — Persistence

Accepted pairs are inserted into `questions.db`:

| column           | type      | description                                  |
|------------------|-----------|----------------------------------------------|
| `question`       | TEXT UNIQUE | question text                              |
| `answer`         | TEXT      | expected answer                              |
| `source_chunk`   | TEXT      | the seed chunk used for generation           |
| `difficulty`     | TEXT      | `easy`, `medium`, or `hard`                  |
| `similarity_count` | INTEGER | number of close chunks (difficulty proxy)    |
| `book`           | TEXT      | book title from PDF TOC                      |
| `chapter`        | TEXT      | chapter title from PDF TOC                   |

The `UNIQUE` constraint on `question` silently discards duplicates across runs.

---

## Parameters

| argument        | default         | effect                                                      |
|-----------------|-----------------|-------------------------------------------------------------|
| `--n`           | 100             | target number of questions to save                          |
| `--output`      | `questions.db`  | SQLite output path                                          |
| `--threshold`   | 1.0             | L2 threshold for grounding/retrieval/difficulty checks      |
| `--k-difficulty`| 100             | number of chunks queried for difficulty estimation          |
| `--k-context`   | 5               | cluster size: anchor + related chunks for generation        |
| `--collection`  | `hp_books`      | ChromaDB collection name                                    |
| `--db`          | `./chroma_db`   | ChromaDB persist directory                                  |

The attempt budget is `n * 3`. If the budget is exhausted before `n` questions are saved,
the script exits with however many were collected. Increase `--n` or lower `--threshold`
to raise yield.

---

## Yield and failure modes

Typical yield with `Qwen2.5-1.5B-Instruct` and `--k-context 5` is ~35% of attempts.
The larger context window (5 chunks vs. the previous default of 1) raises the grounding bar,
which slightly reduces yield compared to smaller contexts but improves question quality.

The dominant rejection reasons in order of frequency:

1. **Retrieval filter** (~40% of rejects) — answer not found in the corpus; the model
   draws on parametric knowledge rather than the passage.
2. **Parse failure** (~20%) — model does not follow the `Q:/A:` format.
3. **Tautology / vague / quality** (~15% combined) — structural issues in the output.
4. **Grounding filter** (~15%) — answer tokens absent from the retrieved context.
5. **Copyright filter** (~10%) — direct quote reproduction.

Yield improves significantly with a stronger model (e.g. `gpt-4o-mini` is a drop-in
replacement: swap `pipeline` for `openai.OpenAI()` and update `generate_raw`).
