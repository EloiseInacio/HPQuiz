# RAG Design — HP Quiz

## Table of contents

1. [What is RAG and why use it](#1-what-is-rag-and-why-use-it)
2. [Core concepts](#2-core-concepts)
   - [Tokens](#tokens)
   - [Embeddings](#embeddings)
   - [Vector similarity search](#vector-similarity-search)
3. [Pipeline overview](#3-pipeline-overview)
4. [Step-by-step breakdown](#4-step-by-step-breakdown)
   - [PDF extraction](#step-1-pdf-extraction)
   - [Regular chunking](#step-2-regular-chunking)
   - [Embedding and indexing](#step-3-embedding-and-indexing)
   - [Retrieval](#step-4-retrieval)
   - [Generation](#step-5-generation)
5. [Component choices](#5-component-choices)
6. [Key parameters and their effects](#6-key-parameters-and-their-effects)
7. [Limitations of this design](#7-limitations-of-this-design)
8. [Swapping to OpenAI](#8-swapping-to-openai)

---

## 1. What is RAG and why use it

A large language model (LLM) like TinyLlama or GPT-4o has knowledge baked in at training time. It cannot reliably answer questions about a specific document — it may hallucinate details, misquote, or simply not know content from a book it was never trained on.

**RAG (Retrieval-Augmented Generation)** solves this by splitting the problem in two:

1. **Retrieve** the passages in the document that are most relevant to the question.
2. **Generate** an answer by showing those passages to the LLM as context.

The LLM never needs to memorise the book. Instead, it reads the relevant excerpts at query time and synthesises an answer from them — much like a student answering open-book exam questions.

Think of it as a two-stage pipeline:

```
Question
   |
   v
[Retriever] -- searches the book --> relevant passages
   |
   v
[LLM] -- reads passages + question --> answer
```

---

## 2. Core concepts

### Tokens

LLMs do not process characters or words — they process **tokens**. A token is a subword unit produced by a tokeniser. As a rough rule of thumb:

- 1 token ≈ 4 characters in English
- 1 token ≈ 0.75 words
- "Dumbledore" → 3 tokens: `D`, `umb`, `ledore`

Tokens matter because both embedding models and LLMs have a maximum **context window** — a hard limit on how many tokens they can process at once. Everything in this pipeline is sized around tokens, not characters.

We use `tiktoken` (OpenAI's tokeniser library) to count and split tokens consistently, even though our models are not from OpenAI. `cl100k_base` is the encoding used by GPT-4 and is a sensible general-purpose choice.

### Embeddings

An **embedding** is a fixed-length list of floating-point numbers (a vector) that represents the *meaning* of a piece of text. The key property is:

> Texts with similar meaning produce vectors that are close together in space.

For example:
- "Harry cast a spell" and "Harry performed magic" → vectors very close together
- "Harry cast a spell" and "Hermione read a book" → vectors further apart
- "The Sorting Hat" and "income tax brackets" → vectors far apart

The embedding model (`all-MiniLM-L6-v2`) was trained specifically to produce this property. It is a small BERT-like neural network that maps any text to a 384-dimensional vector.

You can think of an embedding as a coordinate in a 384-dimensional space where proximity = semantic similarity.

### Vector similarity search

Given a question, we convert it to an embedding (a 384-dim vector), then search the database for the document chunks whose embeddings are closest to the question's embedding. "Closest" is measured by **cosine similarity** — the angle between two vectors.

This is fundamentally a nearest-neighbour search. ChromaDB handles this efficiently so you do not need to implement it yourself.

---

## 3. Pipeline overview

The pipeline has two distinct phases:

**Indexing phase** (runs once, offline):

```
PDF file
   |
   | PyMuPDF
   v
Raw text (6.3M characters)
   |
   | tiktoken fixed-size chunking
   v
3,485 chunks (~500 tokens each)
   |
   | all-MiniLM-L6-v2
   v
3,485 embeddings (384-dim vectors each)
   |
   | ChromaDB
   v
Persistent vector index on disk (./chroma_db)
```

**Query phase** (runs at every question):

```
Question string
   |
   | all-MiniLM-L6-v2
   v
Query embedding (384-dim)
   |
   | ChromaDB cosine similarity search
   v
Top-k chunks (5 most relevant passages)
   |
   | prompt assembly
   v
[system prompt] + [context passages] + [question]
   |
   | TinyLlama-1.1B-Chat
   v
Answer string
```

---

## 4. Step-by-step breakdown

### Step 1: PDF extraction

```python
def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)
```

`fitz` (PyMuPDF) opens the PDF and reads the text layer from each page. The result is a single Python string of 6.3 million characters — all seven books concatenated.

**What is discarded:** page headers/footers, page numbers, images, and any text stored as image (scanned PDFs would need OCR). For a digitally typeset PDF like this one, plain text extraction is clean.

---

### Step 2: Regular chunking

```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start += chunk_size - overlap
    return chunks
```

The full text is tokenised into a flat list of token IDs. Then a sliding window of size `CHUNK_SIZE` steps through that list with a stride of `CHUNK_SIZE - CHUNK_OVERLAP`, decoding each window back to a string.

**Why chunk at all?**
Embedding models have a maximum input length (typically 256–512 tokens). Passing the entire book as one unit is impossible. More importantly, a chunk that is too large produces a "blurry" embedding that mixes many topics — retrieval quality degrades.

**Why overlap?**
Without overlap, a sentence that happens to fall on a chunk boundary gets split in two. Each half ends up in a different chunk, possibly with insufficient context. A 50-token overlap means the last 50 tokens of chunk N are also the first 50 tokens of chunk N+1, preventing hard cuts from destroying meaning at boundaries.

**Visualised:**

```
tokens:  [  0 ...  49 |  50 ... 499 | 500 ... 949 | ...]
                         chunk 0
                                       chunk 1 (starts at 450, not 500)
                                                     chunk 2 (starts at 900)
```

With `CHUNK_SIZE=500` and `CHUNK_OVERLAP=50`, each new chunk starts `450` tokens after the previous one.

This produces **3,485 chunks** for the full HP collection.

---

### Step 3: Embedding and indexing

```python
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def build_index(chunks: list[str], collection) -> None:
    BATCH = 256
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
        ids = [str(i + j) for j in range(len(batch))]
        collection.add(embeddings=embeddings, documents=batch, ids=ids)
```

Each chunk is passed through `all-MiniLM-L6-v2`, which outputs a 384-dimensional vector. Chunks are processed in batches of 256 for efficiency (running the model once per batch is much faster than once per chunk).

ChromaDB stores three things per chunk:
- the **embedding** (the 384 floats, used for similarity search)
- the **document** (the original text string, returned at query time)
- the **id** (a simple string index, required by ChromaDB)

The database is written to `./chroma_db` on disk. On the next notebook run, `collection.count() > 0` short-circuits the indexing, so you pay this cost only once.

---

### Step 4: Retrieval

```python
def retrieve(query: str, k: int = 5) -> list[str]:
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k)
    return results["documents"][0]
```

The question is embedded using the same model that was used at index time. This is critical: the query vector and the document vectors must live in the same embedding space.

ChromaDB then finds the `k` stored vectors with the highest cosine similarity to the query vector and returns their original text strings.

`k=5` means we retrieve five passages. These are the "open book pages" handed to the LLM.

---

### Step 5: Generation

```python
SYSTEM_PROMPT = "You are a Harry Potter expert. Answer concisely based only on the provided context."

def rag_query(question: str, k: int = 5) -> str:
    context = "\n\n---\n\n".join(retrieve(question, k))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    prompt = generator.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = generator(prompt, max_new_tokens=256)
    return out[0]["generated_text"][len(prompt):].strip()
```

The five retrieved chunks are joined into a single context string, then assembled into a **chat prompt** with three roles:

- `system`: sets the LLM's behaviour ("answer only from context")
- `user`: provides the retrieved context and the actual question

`apply_chat_template` formats this list of messages into the exact string format TinyLlama was trained on — each model family has its own template. Using the wrong format degrades output quality significantly.

The model generates up to 256 new tokens. We then strip the prompt prefix from the output (the `transformers` pipeline returns the full text including the input by default).

---

## 5. Component choices

| Component | Choice | Alternatives | Reason |
|-----------|--------|--------------|--------|
| PDF extraction | PyMuPDF | pdfplumber, pypdf | Fastest, cleanest text extraction for digitally typeset PDFs |
| Tokeniser | tiktoken `cl100k_base` | HuggingFace tokenisers | Simple API, no model to load, consistent token counts |
| Chunking strategy | Fixed-size with overlap | Sentence-based, paragraph-based, recursive | Simplest to reason about; predictable chunk sizes; no dependency on sentence structure |
| Embedding model | `all-MiniLM-L6-v2` (384 dim) | `all-mpnet-base-v2` (768 dim), `text-embedding-3-small` | Small footprint (~90 MB), fast on CPU, strong retrieval quality for the size |
| Vector store | ChromaDB | FAISS, Qdrant, Pinecone | Local persistent storage, no server to run, simple Python API |
| LLM | TinyLlama-1.1B-Chat | Phi-3-mini, Mistral-7B, GPT-4o | Runs on CPU, no API key, small enough for prototyping |

---

## 6. Key parameters and their effects

### `CHUNK_SIZE` (default: 500 tokens)

Controls how much text the LLM receives per retrieved chunk.

| Value | Effect |
|-------|--------|
| Too small (< 100) | Chunks lack context; a sentence about "the castle" with no surrounding text is ambiguous |
| Too large (> 1000) | Embeddings become blurry; one chunk covers multiple unrelated topics; retrieval precision drops |
| 500 | ~375 words; typically 2–4 paragraphs; captures a scene or a focused passage |

### `CHUNK_OVERLAP` (default: 50 tokens)

Controls how much adjacent chunks share.

- **Too low:** meaningful sentences at chunk boundaries get split across two chunks, neither of which has full context.
- **Too high:** more redundant content in the index; marginally increases indexing cost.
- **10% of chunk size** (50/500) is a standard starting point.

### `k` in `retrieve()` (default: 5)

The number of chunks passed to the LLM as context.

| Value | Effect |
|-------|--------|
| Low (1–2) | Fast; risk of missing relevant information |
| High (10+) | More context; but the prompt grows; small LLMs struggle with long contexts |
| 5 | Safe default; ~2,500 tokens of context, well within TinyLlama's window |

### `max_new_tokens` in `generator()` (default: 256)

Hard cap on the generated response length. 256 tokens ≈ ~190 words — enough for a concise answer or a short list of quiz questions. Increase for longer outputs.

---

## 7. Limitations of this design

**Regular chunking ignores document structure.**
Chapters, scene breaks, and dialogue boundaries are invisible to the chunker. A chunk may begin mid-sentence or mid-scene. More sophisticated strategies (paragraph-based, recursive character splitting, or semantic chunking) can address this at the cost of complexity.

**The embedding model determines retrieval quality.**
`all-MiniLM-L6-v2` is fast and compact but was not trained on Harry Potter-specific text. It may fail to retrieve the right chunk for highly specific or paraphrased questions. A larger model (e.g., `all-mpnet-base-v2`) or a domain-fine-tuned one would improve this.

**TinyLlama is a small model.**
At 1.1B parameters, it will sometimes give factually wrong answers, miss nuance in the context, or fail to follow instructions precisely. This is a prototyping model. GPT-4o or Claude produces significantly better answers.

**No re-ranking.**
After retrieval, the top-k chunks are used as-is by insertion order. A re-ranker (a second model that scores chunk relevance given the query) can substantially improve which passages reach the LLM, at extra compute cost.

**No conversation memory.**
Each call to `rag_query()` is stateless. Follow-up questions ("What did he do next?") have no access to the prior turn.

---

## 8. Improvements

HP specific vocabulary: requires vocabulary extension + LLMs finetuning