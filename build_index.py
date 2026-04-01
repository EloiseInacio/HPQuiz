"""
build_index.py — One-time script to extract, chunk, embed, and index harrypotter.pdf.

Usage:
    python build_index.py

The index is written to ./chroma_db and persists across runs.
Re-running is safe: the script skips indexing if the collection already exists.
"""

import argparse
import fitz  # PyMuPDF
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer

PDF_PATH        = "harrypotter.pdf"
COLLECTION_NAME = "hp_books"
CHROMA_PATH     = "./chroma_db"
EMBED_MODEL     = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 500   # tokens
CHUNK_OVERLAP   = 50    # tokens
BATCH_SIZE      = 256


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    print(f"  extracted {len(text):,} characters from {len(doc)} pages")
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start += chunk_size - overlap
    print(f"  {len(chunks):,} chunks ({chunk_size}-token size, {overlap}-token overlap)")
    return chunks


def build_index(chunks: list[str], collection, embedder: SentenceTransformer) -> None:
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
        ids = [str(i + j) for j in range(len(batch))]
        collection.add(embeddings=embeddings, documents=batch, ids=ids)
        done = min(i + BATCH_SIZE, len(chunks))
        print(f"  indexed {done:,} / {len(chunks):,}", end="\r")
    print(f"\n  done — {collection.count():,} vectors stored")


def main(force: bool = False) -> None:
    print("=== HP Quiz — index builder ===\n")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() > 0 and not force:
        print(f"Index already built ({collection.count():,} vectors). Use --force to rebuild.")
        return

    if force and collection.count() > 0:
        print("--force: dropping existing collection...")
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(COLLECTION_NAME)

    print("1/3  Extracting PDF...")
    text = extract_text(PDF_PATH)

    print("2/3  Chunking...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    print("3/3  Embedding & indexing...")
    embedder = SentenceTransformer(EMBED_MODEL)
    build_index(chunks, collection, embedder)

    print("\nIndex ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the HP Quiz ChromaDB index.")
    parser.add_argument("--force", action="store_true", help="Drop and rebuild the index")
    args = parser.parse_args()
    main(force=args.force)
