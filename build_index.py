"""
build_index.py — One-time script to extract, chunk, embed, and index harrypotter.pdf.

Usage:
    python build_index.py
    python build_index.py --force

The index is written to ./chroma_db and persists across runs.
Re-running is safe: the script skips indexing if the collection already exists.
"""

import argparse

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_OVERLAP   = 50    # tokens
CHUNK_SIZE      = 500   # tokens
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "hp_books"
EMBED_MODEL     = "all-MiniLM-L6-v2"
PDF_PATH        = "harrypotter.pdf"


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

    print("1/3  Loading PDF...")
    docs = PyMuPDFLoader(PDF_PATH).load()
    print(f"  loaded {len(docs)} pages")

    print("2/3  Chunking...")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(docs)
    print(f"  {len(splits):,} chunks ({CHUNK_SIZE}-token size, {CHUNK_OVERLAP}-token overlap)")

    print("3/3  Embedding & indexing...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        ids=[str(i) for i in range(len(splits))],
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )
    print(f"  done — {len(splits):,} vectors stored")
    print("\nIndex ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the HP Quiz ChromaDB index.")
    parser.add_argument("--force", action="store_true", help="Drop and rebuild the index")
    args = parser.parse_args()
    main(force=args.force)
