"""
build_index.py — One-time script to extract, chunk, embed, and index harrypotter.pdf.

Usage:
    python build_index.py
    python build_index.py --force
    python build_index.py --strategy semantic --collection hp_semantic --db ./chroma_semantic

The index is written to ./chroma_db and persists across runs.
Re-running is safe: the script skips indexing if the collection already exists.
"""

import argparse

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_OVERLAP   = 50    # tokens
CHUNK_SIZE      = 500   # tokens
DEFAULT_DB      = "./chroma_db"
DEFAULT_COLLECTION = "hp_books"
EMBED_MODEL     = "all-MiniLM-L6-v2"
PDF_PATH        = "harrypotter.pdf"


def main(args) -> None:
    print("=== HP Quiz — index builder ===\n")

    client = chromadb.PersistentClient(path=args.db)
    collection = client.get_or_create_collection(args.collection)

    if collection.count() > 0 and not args.force:
        print(f"Index already built ({collection.count():,} vectors). Use --force to rebuild.")
        return

    if args.force and collection.count() > 0:
        print("--force: dropping existing collection...")
        client.delete_collection(args.collection)

    print("1/3  Loading PDF...")
    docs = PyMuPDFLoader(PDF_PATH).load()
    print(f"  loaded {len(docs)} pages")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("2/3  Chunking...")
    if args.strategy == "semantic":
        splitter = SemanticChunker(embeddings,
                                   breakpoint_threshold_type="percentile",
                                   breakpoint_threshold_amount=95)
        splits = splitter.split_documents(docs)
        print(f"  {len(splits):,} chunks (semantic, 95th-percentile breakpoints)")
    else:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        splits = splitter.split_documents(docs)
        print(f"  {len(splits):,} chunks ({CHUNK_SIZE}-token size, {CHUNK_OVERLAP}-token overlap)")

    print("3/3  Embedding & indexing...")
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        ids=[str(i) for i in range(len(splits))],
        collection_name=args.collection,
        persist_directory=args.db,
    )
    print(f"  done — {len(splits):,} vectors stored")
    print("\nIndex ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the HP Quiz ChromaDB index.")
    parser.add_argument("--force", action="store_true", help="Drop and rebuild the index")
    parser.add_argument("--strategy", choices=["token", "semantic"], default="token",
                        help="Chunking strategy: token (default) or semantic")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help="ChromaDB collection name (default: hp_books)")
    parser.add_argument("--db", default=DEFAULT_DB,
                        help="ChromaDB persist directory (default: ./chroma_db)")
    args = parser.parse_args()
    main(args)
