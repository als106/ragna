# build_index.py
from loaders import load_documents, load_all_csv_documents
from processing import split_documents, calculate_chunk_ids
from embeddings import get_embedding_function
from vectorstore import add_to_chroma, clear_database

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reinicia la base de datos.")
    args = parser.parse_args()

    if args.reset:
        clear_database()

    documents = load_documents()
    csv_documents = load_all_csv_documents("data")
    documents.extend(csv_documents)
    
    if not documents:
        return

    chunks = split_documents(documents)
    chunks = calculate_chunk_ids(chunks)
    add_to_chroma(chunks, get_embedding_function())

if __name__ == "__main__":
    main()
