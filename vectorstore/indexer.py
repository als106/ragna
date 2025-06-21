# vectorstore/indexer.py
import logging
from langchain.schema.document import Document
from langchain_chroma import Chroma
from typing import List

CHROMA_PATH = "chroma"


def add_to_chroma(chunks: List[Document], embedding_function) -> None:
    """Añade documentos a la base vectorial Chroma si aún no existen."""
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items.get("ids", []))
    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]

    logging.info("Chunks existentes en la base vectorial: %d", len(existing_ids))
    if new_chunks:
        logging.info(
            "Añadiendo %d nuevos chunks a la base vectorial", len(new_chunks)
        )
        new_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
    else:
        logging.info("No hay nuevos documentos para añadir.")
