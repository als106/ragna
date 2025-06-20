# processing/metadata.py
from langchain.schema.document import Document
from typing import List


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """Asigna un ID único a cada chunk basado en su fuente, página y posición."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks