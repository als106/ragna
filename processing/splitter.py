# processing/splitter.py

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: list[Document]) -> list[Document]:
    """Separa los documentos en chunks de tamaño fijo"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, is_separator_regex=False, length_function=len
    )

    return splitter.split_documents(documents)
