# loaders/pdf_loader.py
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from typing import List
import os

DATA_PATH = "data"


def load_documents() -> List[Document]:
    """Carga documentos PDF desde el directorio especificado."""
    try:
        loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()
        pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

        if not documents:
            logging.warning("No se encontraron documentos PDF en el directorio: %s", DATA_PATH)
        else:
            logging.info("%d archivo(s) PDF cargado(s) desde %s", len(pdf_files), DATA_PATH)

        return documents
    except Exception as e:
        logging.error("Error al cargar documentos: %s", str(e))
        return []
