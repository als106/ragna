# vectorstore/cleaner.py
import os
import shutil
import logging

CHROMA_PATH = "chroma"

def clear_database():
    """Elimina la base vectorial almacenada en disco."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logging.info("Base de datos vectorial eliminada correctamente.")
    else:
        logging.warning("No se encontró ninguna base de datos para eliminar.")
