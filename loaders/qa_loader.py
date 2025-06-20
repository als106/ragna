# loaders/qa_loader.py

import os
import logging
import pandas as pd
from langchain.schema import Document

def load_all_csv_documents(data_path: str) -> list[Document]:
    """
    Carga todos los archivos CSV del directorio especificado y los convierte
    en objetos Document con formato PREGUNTA/RESPUESTA.
    """
    documents = []
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

    if not csv_files:
        logging.warning("No se encontraron archivos CSV en el directorio: %s", data_path)
        return []

    logging.info("📂 %d archivo(s) CSV encontrados en %s", len(csv_files), data_path)

    for filename in csv_files:
        csv_path = os.path.join(data_path, filename)
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")

            if "question" not in df.columns or "answer" not in df.columns:
                logging.warning("⚠️ El archivo %s no contiene columnas 'question' y 'answer'", filename)
                continue

            for _, row in df.iterrows():
                question = str(row.get("question", "")).strip()
                answer = str(row.get("answer", "")).strip()
                if question and answer:
                    content = f"PREGUNTA: {question}\nRESPUESTA: {answer}"
                    doc = Document(
                        page_content=content,
                        metadata={"source": filename, "type": "qa_pair"}
                    )
                    documents.append(doc)

            logging.info("✅ %d pares Q&A cargados desde %s", len(df), filename)

        except Exception as e:
            logging.error("❌ Error al procesar %s: %s", filename, str(e))

    logging.info("📄 Total de documentos Q&A cargados: %d", len(documents))
    return documents
