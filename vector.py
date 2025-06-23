from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
import os
import pandas as pd
import json

# === CONFIGURACIÓN ===
data_dir = "data"
db_location = "./chrome_langchain_db"
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# === EMBEDDINGS Y BASE VECTORIAL ===
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
vector_store = Chroma(
    collection_name="ragna",
    persist_directory=db_location,
    embedding_function=embeddings
)

retriever = VectorStoreRetriever(
    vectorstore=vector_store,
    search_type="mmr",
    search_kwargs={"k": 10}
)


def update_vector_store():
    all_documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    processed_log = "processed_docs.json"
    if os.path.exists(processed_log):
        with open(processed_log, "r") as f:
            processed = set(json.load(f))
    else:
        processed = set()

    for root, dirs, files in os.walk(data_dir):
        tema = os.path.relpath(root, data_dir)

        for file in files:
            path = os.path.join(root, file)
            if path in processed:
                continue

            if file.endswith(".csv"):
                df = pd.read_csv(path)
                for i, row in df.iterrows():
                    doc = Document(
                        page_content = f"Pregunta: {row['Pregunta']}\nRespuesta: {row['Respuesta']}",
                        metadata={"source": file, "tema": tema}
                    )
                    all_documents.append(doc)

            elif file.endswith(".pdf"):
                loader = PyMuPDFLoader(path)
                pdf_docs = loader.load()
                pdf_chunks = splitter.split_documents(pdf_docs)
                for chunk in pdf_chunks:
                    chunk.metadata.update({"source": file, "tema": tema})
                all_documents.extend(pdf_chunks)

            processed.add(path)

    if all_documents:
        vector_store.add_documents(all_documents)
        print(f"{len(all_documents)} documentos añadidos.")
    else:
        print("No hay nuevos documentos para añadir.")

    with open(processed_log, "w") as f:
        json.dump(list(processed), f)