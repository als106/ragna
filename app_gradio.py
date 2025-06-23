import gradio as gr
import argparse
import os
import shutil
import logging
import atexit
import subprocess
from main import chat_chain, reranked_retriever, MODEL_NAME
from vector import update_vector_store

CHROMA_PATH = "./chrome_langchain_db"
PROCESSED_LOG = "processed_docs.json"
PY_CACHE_PATH = "./__pycache__"


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logging.info("✅ Base vectorial eliminada.")
    if os.path.exists(PROCESSED_LOG):
        os.remove(PROCESSED_LOG)
        logging.info("✅ Registro eliminado.")
    if os.path.exists(PY_CACHE_PATH):
        shutil.rmtree(PY_CACHE_PATH)
        logging.info("✅ Cache eliminada.")


def chat_rag(message, history):
    question = message["content"] if isinstance(message, dict) else message
    
    # 🔍 Reranked context
    context_docs = reranked_retriever.invoke(question)

    # Generar el contexto concatenado
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    # Invocar al modelo
    response = chat_chain.invoke({
        "context": context_text,
        "question": question
    })
    return {"role": "assistant", "content": response}


def stop_model():
    subprocess.run(["ollama", "stop", MODEL_NAME])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interfaz Gradio para RAG")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Elimina la base vectorial y el log de procesados.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Procesa nuevos documentos del directorio data/",
    )
    args = parser.parse_args()

    if args.reset:
        clear_database()
        exit()

    if args.update:
        update_vector_store()
        exit()

    gr.ChatInterface(
        fn=chat_rag,
        title="RagNa",
        description="Asistente conversacional UA.",
        chatbot=gr.Chatbot(height=600, show_copy_button=True, type="messages"),
        theme=gr.themes.Soft(),
        type="messages",
    ).launch(share=False)

    atexit.register(stop_model)
