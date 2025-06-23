import gradio as gr
import argparse
import os
import shutil
import logging
import atexit
import subprocess
from main import chat_with_history, retriever, SESSION_ID
from vector import update_vector_store

CHROMA_PATH = "./chrome_langchain_db"
PROCESSED_LOG = "processed_docs.json"

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logging.info("✅ Base vectorial eliminada.")
    if os.path.exists(PROCESSED_LOG):
        os.remove(PROCESSED_LOG)
        logging.info("✅ Registro eliminado.")

def chat_rag(message, history):
    question = message["content"] if isinstance(message, dict) else message
    context = retriever.invoke(question)
    response = chat_with_history.invoke(
        {"context": context, "question": question},
        config={"configurable": {"session_id": SESSION_ID}},
    )
    return {"role": "assistant", "content": response}


def stop_model():
    subprocess.run(["ollama", "stop", "llama3.2:1b"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interfaz Gradio para RAG")
    parser.add_argument("--reset", action="store_true", help="Elimina la base vectorial y el log de procesados.")
    parser.add_argument("--update", action="store_true", help="Procesa nuevos documentos del directorio data/")
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
        type="messages"
    ).launch(share=True)
    
    atexit.register(stop_model)
