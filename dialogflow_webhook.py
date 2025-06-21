# dialogflow_webhook.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
import uvicorn
import logging
import os
import torch

app = FastAPI()
logging.basicConfig(level=logging.INFO)


# --- Modelo de entrada esperado por Dialogflow ---
class Query(BaseModel):
    queryResult: dict


# --- Configuración del modelo y vectorstore ---
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(persist_directory="chroma", embedding_function=embedding_function)

# Descarga el archivo desde Hugging Face Hub
model_path = hf_hub_download(
    repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
    filename="Llama-3.2-1B-Instruct-Q8_0.gguf"
)

# --- Cargar modelo GGUF con llama-cpp ---
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repeat_penalty=1.1,
    min_p=0.05,
    # max_tokens=512,
    n_ctx=4096,
    n_gpu_layers=999,      # Fuerza uso total de VRAM
    n_batch=512,           # Procesa tokens en lote (muy importante)
    n_threads=4,
    f16_kv=True,           # Cache de atención en FP16
    use_mmap=False,        # Evita memory mapping para mantener en RAM
    use_mlock=True,        # Bloquea el modelo en memoria
    use_flash_attention_2=True,  # Activa flash attention (si está compilado)
    verbose=True
)


memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    output_key="answer",
)


# --- Endpoint que recibe solicitudes desde Dialogflow ---
@app.post("/webhook")
async def webhook(request: Request, query: Query):
    user_input = query.queryResult["queryText"]
    logging.info(f"🔍 Pregunta del usuario: {user_input}")

    result = qa_chain.invoke({"question": user_input})
    respuesta = result["answer"]
    fuentes = list(
        dict.fromkeys(
            [
                f"{os.path.basename(doc.metadata.get('source'))} (página {doc.metadata.get('page')})"
                for doc in result.get("source_documents", [])
            ]
        )
    )

    logging.info(f"🤖 Respuesta: {respuesta}")
    return {"fulfillmentText": respuesta, "sourceDocuments": fuentes}


if __name__ == "__main__":
    uvicorn.run("dialogflow_webhook:app", host="0.0.0.0", port=8080)
