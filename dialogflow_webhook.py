# dialogflow_webhook.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
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
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory="chroma",
    embedding_function=embedding_function
)

model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    device_map="auto",               # Detecta y usa GPU si está disponible
    torch_dtype=torch.float16        # Optimiza memoria en GPU compatible
)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.0
)
llm = HuggingFacePipeline(pipeline=generator)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# --- Endpoint que recibe solicitudes desde Dialogflow ---
@app.post("/webhook")
async def webhook(request: Request, query: Query):
    user_input = query.queryResult["queryText"]
    logging.info(f"🔍 Pregunta del usuario: {user_input}")

    result = qa_chain.invoke({"question": user_input})
    respuesta = result["answer"]
    fuentes = list(dict.fromkeys([
        f"{os.path.basename(doc.metadata.get('source'))} (página {doc.metadata.get('page')})"
        for doc in result.get("source_documents", [])
    ]))

    logging.info(f"🤖 Respuesta: {respuesta}")
    return {
        "fulfillmentText": respuesta,
        "sourceDocuments": fuentes
    }

if __name__ == "__main__":
    uvicorn.run("dialogflow_webhook:app", host="0.0.0.0", port=8080)
