from fastapi import FastAPI, Request
from pydantic import BaseModel
from vector import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# -------- CONFIGURACIÓN --------
SESSION_ID = "default"
MODEL_NAME = "llama3.2:1b"
app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str = "default"

model = OllamaLLM(model="llama3.2:1b")
prompt = ChatPromptTemplate.from_template("""
Eres un asistente experto en temas universitarios de la Universidad de Alicante (UA).
Usa solo el contexto que se te da. No inventes ni hagas suposiciones.

{context}

Historial:
{history}

Pregunta: {question}
""")

# --- Cadena básica ---
chain = prompt | model

def memory_factory() -> BaseChatMessageHistory:
    return ChatMessageHistory()

chat_with_history = RunnableWithMessageHistory(
    prompt | model,
    lambda session_id: memory_factory(),
    input_messages_key="question",
    history_messages_key="history",
)

@app.post("/chat")
def chat_endpoint(query: Query):
    context = retriever.invoke(query.question)
    response = chat_with_history.invoke(
        {"context": context, "question": query.question},
        config={"configurable": {"session_id": query.session_id}},
    )
    return {"response": response}

